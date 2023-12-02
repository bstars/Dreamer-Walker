import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.distributions import Normal
from torch.distributions.independent import Independent

from config import Config
from helpers import horizontal_forward

class SampleDist:

	def __init__(self, dist, samples=100):
		self._dist = dist
		self._samples = samples

	@property
	def name(self):
		return 'SampleDist'

	def __getattr__(self, name):
		return getattr(self._dist, name)

	def mean(self):
		sample = self._dist.rsample(self._samples)
		return torch.mean(sample, 0)

	def mode(self):
		dist = self._dist.expand((self._samples, *self._dist.batch_shape))
		sample = dist.rsample()
		logprob = dist.log_prob(sample)
		batch_size = sample.size(1)
		feature_size = sample.size(2)
		indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
		return torch.gather(sample, 0, indices).squeeze(0)

	def entropy(self):
		dist = self._dist.expand((self._samples, *self._dist.batch_shape))
		sample = dist.rsample()
		logprob = dist.log_prob(sample)
		return -torch.mean(logprob, 0)

	def sample(self):
		return self._dist.sample()


class VisualDecoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.dense = nn.Sequential(
			nn.Linear(Config.state_dim + Config.gru_dim, 1024), nn.ReLU()
		)

		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(1024, 64, 5, stride=2, padding=0), nn.ReLU(),
			nn.ConvTranspose2d(64, 32, 5, stride=2, padding=0), nn.ReLU(),
			nn.ConvTranspose2d(32, 16, 6, stride=2, padding=0), nn.ReLU(),
			nn.ConvTranspose2d(16, 3, 6, stride=2, padding=0)
		)

	def forward(self, x):
		x = self.dense(x)
		x = x.view(-1, 1024, 1, 1)
		x = self.deconv(x)
		return x

class VisualEncoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(3, 16, 4, stride=2, padding=0), nn.ReLU(),
			nn.Conv2d(16, 32, 4, stride=2, padding=0), nn.ReLU(),
			nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
			nn.Conv2d(64, 64, 4, stride=2, padding=0), nn.ReLU(),
			nn.Flatten(start_dim=1),
			nn.Linear(64*2*2, Config.obs_embed_dim)
		)

	def forward(self, x):
		return self.net(x)

class RSSM(nn.Module):
	def __init__(self):
		super().__init__()

		self.encoder = VisualEncoder()
		self.decoder = VisualDecoder()

		self.grusa = nn.Sequential(
			nn.Linear(Config.state_dim + Config.action_dim, Config.gru_dim), nn.ELU(),
			# nn.Linear(Config.gru_dim, Config.gru_dim), nn.ELU(),
		)
		self.gru = nn.GRUCell(Config.gru_dim, Config.gru_dim)

		self.prior = nn.Sequential(
			nn.Linear(Config.gru_dim, Config.hidden_dim), nn.ELU(),
			nn.Linear(Config.hidden_dim, Config.state_dim * 2)
		)

		self.posterior = nn.Sequential(
			nn.Linear(Config.gru_dim + Config.obs_embed_dim, Config.hidden_dim), nn.ELU(),
			nn.Linear(Config.hidden_dim, Config.state_dim * 2)
		)

		# reward
		self.reward = nn.Sequential(
			nn.Linear(Config.gru_dim + Config.state_dim, Config.hidden_dim), nn.ELU(),
			nn.Linear(Config.hidden_dim, Config.hidden_dim), nn.ELU(),
			# nn.Linear(Config.hidden_dim, Config.hidden_dim), nn.ELU(),
			nn.Linear(Config.hidden_dim, 1)
		)

	def prior_forward(self, h, sample=True):
		x = self.prior(h)
		mean, std = torch.chunk(x, 2, dim=-1)
		std = F.softplus(std) + Config.latent_min_std
		if sample:
			s = torch.randn_like(mean) * std + mean
			return Independent(Normal(mean, std), 1), s
		else:
			return Independent(Normal(mean, std),1)

	def posterior_forward(self, h, e, sample=True):
		x = self.posterior(torch.cat([h, e], dim=-1))
		mean, std = torch.chunk(x, 2, dim=-1)
		std = F.softplus(std) + Config.latent_min_std
		if sample:
			s = torch.randn_like(mean) * std + mean
			return Independent(Normal(mean, std), 1), s
		else:
			return Independent(Normal(mean, std), 1)

	def deterministic_forward(self, h, s, a):
		sa = self.grusa(torch.cat([s, a], dim=-1))
		h = self.gru(sa, h)
		return h

	def get_init_state(self, e):
		"""
		:param e:
		:return:
		"""
		batch, _ = e.shape
		s = torch.zeros(batch, Config.state_dim).to(Config.device)
		a = torch.zeros(batch, Config.action_dim).to(Config.device)
		h = torch.zeros(batch, Config.gru_dim).to(Config.device)
		h = self.deterministic_forward(h, s, a)
		_, s = self.posterior_forward(h, e)
		return h, s

	def encoder_forward(self, obs):
		"""
		:param obs: [batch, img_c, img_h, img_w] or [batch, T, img_c, img_h, img_w]
		:return: [batch, obs_embed_dim] or [batch, T, obs_embed_dim]
		"""
		if len(obs.shape) == 4:
			return self.encoder(obs)
		else:
			return horizontal_forward(self.encoder, obs)

	def observation_forward(self, h, s):
		"""
		:param h: [batch, gru_dim] or [batch, T, gru_dim]
		:param s: [batch, state_dim] or [batch, T, state_dim]
		:return:
		:rtype:
		"""
		hs = torch.cat([h, s], dim=-1)
		if len(hs.shape) == 2:
			return self.decoder(hs)
		else:
			return horizontal_forward(self.decoder, hs)

	def reward_forward(self, h, s):
		"""
		:param h: [batch, gru_dim]
		:param s: [batch, state_dim]
		:return:
		"""
		z = torch.cat([h, s], dim=-1)
		if len(z.shape) == 2:
			return self.reward(z).squeeze(-1)
		else:
			return horizontal_forward(self.reward, z).squeeze(-1)

	def rollout_obs(self, e, a):
		"""
		input:
			(e[0], a[0]) & -> (e[1], a[1]) -> .... -> (e[-2], a[-1]) -> (e[-1])

		output
						& (pri[1], post[1], h[1], s[1]) -> ...
		:param e: [batch, T+1, obs_embed_dim]
		:param a: [batch, T, action_dim]
		:return:
			priors: A list of Normal distributions, len = T
			posteriors: A list of Normal distributions, len = T
			hs: [batch, T, gru_dim]
			states: [batch, T, state_dim]
		"""
		batch, T, _ = a.shape

		priors = []
		posteriors = []
		hs = []
		states = []

		h, s = self.get_init_state(e[:, 0, :])

		for t in range(T):
			h = self.deterministic_forward(h, s, a[:, t, :])
			pri = self.prior_forward(h, sample=False)
			post, s = self.posterior_forward(h, e[:, t+1, :], sample=True)

			hs.append(h.clone())
			states.append(s.clone())
			priors.append(pri)
			posteriors.append(post)

		hs = torch.stack(hs, dim=1)
		states = torch.stack(states, dim=1)
		return hs, states, priors, posteriors