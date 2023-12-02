import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform, Independent

from config import Config
from helpers import horizontal_forward
from rssm import RSSM, SampleDist


class Dreamer(nn.Module):
	def __init__(self):
		super().__init__()
		self.rssm = RSSM()
		self.actor = nn.Sequential(
			nn.Linear(Config.state_dim + Config.gru_dim, Config.hidden_dim), nn.ELU(),
			nn.Linear(Config.hidden_dim, Config.hidden_dim), nn.ELU(),
			nn.Linear(Config.hidden_dim, Config.hidden_dim), nn.ELU(),
			nn.Linear(Config.hidden_dim, Config.hidden_dim), nn.ELU(),
			nn.Linear(Config.hidden_dim, Config.action_dim * 2)
		)
		self.actor_transform = [TanhTransform()]

		self.critic = nn.Sequential(
			nn.Linear(Config.state_dim + Config.gru_dim, Config.hidden_dim), nn.ELU(),
			nn.Linear(Config.hidden_dim, Config.hidden_dim), nn.ELU(),
			nn.Linear(Config.hidden_dim, Config.hidden_dim), nn.ELU(),
			nn.Linear(Config.hidden_dim, 1)
		)

	def actor_forward(self, h, s, deter=False):
		hs = torch.cat([h, s], dim=-1)
		mu_std = self.actor(hs)
		mu, std = torch.chunk(mu_std, 2, dim=-1)
		mu = torch.tanh(mu)
		std = F.softplus(std + Config.actor_init_std) + Config.actor_min_std
		dist = TransformedDistribution(Normal(mu, std), self.actor_transform)
		dist = Independent(dist, 1)
		dist = SampleDist(dist)
		if deter:
			return dist.mode()
		else:
			return dist.rsample()

	def critic_forward(self, h, s):
		"""
		:param h: [batch, gru_dim] or [batch, T, gru_dim]
		:param s: [batch, state_dim] or [batch, T, state_dim]
		:return:
		"""
		hs = torch.cat([h, s], dim=-1)
		if len(hs.shape) == 2:
			return self.critic(hs).squeeze()
		else:
			return horizontal_forward(self.critic, hs).squeeze()

	def rollout_from_hs(self, h, s, horizon):
		"""
		:param h: [batch, gru_dim]
		:param s: [batch, state_dim]
		:param horizon: int
		:return:
		"""
		hs = [h.clone()]
		states = [s.clone()]
		actions = []

		for _ in range(horizon):
			a = self.actor_forward(h.detach(), s.detach(), deter=True)
			h = self.rssm.deterministic_forward(h, s, a)
			_, s = self.rssm.prior_forward(h)
			hs.append(h.clone())
			states.append(s.clone())
			actions.append(a.clone())

		return torch.stack(hs, dim=1), torch.stack(states, dim=1), torch.stack(actions, dim=1)