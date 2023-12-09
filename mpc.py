import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions import Normal, TransformedDistribution, TanhTransform, Independent
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from tqdm import tqdm
import imageio

from trainer import lambda_return
from dreamer import Dreamer
from config import Config
from helpers import np_to_torch, torch_to_np, preprocess_img, postprocess_img
from replay import Episode
from env_wrapper import make_env


class MPCPolicy():
	def __init__(self, dreamer):
		self.dreamer = dreamer

	def latent_rollout(self, h, s, actions):
		"""
		:param h: [batch, gru_dim]
		:param s: [batch, state_dim]
		:param actions: [batch, T, action_dim]
		:return:
			hs: [batch, T+1, gru_dim]
			ss: [batch, T+1, state_dim]
		"""
		batch, T, _ = actions.shape
		hs = [h.clone()]
		ss = [s.clone()]

		for t in range(T):
			h = self.dreamer.rssm.deterministic_forward(h, s, actions[:, t, :])
			_, s = self.dreamer.rssm.prior_forward(h, sample=True)
			hs.append(h.clone())
			ss.append(s.clone())

		hs = torch.stack(hs, dim=1)
		ss = torch.stack(ss, dim=1)
		return hs, ss


	@torch.no_grad()
	def act(self, h, s, horizon=20, num_samples=100, num_iteration=15, topk=20):
		"""
		:param h: [1, gru_dim]
		:param s: [1, state_dim]
		:param horizon:
		:param num_samples:
		:param num_iteration:
		:param topk:
		:return:
			[1, action_dim]
		"""

		hs, states, actions = self.dreamer.rollout_from_hs(h, s, horizon)
		actions = actions[0]
		h_extend = h.repeat(num_samples, 1)
		s_extend = s.repeat(num_samples, 1)

		action_mean = actions
		action_std = torch.ones_like(actions) * 0.5

		for _ in range(num_iteration):
			dist = Independent(
				Normal(action_mean, action_std), 1
			)
			actions = dist.sample([num_samples])
			actions = torch.clamp(actions, min=-0.9999, max=0.9999)
			hs_extend, ss_extend = self.latent_rollout(h_extend, s_extend, actions)
			r_pred = self.dreamer.rssm.reward_forward(hs_extend[:, :-1], ss_extend[:, :-1])
			v_pred = self.dreamer.critic_forward(hs_extend, ss_extend)
			G_pred = lambda_return(r_pred, v_pred)[:,0]
			_, idx = torch.topk(G_pred, k=topk)

			action_mean = torch.mean(actions[idx], dim=0)
			action_std = torch.std(actions[idx], dim=0)

		return action_mean[0][None,:]



if __name__ == '__main__':
	dreamer = Dreamer().to(Config.device)
	mpc = MPCPolicy(dreamer)
	mpc.act(
		h = torch.zeros(1, Config.gru_dim).to(Config.device),
		s = torch.zeros(1, Config.state_dim).to(Config.device),
	)

