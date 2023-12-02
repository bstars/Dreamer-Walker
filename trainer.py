import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from dreamer import Dreamer
from helpers import np_to_torch, torch_to_np, FreezeParameters, preprocess_img
from replay import Memory, Episode
from env_wrapper import make_env

def lambda_return(rewards, values):
	"""
	(v(s), r) -> (v(s), r,) -> ... (v(s))

	:param rewards: [batch, T]
	:param values: [batch, T+1]
	:return:
	:rtype:
	"""
	batch, T = rewards.shape

	Gs = [values[:, -1]]
	for t in reversed(range(T)):
		Gs.append(
			rewards[:,t] + (1 - Config.td_lamb) * Config.gamma * values[:,t+1] + Config.td_lamb * Config.gamma * Gs[-1]
		)

	Gs = Gs[1:][::-1]
	Gs = torch.stack(Gs, dim=1)
	return Gs


class Trainer:
	def __init__(self):
		self.dreamer = Dreamer().to(Config.device)
		self.rssm_optim = torch.optim.Adam(self.dreamer.rssm.parameters(), lr=Config.rssm_learning_rate)
		self.actor_optim = torch.optim.Adam(self.dreamer.actor.parameters(), lr=Config.actor_learning_rate)
		self.critic_optim = torch.optim.Adam(self.dreamer.critic.parameters(), lr=Config.value_learning_rate)
		self.env = make_env()
		self.memory = Memory(max_size=Config.buffer_size)

	def sample_random_episode(self):
		epi = Episode()
		obs = self.env.reset()
		done = False
		while not done:
			action = self.env.action_space.sample()
			obs_next, reward, done, info = self.env.step(action, render=False)
			epi.append(preprocess_img(obs['image']), action, reward, done)
			if done:
				epi.terminal(preprocess_img(obs_next['image']))
				self.memory.append(epi)
				return np.sum(epi.rewards)
		self.memory.append(epi)

	@torch.no_grad()
	def sample_episode(self):
		obs = self.env.reset()
		obs_prep = preprocess_img(obs['image'])
		done = False
		epi = Episode()

		e = self.dreamer.rssm.encoder_forward(np_to_torch(obs_prep)[None,...].float())
		h, s = self.dreamer.rssm.get_init_state(e)

		while not done:
			action = self.dreamer.actor_forward(h, s, deter=False)
			action_np = torch_to_np(action[0])
			action_np = action_np + np.random.normal(0, Config.exploration_noise, action_np.shape)

			obs_next, reward, done, info = self.env.step(action_np, render=False)
			epi.append(obs_prep, action_np, reward, done)

			obs_prep = preprocess_img(obs_next['image'])
			if done:
				epi.terminal(obs_prep)
				self.memory.append(epi)
				return np.sum(epi.rewards)
			h = self.dreamer.rssm.deterministic_forward(h, s, action)
			e = self.dreamer.rssm.encoder_forward(np_to_torch(obs_prep)[None,...].float())
			_, s = self.dreamer.rssm.posterior_forward(h, e)

	def train(self):
		for _ in range(3):
			self.sample_random_episode()
		num_ckpt = 0
		free_kl = Config.free_nats * torch.ones(1, device=Config.device).detach()
		for num_iter in range(1, int(1e5)+1):
			o, a, r, t = self.memory.sample(Config.batch_size, Config.train_seq_len)
			o = np_to_torch(o).float()
			a = np_to_torch(a).float()
			r = np_to_torch(r).float()

			# train rssm
			e = self.dreamer.rssm.encoder_forward(o)
			hs, states, priors, posteriors = self.dreamer.rssm.rollout_obs(e, a,)

			obs_pred = self.dreamer.rssm.observation_forward(hs, states)
			r_pred = self.dreamer.rssm.reward_forward(hs, states)

			obs_loss = F.mse_loss(obs_pred, o[:,1:,...], reduction='none').sum(dim=[-1, -2, -3]).mean()
			r_loss = F.mse_loss(r_pred, r)
			kl = torch.stack([kl_divergence(p, q) for p, q in zip(posteriors, priors)])
			# kl_loss = torch.max(kl, free_kl).mean()
			kl_loss = torch.max(kl.mean(), free_kl)
			loss = obs_loss + r_loss + Config.kl_loss_coef * kl_loss

			self.rssm_optim.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.dreamer.rssm.parameters(), Config.grad_clip_norm)
			self.rssm_optim.step()

			# actor
			with torch.no_grad():
				hs = torch.reshape(hs, [-1, Config.gru_dim]).detach()
				states = torch.reshape(states, [-1, Config.state_dim]).detach()

			with FreezeParameters([self.dreamer.rssm, self.dreamer.critic]):
				hs, states, _ = self.dreamer.rollout_from_hs(hs, states, Config.imagine_horizon)
				r_pred = self.dreamer.rssm.reward_forward(hs, states)
				v_pred = self.dreamer.critic_forward(hs, states)
				Gs = lambda_return(r_pred[:, :-1], v_pred)

			actor_loss = -torch.mean(Gs)
			self.actor_optim.zero_grad()
			actor_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.dreamer.actor.parameters(), Config.grad_clip_norm)
			self.actor_optim.step()


			# critic
			v_pred = self.dreamer.critic_forward(hs[:,:-1].detach(), states[:,:-1].detach())
			v_loss = F.mse_loss(v_pred, Gs.detach())
			self.critic_optim.zero_grad()
			v_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.dreamer.critic.parameters(), Config.grad_clip_norm)
			self.critic_optim.step()

			if num_iter % 1 == 0:
				print(
					'iter %d, obs_loss %.2f, r_loss %.4f, kl_loss %.2f, actor_loss %.4f, v_loss %.4f' %
					(num_iter, obs_loss.item(), r_loss.item(), kl.mean().item(), actor_loss.item(), v_loss.item())
				)
			if num_iter % 100 == 0:
				r = self.sample_episode()
				print('reward %.2f' % r)

			if num_iter % 2000 == 0:
				num_ckpt += 1
				torch.save(self.dreamer.state_dict(), './drive/MyDrive/RL/ckpts/dreamer_%d.pt' % (num_ckpt))
				print('model saved')

if __name__ == '__main__':
	trainer = Trainer()
	trainer.train()