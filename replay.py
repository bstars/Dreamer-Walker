import torch
import numpy as np


class Episode():
	def __init__(self):
		self.obs = []
		self.actions = []
		self.rewards = []
		self.terminals = []


	@property
	def length(self):
		return len(self.obs)

	def append(self, obs, act, reward, terminal):
		self.obs.append(obs)
		self.actions.append(act)
		self.rewards.append(reward)
		self.terminals.append(terminal)

	def terminal(self, obs):
		self.obs.append(obs)

		self.obs = np.stack(self.obs, axis=0)
		self.actions = np.stack(self.actions, axis=0)
		self.rewards = np.stack(self.rewards, axis=0)
		self.terminals = np.stack(self.terminals, axis=0)

class Memory():
	def __init__(self, max_size):
		self.max_size = max_size
		self.episodes = []

	def append(self, episode):
		self.episodes.append(episode)
		if len(self.episodes) > self.max_size:
			self.episodes.pop(0)


	def sample(self, batch_size, T):
		episode_idx = np.random.choice(len(self.episodes), batch_size, replace=len(self.episodes) < batch_size)
		T = min(T, *[self.episodes[i].length-2 for i in episode_idx])

		obs, actions, rewards, terminals = [], [], [], []
		for i in episode_idx:
			episode = self.episodes[i]
			idx = np.random.randint(0, episode.length-T-1)
			obs.append(episode.obs[idx: idx+T+1])
			actions.append(episode.actions[idx: idx+T])
			rewards.append(episode.rewards[idx: idx+T])
			terminals.append(episode.terminals[idx: idx+T])
		obs = np.stack(obs, axis=0)
		actions = np.stack(actions, axis=0)
		rewards = np.stack(rewards, axis=0)
		terminals = np.stack(terminals, axis=0)
		return obs, actions, rewards, terminals