import gym
import sys
import torch
import numpy as np
from PIL import Image
from dm_control import suite
import cv2

from config import Config



class DeepMindControl:


	def __init__(self):
		self._env = suite.load('walker', 'walk')
		self._camera = dict(quadruped=2).get('walker', 0)
		self._size = (Config.img_w, Config.img_h)
	@property
	def observation_space(self):
		spaces = {}
		for key, value in self._env.observation_spec().items():
			spaces[key] = gym.spaces.Box(
			  -np.inf, np.inf, value.shape, dtype=np.float32)
		spaces['image'] = gym.spaces.Box(
			0, 255, (3,) + self._size , dtype=np.uint8)
		return gym.spaces.Dict(spaces)

	@property
	def action_space(self):
		spec = self._env.action_spec()
		return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

	def step(self, action, render=False):
		time_step = self._env.step(action)
		obs = dict(time_step.observation)
		obs['image'] = self.render().transpose(2, 0, 1).copy()
		if render:
			img = obs['image'].transpose(1, 2, 0)
			img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			cv2.imshow('image', img)
			cv2.waitKey(30)
		reward = time_step.reward or 0
		done = time_step.last()
		info = {'discount': np.array(time_step.discount, np.float32)}
		return obs, reward, done, info

	def reset(self):
		time_step = self._env.reset()
		obs = dict(time_step.observation)
		obs['image'] = self.render().transpose(2, 0, 1).copy()
		return obs

	def render(self, *args, **kwargs):
		if kwargs.get('mode', 'rgb_array') != 'rgb_array':
			raise ValueError("Only render mode 'rgb_array' is supported.")
		return self._env.physics.render(*self._size, camera_id=self._camera)


class TimeLimit:

	def __init__(self, env, duration):
		self._env = env
		self._duration = duration
		self._step = None

	def __getattr__(self, name):
		return getattr(self._env, name)

	def step(self, action, render=False):
		assert self._step is not None, 'Must reset environment.'
		obs, reward, done, info = self._env.step(action, render=render)
		self._step += 1
		if self._step >= self._duration:
			done = True
			if 'discount' not in info:
				info['discount'] = np.array(1.0).astype(np.float32)
			self._step = None
		return obs, reward, done, info

	def reset(self):
		self._step = 0
		return self._env.reset()


class ActionRepeat:

	def __init__(self, env, amount):
		self._env = env
		self._amount = amount

	def __getattr__(self, name):
		return getattr(self._env, name)

	def step(self, action, render=False):
		done = False
		total_reward = 0
		current_step = 0
		while current_step < self._amount and not done:
			obs, reward, done, info = self._env.step(action, render=render)
			total_reward += reward
			current_step += 1
		return obs, total_reward, done, info


class NormalizeActions:

	def __init__(self, env):
		self._env = env
		self._mask = np.logical_and(
			np.isfinite(env.action_space.low),
			np.isfinite(env.action_space.high))
		self._low = np.where(self._mask, env.action_space.low, -1)
		self._high = np.where(self._mask, env.action_space.high, 1)

	def __getattr__(self, name):
		return getattr(self._env, name)

	@property
	def action_space(self):
		low = np.where(self._mask, -np.ones_like(self._low), self._low)
		high = np.where(self._mask, np.ones_like(self._low), self._high)
		return gym.spaces.Box(low, high, dtype=np.float32)

	def step(self, action, render=False):
		original = (action + 1) / 2 * (self._high - self._low) + self._low
		original = np.where(self._mask, original, action)
		return self._env.step(original, render=render)

def make_env():

	env = DeepMindControl()
	env = ActionRepeat(env, Config.repeat_action)
	env = NormalizeActions(env)
	env = TimeLimit(env, Config.time_limit // Config.repeat_action)
	return env