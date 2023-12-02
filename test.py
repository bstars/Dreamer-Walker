import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from tqdm import tqdm
import imageio

from dreamer import Dreamer
from config import Config
from helpers import np_to_torch, torch_to_np, preprocess_img, postprocess_img
from replay import Episode
from env_wrapper import make_env


class Tester():
	def __init__(self):
		self.dreamer = Dreamer().to(Config.device)

	def load_ckpt(self, path):
		ckpt = torch.load(path, map_location=Config.device)
		self.dreamer.load_state_dict(ckpt)


	def sample_episode(self, render=False):
		env = make_env()
		obs = env.reset()
		obs_prep = preprocess_img(obs['image'])
		done = False
		total_reward = 0
		np.set_printoptions(3)
		epi = Episode()

		e = self.dreamer.rssm.encoder_forward(np_to_torch(obs_prep)[None, ...].float())
		h, s = self.dreamer.rssm.get_init_state(e)

		while not done:
			action = self.dreamer.actor_forward(h, s, deter=True)
			action_np = torch_to_np(action[0])
			# action_np = action_np + np.random.normal(0, Config.exploration_noise, action_np.shape)

			obs_next, reward, done, info = env.step(action_np, render=render)
			total_reward += reward
			epi.append(obs_prep, action_np, reward, done)

			if render:
				print('action: ', action_np, 'reward: ', reward, 'total_reward: ', total_reward)

			obs_prep = preprocess_img(obs_next['image'])
			if done:
				epi.terminal(obs_prep)
				return total_reward, epi
			h = self.dreamer.rssm.deterministic_forward(h, s, action)
			e = self.dreamer.rssm.encoder_forward(np_to_torch(obs_prep)[None, ...].float())
			_, s = self.dreamer.rssm.posterior_forward(h, e)

	def show_path(self):
		env = make_env()
		obs = env.reset()
		obs_prep = preprocess_img(obs['image'])
		done = False
		total_reward = 0

		e = self.dreamer.rssm.encoder_forward(np_to_torch(obs_prep)[None, ...].float())
		h, s = self.dreamer.rssm.get_init_state(e)

		for _ in range(50):
			action = self.dreamer.actor_forward(h, s, deter=True)
			action_np = torch_to_np(action[0])
			# action_np = action_np + np.random.normal(0, Config.exploration_noise, action_np.shape)

			obs_next, reward, done, info = env.step(action_np, render=False)
			total_reward += reward
			obs_prep = preprocess_img(obs_next['image'])
			if done:
				raise ValueError('Episode ends early')
			h = self.dreamer.rssm.deterministic_forward(h, s, action)
			e = self.dreamer.rssm.encoder_forward(np_to_torch(obs_prep)[None, ...].float())
			_, s = self.dreamer.rssm.posterior_forward(h, e)

		hs, states, actions = self.dreamer.rollout_from_hs(h, s, Config.imagine_horizon)

		# rollout a trajectory with dreamer
		obs_pred = self.dreamer.rssm.observation_forward(hs[:,1:,:], states[:,1:,:])
		obs_pred = torch_to_np(obs_pred[0]).transpose(0, 2, 3, 1)
		obs_pred = postprocess_img(obs_pred)

		# rollout a trajectory with env
		obs_true = []
		for a in torch_to_np(actions[0]):
			o, *_ = env.step(a, False)
			obs_true.append(o['image'])
		obs_true = np.stack(obs_true, axis=0).transpose(0, 2, 3, 1)

		fig, (ax1, ax2) = plt.subplots(2, 5)
		for i in range(5):
			ax1[i].set_axis_off()
			ax2[i].set_axis_off()
			ax1[i].imshow(obs_pred[2 * i])
			ax2[i].imshow(obs_true[2 * i])
			ax1[i].set_title('T=%d' % (2 * i + 1))
			ax2[i].set_title('T=%d' % (2 * i + 1))
		plt.axis('off')
		plt.show()


def plot_train_history():
	# tester = Tester()
	historys = []
	num_ckpt = 10
	# for i in tqdm(range(1, num_ckpt + 1)):
	# 	tester.load_ckpt('ckpts/dreamer_%d.pt' % (i*5))
	# 	historys.append(
	# 		[tester.sample_episode(render=False)[0] for _ in range(10)]
	# 	)
	#
	# historys = np.array(historys)
	# savemat('historys.mat', {'historys': historys})
	historys = loadmat('historys.mat')['historys']
	plt.plot(
		np.arange(1, num_ckpt+1, 1) * 100,
		np.mean(historys, axis=1),
		'o--'
	)
	plt.fill_between(
		np.arange(1, num_ckpt+1, 1) * 100,
		np.mean(historys, axis=1) - np.std(historys, axis=1),
		np.mean(historys, axis=1) + np.std(historys, axis=1),
		alpha=0.2
	)
	plt.xlabel('# episodes')
	plt.ylabel('total reward')
	plt.savefig('train_history.png', dpi=300)

def make_gif():
	tester = Tester()
	for i in [10, 30, 50]:
		tester.load_ckpt('ckpts/dreamer_%d.pt' % i)
		_, epi = tester.sample_episode(render=False)
		gif = [ postprocess_img(epi.obs[i].transpose(1, 2, 0)) for i in range(100, 400)]
		imageio.mimsave('%d.gif' % i, gif, duration=50)



if __name__ == '__main__':
	# tester = Tester()
	# tester.load_ckpt('ckpts/dreamer_5.pt')
	# tester.sample_episode(render=True)
	# tester.show_path()
	plot_train_history()
	# make_gif()
