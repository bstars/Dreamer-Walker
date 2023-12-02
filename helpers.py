from typing import Iterable
from torch.nn import Module
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2

from config import Config


############################### torch helpers ###############################

def torch_to_np(x):
	return x.detach().cpu().numpy()

def np_to_torch(x):
	return torch.from_numpy(x).to(Config.device)

def horizontal_forward(func, x):
	batch, T, *others = x.shape
	x = torch.reshape(x, [batch * T, *others])
	x = func(x)
	x = torch.reshape(x, [batch, T, *x.shape[1:]])
	return x

def initialize_weight(m):
	if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
		nn.init.kaiming_uniform_(m.weight.data)
		nn.init.constant_(m.bias.data, 0)

def get_parameters(modules: Iterable[Module]):
	"""
	Given a list of torch modules, returns a list of their parameters.
	:param modules: iterable of modules
	:returns: a list of parameters
	"""
	model_parameters = []
	for module in modules:
		model_parameters += list(module.parameters())
	return model_parameters

class FreezeParameters:
	def __init__(self, modules: Iterable[Module]):
		"""
		Context manager to locally freeze gradients.
		In some cases with can speed up computation because gradients aren't calculated for these listed modules.
		example:
		```
		with FreezeParameters([module]):
		  output_tensor = module(input_tensor)
		```
		:param modules: iterable of modules. used to call .parameters() to freeze gradients.
		"""
		self.modules = modules
		self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

	def __enter__(self):

		for param in get_parameters(self.modules):
			param.requires_grad = False

	def __exit__(self, exc_type, exc_val, exc_tb):
		for i, param in enumerate(get_parameters(self.modules)):
			param.requires_grad = self.param_states[i]


class SampleDist():
	def __init__(self, dist, num_sample=100):
		self._dist = dist
		self.num_sample = num_sample

	@property
	def name(self):
		return 'SampleDist'

	def __getattr__(self, name):
		return getattr(self._dist, name)

	def mode(self):
		samples = self._dist.rsample([self.num_sample])
		log_probs = self._dist.log_prob(samples)
		batch_size = samples.size(1)
		feature_size = samples.size(2)
		indices = torch.argmax(log_probs, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
		return torch.gather(samples, 0, indices).squeeze(0)

	def sample(self):
		return self._dist.sample()


############################### image helpers ###############################
def preprocess_img(img):
	img = img / 255.0 - 0.5
	return img

def postprocess_img(img):
	img = (img + 0.5) * 255.0
	img = np.clip(img, 0, 255).astype(np.uint8)
	return img