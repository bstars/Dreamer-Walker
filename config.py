import torch

class Config:

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# env settings
	img_c = 3
	img_w = 64
	img_h = 64
	repeat_action = 2
	time_limit = 1000
	gamma = 0.99
	td_lamb = 0.95

	# model settings
	obs_embed_dim = 1024
	state_dim = 30
	gru_dim = 200
	hidden_dim = 400
	action_dim = 6
	latent_min_std = 0.1
	actor_min_std = 0.1
	actor_init_std = 5


	# training settings
	buffer_size = int(1e6)
	train_seq_len = 50
	imagine_horizon = 15
	batch_size = 8
	rssm_learning_rate = 6e-4
	actor_learning_rate = 8e-5
	value_learning_rate = 8e-5
	free_nats = 3.
	kl_loss_coef = 1.
	grad_clip_norm = 100.
	exploration_noise = 0.3