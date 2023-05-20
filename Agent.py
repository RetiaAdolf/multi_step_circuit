import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from Model import PPOPolicy
from collections import deque
import copy
import random
import numpy as np

def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)

'''
class SAC(object):
	def __init__(self, input_dim, action_space, hidden_dim, batch_size, buffer_size):

		self.alpha = 0.1
		self.lr = 5e-6
		self.gamma = 0.99
		self.target_update_interval = 100
		self.device = torch.device("cuda")

		self.critic = QNetwork(input_dim, action_space.shape[0], hidden_dim).to(device=self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		self.policy = GaussianPolicy(input_dim, action_space.shape[0], hidden_dim, action_space).to(self.device)
		self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
		self.buffer = deque(maxlen=buffer_size)
		self.batch_size = batch_size

	def select_action(self, state, evaluate=False):
		state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
		if evaluate is False:
			action, _, _ = self.policy.sample(state)
		else:
			_, _, action = self.policy.sample(state)
		return action.detach().cpu().numpy()[0].round(2)

	def store(self, data):
		state, action, reward, next_state = copy.deepcopy(data)
		self.buffer.append([state, action, reward, next_state])

	def learn(self, n_iter):
		if len(self.buffer) < self.batch_size:
			return 0, 0, 0
		else:
			batch = random.sample(self.buffer, self.batch_size)
			states, actions, rewards, next_state = zip(*batch)
			batch = (np.array(states), np.array(actions), np.array(rewards), np.array(next_state))
			log_data =  self.update_parameters(batch)
			if n_iter % 100 == 0:
				hard_update(self.critic_target, self.critic)
			return log_data

	def update_parameters(self, batch):
		# Sample a batch from memory
		state_batch, action_batch, reward_batch, next_state_batch = batch

		state_batch = torch.FloatTensor(state_batch).to(self.device)
		action_batch = torch.FloatTensor(action_batch).to(self.device)
		reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
		state_batch = torch.FloatTensor(next_state_batch).to(self.device)

		with torch.no_grad():
			next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
			qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
			min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
			next_q_value = reward_batch + self.gamma * (min_qf_next_target)
		next_q_value = next_q_value.detach()

		qf1, qf2 = self.critic(state_batch, action_batch)
		qf1_loss = F.mse_loss(qf1, next_q_value)
		qf2_loss = F.mse_loss(qf2, next_q_value)
		qf_loss = qf1_loss + qf2_loss

		self.critic_optim.zero_grad()
		qf_loss.backward()
		self.critic_optim.step()

		pi, log_pi, _ = self.policy.sample(state_batch)

		qf1_pi, qf2_pi = self.critic(state_batch, pi)
		min_qf_pi = torch.min(qf1_pi, qf2_pi)

		policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

		self.policy_optim.zero_grad()
		policy_loss.backward()
		self.policy_optim.step()

		return qf1_loss.item(), qf2_loss.item(), policy_loss.item()

	# Save model parameters
	def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
		if not os.path.exists('models/'):
			os.makedirs('models/')
		if ckpt_path is None:
			ckpt_path = "models/{}_{}".format(env_name, suffix)
		print('Saving models to {}'.format(ckpt_path))
		torch.save({'policy_state_dict': self.policy.state_dict(),
					'critic_state_dict': self.critic.state_dict()}, ckpt_path)

	# Load model parameters
	def load_checkpoint(self, ckpt_path, evaluate=False):
		print('Loading models from {}'.format(ckpt_path))
		if ckpt_path is not None:
			checkpoint = torch.load(ckpt_path)
			self.policy.load_state_dict(checkpoint['policy_state_dict'])
			self.critic.load_state_dict(checkpoint['critic_state_dict'])
			if evaluate:
				self.policy.eval()
				self.critic.eval()
			else:
				self.policy.train()
				self.critic.train()
'''


class PPO(object):
	"""docstring for PPO"""
	def __init__(self, input_dim, action_dim, hidden_dim, gamma):
		super(PPO, self).__init__()
		self.lr = 3e-5
		self.gamma = gamma
		self.K = 5
		self.ratio_clip = 0.2
		self.alpha = 0.5
		self.beta = 1e-2
		self.device = torch.device("cuda")

		self.policy = PPOPolicy(input_dim, hidden_dim, action_dim).to(self.device)
		self.optimizer = Adam(self.policy.parameters(), lr=self.lr)

	def get_action(self, state):
		state = torch.tensor(state, dtype=torch.float, device=self.device)
		state = state.unsqueeze(0)
		action, log_prob, _, _ = self.policy(state)
		return action.squeeze(0).detach().cpu().numpy(), log_prob.squeeze(0).detach().cpu().numpy()

	def learn(self, batch):
		old_states = batch['state']
		old_states = torch.tensor(old_states, dtype=torch.float, device=self.device)
		old_actions = batch['action']
		old_actions = torch.tensor(old_actions, dtype=torch.float, device=self.device)
		old_log_probs = batch['log_prob']
		old_log_probs = torch.tensor(old_log_probs, dtype=torch.float, device=self.device)
		rewards = batch['return']
		rewards = torch.tensor(rewards, dtype=torch.float, device= self.device)
		mse_loss = 0
		for _ in range(self.K):
			_, log_probs, state_values, entropys = self.policy(old_states, old_actions)
			ratios = torch.exp(log_probs - old_log_probs)
			advantages = (rewards - state_values).detach()
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1 - self.ratio_clip, 1 + self.ratio_clip) * advantages
			mse_loss_cur = (state_values - rewards)**2
			mse_loss += mse_loss_cur.mean().item()
			loss = - torch.min(surr1, surr2) + self.alpha * mse_loss_cur - self.beta * entropys
			loss = loss.mean()
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
		return mse_loss / self.K

	def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
		if not os.path.exists('models/'):
			os.makedirs('models/')
		if ckpt_path is None:
			ckpt_path = "models/{}_{}".format(env_name, suffix)
		print('Saving models to {}'.format(ckpt_path))
		torch.save({'policy_state_dict': self.policy.state_dict()}, ckpt_path)

	# Load model parameters
	def load_checkpoint(self, ckpt_path, evaluate=False):
		print('Loading models from {}'.format(ckpt_path))
		if ckpt_path is not None:
			checkpoint = torch.load(ckpt_path)
			self.policy.load_state_dict(checkpoint['policy_state_dict'])
			if evaluate:
				self.policy.eval()
			else:
				self.policy.train()