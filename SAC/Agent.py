import os
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from Model import Critic, Actor
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

class SAC(object):
	def __init__(self, input_dim, action_dim, hidden_dim, gamma, batch_size, buffer_size):

		self.alpha = 0.1
		self.lr = 3e-5
		self.gamma = gamma
		self.eps = 1.0
		self.eps_min = 0.05
		self.target_update_interval = 200
		self.device = torch.device("cuda")
		self.batch_size = batch_size
		self.buffer_size = buffer_size

		self.critic = Critic(input_dim, action_dim, hidden_dim).to(device=self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		self.policy = Actor(input_dim, action_dim, hidden_dim).to(self.device)
		self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
		self.buffer = deque(maxlen=self.buffer_size)
		

	def select_action(self, state, evaluate=False):
		self.eps = max(self.eps * 0.99, self.eps_min)
		state = torch.tensor(state, dtype=torch.float, device=self.device) #bv
		logit = self.policy(state) #ba

		if evaluate:
			action = logit.max(dim=-1)
		else:
			dist = Categorical(logits=logit)
			action = dist.sample()
			rand_logit = torch.ones_like(logit)
			rand_dist = Categorical(logits=rand_logit)
			rand_action = rand_dist.sample()
			if random.random() < self.eps:
				action = rand_action

		return action.detach().cpu().numpy()

	def store(self, data):
		state, action, reward, next_state, done, mask = copy.deepcopy(data)
		for i in range(state.shape[0]):
			s = state[i]
			a = action[i]
			r = reward[i]
			s_next = next_state[i]
			d = done[i]
			m = mask[i]
			transition = (s, a, r, s_next, d, m)
			self.buffer.append(transition)


	def learn(self, n_iter):
		if len(self.buffer) < self.batch_size:
			return 0
		else:
			batch = random.sample(self.buffer, self.batch_size)
			states, actions, rewards, next_state, done, mask = zip(*batch)
			batch = (np.array(states), np.array(actions), np.array(rewards), np.array(next_state), np.array(done), np.array(mask))
			log_data =  self.update_parameters(batch)
			if n_iter % 100 == 0:
				hard_update(self.critic_target, self.critic)
			return log_data

	def update_parameters(self, batch):
		# Sample a batch from memory
		state, action, reward, next_state, done, mask = batch

		state = torch.tensor(state, dtype=torch.float, device=self.device) #bv
		action = torch.tensor(action, dtype=torch.float, device=self.device) #b
		reward = torch.tensor(reward, dtype=torch.float, device=self.device) #b
		next_state = torch.tensor(next_state, dtype=torch.float, device=self.device) #b
		done = torch.tensor(done, dtype=torch.float, device=self.device) #b
		mask = torch.tensor(mask, dtype=torch.float, device=self.device) 

		with torch.no_grad():
			next_logit = self.policy(next_state)
			dist = Categorical(logits=next_logit)
			next_action = dist.sample()
			next_log_pi = dist.log_prob(next_action)
			qf1_next, qf2_next = self.critic_target(next_state)
			qf1_next_target = torch.gather(qf1_next, dim=-1, index=next_action.unsqueeze(-1)).squeeze(-1)
			qf2_next_target = torch.gather(qf2_next, dim=-1, index=next_action.unsqueeze(-1)).squeeze(-1)
			min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi
			next_q_value = reward + (1 - done) * self.gamma * (min_qf_next_target)
		next_q_value = next_q_value.detach()

		qf1, qf2 = self.critic(state)
		qf1_selected = torch.gather(qf1, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
		qf2_selected = torch.gather(qf2, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
		qf1_loss = ((qf1_selected - next_q_value) ** 2) * mask
		qf2_loss = ((qf2_selected - next_q_value) ** 2) * mask
		qf_loss = (qf1_loss.sum() + qf2_loss.sum()) / mask.sum()

		self.critic_optim.zero_grad()
		qf_loss.backward()
		self.critic_optim.step()

		logit = self.policy(state)
		dist = Categorical(logits=logit)
		pi = dist.probs()
		log_pi = pi.log()

		qf1_pi, qf2_pi = self.critic(state)
		min_qf_pi = torch.min(qf1_pi, qf2_pi)

		policy_loss = ((pi * (self.alpha * log_pi - min_qf_pi)).sum(dim=-1) * mask) / mask.sum()

		self.policy_optim.zero_grad()
		policy_loss.backward()
		self.policy_optim.step()

		return qf1_loss.item()

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