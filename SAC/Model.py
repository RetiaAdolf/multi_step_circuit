import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
	def __init__(self, num_inputs, hidden_dim):
		super(ValueNetwork, self).__init__()

		self.linear1 = nn.Linear(num_inputs, hidden_dim)
		self.linear2 = nn.Sequential(
					   nn.Linear(hidden_dim, hidden_dim * 2),
					   nn.ELU(),
					   nn.Linear(hidden_dim * 2, hidden_dim))
		self.linear3 = nn.Linear(hidden_dim, 1)

		self.apply(weights_init_)

	def forward(self, state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x


class PPOPolicy(nn.Module):
	def __init__(self, num_inputs, hidden_dim, action_dim):
		super(PPOPolicy, self).__init__()
		self.linear1 = nn.Linear(num_inputs, hidden_dim)
		self.MHA = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=16)
		self.linear2 = nn.Sequential(
					   nn.Linear(hidden_dim, hidden_dim * 2),
					   nn.ELU(),
					   nn.Linear(hidden_dim * 2, hidden_dim))
		self.linear_pi = nn.Linear(hidden_dim, action_dim)
		self.linear_v = nn.Linear(hidden_dim, 1)

		self.apply(weights_init_)

	def forward(self, state, action=None):
		b,l,v = state.shape
		state = state.reshape(b*l, v)
		embed = F.relu(self.linear1(state)).unsqueeze(0)
		x, _ = self.MHA(embed,embed,embed)
		x = x.squeeze(0)
		x = x. reshape(b, l, -1)
		x = F.relu(self.linear2(x))
		logit = self.linear_pi(x)
		value = self.linear_v(x)
		dist = Categorical(logits=logit)
		#print(logit)
		if action is None:
			action = dist.sample()
		else:
			action = action
		log_prob = dist.log_prob(action)

		return action, log_prob, value.squeeze(-1), dist.entropy()


class Critic(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.linear2 = nn.Sequential(
					   nn.Linear(hidden_dim, hidden_dim * 2),
					   nn.ELU(),
					   nn.Linear(hidden_dim * 2, hidden_dim))
		self.linear3 = nn.Linear(hidden_dim, action_dim)

		# Q2 architecture
		self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
		self.linear5 = nn.Sequential(
					   nn.Linear(hidden_dim, hidden_dim * 2),
					   nn.ELU(),
					   nn.Linear(hidden_dim * 2, hidden_dim))
		self.linear6 = nn.Linear(hidden_dim, action_dim)

		self.apply(weights_init_)

	def forward(self, state):
		
		x1 = F.relu(self.linear1(state))
		x1 = F.relu(self.linear2(x1))
		x1 = self.linear3(x1)

		x2 = F.relu(self.linear4(state))
		x2 = F.relu(self.linear5(x2))
		x2 = self.linear6(x2)

		return x1, x2


class Actor(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.linear2 = nn.Sequential(
					   nn.Linear(hidden_dim, hidden_dim * 2),
					   nn.ELU(),
					   nn.Linear(hidden_dim * 2, hidden_dim))
		self.linear3 = nn.Linear(hidden_dim, action_dim)

		self.apply(weights_init_)

	def forward(self, state):
		
		x1 = F.relu(self.linear1(state))
		x1 = F.relu(self.linear2(x1))
		x1 = self.linear3(x1)

		return x1