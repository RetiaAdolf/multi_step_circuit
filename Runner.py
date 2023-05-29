import concurrent.futures
from concurrent.futures import wait
import numpy as np
import time
class runner(object):
	"""docstring for runner"""
	def __init__(self, envs, agent, gamma):
		super(runner, self).__init__()
		self.envs = envs
		self.agent = agent
		self.worker_num = len(self.envs)
		self.state_dim = self.envs[0].state_dim
		self.action_dim = self.envs[0].action_dim
		self.horizon = 50
		self.gamma = gamma
		self.batch = {"state": np.zeros((self.horizon, self.worker_num, self.state_dim)),
					  "action": np.zeros((self.horizon, self.worker_num, 1)),
					  "log_prob": np.zeros((self.horizon, self.worker_num, 1)),
					  "return": np.zeros((self.horizon, self.worker_num, 1)),}

	def GetDiscountedReward(self, reward_list, gamma):
		delta_list = np.zeros((self.horizon, self.worker_num))
		delta_list[-1] = reward_list[-1]
		for t in range(len(reward_list) - 2, -1 ,-1):
			delta_list[t] = gamma * delta_list[t + 1] + reward_list[t]
		return delta_list

	def reset(self, bool_eval):
		state_list = []
		with concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_num) as executor:
			# Start the load operations and mark each future with its URL
			if bool_eval:
				futures = [executor.submit(env.reset_eval) for env in self.envs]
			else:
				futures = [executor.submit(env.reset) for env in self.envs]
			wait(futures)
			for future in futures:
				try:
					state = future.result()
					state_list.append(state)
				except Exception as exc:
					print('exception in reset')
		

		self.batch = {"state": np.zeros((self.horizon, self.worker_num, self.state_dim)),
					  "action": np.zeros((self.horizon, self.worker_num)),
					  "log_prob": np.zeros((self.horizon, self.worker_num)),
					  "reward": np.zeros((self.horizon, self.worker_num)),}

		return np.array(state_list)

	def step(self, actions):
		next_state_list = []
		reward_list = []
		with concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_num) as executor:
			# Start the load operations and mark each future with its URL
			futures = [executor.submit(env.step, actions[i]) for i,env in enumerate(self.envs)]
			wait(futures)
			for future in futures:
				if True:
					next_state, reward = future.result()
					next_state_list.append(next_state)
					reward_list.append(reward)
				else:
				#except Exception as exc:
					print('exception in step')
		return np.array(next_state_list), np.array(reward_list)

	def run(self, bool_eval=False):
		rewards = []
		state = self.reset(bool_eval)
		for h in range(self.horizon):
			start_time = time.time()
			action, log_prob = self.agent.get_action(state)
			next_state, reward = self.step(action)
			print("step {}, reward {}, time {}".format(h, reward, time.time()-start_time))
			self.batch['state'][h] = state
			self.batch['action'][h] = action
			self.batch['log_prob'][h] = log_prob
			self.batch['reward'][h] = reward
			state = next_state
		self.batch['return'] = self.GetDiscountedReward(self.batch['reward'], self.gamma)
		return self.batch, {'return': self.batch['return'][0].mean(), 'best_reward': self.batch['reward'].max(axis=0).mean()}


		