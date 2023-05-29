import concurrent.futures
from concurrent.futures import wait
import numpy as np
import time
class runner(object):
	"""docstring for runner"""
	def __init__(self, envs, agent):
		super(runner, self).__init__()
		self.envs = envs
		self.agent = agent
		self.worker_num = len(self.envs)
		self.state_dim = self.envs[0].state_dim
		self.action_dim = self.envs[0].action_dim
		self.horizon = 50

	def reset(self, bool_eval):
		state_list = np.zeros((self.worker_num, self.state_dim))
		with concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_num) as executor:
			# Start the load operations and mark each future with its URL
			if bool_eval:
				futures = [executor.submit(env.reset_eval) for env in self.envs]
			else:
				futures = [executor.submit(env.reset) for env in self.envs]
			wait(futures)
			for i, future in enumerate(futures):
				state = future.result()
				state_list[i] = state

		return np.array(state_list)

	def step(self, actions):
		next_state_list = np.zeros((self.worker_num, self.state_dim))
		reward_list = np.zeros((self.worker_num))
		done_list = np.zeros((self.worker_num))
		mask_list = np.zeros((self.worker_num))
		with concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_num) as executor:
			# Start the load operations and mark each future with its URL
			futures = [executor.submit(env.step, actions[i]) for i,env in enumerate(self.envs)]
			wait(futures)
			for i, future in enumerate(futures):
				next_state, reward, done, mask = future.result()
				next_state_list[i] = next_state
				reward_list[i] = reward
				done_list[i] = done
				mask_list[i] = mask
		return np.array(next_state_list), np.array(reward_list), np.array(done_list), np.array(mask_list)

	def run(self, evaluate=False):
		log_rewards = np.zeros((self.horizon, self.worker_num))
		state = self.reset()
		for h in range(self.horizon):
			start_time = time.time()
			action = self.agent.get_action(state, evaluate=evaluate)
			next_state, reward, done, mask = self.step(action)
			print("step {}, reward {}, time {}".format(h, reward, time.time()-start_time))
			self.agent.store((state, action, reward, next_state, done, mask))
			state = next_state
			log_rewards[h] = reward
		return self.batch, {'best_reward': log_rewards.max(axis=0).mean()}


		