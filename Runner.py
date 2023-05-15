import concurrent.futures
class runner(object):
	"""docstring for runner"""
	def __init__(self, envs, agent):
		super(runner, self).__init__()
		self.envs = envs
		self.agent = agent
		self.worker_num = len(self.envs)
		self.state_dim = self.envs[0].state_dim
		self.action_dim = self.envs[0].action_dim
		self.horizon = 25
		self.batch = {"state": np.zeros((self.horizon, self.worker_num, self.state_dim)),
					  "action": np.zeros((self.horizon, self.worker_num, 1)),
					  "log_prob": np.zeros((self.horizon, self.worker_num, 1)),
					  "return": np.zeros((self.horizon, self.worker_num, 1)),}

	def GetDiscountedReward(reward_list, gamma):
		delta_list = np.zeros((self.horizon, self.worker_num, 1))
		delta_list[-1] = reward_list[-1]
		for t in range(len(reward_list) - 2, -1 ,-1):
			delta_list[t] = gamma * delta_list[t + 1] + reward_list[t]
		return delta_list

	def reset(self, bool_eval):
		state_list = []
		with concurrent.futures.ProcessPoolExecutor() as executor:
			if bool_eval:
				ids = [i for i in range(len(self.envs))]
				results = executor.map(lambda x: x[0].reset_eval(index=x[1]), zip(self.envs, ids))
			else:
				results = executor.map(lambda x: x[0].reset(), self.envs)
		for result in results:
			state_list.append(result)

		self.batch = {"state": np.zeros((self.horizon, self.worker_num, self.state_dim)),
					  "action": np.zeros((self.horizon, self.worker_num, 1)),
					  "log_prob": np.zeros((self.horizon, self.worker_num, 1)),
					  "reward": np.zeros((self.horizon, self.worker_num, 1)),}
		return np.array(state_list)

	def step(self, actions):
		next_state_list = []
		reward_list = []
		with concurrent.futures.ProcessPoolExecutor() as executor:
			results = executor.map(lambda x: x[0].step(x[1]), zip(self.envs, actions))
		for result in results:
			next_state, reward = result
			next_state_list.append(next_state)
			reward.append(reward)
		return np.array(next_state_list), np.array(reward_list)

	def run(self, bool_eval=False):
		rewards = []
		state = self.reset(bool_eval)
		for h in range(self.horizon):
			action, log_prob = self.agent.get_action(state)
			next_state, reward = self.step(action)
			self.batch['state'][h] = state
			self.batch['action'][h] = action
			self.batch['log_prob'][h] = log_prob
			self.batch['reward'][h] = reward
			state = next_state
		self.batch['return'] = GetDiscountedReward(self.batch['reward'], 0.99)
		return self.batch, {'return': self.batch['return'][-1].mean()}


		