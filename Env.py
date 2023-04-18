import subprocess
import os
import random
import numpy as np
import copy
class Env(object):
	"""docstring for Env"""
	def __init__(self):
		super(Env, self).__init__()
		self.action_space = np.array([[-1.0, 1.0], [-1.0, 1.0], [-0.1, 0.1]])
		self.state_dim = 4 + 3 + 4

		self.eval_list = []
		for _ in range(10):
			_ = self.reset()
			self.eval_list.append([self.weight,self.control,self.output])

	def reset(self):
		_weight = np.random.rand(4)
		_weight = np.exp(_weight)
		_weight = _weight / _weight.sum()
		self.weight = _weight

		_action_space = np.array([[12, 60], [12, 60], [0.00, 0.50]])
		_control = self.random_action(_action_space)
		self.control = _control

		_output = self.__simulator_step__(self.control)
		self.output = _output

		self.state = self.__get_state__(self.weight, self.control, self.output)

		return self.state

	def reset_eval(self, idx):
		eval_instance = self.eval_list[idx]
		self.weight = eval_instance[0]
		self.control = eval_instance[1]
		self.output = eval_instance[2]
		self.state = self.__get_state__(self.weight, self.control, self.output)
		return self.state

	def random_action(self, action_space):
		rand_aciton = []
		for action in action_space:
			rand_aciton.append(round(random.uniform(action[0],action[1]), 2))
		return np.array(rand_aciton)


	def __simulator_step__(self, action):
		M3_W, M7_W, IN_OFST = action
		M3_W = str(M3_W)
		M7_W = str(M7_W)
		IN_OFST = str(IN_OFST)
		file_path = 'data/M3W_{}_M7W_{}_INOFST_{}.txt'.format(M3_W, M7_W, IN_OFST)
		while not os.path.exists(file_path):
			subprocess.run(['make', 'M3_W={}'.format(M3_W), 'M7_W={}'.format(M7_W), 'IN_OFST={}'.format(IN_OFST)], stdout=subprocess.PIPE)
		with open(file_path, 'r') as f:
			data = f.readline().split()
			while not data:
				data = f.readline().split()
			data = f.readline().split()
			PowerDC, GBW, RmsNoise, SettlingTime = self.__read_data__(data)
		f.close()
		return [PowerDC, GBW, RmsNoise, SettlingTime]

	def __read_data__(self, data):
		PowerDC = float(data[4][:-1])
		PowerDC_unit = data[4][-1]
		if PowerDC_unit == "u":
			PowerDC = PowerDC
		elif PowerDC_unit == "n":
			PowerDC = PowerDC * 1e-3
		elif PowerDC_unit == "m":
			PowerDC = PowerDC * 1e3

		GBW = float(data[5][:-1])
		GBW_unit = data[5][-1]
		if GBW_unit == "M":
			GBW = GBW
		elif GBW_unit == "K":
			GBW = GBW * 1e-3
		elif GBW_unit == "G":
			GBW = GBW * 1e3

		RmsNoise = float(data[6][:-1])
		RmsNoise_unit = data[6][-1]
		if RmsNoise_unit == "n":
			RmsNoise = RmsNoise
		elif RmsNoise_unit == "p":
			RmsNoise = RmsNoise * 1e-3
		elif RmsNoise_unit == "u":
			RmsNoise = RmsNoise * 1e3


		SettlingTime = float(data[7][:-1])
		SettlingTime_unit = data[7][-1]
		if SettlingTime_unit == "u":
			SettlingTime = SettlingTime
		elif SettlingTime_unit == "n":
			SettlingTime = SettlingTime * 1e-3
		elif SettlingTime_unit == "m":
			SettlingTime = SettlingTime * 1e3

		return PowerDC, GBW, RmsNoise, SettlingTime

	def __get_state__(self, weight, control, output):
		return np.concatenate([weight, control, output], axis=-1) 

	def __get_reward__(self, weight, output):
		return (weight * output).sum(axis=-1)

	def __normalization__(self, output):
		PowerDC, GBW, RmsNoise, SettlingTime = output

		PowerDC = ((35 - PowerDC) / (35 - 31.8))
		GBW = ((GBW - 1.17) / (10.098753 - 1.175))
		RmsNoise = ((14.7 - RmsNoise) / (14.7 - 11))
		SettlingTime = ((13.5 - SettlingTime) / (13.5 - 1.72))

		return np.array([PowerDC, GBW, RmsNoise, SettlingTime])

	def step(self, action):

		self.control = self.control + action
		self.output = self.__simulator_step__(self.control)
		self.state = self.__get_state__(self.weight, self.control, self.output)

		_normalize_output = self.__normalization__(self.output)

		reward = self.__get_reward__(self.weight, _normalize_output)

		return self.state, reward

	def dummy_step(self, state, action, next_state):

		_state = copy.deepcopy(state)
		_action = copy.deepcopy(action)
		_next_state = copy.deepcopy(next_state)

		_weight = np.random.rand(4)
		_weight = np.exp(_weight)
		_weight = _weight / _weight.sum()
		_state[:4] = _weight
		_next_state[:4] = _weight
		_output = _next_state[-4:]
		_normalize_output = self.__normalization__(_output)
		_reward = self.__get_reward__(_weight, _normalize_output)

		return _state, _action, _reward, _next_state




if __name__ == '__main__':
	test_env = Env()
	state = test_env.reset()
	print(test_env.state)
	print(test_env.weight)
	print(test_env.control)
	print(test_env.output)
	for h in range(10):
		print("------------------{}---------------".format(h))
		action = test_env.random_action(test_env.action_space)
		next_state, reward = test_env.step(action)
		dummy_state, dummy_action, dummy_reward, dummy_next_state = test_env.dummy_step(state, action, next_state)
		print(state, action, reward, next_state)
		print(dummy_state, dummy_action, dummy_reward, dummy_next_state)
		state = next_state
