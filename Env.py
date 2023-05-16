import subprocess
import signal
import os
import random
import numpy as np
import copy
import itertools
import time
class Env(object):
	"""docstring for Env"""
	def __init__(self, env_id):
		super(Env, self).__init__()
		__input_num__ = 3
		__action_num__ = 3
		__ouput_num__ = 4

		self.env_id = env_id
		print("init env id {}".format(self.env_id))
		self.action_dim = __input_num__ ** __action_num__
		self.state_dim = __ouput_num__* 2 + __input_num__

		self.joint_action_mapping = list(itertools.product([-1,0,1],repeat=3))

		self.input_range = np.array([[12, 60], [12, 60], [0, 50]])
		self.output_range = np.array([[31.8, 35], [-10.098753, -1.175], [11, 14.7], [1.72, 13.5]])

		self.target_output = None
		self.normalized_target_output = None
		self.cur_input = None
		self.cur_output = None
		self.normalized_cur_output = None

		self.eval_targets = []
		for _ in range(6):
			self.eval_targets.append(self.random_target())

	def reset(self):
		self.target_output = self.random_target()

		self.cur_input = np.array([32, 32, 25])#self.random_input()

		self.cur_output = self.__simulator_step__(self.cur_input)

		state = self.__get_state__(self.cur_input, self.cur_output, self.target_output)

		return state


	def reset_eval(self):
		self.target_output = self.eval_targets[self.env_id-1]

		self.cur_input = np.array([32, 32, 25])

		self.cur_output = self.__simulator_step__(self.cur_input)

		state = self.__get_state__(self.cur_input, self.cur_output, self.target_output)

		return state

	def reset_test(self, target_output):
		self.target_output = target_output
		self.cur_input = np.array([32, 32, 25])
		self.cur_output = self.__simulator_step__(self.cur_input)
		state = self.__get_state__(self.cur_input, self.cur_output, self.target_output)
		return state


	def random_target(self):
		rand_output = []
		for output_range in self.output_range:
			rand_output.append(round(random.uniform(output_range[0],output_range[1]), 2))
		return np.array(rand_output)

	def random_input(self):
		rand_input = []
		for input_range in self.input_range:
			rand_input.append(random.randint(input_range[0],input_range[1]))
		return np.array(rand_input)


	def run_command(self, command, makefile_dir):
		with open('/dev/null', 'w') as f:
			process = subprocess.Popen(command, cwd=makefile_dir, stdout=f, stderr=f)
		return process

	def __read_file__(self, file_path):
		with open(file_path, 'r') as f:
			#print(file_path)
			data = f.readline().split()
			while not data:
				data = f.readline().split()
			data = f.readline().split()
		f.close()
		return data

	def __simulator_step__(self, action):
		M3_W, M7_W, IN_OFST = action
		IN_OFST = IN_OFST / 100.0
		M3_W = str(M3_W)
		M7_W = str(M7_W)
		IN_OFST = str(IN_OFST)
		file_path = '../data/M3W_{}_M7W_{}_INOFST_{}.txt'.format(M3_W, M7_W, IN_OFST)
		while not os.path.exists(file_path):
			#command = "make -C /mnt/mydata/RL_{}/run/ M3_W={} M7_W={} IN_OFST={}".format(self.env_id, M3_W, M7_W, IN_OFST)
			command = ["make", "M3_W={}".format(M3_W), "M7_W={}".format(M7_W), "IN_OFST={}".format(IN_OFST)]
			makefile_dir = "/mnt/mydata/RL_{}/run/".format(self.env_id)
			process = self.run_command(command,makefile_dir)
			try:
				process.wait(timeout=100)
			except:
				pgid = os.getpgid(process.pid)
				os.killpg(pgid, signal.SIGTERM)

		data = self.__read_file__(file_path)
		PowerDC, GBW, RmsNoise, SettlingTime = self.__read_data__(data)

		return np.array([PowerDC, GBW, RmsNoise, SettlingTime])

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
		GBW = -GBW

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

	def __get_state__(self, input, output, target):
		normalized_output = self.__normalization__(output)
		normalized_target = self.__normalization__(target)

		return np.concatenate([input, normalized_output, normalized_target]) 

	def __get_reward__(self, output, target):
		normalized_output = self.__normalization__(output)
		normalized_target = self.__normalization__(target)
		reward = -np.abs((normalized_output - normalized_target)).sum()

		return reward if reward < -0.01 else 100

	def __normalization__(self, output):
		output_min = self.output_range[:, 0]
		output_max = self.output_range[:, 1]
		normalized_output = (output - output_min) / (output_max - output_min)

		return normalized_output

	def __modify_input__(self, cur_input, action_id):
		actions = np.array(self.joint_action_mapping[action_id])
		modified_action = cur_input + actions
		clipped_action = np.clip(modified_action, self.input_range[:, 0], self.input_range[:, 1])
		return clipped_action

	def step(self, action):


		self.cur_input = self.__modify_input__(self.cur_input, action)
		start_time = time.time()
		self.cur_output = self.__simulator_step__(self.cur_input)
		print("env {}, time {}".format(self.env_id, time.time() - start_time))
		state = self.__get_state__(self.cur_input, self.cur_output, self.target_output)

		reward = self.__get_reward__(self.cur_output, self.target_output) * 0.1

		return state, reward




if __name__ == '__main__':
	input_range = np.array([[12, 60], [12, 60], [0, 50]])
	modified_action = np.array([11, 47, 36])
	print(np.clip(modified_action, input_range[:, 0], input_range[:, 1]))