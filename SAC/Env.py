import random
import numpy as np
import copy
import itertools
import time
from Utils import __normalization__, __random_target__
from Utils import Simulator

__input_num__ = 3
__action_num__ = 3
__ouput_num__ = 4

class Env(object):
	"""docstring for Env"""
	def __init__(self, env_id, target_selection):
		super(Env, self).__init__()


		self.Sim = Simulator()
		self.env_id = env_id
		self.action_dim = __input_num__ ** __action_num__
		self.state_dim = __ouput_num__* 2 + __input_num__
		self.target_selection = target_selection

		self.joint_action_mapping = list(itertools.product([-1,0,1],repeat=3))
		
		self.input_range = np.array([[12, 60], [12, 60], [0, 50]])
		self.output_range = np.array([[28.3137, 78.6168], [-11.6272, -0.156108], [1.86478, 183.368], [1.62915, 16.917]])
		#self.target_range = np.array([[31.8, 35], [-10.098753, -1.175], [11, 14.7], [1.72, 13.5]])
		self.target_range = np.array([[31.87, 69.59], [-10.1, -0.6052], [8.24, 125.3], [1.72, 14.62]])

		self.target_output = None
		self.cur_input = None
		self.cur_output = None
		self.done = None
		self.mask = None


		self.eval_targets = [[69.59, -0.6052, 125.3, 14.62],
							 [31.87, -0.6052, 125.3, 14.62],
							 [69.59, -10.100, 125.3, 14.62],
							 [69.59, -0.6052, 8.240, 14.62],
							 [69.59, -0.6052, 125.3, 1.720],
							 [50.59, -5.6052, 65.30, 7.625]]

		self.target_sweep = self.__sweep_target__()

	def __sweep_target__(self):
		PowerDC_vals = np.linspace(self.target_range[0,0], self.target_range[0,1], 10).round(2)
		GBW_vals = np.linspace(self.target_range[1,0], self.target_range[1,1], 10).round(2)
		RmsNoise_vals = np.linspace(self.target_range[2,0], self.target_range[2,1], 10).round(2)
		SettlingTime_vals = np.linspace(self.target_range[3,0], self.target_range[3,1], 10).round(2)
		sweep_target_list = list(itertools.product(PowerDC_vals, GBW_vals, RmsNoise_vals, SettlingTime_vals))
		return sweep_target_list	

	def reset(self, target=None):

		if self.target_selection == "random_sweep":
			target_output = random.choice(self.target_sweep)
		elif self.target_selection == "random_range":
			target_output = __random_target__(self.target_range, "float")
		elif self.target_selection == "fix_sweep":
			target_output = self.eval_targets[self.env_id-1]
		elif self.target_selection == "fix_given":
			target_output = target


		self.target_output = np.array(target_output)
		self.cur_input = np.array([32, 32, 25])
		self.cur_output = self.Sim.step(self.cur_input, self.env_id)
		self.done = 0
		self.mask = 1

		state = self.__get_state__(self.cur_input, self.cur_output, self.target_output)

		return state


	def __get_state__(self, input, output, target):
		normalized_input = __normalization__(input, self.input_range)
		normalized_output = __normalization__(output, self.output_range)
		normalized_target = __normalization__(target, self.output_range)

		return np.concatenate([normalized_input, normalized_output, normalized_target])

	def __get_reward__(self, output, target):
		normalized_output = __normalization__(output, self.output_range)
		normalized_target = __normalization__(target, self.output_range)
		relative_output = normalized_output - normalized_target

		reward = 0
		count = 0

		for element in relative_output:
			if element > 0:
				reward -= element
			else:
				count += 1

		if count < 4:
			reward = 10

		return reward, count

	def __modify_input__(self, cur_input, action_id):
		actions = np.array(self.joint_action_mapping[action_id])
		modified_action = cur_input + actions
		clipped_action = np.clip(modified_action, self.input_range[:, 0], self.input_range[:, 1])
		return clipped_action

	def step(self, action):

		#start_time = time.time()
		self.target_output = self.target_output
		self.cur_input = self.__modify_input__(self.cur_input, action)
		self.cur_output = self.Sim.step(self.cur_input, self.env_id)
		self.done = self.done
		self.mask = 1 - self.done

		state = self.__get_state__(self.cur_input, self.cur_output, self.target_output)
		reward, count = self.__get_reward__(self.cur_output, self.target_output)
		
		if count == 4 and self.done == 0:
			self.done = 1
		
		#print("env {}, time {}, action {}, output {}, reward {}".format(self.env_id, time.time() - start_time, self.cur_input, self.cur_output, reward))

		return state, reward, self.done, self.mask



if __name__ == '__main__':
	input_range = np.array([[12, 60], [12, 60], [0, 50]])
	modified_action = np.array([11, 47, 36])
	print(np.clip(modified_action, input_range[:, 0], input_range[:, 1]))