from Env import Env
from Agent import PPO
import numpy as np
import random
import sys
# 7000, 24130
'''
w_PowerDC, w_GBW, w_RmsNoise, w_SettlingTime, iters, alpha = sys.argv[1:]
model_path = "./checkpoints/sac_checkpoint_EDA_iter_{}".format(iters)
hidden_size = 256
batch_size = 1024
buffer_size = 50000


state = np.array([float(w_PowerDC), float(w_GBW), float(w_RmsNoise), float(w_SettlingTime)])
alpha = float(alpha)
state = state / alpha
state = (np.exp(state) / np.sum(np.exp(state))) * 10
print(state)
SimEnv = Env()
agent = SAC(input_dim=SimEnv.state_dim, 
            action_space=SimEnv.action_space, 
            hidden_dim=hidden_size, 
            batch_size=batch_size,
            buffer_size=buffer_size)
if iters == "0":
	pass
else:
	agent.load_checkpoint(ckpt_path=model_path, evaluate=True)

action = agent.select_action(state, evaluate=True)
SimEnv.state = state
output, reward = SimEnv.step(action)
M3_W, M7_W, IN_OFST = action
M3_W = str(M3_W)
M7_W = str(M7_W)
IN_OFST = str(IN_OFST)
print("M3_W = {}, M7_W = {}, IN_OFST = {}".format(M3_W, M7_W, IN_OFST))
PowerDC, GBW, RmsNoise, SettlingTime = output
print("PowerDC = {}, GBW = {}, RmsNoise = {}, SettlingTime = {}".format(PowerDC, GBW, RmsNoise, SettlingTime))
print(reward)
'''

from Env import Env
from Agent import PPO
from Runner import runner
import numpy as np
import random

PowerDC, GBW, RmsNoise, SettlingTime, iters = sys.argv[1:]
target_state = np.array([float(PowerDC), float(GBW), float(RmsNoise), float(SettlingTime)])
model_path = "EDA_iter_{}".format(iters)
hidden_size = 1024
NUM_WORKERS = 6
#'''
SimEnv = Env(1)
agent = PPO(input_dim=SimEnv.state_dim, 
			action_dim=SimEnv.action_dim, 
			hidden_dim=hidden_size,
			gamma=0.99)
if int(iters) > 0:
	agent.load_checkpoint(ckpt_path=model_path, evaluate=True)
state = SimEnv.reset_test(target_state)
best_reward = -999999
best_input = None
best_output = None
for i in range(50):
	action, _ = agent.get_action(state[np.newaxis,])
	next_state, reward = SimEnv.step(action[0])
	state = next_state
	print("step {}, input {}, output {}, reward {}".format(i, SimEnv.cur_input, SimEnv.cur_output, reward))
	if reward > best_reward:
		best_reward = reward
		best_input = SimEnv.cur_input
		best_output = SimEnv.cur_output
	if reward == 10:
		break
print("input {}, output {}, reward {}".format(best_input, best_output, best_reward))