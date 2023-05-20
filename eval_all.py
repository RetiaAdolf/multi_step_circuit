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
model_path = "models/EDA_iter_{}".format(iters)
hidden_size = 512
NUM_WORKERS = 6

Envs = [Env(i) for i in range(1, 1 + NUM_WORKERS)]
agent = PPO(input_dim=Envs[0].state_dim, 
			action_dim=Envs[0].action_dim, 
			hidden_dim=hidden_size,
			gamma=0.99)

best_reward = -9999
best_model = 0
runner = runner(Envs, agent, 0.99)
for i in range(10000, 50000, 100):
	model_path = "models/EDA_iter_{}".format(i)
	try:
		agent.load_checkpoint(ckpt_path=model_path, evaluate=True)
	except:
		continue
	_, log = runner.run(bool_eval=True)
	if log['best_reward'] > best_reward:
		best_reward = log['best_reward']
		best_model = i
print(best_model, best_reward)
'''