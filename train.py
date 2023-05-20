from Env import Env
from Agent import PPO
from Runner import runner
import numpy as np
import random

TRAIN_ITER = 300000
EVAL_INTERVAL = 100
SAVE_INTERVAL = 100
NUM_WORKERS = 6

start_iters = 0
model_path = "EDA_iter_{}".format(start_iters)
hidden_size = 1024
train_log = open('train_multi_log.txt', 'a')

Envs = [Env(i) for i in range(1, 1 + NUM_WORKERS)]
agent = PPO(input_dim=Envs[0].state_dim, 
			action_dim=Envs[0].action_dim, 
			hidden_dim=hidden_size,
			gamma=0.99)
runner = runner(Envs, agent, 0.99)
if start_iters > 0:
	agent.load_checkpoint(ckpt_path=model_path, evaluate=True)
v_loss_log = []
r_log = []
for i in range(start_iters + 1, TRAIN_ITER + 1):
	batch, log = runner.run(bool_eval=False)
	loss = agent.learn(batch)
	v_loss_log.append(loss)
	r_log.append(log['return'])

	if i % SAVE_INTERVAL == 0:
		agent.save_checkpoint(env_name="EDA", suffix="iter_{}".format(i))
		print("current iter {}, eval mean return {}, loss {}".format(i, np.mean(r_log), np.mean(v_loss_log)))
		train_log.write("current iter {}, eval mean return {}, loss {}".format(i, np.mean(r_log), np.mean(v_loss_log)))
		train_log.write('\n')
		train_log.flush()