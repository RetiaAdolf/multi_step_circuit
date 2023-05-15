from Env import Env
from Agent import PPO
from Runner import runner
import numpy as np
import random

TRAIN_ITER = 1000000
EVAL_INTERVAL = 100
PRINT_INTERVAL = 10
NUM_WORKERS = 8

eps = 1
start_iters = 0
model_path = "models/EDA_iter_{}".format(start_iters)
hidden_size = 64
log = open('train_multi_log.txt', 'a')

Envs = [Env(i) for i in range(1, 1 + NUM_WORKERS)]
agent = PPO(input_dim=Envs[0].state_dim, 
			action_dim=Envs[0].action_dim, 
			hidden_dim=hidden_size)
runner = runner(Envs, agent)
if start_iters > 0:
	agent.load_checkpoint(ckpt_path=model_path, evaluate=True)
for i in range(start_iters + 1, TRAIN_ITER + 1):
	batch, _ = runner.run(bool_eval=False)
	agent.learn(batch)

	if i % EVAL_INTERVAL == 0:
		agent.save_checkpoint(env_name="EDA", suffix="iter_{}".format(i))
		_, log = runner.run(bool_eval=True)
		print("current iter {}, eval mean return {}".format(i, log['return']))
		log.write("current iter {}, eval mean return {}".format(i, log['return']))
		log.write('\n')
		log.flush()