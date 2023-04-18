from Env import Env
from Agent import SAC
import numpy as np
import random

TRAIN_ITER = 1000000
EVAL_INTERVAL = 100
PRINT_INTERVAL = 10
H = 10

eps = 1
start_iters = 0
model_path = "models/EDA_iter_{}".format(start_iters)
hidden_size = 512
batch_size = 2048
buffer_size = 100000
n_augs = 128
log = open('train_multi_log.txt', 'a')

qf1_loss_log = []
qf2_loss_log = []
policy_loss_log = []
eval_reward_log = []

SimEnv = Env()
agent = SAC(input_dim=SimEnv.state_dim, 
            action_space=SimEnv.action_space, 
            hidden_dim=hidden_size, 
            batch_size=batch_size,
            buffer_size=buffer_size)
if start_iters > 0:
	agent.load_checkpoint(ckpt_path=model_path, evaluate=True)
for i in range(start_iters + 1, TRAIN_ITER + 1):
	state = SimEnv.reset()
	for h in range(H):
		if random.random() < eps:
			action = SimEnv.random_action(SimEnv.action_space)
		else:
			action = agent.select_action(state, evaluate=False)
		next_state, reward  = SimEnv.step(action)
		agent.store([state, action, reward, next_state])
		for _ in range(n_augs):
			dummy_state, dummy_action, dummy_reward, dummy_next_state = SimEnv.dummy_step(state, action, next_state)
			agent.store([dummy_state, dummy_action, dummy_reward, dummy_next_state])
		ep_return += reward
		state = next_state
	if i % PRINT_INTERVAL == 0:
		print("current iter {}, weight {}, action {}, reward {}, output {}".format(i, SimEnv.weight, SimEnv.control, reward, SimEnv.output))
	
	qf1_loss, qf2_loss, policy_loss = agent.learn(i)
	qf1_loss_log.append(qf1_loss)
	qf2_loss_log.append(qf2_loss)
	policy_loss_log.append(policy_loss)
	eps = max(0.05, eps * 0.999)

	if i % EVAL_INTERVAL == 0:
		agent.save_checkpoint(env_name="EDA", suffix="iter_{}".format(i))
		for idx in range(len(SimEnv.eval_list)):
			state = SimEnv.reset_eval(idx)
			for _ in range(H):
				action = agent.select_action(state, evaluate=True)
				next_state, reward = SimEnv.step(action)
		eval_reward_log.append(reward)
		qf1_loss_log = np.array(qf1_loss_log)
		qf2_loss_log = np.array(qf2_loss_log)
		policy_loss_log = np.array(policy_loss_log)
		eval_reward_log = np.array(eval_reward_log)
		print("current iter {}, qf1 mean loss {}, qf2 mean loss {}, policy mean loss {}, eval mean reward {}".format(
			i,
			np.mean(qf1_loss_log),
			np.mean(qf2_loss_log),
			np.mean(policy_loss_log),
			np.mean(eval_reward_log),))
		log.write("current iter {}, qf1 mean loss {}, qf2 mean loss {}, policy mean loss {}, eval mean reward {}".format(
			i,
			np.mean(qf1_loss_log),
			np.mean(qf2_loss_log),
			np.mean(policy_loss_log),
			np.mean(eval_reward_log),))
		log.write('\n')
		log.flush()
		qf1_loss_log = []
		qf2_loss_log = []
		policy_loss_log = []
		eval_reward_log = []