import sys
from environment import Environment
import random


dx = [0, -1, 0, 1]
dy = [-1, 0, 1, 0]

def q_learning(env, q_table, num_episodes, max_episode_length, learning_rate, discount_factor, epsilon):
	total_step = 0.0
	for i in range(num_episodes):
		env.reset()
		for step in range(max_episode_length):
			i, j = env.current_x, env.current_y
			explore = random.random() <= epsilon
			if explore:
				action = random.randint(0, 3)
			else:
				action = 0
				max_q = -float("inf")
				for move in range(0, 4):
					if q_table[(i, j)][move] > max_q:
						max_q = q_table[(i, j)][move]
						action = move
			i_, j_, reward, is_terminal = env.step(action)
			action_ = 0
			max_q_ = -float("inf")
			for move in range(0, 4):
				if q_table[(i_, j_)][move] > max_q_:
					max_q_ = q_table[(i_, j_)][move]
					action_ = move
			q_table[(i, j)][action] = (1 - learning_rate) * q_table[(i, j)][action] + learning_rate * (reward + discount_factor * q_table[(i_, j_)][action_])
			if is_terminal:
				break
		total_step += float(step)
	print("Average step", total_step / float(num_episodes))
	return q_table


if __name__ == "__main__":
	# python q_learning.py maze1.txt value_output.txt q_value_output.txt policy_output.txt 2000 20 0.1 0.9 0.2
	_, maze_input, value_file, q_value_file, policy_file, num_episodes, max_episode_length, learning_rate, discount_factor, epsilon = sys.argv
	env = Environment(maze_input)
	q_table = {}
	for i in range(len(env.maze)):
		for j in range(len(env.maze[0])):
			if env.maze[i][j] == "*":
				continue
			q_table[(i, j)] = [0.0 for action in range(0, 4)]
	import time
	start = time.clock()
	q_table = q_learning(env, q_table, int(num_episodes), int(max_episode_length), float(learning_rate), float(discount_factor), float(epsilon))
	elapsed = (time.clock() - start)
	print("Time used:",elapsed)
	value = []
	policy = []
	for key in q_table.keys():
		i, j = key
		best_action = 0
		max_q = -float("inf")
		for action in range(0, 4):
			if q_table[key][action] > max_q:
				max_q = q_table[key][action]
				best_action = action
		value.append([i, j, max_q])
		policy.append([i, j, best_action])
	with open(value_file, "w") as f:
		for line in value:
			line = [str(item) for item in line]
			f.write("{}\n".format(" ".join(line)))
	with open(policy_file, "w") as f:
		for line in policy:
			line = [str(item) for item in line]
			f.write("{}\n".format(" ".join(line)))
	with open(q_value_file, "w") as f:
		for key in q_table.keys():
			i, j = key
			for action in range(0, 4):
				f.write("{} {} {} {}\n".format(i, j, action, q_table[key][action]))



