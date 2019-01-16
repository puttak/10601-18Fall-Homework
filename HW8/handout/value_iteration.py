import sys
import time

def load_maze(maze_input):
	with open(maze_input, "r") as f:
		lines = f.readlines()
	maze = []
	for line in lines:
		maze.append([char for char in line[:-1]])
	start_point, end_point = None, None
	for i, row in enumerate(maze):
		for j, char in enumerate(row):
			if char == "S":
				start_point = (i, j)
			if char == "G":
				end_point = (i, j)
	return maze, start_point, end_point

dx = [0, -1, 0, 1]
dy = [-1, 0, 1, 0]
def update_state(maze, value_matrix, i, j, discount_factor):
	new_value = -float('inf')
	action = None
	for move in range(0, 4):
		x = i + dx[move]
		y = j + dy[move]
		value = None
		if 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] != "*":
			value = -1.0 + discount_factor * float(value_matrix[x][y])
		else:
			value = -1.0 + discount_factor * float(value_matrix[i][j])
		if value > new_value:
			new_value = value
			action = move
	return new_value




def update_value(maze, value_matrix, start_point, end_point, discount_factor):
	ans = [[0.0] * len(maze[0]) for i in range(len(maze))]
	for i, row in enumerate(maze):
		for j, char in enumerate(row):
			if (i, j) == end_point:
				continue
			if char == "*":
				continue
			ans[i][j] = update_state(maze, value_matrix, i, j, discount_factor)
	return ans


def value_iteration(maze, start_point, end_point, num_epoch, discount_factor):
	value_matrix = [[0.0] * len(maze[0]) for i in range(len(maze))]
	for epoch in range(num_epoch):
		value_matrix = update_value(maze, value_matrix, start_point, end_point, discount_factor)
	return value_matrix

def get_q_value(maze, value_matrix, start_point, end_point, discount_factor):
	q_value = []
	optimal_action = []
	for i, row in enumerate(maze):
		for j, char in enumerate(row):
			if char == "*":
				continue
			new_value = -float('inf')
			best_action = 0
			for action in range(0, 4):
				if (i, j) == end_point:
					q_value.append([i, j, action, 0.0])
				else:
					x = i + dx[action]
					y = j + dy[action]
					value = None
					if 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] != "*":
						value = -1.0 + discount_factor * float(value_matrix[x][y])
					else:
						value = -1.0 + discount_factor * float(value_matrix[i][j])
					q_value.append([i, j, action, value])
					if value > new_value:
						new_value = value
						best_action = action
			optimal_action.append([i, j, float(best_action)])
	return q_value, optimal_action





if __name__ == "__main__":
	# python value_iteration.py maze1.txt value_output.txt q_value_output.txt policy_output.txt 5 0.9
	_, maze_input, value_file, q_value_file, policy_file, num_epoch, discount_factor = sys.argv
	maze, start_point, end_point = load_maze(maze_input)
	start = time.clock()
	value_matrix = value_iteration(maze, start_point, end_point, int(num_epoch), float(discount_factor))
	elapsed = (time.clock() - start)
	print("Time used:",elapsed)
	q_value, optimal_action = get_q_value(maze, value_matrix, start_point, end_point, float(discount_factor))
	with open(value_file, "w") as f:
		for i, row in enumerate(maze):
			for j, char in enumerate(row):
				if char == "*":
					continue
				f.write("{} {} {}\n".format(i, j, value_matrix[i][j]))
	with open(q_value_file, "w") as f:
		for line in q_value:
			line = [str(item) for item in line]
			f.write("{}\n".format(" ".join(line)))
	with open(policy_file, "w") as f:
		for line in optimal_action:
			line = [str(item) for item in line]
			f.write("{}\n".format(" ".join(line)))






