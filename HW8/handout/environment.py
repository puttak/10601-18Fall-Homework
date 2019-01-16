import sys

class Environment:
	def __init__(self, maze_input):
		with open(maze_input, "r") as f:
			lines = f.readlines()
		self.maze = []
		for line in lines:
			self.maze.append([char for char in line[:-1]])
		self.start_point, self.end_point = None, None
		for i, row in enumerate(self.maze):
			for j, char in enumerate(row):
				if char == "S":
					self.start_point = (i, j)
				if char == "G":
					self.end_point = (i, j)

		self.current_x, self.current_y = self.start_point
		self.dx = [0, -1, 0, 1]
		self.dy = [-1, 0, 1, 0]

	def step(self, action):
		x = self.current_x + self.dx[action]
		y = self.current_y + self.dy[action]
		value = None
		if 0 <= x < len(self.maze) and 0 <= y < len(self.maze[0]) and self.maze[x][y] != "*":
			self.current_x = x
			self.current_y = y
		else:
			pass
		is_terminal = (self.current_x, self.current_y) == self.end_point
		reward = -1
		return self.current_x, self.current_y, reward, is_terminal

	def reset(self):
		self.current_x, self.current_y = self.start_point
		


if __name__ == "__main__":
	# python environment.py medium_maze.txt output.feedback action_seq.txt
	_, maze_input, output_file, action_seq_file = sys.argv
	env = Environment(maze_input)
	with open(action_seq_file, "r") as f:
		action_seq = f.readlines()[0].replace("\n", "").split(" ")
	with open(output_file, "w") as f:
		for action in action_seq:
			x, y, reward, is_terminal = env.step(int(action))
			f.write("{} {} {} {}\n".format(x, y, reward, int(is_terminal)))
		

