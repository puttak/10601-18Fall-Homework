import sys

def main():
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	lines = read(input_file)
	lines = reverse(lines)
	write(output_file, lines)
	return

def read(input_file):
	file = open(input_file, "r")
	lines = file.readlines()
	return lines

def reverse(lines):
	ans = [lines[- i - 1] for i in range(len(lines))]
	return ans

def write(output_file, lines):
	file = open(output_file, "w")
	for line in lines:
		file.write(line)
	return

if __name__ == "__main__":
	main()

