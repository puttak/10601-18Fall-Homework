import csv
import math
import sys




def inspect_data(csv_file_path):
	with open(csv_file_path, "r") as csv_file:
		reader = csv.reader(csv_file)
		root_value = [row[-1] for row in reader][1:]
	root_value_cnt = {}
	for value in root_value:
		if value not in root_value_cnt:
			root_value_cnt[value] = 1
		else:
			root_value_cnt[value] += 1
	entropy = calculate_margin_entropy(root_value_cnt.values())
	error_rate = calculate_majority_vote_error(root_value_cnt.values())
	return entropy, error_rate


def calculate_margin_entropy(values):
	entropy = 0
	for value in values:
		entropy += -((value) / sum(values)) * math.log2((value) / sum(values))
	return entropy


def calculate_majority_vote_error(values):
	return 1 - max(values) / sum(values)


def main():
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	entropy, error_rate = inspect_data(input_file)
	output_file = open(output_file, "w")
	output_file.write("entropy: {}\n".format(entropy))
	output_file.write("error: {}\n".format(error_rate))
	output_file.close()


if __name__ == "__main__":
	main()