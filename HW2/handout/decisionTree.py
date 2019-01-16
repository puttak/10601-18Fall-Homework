import numpy as np
import csv
import math
import sys

class Node:
	def __init__(self, training_data, depth, max_depth, attrs, used_attrs, 
				 msg = ""):
		self.training_data = training_data
		self.depth = depth
		self.max_depth = max_depth
		self.attrs = attrs
		self.used_attrs = used_attrs
		self.split_attr = None
		self.split_attr_values = None
		self.sons = []
		self.msg = msg
		self.label_value_cnt = self.calculate_label_value_cnt(training_data)
		self.majority_vote_result = max(self.label_value_cnt.keys(), key = lambda x: self.label_value_cnt[x])

	def calculate_label_value_cnt(self, training_data):
		label_value = training_data[:, -1]
		label_value_cnt = {}
		for value in label_value:
			if value not in label_value_cnt:
				label_value_cnt[value] = 1
			else:
				label_value_cnt[value] += 1
		return label_value_cnt

	def calculate_margin_entropy(self, label_value_cnt):
		entropy = 0
		for value in label_value_cnt.values():
			entropy += -((value) / sum(label_value_cnt.values())) * math.log2((value) / sum(label_value_cnt.values()))
		return entropy

	def calculate_conditional_entropy(self, attr_label):
		attr_label_cnt = {}
		total_entry_num = attr_label.shape[0]
		for tup in attr_label:
			attr_value = tup[0]
			label = tup[1]
			if attr_value not in attr_label_cnt.keys():
				attr_label_cnt[attr_value] = {}
			if label not in attr_label_cnt[attr_value].keys():
				attr_label_cnt[attr_value][label] = 1
			else:
				attr_label_cnt[attr_value][label] += 1
		conditional_entropy = 0
		for attr_value in attr_label_cnt.keys():
			p_attr = sum(attr_label_cnt[attr_value].values()) / total_entry_num
			specific_entropy = 0
			for value in attr_label_cnt[attr_value].values():
				specific_entropy += -(value / sum(attr_label_cnt[attr_value].values())) * math.log2(value / sum(attr_label_cnt[attr_value].values()))
			conditional_entropy += p_attr * specific_entropy
		return conditional_entropy
		

	def train(self):
		if self.depth == self.max_depth:
			return
		if len(self.attrs) == len(self.used_attrs):
			return
		margin_entropy = self.calculate_margin_entropy(self.label_value_cnt)
		if margin_entropy == 0:
			return
		max_infomation_gain = -1
		split_attr_index = -1
		for i in range(len(self.attrs)):
			if self.attrs[i] not in self.used_attrs:
				conditional_entropy = self.calculate_conditional_entropy(self.training_data[:, [i, -1]])
				if margin_entropy - conditional_entropy > max_infomation_gain:
					split_attr_index = i
					max_infomation_gain = margin_entropy - conditional_entropy
		self.split_attr = self.attrs[split_attr_index]
		self.split_attr_values = list(set(self.training_data[:, split_attr_index]))
		for value in self.split_attr_values:
			son_training_data = []
			for data in self.training_data:
				if data[split_attr_index] == value:
					son_training_data.append(data)
			son_training_data = np.array(son_training_data)
			msg = "{} = {}: ".format(self.split_attr, value)
			self.sons.append(Node(son_training_data, self.depth + 1, self.max_depth, 
								  self.attrs, self.used_attrs + [self.split_attr], msg))
		for son in self.sons:
			son.train()
		return

	def predict(self, attr_dict):
		if self.split_attr == None:
			return self.majority_vote_result
		attr_value = attr_dict[self.split_attr]
		for i in range(len(self.split_attr_values)):
			if attr_value == self.split_attr_values[i]:
				return self.sons[i].predict(attr_dict)
		return self.majority_vote_result


	def print(self):
		print("{}{}".format("| " * self.depth + self.msg, self.label_value_cnt))
		for son in self.sons:
			son.print()
		return
		

class DecisionTree:
	def __init__(self, training_data_file, max_depth):
		self.training_data, self.attributes, self.label = self.parse_training_data_file(training_data_file)
		self.root = Node(self.training_data, 0, max_depth, self.attributes, [])

	def parse_training_data_file(self, training_data_file):
		with open(training_data_file, "r") as csv_file:
			reader = csv.reader(csv_file)
			header = next(reader)
			attributes = header[:-1]
			label = header[-1]
			training_data = np.array([row for row in reader])
		return training_data, attributes, label

	def train(self):
		self.root.train()

	def predict(self, attr_dict):
		return self.root.predict(attr_dict)

	def pretty_print(self):
		self.root.print()


def load_test_data(test_data_path):
	with open(test_data_path, "r") as csv_file:
		reader = csv.reader(csv_file)
		header = next(reader)
		attributes = header[:-1]
		label = header[-1]
		training_data = np.array([row for row in reader])
	attr_dict_list = []
	label_list = []
	for data in training_data:
		attr_dict = {}
		for i in range(len(attributes)):
			attr_dict[attributes[i]] = data[i]
		attr_dict_list.append(attr_dict)
		label_list.append(data[-1])
	return attr_dict_list, label_list


def main():
	train_data_file = sys.argv[1]
	test_data_file = sys.argv[2]
	max_depth = int(sys.argv[3])
	train_predict_output = sys.argv[4]
	test_predict_output = sys.argv[5]
	maxtrx_output = sys.argv[6]

	tree = DecisionTree(train_data_file, max_depth)
	tree.train()
	tree.pretty_print()

	file = open(train_predict_output, "w")
	error_cnt = 0
	attr_dict_list, label_list = load_test_data(train_data_file)
	for label, attr_dict in zip(label_list, attr_dict_list):
		pred = tree.predict(attr_dict)
		if pred != label:
			error_cnt += 1
		file.write("{}\n".format(pred))
	file.close()
	train_error_rate = error_cnt / len(label_list)

	file = open(test_predict_output, "w")
	error_cnt = 0
	attr_dict_list, label_list = load_test_data(test_data_file)
	for label, attr_dict in zip(label_list, attr_dict_list):
		pred = tree.predict(attr_dict)
		if pred != label:
			error_cnt += 1
		file.write("{}\n".format(pred))
	file.close()
	test_error_rate = error_cnt / len(label_list)

	file = open(maxtrx_output, "w")
	file.write("error(train): {}\n".format(train_error_rate))
	file.write("error(test): {}\n".format(test_error_rate))
	file.close()



if __name__ == "__main__":
	main()






