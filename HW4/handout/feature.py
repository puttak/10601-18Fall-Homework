import numpy as np
import sys

class Model1:
	def __init__(self, train_input, validation_input, test_input, dic_input,
				 formatted_train_out, formatted_validation_out, formatted_test_out):
		self.train_input = train_input
		self.validation_input = validation_input
		self.test_input = test_input
		self.dic_input = dic_input
		self.formatted_train_out = formatted_train_out
		self.formatted_validation_out = formatted_validation_out
		self.formatted_test_out = formatted_test_out
		self.dic = None

	def get_dic(self, dic_path):
		if self.dic != None:
			return self.dic
		file = open(dic_path, "r")
		lines = file.readlines()
		file.close()
		dic = {}
		for line in lines:
			key, value = line.split(" ")
			dic[key] = value[:-1]
		self.dic = dic
		return self.dic

	def read_input(self, input_path):
		file = open(input_path, "r")
		lines = file.readlines()
		file.close()
		word_bags = []
		labels = []
		for line in lines:
			labels.append(line.split("\t")[0])
			word_bags.append(line.split("\t")[1].split(" "))
		return labels, word_bags

	def feature_extract(self, word_bags, dic):
		feats = []
		for word_bag in word_bags:
			feat = []
			for word in word_bag:
				if word in dic.keys() and dic[word] not in feat:
					feat.append(dic[word])
			feats.append(feat)

		return feats

	def formatted_output(self, input_path, output_path):
		dic = self.get_dic(self.dic_input)
		labels, word_bags = self.read_input(input_path)
		feats = self.feature_extract(word_bags, dic)
		file = open(output_path, "w")
		for label, feat in zip(labels, feats):
			line = label
			for value in feat:
				line += "\t{}:1".format(value)
			line += "\n"
			file.write(line)
		file.close()
		return

	def run(self):
		self.formatted_output(self.train_input, self.formatted_train_out)
		self.formatted_output(self.validation_input, self.formatted_validation_out)
		self.formatted_output(self.test_input, self.formatted_test_out)
		return

class Model2:
	def __init__(self, train_input, validation_input, test_input, dic_input,
				 formatted_train_out, formatted_validation_out, formatted_test_out):
		self.train_input = train_input
		self.validation_input = validation_input
		self.test_input = test_input
		self.dic_input = dic_input
		self.formatted_train_out = formatted_train_out
		self.formatted_validation_out = formatted_validation_out
		self.formatted_test_out = formatted_test_out
		self.dic = None
		self.threshold = 4

	def get_dic(self, dic_path):
		if self.dic != None:
			return self.dic
		file = open(dic_path, "r")
		lines = file.readlines()
		file.close()
		dic = {}
		for line in lines:
			key, value = line.split(" ")
			dic[key] = value[:-1]
		self.dic = dic
		return self.dic

	def read_input(self, input_path):
		file = open(input_path, "r")
		lines = file.readlines()
		file.close()
		word_bags = []
		labels = []
		for line in lines:
			labels.append(line.split("\t")[0])
			word_bags.append(line.split("\t")[1].split(" "))
		return labels, word_bags

	def feature_extract(self, word_bags, dic):
		feats = []
		for word_bag in word_bags:
			feat = {}
			for word in word_bag:
				if word in dic.keys():
					if word not in feat.keys():
						feat[word] = 1
					else:
						feat[word] += 1
			feats.append([dic[word] for word in feat if feat[word] < self.threshold])

		return feats

	def formatted_output(self, input_path, output_path):
		dic = self.get_dic(self.dic_input)
		labels, word_bags = self.read_input(input_path)
		feats = self.feature_extract(word_bags, dic)
		file = open(output_path, "w")
		for label, feat in zip(labels, feats):
			line = label
			for value in feat:
				line += "\t{}:1".format(value)
			line += "\n"
			file.write(line)
		file.close()
		return

	def run(self):
		self.formatted_output(self.train_input, self.formatted_train_out)
		self.formatted_output(self.validation_input, self.formatted_validation_out)
		self.formatted_output(self.test_input, self.formatted_test_out)
		return



def main():
	train_input = sys.argv[1]
	validation_input = sys.argv[2]
	test_input = sys.argv[3]
	dic_input = sys.argv[4]
	formatted_train_out = sys.argv[5]
	formatted_validation_out = sys.argv[6]
	formatted_test_out = sys.argv[7]
	feature_flag = sys.argv[8]

	if feature_flag == "1":
		model = Model1(train_input, validation_input, test_input, dic_input, 
					   formatted_train_out, formatted_validation_out, formatted_test_out)
	else:
		model = Model2(train_input, validation_input, test_input, dic_input, 
					   formatted_train_out, formatted_validation_out, formatted_test_out)
	model.run()


if __name__ == "__main__":
	main()