import numpy as np
import sys
import matplotlib.pyplot as plt
def load_data(data_path):
	with open(data_path, "r") as file:
		lines = file.readlines()
	labels = []
	feats_index = []
	for i, line in enumerate(lines):
		instance = line.split("\t")
		labels.append(int(instance[0]))
		feats_index.append([int(item.split(":")[0]) for item in instance[1:]])
		feats_index[i].append(39176)
	return labels, feats_index

def sparse_dot_product(feat_index, weights):
	ans = 0.0
	for index in feat_index:
		ans += weights[index]
	return ans

def train(feats_index, labels, epochs, lr):
	weights = np.zeros(39177)
	for epoch in range(epochs):
		for feat_index, label in zip(feats_index, labels):
			feat_vec = np.zeros(39177)
			feat_vec[feat_index] = 1.0
			dot_product = sparse_dot_product(feat_index, weights)
			weights += lr * feat_vec * (label - (np.exp(dot_product) / (1 + np.exp(dot_product))))
	return weights

def predict(feats_index, labels, weights, out):
	correct = 0
	file = open(out, "w")
	for feat_index, label in zip(feats_index, labels):
		dot_product = sparse_dot_product(feat_index, weights)
		pos_prob = np.exp(dot_product) / (1 + np.exp(dot_product))
		pred = 1 if pos_prob >= 0.5 else 0
		if pred == label:
			correct += 1
		file.write("{}\n".format(pred))
	file.close()
	return correct / len(feats_index)

def plot(train_feats_index, train_labels, valid_feats_index, valid_labels, epochs, lr):
	weights = np.zeros(39177)
	train_loss = []
	valid_loss = []
	for epoch in range(epochs):
		for feat_index, label in zip(train_feats_index, train_labels):
			feat_vec = np.zeros(39177)
			feat_vec[feat_index] = 1.0
			dot_product = sparse_dot_product(feat_index, weights)
			weights += lr * feat_vec * (label - (np.exp(dot_product) / (1 + np.exp(dot_product))))
		train_loss.append(loss(train_feats_index, weights, train_labels))
		valid_loss.append(loss(valid_feats_index, weights, valid_labels))
	x = np.linspace(0, epochs - 1, epochs)
	plt.xlabel("Epochs")
	plt.ylabel("Negative Log Likelihood")
	plt.plot(x[100:], train_loss[100:], "r", linewidth = 3.0, label = "Training set")
	#plt.plot(x[150:], valid_loss[150:], "b", linewidth = 3.0, label = "Validation set")
	plt.legend(loc='upper right')
	plt.show()
	return

def loss(feats_index, weights, labels):
	loss = 0.0
	for feat_index, label in zip(feats_index, labels):
		dot_product = sparse_dot_product(feat_index, weights)
		loss += (-label * dot_product + np.log(1 + np.exp(dot_product)))
	return loss

def main():
	formatted_train_input = sys.argv[1]
	formatted_validation_input = sys.argv[2]
	formatted_test_input = sys.argv[3]
	dict_input = sys.argv[4]
	train_out = sys.argv[5]
	test_out = sys.argv[6]
	metrics_out = sys.argv[7]
	num_epoch = int(sys.argv[8])

	train_labels, train_feats_index = load_data(formatted_train_input)
	test_labels, test_feats_index = load_data(formatted_test_input)
	weights = train(train_feats_index, train_labels, num_epoch, 0.1)
	train_accuracy = predict(train_feats_index, train_labels, weights, train_out)
	test_accuracy = predict(test_feats_index, test_labels, weights, test_out)

	file = open(metrics_out, "w")
	file.write("error(train): {}\n".format(1 - train_accuracy))
	file.write("error(test): {}\n".format(1 - test_accuracy))
	file.close()

	# train_labels, train_feats_index = load_data(formatted_train_input)
	# valid_labels, valid_feats_index = load_data(formatted_validation_input)
	# plot(train_feats_index, train_labels, valid_feats_index, valid_labels, num_epoch, 0.1)



	


if __name__ == "__main__":
	main()











