import numpy as np 
import sys


def load_hmm_prior(hmmprior):
	with open(hmmprior, "r") as f:
		return np.array([float(item.replace("\n", "")) for item in f.readlines()])

def load_hmm_args(path):
	ans = []
	with open(path, "r") as f:
		lines = [line.replace("\n", "") for line in f.readlines()]
	for line in lines:
		ans.append(np.array([float(num) for num in line.split(" ")]))
	return np.array(ans)

def load_data(test_input, index_to_word, index_to_tag):
	with open(test_input) as f:
		sentences = [item.replace("\n", "").split(" ") for item in f.readlines()]
	with open(index_to_word) as f:
		lines = f.readlines()
		word2index = {item.replace("\n", ""):i for i, item in enumerate(lines)}
		index2word = {i:item.replace("\n", "") for i, item in enumerate(lines)}
	with open(index_to_tag) as f:
		lines = f.readlines()
		tag2index = {item.replace("\n", ""):i for i, item in enumerate(lines)}
		index2tag = {i:item.replace("\n", "") for i, item in enumerate(lines)}
	return sentences, word2index, index2word, tag2index, index2tag

def forwardbackward(words, word2index, index2word, tag2index, index2tag, prior, trans, emit):
	alpha = np.zeros((len(words), len(tag2index.keys())))
	for j in range(0, len(tag2index.keys())):
		alpha[0][j] = prior[j] * emit[j][word2index[words[0]]]
	if alpha.shape[0] > 1:
		alpha[0] /= np.sum(alpha[0])
	for t in range(1, len(words)):
		for j in range(0, len(tag2index.keys())):
			tot = 0.0
			for k in range(0, len(tag2index.keys())):
				tot += alpha[t - 1][k] * trans[k][j]
			alpha[t][j] = emit[j][word2index[words[t]]] * tot
		if t != len(words) - 1:
			alpha[t] /= np.sum(alpha[t])
	log_likelihood = np.log(np.sum(alpha[-1]))
	
	#alpha /= np.sum(alpha, axis = 1).reshape(alpha.shape[0], -1)

	beta = np.zeros((len(words), len(tag2index.keys())))
	# for j in range(0, len(tag2index.keys())):
	# 	beta[-1][j] = 1
	# for t in range(len(words) - 2, -1, -1):
	# 	for j in range(0, len(tag2index.keys())):
	# 		for k in range(0, len(tag2index.keys())):
	# 			beta[t][j] += emit[k][word2index[words[t + 1]]] * beta[t + 1][k] * trans[j][k]
	# beta /= np.sum(beta, axis = 1).reshape(beta.shape[0], -1)

	prob = alpha * beta
	tags_index = np.argmax(prob, axis = 1)
	return [index2tag[index] for index in tags_index], log_likelihood


def run(sentences, word2index, index2word, tag2index, index2tag, prior, trans, emit):
	predict = []
	total_likelihood = 0.0
	total_tags = 0
	correct = 0
	for sentence in sentences:
		words = [item.split("_")[0] for item in sentence]
		labels = [item.split("_")[1] for item in sentence]
		tags, log_likelihood = forwardbackward(words, word2index, index2word, tag2index, index2tag, prior, trans, emit)
		predict.append(["{}_{}".format(word, tag) for word, tag in zip(words, tags)])
		total_likelihood += log_likelihood
		total_tags += len(tags)
		for tag, label in zip(tags, labels):
			if tag == label:
				correct += 1
	return predict, total_likelihood / float(len(sentences)), float(correct) / float(total_tags)





# py forwardbackward.py handout/toydata/toytrain.txt handout/toydata/toy_index_to_word.txt handout/toydata/toy_index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt predicted.txt metrics.txt
if __name__ == "__main__":
	args = sys.argv
	_, test_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans, predicted_file, metric_file = args
	prior, trans, emit = load_hmm_prior(hmmprior), load_hmm_args(hmmtrans), load_hmm_args(hmmemit)
	sentences, word2index, index2word, tag2index, index2tag = load_data(test_input, index_to_word, index_to_tag)
	predict, aver_likelihood, accuracy = run(sentences, word2index, index2word, tag2index, index2tag, prior, trans, emit)
	with open(predicted_file, "w") as f:
		lines = [" ".join(item) for item in predict]
		f.write("\n".join(lines))
	with open(metric_file, "w") as f:
		f.write("Average Log-Likelihood: {}\n".format(aver_likelihood))
		f.write("Accuracy: {}".format(accuracy))

	
