import numpy as np 
import sys


def load_train(train_input, index_to_word, index_to_tag):
	with open(train_input) as f:
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

def calculate(sentences, word2index, index2word, tag2index, index2tag):
	prior = np.zeros(len(tag2index.keys()))
	trans = np.zeros((len(tag2index.keys()), len(tag2index.keys())))
	emit = np.zeros((len(tag2index.keys()), len(word2index.keys())))
	for sentence in sentences:
		for i, item in enumerate(sentence):
			word, tag = item.split("_")
			word_id, tag_id = word2index[word], tag2index[tag]
			if i == 0:
				prior[tag_id] += 1
			if i != len(sentence) - 1:
				item_next = sentence[i + 1]
				word_next, tag_next = item_next.split("_")
				word_next_id, tag_next_id = word2index[word_next], tag2index[tag_next]
				trans[tag_id][tag_next_id] += 1
			emit[tag_id][word_id] += 1
	prior += 1
	trans += 1
	emit += 1
	prior /= np.sum(prior)
	trans /= np.sum(trans, axis = 1).reshape(len(tag2index.keys()), -1)
	emit /= np.sum(emit, axis = 1).reshape(len(tag2index.keys()), -1)
	return prior, trans, emit

def write_result(data, path):
	with open(path, "w") as f:
		strings = []
		for item in data:
			try:
				item = ["%.20e"%(num) for num in item]
				strings.append(" ".join(item))
			except:
				strings.append(str(item))
		f.write("\n".join(strings))
	return


	



# python learnhmm.py handout/fulldata/trainwords.txt handout/fulldata/index_to_word.txt handout/fulldata/index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt
if __name__ == "__main__":
	args = sys.argv
	_, train_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans = args
	sentences, word2index, index2word, tag2index, index2tag = load_train(train_input, index_to_word, index_to_tag)
	prior, trans, emit = calculate(sentences[:10000], word2index, index2word, tag2index, index2tag)
	write_result(prior, hmmprior)
	write_result(trans, hmmtrans)
	write_result(emit, hmmemit)
