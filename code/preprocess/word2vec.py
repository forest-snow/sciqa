import json
import random
import _pickle as cPickle

from csv import DictReader
from gensim.models import Word2Vec

DIM = 100
WIN = 5

if __name__ == '__main__':

	with open('../../data/quizbowl/qs_train.txt', 'r') as infile:
		sentences = infile.readlines()

	sentences = [s.strip() for s in sentences]

	# add links between named entities
	chance = .8 / WIN
	wiki_data = list(DictReader(open("../../data/quizbowl/wiki_data.csv", 'r')))

	for wiki in wiki_data:

		title = wiki['title']
		text = ''

		for entity in wiki['links']:
			text += ' ' + entity
			if random.random() < chance:
				text += ' ' + title

		sentences.append(text.strip())

	print(len(sentences))

	model = Word2Vec(sentences, sg=1, size=DIM, window=WIN, negative=10, iter=20)

	vocab = cPickle.load(open('../../data/quizbowl/qb_split', 'rb'))[0]

	# initialize with random values
	r = sqrt(6) / sqrt(51)
	word2vec = random.rand(DIM, len(vocab)) * 2 * r - r

	for index, word in enumerate(vocab):
		if word not in model.wv.vocab:
			print(word)
		else:
			word2vec[:, index] = model.wv[word]

	with open('../../data/quizbowl/qb_We', 'wb') as outfile:
		cPickle.dump(word2vec, outfile)
