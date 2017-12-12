import json
import random
import _pickle as cPickle

from csv import DictReader
from gensim.models import Word2Vec
from math import sqrt
from numpy import random

kDIM = 100
kWIN = 5

if __name__ == '__main__':

	sentences = []

	# add links between named entities
	chance = .8 / kWIN
	wiki_data = list(DictReader(open("../../data/quizbowl/wiki_data.csv", 'r')))

	for wiki in wiki_data:

		title = wiki['title']
		text = []

		for entity in wiki['links']:
			text.append(entity)
			if random.random() < chance:
				text.append(title)

		sentences.append(text)

	print(len(sentences))

	ent_list = json.load(open('../../data/quizbowl/answer_set.json', 'r'))

	model = Word2Vec(sentences, sg=1, size=kDIM, window=kWIN, negative=10, iter=60)

	# initialize with random values
	r = sqrt(6) / sqrt(51)
	word2vec = random.rand(kDIM, len(ent_list)) * 2 * r - r

	for index, word in enumerate(ent_list):
		if word not in model.wv.vocab:
			print(word)
		else:
			word2vec[:, index] = model.wv[word]

	with open('../../data/quizbowl/qb_Wv', 'wb') as outfile:
		cPickle.dump(word2vec, outfile)
