import csv
import gensim
import string
import sys

from collections import defaultdict
from csv import DictReader
from gensim import corpora
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer

csv.field_size_limit(sys.maxsize)

kTOPICS = 100

def get_key_words(wiki):
	punctuation = string.punctuation.replace('-', '').replace('_', '')
	categories = wiki['categories'][2:-2].split("\', \'")

	key_words = []
	for cat in categories:
		cat = cat.translate(str.maketrans('', '', punctuation))
		key_words.extend(cat.split('_'))
	return key_words

if __name__ == "__main__":
	# Cast to list to keep it all in memory.
	wiki_data = list(DictReader(open("../data/quizbowl/wiki_data.csv", 'r')))

	documents = []
	for wiki in wiki_data:
		documents.append(get_key_words(wiki))

	# Create the term dictionary of our courpus, where every unique term is assigned an index.
	dictionary = corpora.Dictionary(documents)

	# Convert list of documents (corpus) into Document Term Matrix using dictionary prepared above.
	doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents]

	# Create the object for LDA model using gensim library.
	Lda = gensim.models.ldamodel.LdaModel

	# Train LDA model on the document term matrix.
	print("Training...")
	lda_model = Lda(doc_term_matrix, num_topics=kTOPICS, id2word=dictionary, passes=20)

	print("Saving...")
	lda_model.save('../data/quizbowl/wiki_topics.model')

	topic_entity = defaultdict(list)
	for i, bow in enumerate(doc_term_matrix):
		for topic in lda_model.get_document_topics(bow):
			tid = topic[0]
			topic_entity[tid].append((wiki_data[i]['title'], topic[1]))
	for tid, items in topic_entity.items():
		sorted_list = sorted(items, key=lambda item: item[1], reverse=True)
		print("Topic #%d:" % tid)
		print(', '.join(item[0] + ': ' + str(item[1]) for item in sorted_list[:10]))
