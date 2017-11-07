# baseline_1 because let's be real. There are gonna be so many versions of these.
# input: question + one of the 4 answers.
# label: 1 if answer is correct, 0 if wrong

# logreg - pick the correct answer for a question to be the answer with highest probability/score
# calculate accuracy based on that

import argparse
import nltk
import numpy as np
import random
import string

from collections import Counter
from csv import DictReader
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

kSEED = 42

class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(
            analyzer='word',
            tokenizer=nltk.word_tokenize,
            max_features=500000)

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % ", ".join(feature_names[top10]))
            print("Neg: %s" % ", ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, ", ".join(feature_names[top10])))


def transform_qa_pair(question, answer):
    # remove punctuations from question and answer (dashes (-) are kept)
    punctuation = string.punctuation.replace('-','')
    question = question.translate(str.maketrans('', '', punctuation))
    answer = answer.translate(str.maketrans('', '', punctuation))

    answer_phrase = '_'.join(answer.split())

    sentence = answer_phrase
    for word in question.split():
        sentence += ' ' + word + '-' + answer_phrase
    return sentence

def transform_data(data, num_wrong_ans, test_flag=False, answer_pool=None):
    """
    Args:
        data: each vector in data is in the form
        (question, correct answer index, answer A, answer B, answer C, answer D)

        num_wrong_ans: number of wrong answers used from answer pool.

        test_flag: true if data is testing data; otherwise, false

        answer_pool: answers to help augment data

    Returns:

        x: transformations of each vector in data
        y: corresponding 0-1 labels of each vector in x
        label: corresponding answer of each vector in x
    """
    x = []
    y = []
    label = []

    def transform_vector(vector,):
        correct_answer = vector['answer' + vector['correctAnswer']]
        for answer in random.sample(answer_pool, num_wrong_ans):
            input_sentence = transform_qa_pair(vector['question'], answer)
            x.append(input_sentence)
            label.append(answer)
            y.append(1) if answer == correct_answer else y.append(0)
        
        #if training data, then add (question, correct answer) pair to x
        if not test_flag:
            input_sentence = transform_qa_pair(vector['question'], correct_answer)
            x.append(input_sentence)
            label.append(correct_answer)
            y.append(1)

    for vector in data:
        transform_vector(vector)

    return x, y, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--limit', type=int, default=-1,
          help="Restrict training to these many examples")

    # Cast to list to keep it all in memory
    data = list(DictReader(open("../data/quizbowl_science/sci_train2.csv", 'r')))

    # list of all answers
    answers = []
    for line in data:
        answers.append(line['answer' + line['correctAnswer']])

    # distribution of answer counts
    stats = Counter(answers)
    answer_pool = list(stats.keys())
    num_answers = len(answer_pool)

    # Split to train and validation.
    train = []
    for line in data[:4000]:
        answer = line['answer' + line['correctAnswer']]
        train.append(line)

    val = []
    for line in data[4000:]:
        answer = line['answer' + line['correctAnswer']]
        val.append(line)

    print("train data", len(train))
    print(len(val))

    num_wrong_ans = 1
    
    # Pre-process data
    x_train_pre, y_train, label_train = transform_data(train, num_wrong_ans, answer_pool=answer_pool)
    x_val_pre, y_val, label_val = transform_data(val, len(answer_pool), True, answer_pool=answer_pool)

    print(len(x_train_pre))
    print(len(x_val_pre))
    
    # get features
    feat = Featurizer()
    print("Featurizering")
    x_train = feat.train_feature(x_train_pre)
    x_val = feat.test_feature(x_val_pre)
    print("Featurizerized")
    
    # train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True, random_state=kSEED)
    lr.fit(x_train, y_train)
    print("training done")
    
    train_acc = lr.score(x_train, y_train)
    print("Train accuracy is " + str(train_acc))

    print("above accuracies don't mean anything but why are they high? One reason is: predict all as 0, that already gives 75% accuracy")

    y_train_probabilities = lr.decision_function(x_train)
    y_val_probabilities = lr.decision_function(x_val)

    # get true training accuracy
    correct_train_answers = []
    for line in train:
        correct_train_answers.append(line['answer' + line['correctAnswer']])

    predicted_train_answers = []
    for i in range(0,len(y_train_probabilities), num_wrong_ans+1):
        index = np.argmax(y_train_probabilities[i: i + num_wrong_ans+1]) # position of max
        predicted_train_answers.append(label_train[i + index])

    true_val_acc = accuracy_score(correct_train_answers, predicted_train_answers)
    print("True train accuracy is " + str(true_val_acc))

    # get true validation accuracy
    correct_test_answers = []
    for line in val:
        correct_test_answers.append(line['answer' + line['correctAnswer']])

    predicted_test_answers = []
    for i in range(0,len(y_val_probabilities), num_answers):
        index = np.argmax(y_val_probabilities[i: i + num_answers]) # position of max
        predicted_test_answers.append(label_val[i + index])

    # true validation accuracy
    true_val_acc = accuracy_score(correct_test_answers, predicted_test_answers)
    print("True val accuracy is " + str(true_val_acc))
    print("This is much better than random guess now :P")

    feat.show_top10(lr, ['Pos', 'Neg'])
