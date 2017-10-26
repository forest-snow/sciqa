# baseline_1 because let's be real. There are gonna be so many versions of these.
# input: question + one of the 4 answers.
# label: 1 if answer is correct, 0 if wrong

# logreg - pick the correct answer for a question to be the answer with highest probability/score
# calculate accuracy based on that

import argparse
import nltk
import numpy as np
import random

from csv import DictReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

kSEED = 42

class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(
            analyzer='word',
            tokenizer=nltk.word_tokenize,
            stop_words='english',
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

def transform_QA_pair(question, answer):
    answer_phrase = ''
    for word in answer.split():
        if word.startswith('('):
            word = word[1:]
        if word.endswith(')'):
            word = word[:-1]
        answer_phrase += '.' + word

    sentence = ''
    for word in question:
        sentence += ' ' + word + answer_phrase
    return sentence

def transform_data(data, augmentation=False, answer_pool=None):
    X = []
    y = []
    for x in data:
        question = nltk.word_tokenize(x['question'])
        classifier_input = transform_QA_pair(question, x['answerA'])
        X.append(classifier_input)
        if  x['correctAnswer'] == 'A':
            y.append(1)
        else:
            y.append(0)

        classifier_input = transform_QA_pair(question, x['answerB'])
        X.append(classifier_input)
        if  x['correctAnswer'] == 'B':
            y.append(1)
        else:
            y.append(0)

        classifier_input = transform_QA_pair(question, x['answerC'])
        X.append(classifier_input)
        if  x['correctAnswer'] == 'C':
            y.append(1)
        else:
            y.append(0)

        classifier_input = transform_QA_pair(question, x['answerD'])
        X.append(classifier_input)
        if  x['correctAnswer'] == 'D':
            y.append(1)
        else:
            y.append(0)

        if augmentation:
            answer_list = [x['answerA'], x['answerB'], x['answerC'], x['answerD']]
            for answer in random.sample(answer_pool, 20):
                if answer not in answer_list:
                    classifier_input = transform_QA_pair(question, answer)
                    X.append(classifier_input)
                    y.append(0)

    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--limit', type=int, default=-1,
          help="Restrict training to these many examples")

    # Cast to list to keep it all in memory
    data = list(DictReader(open("../data/quizbowl_science/sci_train.csv", 'r')))
    # Split to train and validation.
    train = data[:4000]
    val = data[4000:]

    """
    answer_pool = set()
    for x in train:
        if x['answerA'] not in answer_pool:
            answer_pool.add(x['answerA'])
        if x['answerB'] not in answer_pool:
            answer_pool.add(x['answerB'])
        if x['answerC'] not in answer_pool:
            answer_pool.add(x['answerC'])
        if x['answerD'] not in answer_pool:
            answer_pool.add(x['answerD'])
    """

    X_train, y_train = transform_data(train, augmentation=False)
    X_val, y_val = transform_data(val, augmentation=False)

    # get features
    feat = Featurizer()
    x_train = feat.train_feature(X_train)
    x_val = feat.test_feature(X_val)

    # train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True, random_state = kSEED)
    lr.fit(x_train, y_train)

    train_acc = lr.score(x_train, y_train)
    val_acc = lr.score(x_val, y_val)
    print("Train accuracy is " + str(train_acc))
    print("Val accuracy is " + str(val_acc))

    print("above accuracies don't mean anything but why are they high? One reason is: predict all as 0, that already gives 75% accuracy")

    y_train_probabilities = lr.decision_function(x_train)
    y_val_probabilities = lr.decision_function(x_val)

    # get true training accuracy
    correct_train_answers = []
    for line in train:
        correct_train_answers.append(line['correctAnswer'])

    predicted_train_answers = []
    for  i in range(0,len(y_train_probabilities),4):
        pos = np.argmax(y_train_probabilities[i:i+4]) # position of max
        if pos == 0:
            predicted_train_answers.append('A')
        elif pos == 1:
            predicted_train_answers.append('B')
        elif pos == 2:
            predicted_train_answers.append('C')
        else:
            predicted_train_answers.append('D')

    indicator_correct = [1 for a, b in zip(correct_train_answers, predicted_train_answers) if a==b]
    true_val_acc = sum(indicator_correct) / len(correct_train_answers)
    print("True train accuracy is " + str(true_val_acc))

    # get true validation accuracy
    correct_test_answers = []
    for line in val:
        correct_test_answers.append(line['correctAnswer'])

    predicted_test_answers = []
    for  i in range(0,len(y_val_probabilities),4):
        pos = np.argmax(y_val_probabilities[i:i+4]) # position of max
        if pos == 0:
            predicted_test_answers.append('A')
        elif pos == 1:
            predicted_test_answers.append('B')
        elif pos == 2:
            predicted_test_answers.append('C')
        else:
            predicted_test_answers.append('D')

    # true validation accuracy
    indicator_correct = [1 for a, b in zip(correct_test_answers, predicted_test_answers) if a==b]
    true_val_acc = sum(indicator_correct) / len(correct_test_answers)
    print("True val accuracy is " + str(true_val_acc))
    print("This is much better than random guess now :P")

    feat.show_top10(lr, ['Positive', 'Negative'])
