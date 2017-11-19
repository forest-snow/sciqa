import argparse
import nltk
import numpy as np
import random
import string

from csv import DictReader
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold

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
    """
    Transforms question/answer pair into a string such that:
    - answers with multiple words have underscores between words
    - string contains the answer and every word in question with answer appended to the end

    Args:
        question: string
        answer: string

    Returns:
        sentence: string
    """
    # remove punctuations from question and answer (dashes (-) are kept)
    punctuation = string.punctuation.replace('-','')
    question = question.translate(str.maketrans('', '', punctuation))
    answer = answer.translate(str.maketrans('', '', punctuation))

    # answers with multiple words have underscores between words instead of spaces
    answer_phrase = '_'.join(answer.split())
    sentence = answer_phrase

    # append answer to end of each word in question
    for word in question.split():
        if word not in stopwords.words('english'):
            sentence += ' ' + word + '-' + answer_phrase
    return sentence


def transform_data(data, augment=False, answer_pool=None):
    """
    Science questions, along with 4 answers (3 wrong, 1 correct) are given as input.
    Function transforms data such that each question corresponds to 4 different
    question/answer pairs. These pairs are then transformed into strings which will then
    be featurized into vectors for further classification.

    Function also returns labels corresponding to the string representation of each
    question/answer pair.  If the answer is the correct answer to the question, then
    label is 1; otherwise, 0.

    Args:
        data: each vector in data is in the form
        (question, correct answer index, answer A, answer B, answer C, answer D)

        augment: augment data only if true

        answer_pool: answers to help augment data

    Returns:
        (x,y) where x contains the featurized, transformation of each
        vector in data and y contains the corresponding 0-1 labels
    """
    x = []
    y = []
    def transform_vector(vector):
        answers = {'A': vector['answerA'], 'B': vector['answerB'], 'C': vector['answerC'],'D': vector['answerD']}
        for i in answers:
            qa_string = transform_qa_pair(vector['question'], answers[i])
            x.append(qa_string)
            y.append(1) if vector['correctAnswer'] == i else y.append(0)
        if augment:
            for answer in random.sample(answer_pool, 20):
                if answer not in answers.values():
                    qa_string = transform_qa_pair(vector['question'], answer)
                    x.append(qa_string)
                    y.append(0)
    for v in data:
        transform_vector(v)
    return x,y

def accuracy(data, prob):
    """
    Given list of probabilities of being correct for every question/answer pair in dataset,
    function computes accuracy of classification by comparing the correct answer with the
    answer that is classified as most likely correct.

    Args:
        data: each vector in data is in the form
        (question, correct answer index, answer A, answer B, answer C, answer D)

        prob: list of probabilities of being correct for every question/answer pair in data

    Returns:
        acc: accuracy of classification
    """
    assert 4*len(data) == len(prob), "List of probabilities does not correspond to data"
    correct_inds = []
    predicted_inds = []
    index = {'A': 0, 'B':1, 'C':2, 'D':3}
    for i in range(len(data)):
        # list of indices of correct answers to every question
        correct_inds.append(index[data[i]['correctAnswer']])
        # finds index of answer classified as most likely correct
        max_ind = np.argmax(prob[4*i:4*i+4])
        predicted_inds.append(max_ind)
    count = [1 for a,b in zip(correct_inds, predicted_inds) if a==b]
    acc = sum(count)/len(data)
    return acc

def classify(x_train, y_train, x_test):
    """
    Trains logistic regression classifier on training set and then returns the probabilities
    of being the correct answer for points in training and test sets.

    Args:
        x_train: features of training set
        y_train: labels of training set
        x_test: features of test set

    Returns:
        y_train_prob: probabilities of training set points
        y_test_prob: probabilities of testing set points
        lr: classifier
    """
    # train classifier
    lr = SGDClassifier(loss='log', penalty='l2', max_iter = 5, tol = None)
    lr.fit(x_train, y_train)

    # obtain probabilities from decision boundary of classifier
    y_train_prob = lr.decision_function(x_train)
    y_test_prob = lr.decision_function(x_test)
    return y_train_prob, y_test_prob, lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--limit', type=int, default=-1,
          help="Restrict training to these many examples")
    parser.add_argument('--k', type=int, default=10,
          help="Define how we split the data for cross validation")
    args = parser.parse_args()

    # Cast to list to keep it all in memory
    data = list(DictReader(open("../data/quizbowl_science/sci_train2.csv", 'r')))

    n_splits = args.k
    kf = KFold(n_splits=n_splits, shuffle = True, random_state = kSEED)
    train_acc = 0
    val_acc = 0
    k = 0
    for train_index, val_index in kf.split(data):

        # split into train and validation sets
        train = [data[i] for i in train_index]
        val = [data[i] for i in val_index]

        # transform data into features and labels
        x_train_strings, y_train = transform_data(train)
        x_val_strings, y_val = transform_data(val)
        feat = Featurizer()
        x_train = feat.train_feature(x_train_strings)
        x_val = feat.test_feature(x_val_strings)

        # train classifier and classify points
        y_train_prob, y_val_prob, lr = classify(x_train, y_train, x_val)

        # obtain accuracies
        train_acc += accuracy(train, y_train_prob)
        val_acc += accuracy(val, y_val_prob)

        # print top features for some folds
        if k % 3 == 0:
            feat.show_top10(lr, ['Positive', 'Negative'])
        k += 1
    train_acc /= n_splits
    val_acc /= n_splits
    print("Training accuracy is " + str(train_acc))
    print("Validation accuracy is " + str(val_acc))
