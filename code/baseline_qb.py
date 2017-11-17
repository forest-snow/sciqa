# baseline for qb_train.json
# TODO 1: remove stopwords from data and save (taking forever)
# TODO 2: stats on answer distributions
# TODO 3: cross validation if this works

import json
import nltk
import numpy as np
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

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


def transform_data(data, augment=False):
    """
    Questions, along with 50 guesses (guesses may or may not have correct answer) 
    are given as input.
    Function transforms data such that each question corresponds to 50 different
    question/answer pairs. These pairs are then transformed into strings which will then
    be featurized into vectors for further classification.  

    Function also returns labels corresponding to the string representation of each
    question/answer pair.  If the answer is the correct answer to the question, then 
    label is 1; otherwise, 0.

    Args:
        data: each vector in data is in the form 
        (question, correct answer index, answer A, answer B, answer C, answer D)

    Returns: 
        (x,y) where x contains the featurized, transformation of each 
        vector in data and y contains the corresponding 0-1 labels 
    """
    x = []
    y = []
    def transform_vector(vector):
        guesses = vector['guesses']
        for choice in guesses:
            qa_string = transform_qa_pair(vector['text'], choice['guess'])
            x.append(qa_string)
            y.append(1) if vector['answer'] == choice['guess'] else y.append(0)
    
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
        (question, correct answer, guesses (50))

        prob: list of probabilities of being correct for every question/answer pair in data

    Returns: 
        acc: accuracy of classification  
    """
    assert 50*len(data) == len(prob), "List of probabilities does not correspond to data"
    correct_answers = []
    predicted_answers = []
    for i in range(len(data)):
        # list of correct answers to every question
        correct_answers.append(data[i]['answer'])
        # finds guesses classified as most likely correct
        guesses = data[i]['guesses']
        max_ind = np.argmax(prob[50*i:50*i+50])
        predicted_answers.append(guesses[max_ind]['guess'])
    count = [1 for a,b in zip(correct_answers, predicted_answers) if a==b]
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
   
    # Cast to list to keep it all in memory
    data = qb_train = json.load(open('../data/quizbowl/qb_train.json', 'r'))
       
    # split into train and validation sets
    train, val = train_test_split(data, test_size = 0.2)

    # transform data into features and labels
    x_train_strings, y_train = transform_data(train)
    x_val_strings, y_val = transform_data(val)
    feat = Featurizer()
    x_train = feat.train_feature(x_train_strings)
    x_val = feat.test_feature(x_val_strings)    

    '''
    # train classifier and classify points
    y_train_prob, y_val_prob, lr = classify(x_train, y_train, x_val)
    
    # obtain accuracies 
    train_acc = accuracy(train, y_train_prob)
    val_acc = accuracy(val, y_val_prob)
    
    feat.show_top10(lr, ['Positive', 'Negative'])
    
    print("Training accuracy is " + str(train_acc))
    print("Validation accuracy is " + str(val_acc))
    '''


