from numpy import *
from rnn.propagation import *
from scipy.spatial.distance import cosine
from sklearn.feature_extraction import DictVectorizer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from collections import Counter
import _pickle as cPickle

kTHRESHOLD = .2
kTOPICS = 108

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return exp(x) / sum(exp(x), axis=0)

# Topic classifier
class TopicClassifier:

    def __init__(self, estimator, answer_topic_map, ans_list):
        self._estimator = estimator
        self._answer_topic_map = answer_topic_map
        self._ans_list = ans_list
        self._vectorizer = DictVectorizer()
        self._encoder = LabelEncoder()


    def train(self, feature_sets):
        X, y, weights = [], [], []
        for feats, answer in feature_sets:
            topics = self._answer_topic_map[answer]
            for key, prob in topics:
                if prob > kTHRESHOLD:
                    X.append(feats)
                    y.append(key)
                    weights.append(prob)

        X = self._vectorizer.fit_transform(X)
        y = self._encoder.fit_transform(y)
        self._estimator.fit(X, y, sample_weight=weights)
        return self


    def accuracy(self, lr, feature_sets, ans_list, qid_list, T=5):
        classes = list(self._encoder.classes_)

        gold_answers = []
        predicted_answers = []

        log = []

        for index, guesses in enumerate(ans_list):
            curr_feats, answer = feature_sets[index]
            gold_answers.append(answer)

            X = self._vectorizer.transform(curr_feats)
            probas = self._estimator.predict_proba(X)[0]
            lr_probas = lr.prob_classify(curr_feats)

            max_lr = max([lr_probas.prob(guess[0]) for guess in guesses])

            est_scores = []
            scores = softmax([item[1] for item in guesses])
            for guess, score in zip(guesses, scores):
                vec = zeros(kTOPICS)
                topics = self._answer_topic_map[guess[0]]
                for key, prob in topics:
                    vec[classes.index(key)] = prob

                if max_lr > .5:
                    est_scores.append(lr_probas.prob(guess[0]))
                else:
                    est_scores.append(- cosine(vec, probas) + 0.1 * lr_probas.prob(guess[0]))

            pred = guesses[argmax(est_scores)][0]
            predicted_answers.append(pred)
            '''
            if answer == pred:
                top5 = argsort(est_scores)[-5:]
                log.append(str(qid_list[index]) + ', ' + ', '.join([guesses[idx][0] + ': ' + str(est_scores[idx]) for idx in top5]))
            '''
        '''
        with open('logs/topic_correct.log', 'w') as f:
            for line in log:
                f.write(line + '\n')
        '''
        return accuracy_score(gold_answers, predicted_answers)


# create question dictionary such that sentences belonging to the same
# question are grouped together, {question ID: {sentence position: tree}}
def collapse_questions(train_trees, test_trees):
    train_q = {}
    for tree in train_trees:
        if tree.qid not in train_q:
            train_q[tree.qid] = {}

        train_q[tree.qid][tree.dist] = tree

    test_q = {}
    for tree in test_trees:
        if tree.qid not in test_q:
            test_q[tree.qid] = {}

        test_q[tree.qid][tree.dist] = tree

    return train_q, test_q


# - full evaluation on test data, returns accuracy on all sentence positions 
#   within a question including full question accuracy
# - can add / remove features to replicate baseline models described in paper
# - bow_feats is unigrams, rel_feats is dependency relations
def evaluate(data_split, model_file, topic_file, d, rnn_feats=True, bow_feats=False, rel_feats=False):

    stop = stopwords.words('english')

    vocab, rel_list, ans_list, tree_dict = \
        cPickle.load(open(data_split, 'rb'))

    train_trees = tree_dict['train']
    test_trees = tree_dict['val']

    params, vocab, rel_list = cPickle.load(open(model_file, 'rb'))

    answer_topic_map = cPickle.load(open(topic_file, 'rb'))

    (rel_dict, Wv, b, We) = params

    data = [train_trees, test_trees]

    # get rid of trees that the parser messed up on
    for sn, split in enumerate(data):

        bad_trees = []

        for ind, tree in enumerate(split):
            if tree.get(0).is_word == 0:
                # print tree.get_words()
                bad_trees.append(ind)
                continue

        # print 'removed', len(bad_trees)
        for ind in bad_trees[::-1]:
            split.pop(ind)

    for split in data:
        for tree in split:
            for node in tree.get_nodes():
                node.vec = We[:, node.ind].reshape( (d, 1))

            tree.ans_list = tree.guesses

    train_q, test_q = collapse_questions(train_trees, test_trees)

    # print 'number of training questions:', len(train_q)
    # print 'number of testing questions:', len(test_q)

    train_feats = []
    test_feats = []
    train_ans_list = []
    test_ans_list = []
    test_qid_list = []

    for tt, split in enumerate([train_q, test_q]):

        if tt == 0:
            print('processing train')

        else:
            print('processing test')

        # for each question in the split
        for qid in split:

            q = split[qid]
            ave = zeros( (d, 1))
            words = zeros ( (d, 1))
            bow = []
            count = 0.
            tree = None
            curr_ave = None
            curr_words = None

            # for each sentence in the question, generate features
            for i in range(0, len(q)):

                try:
                    tree = q[i]
                except:
                    continue

                forward_prop(params, tree, d, labels=False)

                # features: average of hidden representations and average of word embeddings
                for ex, node in enumerate(tree.get_nodes()):

                    if node.word not in stop:
                        ave += node.p_norm
                        words += node.vec
                        count += 1.

                if count > 0:
                    curr_ave = ave / count
                    curr_words = words / count

                featvec = concatenate([curr_ave.flatten(), curr_words.flatten()])
                curr_feats = {}

                # add QANTA's features to the current feature set
                if rnn_feats:
                    for dim, val in ndenumerate(featvec):
                        curr_feats['__' + str(dim)] = val

                # add unigram indicator features to the current feature set
                if bow_feats:
                    bow += [l.word for l in tree.get_nodes()]
                    for word in bow:
                        curr_feats[word] = 1.0

                # add dependency relation indicator features to the current feature set
                if rel_feats:
                    for l in tree.get_nodes():
                        if len(l.parent) > 0:
                            par, rel = l.parent[0]
                            if rel == 'det':
                                this_rel = l.word + '__' + rel + '__' + tree.get(par).word
                                curr_feats[this_rel] = 1.0
                            elif rel == 'nummod':
                                curr_feats['__nummod__' + tree.get(par).word] = 1.0

                if tt == 0:
                    train_feats.append( (curr_feats, tree.ans) )
                    train_ans_list.append(tree.ans_list)

                elif i + 1 == len(q):
                    test_feats.append( (curr_feats, tree.ans) )
                    test_ans_list.append(tree.ans_list)
                    test_qid_list.append(tree.qid)

    print('total training instances:', len(train_feats))
    print('total testing instances:', len(test_feats))

    lr = SklearnClassifier(LogisticRegression(C=100))
    lr.train(train_feats)

    # can modify this classifier / do grid search on regularization parameter using sklearn
    classifier = TopicClassifier(LogisticRegression(C=10), answer_topic_map, ans_list)
    classifier.train(train_feats)

    # classifier.predict(test_feats, test_ans_list, test_qid_list)
    # print('accuracy train:', classifier.accuracy(lr, train_feats, train_ans_list))
    print('accuracy test:', classifier.accuracy(lr, test_feats, test_ans_list, test_qid_list))
