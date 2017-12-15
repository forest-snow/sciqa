from numpy import *
from rnn.propagation import *
import nltk.classify.util
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from collections import Counter
import _pickle as cPickle


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

# compute accuracy given the classifier and test data
def accuracy(classifier, data, ans_list, qid_list):
    gold_answers = []
    predicted_answers = []

    log = []

    for index, guesses in enumerate(ans_list):
        curr_feats, answer = data[index]
        gold_answers.append(answer)
        probs = classifier.prob_classify(curr_feats)

        est_scores = []
        scores = softmax([item[1] for item in guesses])
        for guess, score in zip(guesses, scores):
            est_scores.append(probs.prob(guess[0]))

        pred = guesses[argmax(est_scores)][0]
        predicted_answers.append(pred)

        if answer != pred:
            top5 = argsort(est_scores)[-5:]
            log.append(str(qid_list[index]) + ', ' + ', '.join([guesses[idx][0] + ': ' + str(est_scores[idx]) for idx in top5]))

    with open('logs/bow_wrong.log', 'w') as f:
        for line in log:
            f.write(line + '\n')
    return accuracy_score(gold_answers, predicted_answers)

def predict(classifier, data, ans_list, qid_list):
    predicted_answers = []

    for index, guesses in enumerate(ans_list):
        curr_feats, answer = data[index]
        probs = classifier.prob_classify(curr_feats)

        est_scores = []
        scores = softmax([item[1] for item in guesses])
        for guess, score in zip(guesses, scores):
            est_scores.append(probs.prob(guess[0]))

        predicted_answers.append(guesses[argmax(est_scores)][0])

    with open('../data/lr+score_answer.csv', 'w') as f:
        f.write('id,answer\n')
        for ID, answer in zip(qid_list, predicted_answers):
            if ',' in answer:
                f.write(str(ID) + ',\"' + answer + '\"\n')
            else:
                f.write(str(ID) + ',' + answer + '\n')

# - full evaluation on test data, returns accuracy on all sentence positions 
#   within a question including full question accuracy
# - can add / remove features to replicate baseline models described in paper
# - bow_feats is unigrams, rel_feats is dependency relations
def evaluate(data_split, model_file, d, rnn_feats=True, bow_feats=False, rel_feats=False):

    stop = stopwords.words('english')

    vocab, rel_list, ans_list, tree_dict = \
        cPickle.load(open(data_split, 'rb'))

    train_trees = tree_dict['train']
    test_trees = tree_dict['val']

    params, vocab, rel_list = cPickle.load(open(model_file, 'rb'))

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

    train_qid_list = []
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
                    train_qid_list.append(tree.qid)

                elif i + 1 == len(q):
                    test_feats.append( (curr_feats, tree.ans) )
                    test_ans_list.append(tree.ans_list)
                    test_qid_list.append(tree.qid)

    print('total training instances:', len(train_feats))
    print('total testing instances:', len(test_feats))

    # can modify this classifier / do grid search on regularization parameter using sklearn
    classifier = SklearnClassifier(LogisticRegression(C=100))
    classifier.train(train_feats)

    # predict(classifier, test_feats, test_ans_list, test_qid_list)
    print('accuracy train:', accuracy(classifier, train_feats, train_ans_list, train_qid_list))
    print('accuracy test:', accuracy(classifier, test_feats, test_ans_list, test_qid_list))


# - returns single sentence accuracy on training / validation set
# - use ONLY for hyperparameter tuning / early stopping criteria
# - this returns single sentence accuracy, not question-level accuracy
# - a logistic regression classifier is trained on the average hidden representation
#   of all nodes in the tree. the full evaluation (in the evaluate method)
# - includes the average word embeddings in addition to collapsing sentences
#   belonging to the same question
def validate(data, params, d):

    stop = stopwords.words('english')

    (rel_dict, Wv, b, L) = params

    print('validating, adding lookup')
    for split in data:
        for tree in split:
            for node in tree.get_nodes():
                node.vec = L[:, node.ind].reshape( (d, 1))

    train_feats = []
    val_feats = []

    for tt, split in enumerate(data):

        if tt == 0:
            print('processing train')

        else:
            print('processing val')

        for num_finished, tree in enumerate(split):

            # process validation trees
            forward_prop(params, tree, d, labels=False)

            ave = zeros( (d, 1))
            words = zeros ( (d, 1))
            count = 0
            wcount = 0
            word_list = []
            for ex, node in enumerate(tree.get_nodes()):

                if ex != 0 and node.word not in stop:
                    ave += node.p_norm
                    count += 1

            ave = ave / count
            featvec = ave.flatten()

            curr_feats = {}
            for dim, val in ndenumerate(featvec):
                curr_feats['_' + str(dim)] = val

            if tt == 0:
                train_feats.append( (curr_feats, tree.ans) )

            else:
                val_feats.append( (curr_feats, tree.ans) )

    print('training')
    classifier = SklearnClassifier(LogisticRegression(C=10))
    classifier.train(train_feats)

    print('predicting...')
    train_acc = nltk.classify.util.accuracy(classifier, train_feats)
    val_acc = nltk.classify.util.accuracy(classifier, val_feats)
    return train_acc, val_acc