import argparse
import json
import os
import random
import sys
import _pickle as cPickle

from dtree_util import *
from nltk.tokenize import sent_tokenize


## CAUTION: you will most likely have to fiddle around with these functions to
##          get them to do what you want. they are meant to help you get your data
##          into the proper format for QANTA. send me an email if you have any questions
##          miyyer@umd.edu


# - given a text file where each line is a question sentence, use the
#   stanford dependency parser to create a dependency parse tree for each sentence
def dparse(question_file, output_file):

    out_file = open(output_file, 'w')

    # change these paths to point to your stanford parser.
    # make sure to use the lexparser.sh file in this directory instead of the default!
    parser_out = os.popen("./lexparser.sh " + question_file).readlines()
    for line in parser_out:
        out_file.write(line)

    out_file.close()


# - function that parses the resulting stanford parses
#   e.g., "nsubj(finalized-5, john-1)"
def split_relation(text):
    rel_split = text.split('(')
    rel = rel_split[0]
    deps = rel_split[1][:-1]
    if len(rel_split) != 2:
        print('error ' + str(rel_split))
        sys.exit(0)

    else:
        dep_split = deps.split(',')

        # more than one comma (e.g. 75,000-19)
        if len(dep_split) > 2:

            fixed = []
            half = ''
            for piece in dep_split:
                piece = piece.strip()
                if '-' not in piece:
                    half += piece

                else:
                    fixed.append(half + piece)
                    half = ''

            print('fixed: ' + str(fixed))
            dep_split = fixed

        final_deps = []
        for dep in dep_split:
            words = dep.split('-')
            word = words[0]
            ind = int(words[len(words) - 1])

            if len(words) > 2:
                word = '-'.join([w for w in words[:-1]])

            final_deps.append( (ind, word.strip()) )

        return rel, final_deps


# - given a list of all the split relations in a particular sentence,
#   create a dtree object from that list
def make_tree(plist):

    # identify number of tokens
    max_ind = -1
    for rel, deps in plist:
        for ind, word in deps:
            if ind > max_ind:
                max_ind = ind

    # load words into nodes, then make a dependency tree
    nodes = [None for i in range(0, max_ind + 1)]
    for rel, deps in plist:
        for ind, word in deps:
            nodes[ind] = word

    tree = dtree(nodes)

    # add dependency edges between nodes
    for rel, deps in plist:
        par_ind, par_word = deps[0]
        kid_ind, kid_word = deps[1]
        tree.add_edge(par_ind, kid_ind, rel)

    return tree  

def process_parsing_trees(data_file, parse_file, is_train=False, hold_out=.2):
    parses = open(parse_file, 'r')
    data = json.load(open(data_file, 'r'))

    parse_text = []
    new = False
    cur_parse = []
    for line in parses:

        line = line.strip()

        if not line:
            new = True

        if new:
            parse_text.append(cur_parse)
            cur_parse = []
            new = False

        else:
            # print line
            rel, final_deps = split_relation(line)
            cur_parse.append( (rel, final_deps) )

    print(len(parse_text))

    # make mapping from answers: questions
    # and questions: [sentence trees]
    count = 0
    tree_list = []

    for line in data:

        text = line['text'].replace('\n', '')

        for i, sentence in enumerate(sent_tokenize(text)):
            tree = make_tree(parse_text[count])
            tree.ans = line['answer']
            tree.guesses = [(guess['guess'], guess['score']) for guess in line['guesses']]
            tree.dist = i
            tree.qid = line['id']
            tree_list.append(tree)
            count += 1

    if is_train:
        random.shuffle(tree_list)
        train_size = int(len(tree_list) * (1 - hold_out))
        return tree_list[:train_size], tree_list[train_size:]
    else:
        return tree_list

# - given all dependency parses of a dataset as well as that dataset (in the same order),
#   dumps a processed dataset that can be fed into QANTA:
#   (vocab, list of dep. relations, list of answers, and dict of {fold: list of dtrees})
def process_question_file(data_files, parse_files, output_file):

    # change this path to point to the answer_list file
    ans_list = json.load(open('../../data/quizbowl/answer_set.json', 'r'))

    tree_dict = {}
    tree_dict['train'], tree_dict['val'] = process_parsing_trees(data_files[0], parse_files[0], is_train=True)
    tree_dict['test'] = process_parsing_trees(data_files[1], parse_files[1])

    vocab = []
    rel_list = []

    for key, tree_list in tree_dict.items():

        print('processing ' + key)

        for tree in tree_list:
            if tree.ans not in vocab:
                vocab.append(tree.ans)
            for guess in tree.guesses:
                if guess[0] not in vocab:
                    vocab.append(guess[0])

            tree.ans_ind = vocab.index(tree.ans)

            for node in tree.get_nodes():
                if node.word not in vocab:
                    vocab.append(node.word)

                node.ind = vocab.index(node.word)

                for ind, rel in node.kids:
                    if rel not in rel_list:
                        rel_list.append(rel)


    print('rels: %d' % len(rel_list))
    print('vocab: %d' % len(vocab))
    print('ans: %d' % len(ans_list))

    cPickle.dump((vocab, rel_list, ans_list, tree_dict), open(output_file, 'wb'))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--train_data', type=str, required=True,
        help="Location of training data")
    parser.add_argument('--train_questions', type=str, required=True,
        help="Location of training questions")
    parser.add_argument('--train_parses', type=str, required=True,
        help="Location of parsed training questions")

    parser.add_argument('--test_data', type=str, required=True,
        help="Location of testing data")
    parser.add_argument('--test_questions', type=str, required=True,
        help="Location of testing questions")
    parser.add_argument('--test_parses', type=str, required=True,
        help="Location of parsed testing questions")

    parser.add_argument('--output_file', type=str, required=True,
        help="Location of the output file")

    args = parser.parse_args()

    '''
    # get dependency tree for training and testing data
    dparse(args.train_questions, args.train_parses)
    dparse(args.test_questions, args.test_parses)
    '''
    process_question_file((args.train_data, args.test_data), (args.train_parses, args.test_parses), args.output_file)
