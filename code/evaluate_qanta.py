import classify.learn_classifiers as lr
import classify.zero_shot_classifiers as zs
import classify.topic_classifiers as tp
import argparse

## - evaluate QANTA's learned representations on history questions
##   and compare performance to bag of words and dependency relation baselines
## - be sure to train a model first by running qanta.py

if __name__ == '__main__':
    
    # command line arguments
    parser = argparse.ArgumentParser(description='QANTA evaluation')
    parser.add_argument('-data', help='location of dataset', default='../data/quizbowl/qb_split')
    parser.add_argument('-model', help='location of trained model', default='models/qb_params_upd')
    parser.add_argument('-We', help='location of We file', default='../data/quizbowl/answer_topic')
    parser.add_argument('-d', help='word embedding dimension', type=int, default=100)

    args = vars(parser.parse_args())

    print('topic performance:')
    tp.evaluate(args['data'], args['model'], args['We'], args['d'], rnn_feats=True, \
              bow_feats=False, rel_feats=True)
    '''
    print('zero-shot performance:')
    zs.evaluate(args['data'], args['model'], args['We'], args['d'], rnn_feats=True, \
              bow_feats=False, rel_feats=False)

    print('logistic-regression performance:')
    lr.evaluate(args['data'], args['model'], args['d'], rnn_feats=True, \
                 bow_feats=False, rel_feats=False)

    print('topic performance:')
    tp.evaluate(args['data'], args['model'], args['We'], args['d'], rnn_feats=True, \
              bow_feats=True, rel_feats=True)
    '''
    '''
    print('\n\nbow performance:')
    evaluate(args['data'], args['model'], args['d'], rnn_feats=False, \
              bow_feats=True, rel_feats=False)

    print('\n\nbow-dt performance:')
    evaluate(args['data'], args['model'], args['d'], rnn_feats=False, \
              bow_feats=True, rel_feats=True)
    '''
