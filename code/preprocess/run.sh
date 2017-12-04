#!/bin/bash

# generate questions
# python3 doc_gen.py --input_file ../../data/quizbowl/qb_test_ner.json --output_name ../../data/quizbowl/qs_test
# python3 doc_gen.py --input_file ../../data/quizbowl/qb_train_ner.json --output_name ../../data/quizbowl/qs_train


# generate dtree
# python3 dparse_to_dtree.py --train_data ../../data/quizbowl/qb_train_ner.json --train_questions ../../data/quizbowl/qs_train.txt \
#	--train_parses ../../data/quizbowl/dptree_train --test_data ../../data/quizbowl/qb_test_ner.json --test_questions ../../data/quizbowl/qs_test.txt \
#	--test_parses ../../data/quizbowl/dptree_test --output_file ../../data/quizbowl/qb_split

# run word2vec
python3 word2vec.py
