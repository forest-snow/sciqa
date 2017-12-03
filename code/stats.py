import json
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from csv import DictReader


if __name__ == "__main__":
    # Cast to list to keep it all in memory
    train = json.load(open('../data/quizbowl/qb_train.json', 'r'))
    test = json.load(open('../data/quizbowl/qb_test.json', 'r'))

    # list of all answers
    answers = []
    for line in train:
        print(line['guesses'][:5])
        print(line['text'])
        print(line['answer'])
        answers.append(line['answer'])

    # distribution of answer counts
    stats = Counter(answers)
    unique = list(stats.keys())
    dist = list(stats.values())

    # plot distribution
    plt.plot(dist)
    plt.title('distribution of answers')

    # sanity check (adding frequency of all unique answers should equal number of answers)
    assert(sum(dist) == len(train))

    print("total number of answers: %d" % len(unique))

    # answers that appear more than a 1000 times
    idx = [[unique[i], v] for i, v in enumerate(dist) if v > 1000]
    print("these appear more than a thousand times: ")
    print(idx)

    # number of answers that appear only once
    idx = [i for i, v in enumerate(dist) if v == 1]
    print("number of answers that appear just once: " + str(len(idx)))

    # number of answers that appear less than 5
    idx = [i for i, v in enumerate(dist) if v < 5]
    print("number of answers that appear less than 5 times: " + str(len(idx)))

    # histogram
    # hist, bin_edges = np.histogram(dist, 'auto')
    # plt.figure()
    # plt.hist(dist, bins = bin_edges)
    plt.figure()
    n, bins, patches = plt.hist(dist)
    plt.title('histogram with 10 total bins')
    plt.figure()
    plt.hist(dist, bins = range(0,21))
    plt.title('histogram with 20 size one bins')
