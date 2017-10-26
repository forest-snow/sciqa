import numpy as np
import argparse
import matplotlib.pyplot as plt

from csv import DictReader
from collections import Counter
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--limit', type=int, default=-1,
          help="Restrict training to these many examples")

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/quizbowl_science/sci_train.csv", 'r')))
    test = list(DictReader(open("../data/quizbowl_science/sci_test.csv", 'r')))

    # list of all answers
    answers = []
    for line in train:
        answers.append(line['answerA'])
        answers.append(line['answerB'])
        answers.append(line['answerC'])
        answers.append(line['answerD'])
            
    # distribution of answer counts
    stats = Counter(answers)
    unique = list(stats.keys())
    dist = list(stats.values())
          
    # plot distribution
    plt.plot(dist)
    plt.title('distribution of answers')
    
    # sanity check (adding frequency of all unique answers should equal number of answers)
    assert(sum(dist) == len(train)*4)
    
    # answers that appear more than a 1000 times
    idx = [[unique[i], v] for i, v in enumerate(dist) if v > 1000]
    print("these appear more than a 1000 times: ")
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
    
    # correct answers
    correct_answers = []
    for line in train:
        if line['correctAnswer'] == 'A':
            correct_answers.append(line['answerA'])
        elif line['correctAnswer'] == 'B':
            correct_answers.append(line['answerB'])
        elif line['correctAnswer'] == 'C':
            correct_answers.append(line['answerC'])
        else:
            correct_answers.append(line['answerD'])
    
    # distribution of correct answers
    c_stats = Counter(correct_answers)
    c_unique = list(c_stats.keys())
    c_dist = list(c_stats.values())
    
    plt.figure()
    plt.plot(c_dist)
    plt.title('distribution of correct answers')
    plt.figure()
    plt.hist(c_dist, bins = range(0,21))
    plt.title('histogram of correct answers with 20 size one bins')