# CS114 Spring 2019 Homework 4
# Part-of-speech Tagging with Hidden Markov Models

import os
import numpy as np
from scipy.special import logsumexp
from pos_tagger import POSTagger

class ForwardBackward(POSTagger):

    def __init__(self):
        self.pos_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        self.word_dict = {}
        self.initial = None
        self.transition = None
        self.emission = None

    '''
    Implements the forward algorithm.
    '''
    def forward(self, sentence):
        alpha = None
        # alpha initialization step
        # alpha recursion step
        # no alpha termination step necessary
        return alpha

    '''
    Implements the backward algorithm.
    '''
    def backward(self, sentence):
        beta = None
        # beta initialization step
        # beta recursion step
        # no beta termination step necessary
        return beta

    '''
    Trains an unsupervised hidden Markov model on a data set.
    '''
    def forward_backward(self, data_set):
        # initialize A and B (and pi)
        converged = False
        # iterate until convergence
        while not converged:
            # E-step
            # M-step
            converged = True

    '''
    Tests the tagger on a data set.
    This is filled in for you.
    '''
    def test(self, data_set):
        print('POS dict:', self.pos_dict)
        print('Word dict:', self.word_dict)
        print('Initial probability distribution:', np.exp(self.initial))
        print('Transition matrix:', np.exp(self.transition))
        print('Emission matrix:', np.exp(self.emission))
        reverse_word_dict = {word: j for j, word in self.word_dict.items()}
        with open(data_set) as f:
            for sentence in f.read().splitlines():
                if sentence:
                    word_list = sentence.split()
                    index_list = [reverse_word_dict[word] for word in word_list]
                    # note that ForwardBackward inherits your POSTagger's
                    # viterbi function
                    pos_list = self.viterbi(index_list)
                    print(' '.join([word + '/' + pos
                                    for word, pos in zip(word_list, pos_list)]))

if __name__ == '__main__':
    fb = ForwardBackward()
    # make sure this points to the right file
    fb.forward_backward('data_set')
    fb.test('data_set')
