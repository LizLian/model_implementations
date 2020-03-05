# CS114 Spring 2019 Homework 4
# Part-of-speech Tagging with Hidden Markov Models

import os, re
import numpy as np
from collections import defaultdict

class POSTagger():

    def __init__(self):
        self.pos_dict = {}
        self.word_dict = {}
        self.initial = None
        self.transition = None
        self.emission = None
        self.UNK = '*UNKNOWN*'

    '''
    Trains a supervised hidden Markov model on a training set.
    self.initial[POS] = log(P(the initial tag is POS))
    self.transition[POS1][POS2] =
    log(P(the current tag is POS2|the previous tag is POS1))
    self.emission[POS][word] =
    log(P(the current word is word|the current tag is POS))
    '''
    def train(self, train_set):
        pos_freq = defaultdict(int) # # of times a pos tag appear in the corpus
        pos_bigram = defaultdict(dict) # pos bigrams
        word_pos_tags = defaultdict(dict) # pos & word
        initial = defaultdict(int) # # of times a pos tag appear in the initial position
        index, index_word, total_initial = 0, 0, 0
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # be sure to split documents into sentences here
                    sentences = f.read().split('\n')
                    sentences = [sent for sent in sentences if sent.strip()]

                    for sent in sentences:
                        if sent:    
                            word_tag = self._normalization(sent.lower())
                            # get the initial tag count
                            initial[word_tag[0][1]] += 1
                            total_initial += 1
                            
                            for word, pos in word_tag:
                                if word not in word_pos_tags[pos]:
                                    word_pos_tags[pos][word] = 1
                                else:
                                    word_pos_tags[pos][word] += 1
                                pos_freq[pos] += 1
                                if pos not in self.pos_dict:
                                    self.pos_dict[pos] = index
                                    index += 1
                                if word not in self.word_dict:
                                    self.word_dict[word] = index_word
                                    index_word += 1
                                    
                            # construct pos bigrams
                            for i in range(len(word_tag) - 1):
                                if word_tag[i+1][1] not in pos_bigram[word_tag[i][1]]:
                                    pos_bigram[word_tag[i][1]][word_tag[i+1][1]] = 1
                                else:
                                    pos_bigram[word_tag[i][1]][word_tag[i+1][1]] += 1
        # add unknown
        self.word_dict[self.UNK] = len(self.word_dict)

        for pos in self.pos_dict:
            if pos in initial:
                initial[pos] = (initial[pos] + 1)/(total_initial + len(self.pos_dict))
            else:
                initial[pos] = 1/(total_initial + len(self.pos_dict))

        self.initial = np.log(list(initial.values()))

        # fill in transition probabilities
        self.transition = np.zeros((len(self.pos_dict), len(self.pos_dict)))
        for row in self.pos_dict: #row
            for col in self.pos_dict: #column
                if (row in pos_bigram) and (col in pos_bigram[row]):
                    self.transition[self.pos_dict[row], self.pos_dict[col]] = \
                            (pos_bigram[row][col] + 1)/(pos_freq[row] + len(self.pos_dict))
                else: # add 1 smoothing
                    self.transition[self.pos_dict[row], self.pos_dict[col]] = \
                        1/(pos_freq[row] + len(self.pos_dict))
                        
        #print(np.sum(np.array(self.transition), axis = 1 ) )
             
        self.transition = np.log(self.transition)
        #print(self.transition, self.initial)
        
        # construct emission prob.
        emission = []
        for pos in self.pos_dict:
            temp = []
            for word in self.word_dict:
                if word in word_pos_tags[pos]:
                    temp.append((word_pos_tags[pos][word] + 1)/(pos_freq[pos] + len(self.word_dict)))
                else: # add 1 smoothing
                    temp.append( 1/(pos_freq[pos] + len(self.word_dict)) )
            emission.append(temp)
        self.emission = np.log(emission)
    
    '''
        normalize the data
        strip off fw- (foreign word)
        strip off hyphenated tags
        strip off the negation * 
        for merged constructions (joined by +), we only care about the first tag
    '''    
    def _normalization(self, sent):
        tokens = sent.split()
        word_pos_tags = []
        for token in tokens:
            #word could also contains /
            #In these cases, it is the last / that separates the word from the pos
            word, pos = token.rsplit("/", 1)
            
            #strip off fw- (foreign word)
            pos = re.sub('fw-', '', pos)
            #strip off hyphenated tags
            pos = re.sub('-\w+', '', pos)
            #strip off the negation *
            pos = re.sub('(\w+)(\*+)', r'\1', pos)
            #for merged constructions (joined by +)
            #we only care about the first tag
            pos = pos.split('+')[0]
            
            word_pos_tags.append((word, pos))
        return word_pos_tags
            


    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        features = {self.word_dict[key]: 0 for key in self.word_dict}
        v = np.zeros((len(self.pos_dict), len(sentence)))
        backpointer = np.zeros((len(self.pos_dict), len(sentence)))

        if sentence[0] in self.word_dict:
            features[self.word_dict[sentence[0]]] = 1
        else:
            features[self.word_dict[self.UNK]] = 1
            
        # initialization step
        state_obs_likelihood = np.dot(self.emission, np.array(list(features.values())))
        v[ :, 0:1] = np.reshape((self.initial + state_obs_likelihood), (len(self.pos_dict), 1))

        # recursion step
        for t in range(1, len(sentence)):
            features = {self.word_dict[key]: 0 for key in self.word_dict}
            
            if sentence[t] in self.word_dict:
                features[self.word_dict[sentence[t]]] = 1
            else:
                features[self.word_dict[self.UNK]] = 1
                
            # values of bs(Ot) for a particular Ot(word), for each s (pos)
            state_obs_likelihood = np.dot(self.emission, np.array(list(features.values())))
            v[:, t:(t+1)] = np.reshape(np.max((self.transition + v[:, (t-1):t] 
                                            + state_obs_likelihood), axis=0), (len(self.pos_dict), 1))
            
            backpointer[:, t:(t+1)] = np.reshape(np.argmax((self.transition + v[:, (t-1):t] 
                                            + state_obs_likelihood), axis=0), (len(self.pos_dict), 1))

        # termination step3
        bestpathPointer = np.argmax(v[:, len(sentence)-1:len(sentence)])
        best_path = [bestpathPointer]
        for t in reversed(range(1, len(sentence))):
            bestpathPointer = int(backpointer[bestpathPointer][t])
            best_path.insert(0, bestpathPointer)
        return best_path

    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of POS tags such that:
    results[sentence_id]['correct'] = correct sequence of POS tags
    results[sentence_id]['predicted'] = predicted sequence of POS tags
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        sent_index = 0
        pos_dict = {self.pos_dict[pos]: pos for pos in self.pos_dict}
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    sents = f.read().split('\n')
                    for sent in sents:
                        if sent:    
                            word_tag = self._normalization(sent.lower())
                            results[sent_index]['correct'] = [pos for word, pos in word_tag]
                            best_path = self.viterbi([word for word, pos in word_tag])
                            results[sent_index]['predicted'] = [pos_dict[pos] for pos in best_path]
                            sent_index += 1
        return results

    '''
    Given results, calculates overall accuracy.
    '''
    def evaluate(self, results):
        correct_count, total_count = 0, 0
        for sent_id in results:
            total_count += len(results[sent_id]['correct'])
                
        for sent_id in results:
            for i in range(0, len(results[sent_id]['correct'])):
                if results[sent_id]['correct'][i] == results[sent_id]['predicted'][i]:
                    correct_count += 1
        accuracy = correct_count/total_count
        return accuracy

if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    pos.train('brown/train')
#    sents = 'learn more change thoroughly'.split()
#    pos.viterbi(sents)
    results = pos.test('brown/dev')
    print('Accuracy:', pos.evaluate(results))
