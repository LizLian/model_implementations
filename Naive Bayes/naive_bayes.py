# CS114 Spring 2019 Homework 3
# Naive Bayes Classifier and Evaluation

import os
import numpy as np
from collections import defaultdict
import math
from nltk.corpus import stopwords
from string import punctuation 

with open('positive-words.txt') as f:
    positive_words = f.read().split()
with open('negative-words.txt') as f:
    negative_words = f.read().split()

opinion_words = positive_words + negative_words

class NaiveBayes():

    def __init__(self):
        self.class_dict = {0: 'neg', 1: 'pos'}
        self.feature_dict = {}
        self.prior = None
        self.likelihood = None
        self.likelihood_pos = []
        self.likelihood_neg = []
        self.opinion_words = []

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[class][feature] = log(P(feature|class))
    '''
    def train(self, train_set, features = []):
        n_doc, n_pos, n_neg= 0, 0, 0
        words_pos = []
        words_neg = []
        bigdoc = defaultdict(dict)
        vocabs = []
        count_pos = defaultdict(int)
        count_neg = defaultdict(int)
        
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            n_doc += len(files) # number of documents in D
            if root.endswith('pos'):
                # number of documents from D in class c
                n_pos += len(files)
                for name in files:                    
                    with open(os.path.join(root, name)) as f:
                        words_pos.extend(f.read().split())
            if root.endswith('neg'):
                n_neg += len(files)
                for name in files:
                    with open(os.path.join(root, name)) as f:
                        words_neg.extend(f.read().split())
        
        vocabs = words_pos + words_neg
        
        if len(features) == 0:
            for vocab in vocabs:
                if vocab in opinion_words and vocab not in self.opinion_words:
                    self.opinion_words.append(vocab)

        #for each word w in V, compute count(w,c)                                       
        for word in words_pos:
            count_pos[word] += 1
        for word in words_neg:
            count_neg[word] += 1
            
        bigdoc_pos_cnt, bigdoc_neg_cnt = 0, 0
        bigdoc_pos_cnt = sum(c for c in count_pos.values())
        bigdoc_neg_cnt = sum(c for c in count_neg.values())
        
        pos_denominator = bigdoc_pos_cnt + len(set(vocabs))
        neg_denominator = bigdoc_neg_cnt + len(set(vocabs))
        
        # get features from the train_set
        feature_index = 0
        if len(features) == 0:
            for vocab in set(vocabs):
                self.feature_dict[feature_index] = vocab
                feature_index += 1
            self._remove_stopwords()
        else:
            print('setting features')
            self._set_features(features)
        
        self.likelihood_pos = []
        self.likelihood_neg = []
        for key, word in self.feature_dict.items():
            self.likelihood_pos.append((count_pos[word] + 1)/pos_denominator)
            self.likelihood_neg.append((count_neg[word] + 1)/neg_denominator)
        
        # normalize counts to probabilities, and take logs
        self.prior = np.log([n_neg/n_doc, n_pos/n_doc])
        self.likelihood = np.log([self.likelihood_neg, self.likelihood_pos])
        
#        for key, value in self.feature_dict.items():
#            if value == 'great':
#                print('likelihood great: ', self.likelihood[[0,1], key])
#            if value == 'poor':
#                print('likelihood poor: ', self.likelihood[[0,1], key])
#            if value == 'long':
#                print('likelihood long: ', self.likelihood[[0,1], key])
                
        return self.prior, self.likelihood
        

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)

        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            if root.endswith('pos') or root.endswith('neg'):
                for name in files:
                    features = {self.feature_dict[key]: 0 for key in self.feature_dict}
                    with open(os.path.join(root, name), encoding = 'latin-1') as f:
                        # create feature vectors for each document
                        for word in f.read().split():
                            if word in self.feature_dict.values():
                                features[word] += 1
                    dot_product = np.dot(self.likelihood, np.array(list(features.values())))

                    # get most likely class
                    results[name]['predicted'] = self.class_dict[np.argmax(np.add(self.prior, dot_product))]
                    
                    if root.endswith('pos'):
                        results[name]['correct'] = 'pos'
                    if root.endswith('neg'):
                        results[name]['correct'] = 'neg'
                
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # you may find this helpful
        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))

        for file in results:
            if results[file]['correct'] == 'neg' and results[file]['predicted'] == 'neg':
                confusion_matrix[1, 1] += 1 
            elif results[file]['correct'] == 'neg' and results[file]['predicted'] == 'pos':
                confusion_matrix[1, 0] += 1 
            elif results[file]['correct'] == 'pos' and results[file]['predicted'] == 'neg':
                confusion_matrix[0, 1] += 1 
            elif results[file]['correct'] == 'pos' and results[file]['predicted'] == 'pos':
                confusion_matrix[0, 0] += 1 
        
        # recall
        recall = confusion_matrix[0, 0]/(confusion_matrix[0, 0] + confusion_matrix[0, 1])
        recall_neg = confusion_matrix[1, 1]/(confusion_matrix[1, 1] + confusion_matrix[1, 0])
        # precision
        precision = confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[1][0])
        precision_neg = confusion_matrix[1][1]/(confusion_matrix[0][1] + confusion_matrix[1][1])
        # F1
        f1 = 2*precision*recall/(precision + recall)
        f1_neg = 2*precision_neg*recall_neg/(precision_neg + recall_neg)
        # accuracy
        accuracy = (confusion_matrix[1][1] + confusion_matrix[0][0])/(confusion_matrix[1][1] + confusion_matrix[1][0] + confusion_matrix[0][0] + confusion_matrix[0][1])
        print('Number of features: ', len(self.feature_dict))
        print()
        print('Positive: ')
        print('Precision: \t\t' + str(round(precision, 5)))
        print('Recall: \t\t' + str(round(recall, 5)))
        print('F1-Score: \t\t' + str(round(f1, 5)))
        print()
        print('Negative: ')
        print('Precision: \t\t' + str(round(precision_neg, 5)))
        print('Recall: \t\t' + str(round(recall_neg, 5)))
        print('F1-Score: \t\t' + str(round(f1_neg, 5)))
        print('Accuracy: \t\t' + str(round(accuracy, 5)))

    def _remove_stopwords(self):
        index = 0
        words = [word for word in self.feature_dict.values() 
                    if word not in punctuation 
                    and word not in set(stopwords.words('english'))]
        self.feature_dict = {}
        for word in words:
            self.feature_dict[index] = word
            index += 1

    def _set_features(self, features):
        index = 0
        self.feature_dict = {}
        for word in features:
            self.feature_dict[index] = word
            index += 1
        
        #self.feature_dict = {0:'great', 1:'poor', 2:'long'}

    '''
    compute the likelihood ratio for each word 
    lr(w) = max p(w|c_i)/p(w|c_j)
    lr is sorted by the words' likelihood ratio
    '''
    def feature_selection(self, train_set, n):
        self.train(train_set)
            
        lr = {}
        for key, word in self.feature_dict.items():
            if word in self.opinion_words:
                if self.likelihood_pos[key] > self.likelihood_neg[key]:
                    ratio = self.likelihood_pos[key]/self.likelihood_neg[key]
                else:
                    ratio = self.likelihood_neg[key]/self.likelihood_pos[key]
                lr[word] = ratio
        lr = {k: v for k, v in sorted(lr.items(), key = lambda x: x[1], reverse= True)}   

        return list(lr.keys())[0:n]


if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
 
    selected_features = nb.feature_selection('movie_reviews/train', 4200)
    
    nb.train('movie_reviews/train', selected_features)
    
    results = nb.test('movie_reviews/dev')
    
    nb.evaluate(results)
