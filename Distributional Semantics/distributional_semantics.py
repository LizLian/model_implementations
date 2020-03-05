#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as scipy_linalg
import random
from collections import defaultdict
from prettytable import PrettyTable

class WordVector():
    
    def __init__(self):
        self.cooccurrence = None #co-occurrence matrix
        self.ppmi = None #positive pointwise mutual information
        self.word_dict = {}
        self.analogy_pairs = []
        
        
    def term_context(self, file):
        with open(file) as f:
            sents = f.read().split('\n')
        
        i = 0
        for sent in sents:
            for word in sent.split():
                if word not in self.word_dict:
                    self.word_dict[word] = i
                    i += 1
        
        #initialize the co-occurrence matrix
        self.cooccurrence = np.zeros((len(self.word_dict), len(self.word_dict)))
        
        #construct the co-occurrence matrix
        for sent in sents:
            words = sent.split()
            for index in range(len(words)-1):
                w = self.word_dict[words[index]]
                c = self.word_dict[words[index+1]]
                self.cooccurrence[w, c] += 1
                self.cooccurrence[c, w] += 1
        
        #multiply the co-occurrence matrix by 10
        self.cooccurrence *= 10
        
        #smooth the counts by adding 1 to all cells
        self.cooccurrence += 1

        #compute the positive pointwise mutual informaiton (PPMI) 
        #for each word w and context word c
        p_wc = self.cooccurrence/np.sum(self.cooccurrence) 
        p_w = np.sum(self.cooccurrence, axis = 1)/np.sum(self.cooccurrence)
        p_c = np.sum(self.cooccurrence, axis = 0)/np.sum(self.cooccurrence)
        self.ppmi = np.log2(p_wc/(p_w*p_c))
        #replace negative numbers with 0s
        self.ppmi[self.ppmi <0] = 0

        return self.cooccurrence, self.ppmi
    
    
    def eucl_dist_c(self):
        #normalize vectors
        women = self.cooccurrence[self.word_dict['women'], :]/scipy_linalg.norm(self.cooccurrence[self.word_dict['women'], :])
        men = self.cooccurrence[self.word_dict['men'], :]/scipy_linalg.norm(self.cooccurrence[self.word_dict['men'], :])
        dogs = self.cooccurrence[self.word_dict['dogs'], :]/scipy_linalg.norm(self.cooccurrence[self.word_dict['dogs'], :])
        feed = self.cooccurrence[self.word_dict['feed'], :]/scipy_linalg.norm(self.cooccurrence[self.word_dict['feed'], :])
        like = self.cooccurrence[self.word_dict['like'], :]/scipy_linalg.norm(self.cooccurrence[self.word_dict['like'], :])
        bite = self.cooccurrence[self.word_dict['bite'], :]/scipy_linalg.norm(self.cooccurrence[self.word_dict['bite'], :])
        #euclidean distance 
        ed_women_men = scipy_linalg.norm(women - men)
        ed_women_dogs = scipy_linalg.norm(women - dogs)
        ed_men_dogs = scipy_linalg.norm(men - dogs)
        ed_feed_like = scipy_linalg.norm(feed - like)
        ed_feed_bite = scipy_linalg.norm(feed - bite)
        ed_like_bite = scipy_linalg.norm(like - bite)
        return ed_women_men,ed_women_dogs,ed_men_dogs,ed_feed_like,ed_feed_bite,ed_like_bite
    
    
    def euclidean_distance(self):
        #normalize vectors
        women = self.ppmi[self.word_dict['women'], :]/scipy_linalg.norm(self.ppmi[self.word_dict['women'], :])
        men = self.ppmi[self.word_dict['men'], :]/scipy_linalg.norm(self.ppmi[self.word_dict['men'], :])
        dogs = self.ppmi[self.word_dict['dogs'], :]/scipy_linalg.norm(self.ppmi[self.word_dict['dogs'], :])
        feed = self.ppmi[self.word_dict['feed'], :]/scipy_linalg.norm(self.ppmi[self.word_dict['feed'], :])
        like = self.ppmi[self.word_dict['like'], :]/scipy_linalg.norm(self.ppmi[self.word_dict['like'], :])
        bite = self.ppmi[self.word_dict['bite'], :]/scipy_linalg.norm(self.ppmi[self.word_dict['bite'], :])
        #euclidean distance 
        ed_women_men = scipy_linalg.norm(women - men)
        ed_women_dogs = scipy_linalg.norm(women - dogs)
        ed_men_dogs = scipy_linalg.norm(men - dogs)
        ed_feed_like = scipy_linalg.norm(feed - like)
        ed_feed_bite = scipy_linalg.norm(feed - bite)
        ed_like_bite = scipy_linalg.norm(like - bite)
        return ed_women_men,ed_women_dogs,ed_men_dogs,ed_feed_like,ed_feed_bite,ed_like_bite
        
    
    def _decompose(self):
        U, E, Vt = scipy_linalg.svd(self.ppmi, full_matrices = False)
        U = np.matrix(U) #compute U
        E = np.matrix(np.diag(E)) #compute E
        Vt = np.matrix(Vt) #compute Vt = conjugate transpose of V
        V = Vt.T #compute V = conjugate transpose of Vt
        if self._sanity_check_decomposition(U, E, Vt):
            print("The original matrix is successfully recovered by multiplying U, E, and Vt")
        return V

    
    def _sanity_check_decomposition(self, U, E, Vt):
        ppmi = np.dot(np.dot(U, E), Vt)
        return np.allclose(ppmi, self.ppmi)
    
    
    def reduced_eucl_distance(self):
        V = self._decompose()
        #reduce the dimensions to 3 to get word vectors
        reduced_ppmi = self.ppmi * V[:, 0:3]

        #normalize vectors
        women = reduced_ppmi[self.word_dict['women'], :]/scipy_linalg.norm(reduced_ppmi[self.word_dict['women'], :])
        men = reduced_ppmi[self.word_dict['men'], :]/scipy_linalg.norm(reduced_ppmi[self.word_dict['men'], :])
        dogs = reduced_ppmi[self.word_dict['dogs'], :]/scipy_linalg.norm(reduced_ppmi[self.word_dict['dogs'], :])
        feed = reduced_ppmi[self.word_dict['feed'], :]/scipy_linalg.norm(reduced_ppmi[self.word_dict['feed'], :])
        like = reduced_ppmi[self.word_dict['like'], :]/scipy_linalg.norm(reduced_ppmi[self.word_dict['like'], :])
        bite = reduced_ppmi[self.word_dict['bite'], :]/scipy_linalg.norm(reduced_ppmi[self.word_dict['bite'], :])
        #euclidean distance 
        ed_women_men = scipy_linalg.norm(women - men)
        ed_women_dogs = scipy_linalg.norm(women - dogs)
        ed_men_dogs = scipy_linalg.norm(men - dogs)
        ed_feed_like = scipy_linalg.norm(feed - like)
        ed_feed_bite = scipy_linalg.norm(feed - bite)
        ed_like_bite = scipy_linalg.norm(like - bite)
        return ed_women_men,ed_women_dogs,ed_men_dogs,ed_feed_like,ed_feed_bite,ed_like_bite
        
    
    def synonym_detection(self, syn_file, word_dict, word_vectors):
        """
        put together _parse_word_vectors, synonym_test, and accuracy methods
        return the accuracy results from euclidean distance and cosine similarity
        applied on the classic word vectors and google word vectors
        
        """

        syn_pairs = self._get_syn_pairs(syn_file)
        
        #self._create_questions(syn_file)
        
        #load test set from file
        with open('synonym_test_set.txt') as infile:
            content = infile.read()
        
        test_set = []
        questions = content.strip().split('\n\n')
        for question in questions:
            item = []
            for q in question.split('\n'):
                item.append((q.split('\t')[0], q.split('\t')[1]))
            test_set.append(item)
                   
        results_ed, results_cosine = self.synonym_test(test_set, word_dict, word_vectors, syn_file)
        accuracy_ed = self.accuracy(syn_pairs, results_ed)
        accuracy_cosine = self.accuracy(syn_pairs, results_cosine)
        
        return accuracy_ed, accuracy_cosine


    def _create_questions(self, syn_file):
        """
        use to create the question data set for synonym detection
        generate file synonym_question_pairs.txt
        
        """
        
        syn_pairs = self._get_syn_pairs(syn_file)
        syn_pairs = syn_pairs*3
        random.shuffle(syn_pairs)
        
        n = 1000
        with open('synonym_test_set.txt', 'w') as f:
            for input_word, answer in syn_pairs:
                if n==0:
                    return
                #exclude 0 and multi-expression words
                if answer != '0' and len(answer.split('-'))==1:
                    #add one synonym
                    f.write(input_word + '\t' + answer + '\n')
                
                    #add four non-synonym pairs along with the one synonym pair
                    for i in range(4):
                        r = random.randint(0, len(syn_pairs)-1)
                        
                        while syn_pairs[r][0] == input_word or syn_pairs[r][1]=='0' or len(syn_pairs[r][1].split('-'))>1:
                            r = random.randint(0, len(syn_pairs)-1)
                        
                        f.write(input_word + '\t' + syn_pairs[r][1] + '\n')
                    f.write('\n')
                    n -= 1
                

    
    def synonym_test(self, test_set, word_dict, word_vectors, syn_file):
        """
        
        read in the classic and google file
        apply eucliean distance and cosine similarity on the word vectors
        compute the accuracy
        return the accuracy using eucliean distance and cosine similarity
        
        """
        #use the word vectors to pick the synonym out of the 5 alternatives for
        #each verb and compute the accuracy. 
        #Euclidean Distance - classic
        ed_results = []
        cs_results = []
        for question in test_set:
            ed_dist = []
            cs_dist = []
            for input_word, answer in question:
                w = input_word
                a = answer
                
                if w in word_dict and a in word_dict:
                    v1 = word_vectors[word_dict[w], :]/np.linalg.norm(word_vectors[word_dict[w], :])
                    v2 = word_vectors[word_dict[a], :]/np.linalg.norm(word_vectors[word_dict[a], :])
                    ed_dist.append(np.linalg.norm(v1-v2))
                    
                    #Cosine similarity 
                    a1 = word_vectors[word_dict[w], :] #input word
                    a2 = word_vectors[word_dict[a], :] #answer
                    cosine_similarity = self._cosine_dist(a1, a2)
                    cs_dist.append(cosine_similarity)
                else:
                    ed_dist.append(1000) #unknown words
                    cs_dist.append(-1) #unknown words

            ed_results.append(question[ed_dist.index(min(ed_dist))])
            cs_results.append(question[cs_dist.index(max(cs_dist))])
        
        return ed_results, cs_results
        
    
    def _get_syn_pairs(self, source_file):
        """
        strip 'to_' from the synonym pairs
        keep only the words
        return tuples of synonym pairs in a list
        """
        syn_pairs = []
        with open(source_file) as f:
            lines = f.read().split('\n')
        for line in lines[1:]:
            if line:
                w = line.split()[0].split('_',1)[1]
                if line.split()[1] =='0':
                    a = '0'
                else:
                    a = line.split()[1].split('_',1)[1]
                syn_pairs.append((w, a))
        return syn_pairs
    
    
    def accuracy(self, syn_pairs, results):
        """
        compute the accuracy after using the word vectors 
        to pick the synonym out of the 5 alternatives for each verb
        
        """
        
        counter = 0
        for result in results:
            if result in syn_pairs:
                counter += 1
        return counter/len(results)
        
    
    def parse_word_vectors(self, file):
        """
        parse the distributional semantics matrix files
        1.Classic distributional semantic matrix
        2. Google’s souped-up hyped-up 300-dimensional 
            word vectors trained by deep learning
        
        """
        word_vectors = []
        word_dict = {}
        i = 0
        with open(file) as f:
            lines = f.readlines()
        for line in lines:
            if line:
                words = line.split()
                word_dict[words[0]] = i
                word_vectors.append(words[1:])
                i += 1
        word_vectors = np.array(word_vectors)
        word_vectors = word_vectors.astype(np.float)
        return word_dict, word_vectors

    
    def parse_sat(self, file):
        """
        parse SAT analogy question data
        each quesiton is saved in a nested list within a list
        ('lull', 'trust', '-1') #example: -1
        ('balk', 'fortitude', '0') #wrong answer: 0
        ('betray', 'loyalty', '1') #correct answer: 1
        
        """
        with open(file) as f:
            content = f.read()
        
        lines = content.split('\n\n')

        for items in lines:
            question = []
            item = items.split('\n')

            if '190 FROM REAL SATs' in item[0]:
                answer_index = ord(item[7]) - 96
                question.append((item[1].split()[0], item[1].split()[1], '-1'))
                question.append((item[answer_index + 1].split()[0], item[answer_index + 1].split()[1], '1'))
                for i in range(2, 7):
                    if i != answer_index + 1:
                        question.append((item[i].split()[0], item[i].split()[1], '0'))
                self.analogy_pairs.append(question)
        
        
    def find_distance_sat(self, word_dict, word_vectors):
        """
        use tranditional and google word vectors to get distance
        1. calculate distance for the two words in a pair
            v1 = (lull, trust) distance between lull and trust
            v2 = (balk, fortitude) distane between balk and fortitude
            v3 = (cajole, compliance) distance between cajole and compliance
        2. compuate the cosine similarity between v1 and v2, v1 and v3 
            where v1 is the example
        3. pick the pair that is closest to the example
        
        """
        
        distance_set = []
        for question in self.analogy_pairs:
            item = []
            
            if question[0][2] == '-1' and (question[0][0] not in word_dict or question[0][1] not in word_dict):
                for w1, w2, is_answer in question[1:]:
                    item.append(((w1, w2), 0, is_answer)) #question is out of vocabulary, pick an answer randomly         
            else:
                for w1, w2, is_answer in question:
                    if w1 in word_dict and w2 in word_dict:
                        diff_vector = word_vectors[word_dict[w1]] - word_vectors[word_dict[w2]]
                        item.append(((w1, w2), diff_vector, is_answer))
            distance_set.append(item)
            
            
        results = []
        for question in distance_set:
            item = []
            
            if type(question[0][1]) == int:
                  
                r = random.randint(1, len(question)-1)
                results.append([(3, question[r][0], question[r][2])])
            else:
                question_vector = question[0][1]
                for (w1, w2), diff_vector, is_answer in question:
                    dist = 1-self._cosine_dist(question_vector, diff_vector)
                    item.append((dist, (w1, w2), is_answer))
            
                #sort the cosine similarity in the ascending order
                #the lowest distance should be the answer
                item.sort(key = lambda tup: tup[0])
                
                results.append(item)
            
        return results
                    
        
    def accuracy_sat(self, results):
        """
        compute the accuracy of the model for SAT Analogy question 
        
        """
        
        counter = 0
        for answer in results:
            if len(answer) == 1: #question is oov
                if answer[0][2] == '1':
                    counter += 1
            elif answer[1][2] == '1':
                counter += 1
        return counter/len(results)
            
        
    def _cosine_dist(self, v1, v2):
        """
        compute the cosine distance between two given vectors v1 and v2
        
        """
        return np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
            
    
        
if __name__ == '__main__':
    wv = WordVector()
    filename = 'data/dist_sim_data.txt'
    print('Section 1: \n')
    c_matrix, ppmi = wv.term_context(filename)
    print('co-occurrence matrix \n', c_matrix)
    print('ppmi \n', ppmi)
    
    d1,d2,d3,d4,d5,d6 = wv.euclidean_distance()
    print()
    print('Euclidean distance using PPMI: \n')
    pt = PrettyTable(['women/men', 'women/dogs', 'men/dogs', 'feed/like', 'feed/bite', 'like/bite' ])
    pt.add_row([round(d1,4), round(d2,4), round(d3,4), round(d4,4), round(d5,4), round(d6,4)])
    print(pt)
    
    print()
    pt = PrettyTable(['women/men', 'women/dogs', 'men/dogs', 'feed/like', 'feed/bite', 'like/bite'])
    rd1,rd2,rd3,rd4,rd5,rd6 = wv.reduced_eucl_distance()
    pt.add_row([round(rd1,4), round(rd2,4), round(rd3,4), round(rd4,4), round(rd5,4), round(rd6,4)])
    print('Euclidean distance using reduced PPMI-weighted count matrix')
    print(pt)
    
    
    print('\n\n')
    print('Section 2: \n')
    syn_file = 'data/EN_syn_verb.txt' 
    composes_file = 'data/EN-wform.w.2.ppmi.svd.500.rcv_vocab.txt' 
    google_file = 'data/GoogleNews-vectors-rcv_vocab.txt'
    word_dict_composes, word_vectors_composes = wv.parse_word_vectors(composes_file)
    word_dict_google, word_vectors_google = wv.parse_word_vectors(google_file)

    #detect synonyms
    composes_accuracy_ed, composes_accuracy_cosine = wv.synonym_detection(syn_file, word_dict_composes, word_vectors_composes)
    google_accuracy_ed, google_accuracy_cosine = wv.synonym_detection(syn_file, word_dict_google, word_vectors_google)
    pt = PrettyTable(['Model', 'Euclidean distance', 'Cosine similarity'])
    pt.add_row(['Composes', round(composes_accuracy_ed,4), round(composes_accuracy_cosine,4)])
    pt.add_row(['Google', round(google_accuracy_ed,4), round(google_accuracy_cosine,4)])
    print(pt)
    print()
    
    #SAT analogy questions
    sat_file = 'data/SAT-package-V3.txt'
    wv.parse_sat(sat_file)
    
    #run the model using the google file
    google_results = wv.find_distance_sat(word_dict_google, word_vectors_google)
    google_accuracy = wv.accuracy_sat(google_results)
    #run the model using the composes file
    composes_results = wv.find_distance_sat(word_dict_composes, word_vectors_composes)
    composes_accuracy = wv.accuracy_sat(composes_results)
    pt = PrettyTable(['Accuracy(Google file)', 'Accuracy(Composes file)'])
    pt.add_row([round(google_accuracy,4), round(composes_accuracy,4)])
    print(pt)
    