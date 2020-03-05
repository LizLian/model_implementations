from collections import defaultdict
from languageModel import LanguageModel
import random
import bisect

'''
Tuan Do, Kenneth Lai
'''
class Bigram(LanguageModel):

    def __init__(self):
        # P(word2|word1) = self.probCounter[word1][word2]
        self.probCounter = defaultdict(lambda: defaultdict(float))
        self.rand = random.Random()
    
    '''
    return the probability of bigrams (the probability of word 2 given word 1)
    '''
    def train(self, trainingSentences):
        self.accu = defaultdict(list)
        self.total = defaultdict(int)
        
        for sentence in trainingSentences:
            for i in range(1, len(sentence)):
                self.probCounter[sentence[i - 1]][sentence[i]] += 1
                self.total[sentence[i - 1]] += 1
                 
                #word 1 (given word) is unknown
                if self.probCounter[LanguageModel.UNK][sentence[i]] != 1:
                    self.probCounter[LanguageModel.UNK][sentence[i]] = 1
                    self.total[LanguageModel.UNK] += 1 
                                
            #start of the context
            self.probCounter[LanguageModel.START][sentence[0]] += 1
            self.total[LanguageModel.START] += 1
                    
            #end of the context                
            self.probCounter[sentence[-1]][LanguageModel.STOP] += 1
            self.total[sentence[-1]] += 1
            
        #word2 is unknown
        for word1 in self.probCounter: #this includes LanguageModel.START and LanguageModel.UNK
            if word1 != LanguageModel.STOP: #exclude LanguageModel.STOP
                self.probCounter[word1][LanguageModel.UNK] = 1
                self.total[word1] += 1

        #last word unknown
        self.probCounter[LanguageModel.UNK][LanguageModel.STOP] = 1
        self.total[LanguageModel.UNK] += 1

        #self.accu --> 
        #{'restructured': [1.0], 'situations': [4.0, 5.0, 6.0, 10.0, 14.0, 16.0, 18.0, 27.0, 28.0, 
        #29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0]}
        for word1 in self.probCounter: 
            for word2 in self.probCounter[word1]:
                if len(self.accu[word1]) == 0:
                    self.accu[word1] = [self.probCounter[word1][word2]]
                else:
                    self.accu[word1].append(self.accu[word1][-1] + self.probCounter[word1][word2])
                                               
            for word2 in self.probCounter[word1]:
                self.probCounter[word1][word2] /= self.total[word1]

        
    '''return the probablity of the word at index, given word at index - 1
    '''
    def getWordProbability(self, sentence, index):
        vocab = self.getVocabulary(sentence[0:index])
        #start of the sentence
        if index == 0:
            if sentence[index] in self.probCounter[LanguageModel.START]:
                return self.probCounter[LanguageModel.START][sentence[index]]
            else:
                return self.probCounter[LanguageModel.START][LanguageModel.UNK]
                  
        #end of the sentence
        elif index == len(sentence):
            if sentence[index-1] in self.probCounter and LanguageModel.STOP in vocab:
                return self.probCounter[sentence[index-1]][LanguageModel.STOP]
            elif sentence[index-1] in self.probCounter and LanguageModel.STOP not in vocab:
                return self.probCounter[sentence[index-1]][LanguageModel.UNK]
            else:
                return self.probCounter[LanguageModel.UNK][LanguageModel.STOP]
        
#        if sentence[index-1] not in self.probCounter:
#            return 1/len(self.probCounter[LanguageModel.UNK])
        if sentence[index-1] in self.probCounter:
            if sentence[index] in self.probCounter[sentence[index-1]]: #the bigram is in the training data
                return self.probCounter[sentence[index - 1]][sentence[index]]
            else:
                return self.probCounter[sentence[index-1]][LanguageModel.UNK]
        elif sentence[index] in self.probCounter[LanguageModel.UNK]:
            return self.probCounter[LanguageModel.UNK][sentence[index]]
            
        return self.probCounter[LanguageModel.UNK][LanguageModel.UNK]
            
    
    '''return a list of possible words that could possible be 
    given the list of previous words in the sentence
    '''    
    def getVocabulary(self, context):
        vocab = []
        if len(context) == 0:
            vocab = [word2 for word2 in self.probCounter[LanguageModel.START]
                        if self.probCounter[LanguageModel.START][word2] > 0]
        elif context[-1] in self.probCounter:
            vocab = list(self.probCounter[context[-1]].keys())
        else:
            vocab = self.probCounter[LanguageModel.UNK] #word1 is unknown
            #vocab = [LanguageModel.UNK]
        return vocab

    '''generate a word
    '''
    def generateWord(self, context):   
        #generate random integer
        if len(context) == 0:
            i = self.rand.randint(0, self.total[LanguageModel.START] - 1)
            index = bisect.bisect(self.accu[LanguageModel.START], i)
            return list(self.probCounter[LanguageModel.START])[index]
        else:
            i = self.rand.randint(0, self.total[context[-1]] - 1)
            index = bisect.bisect(self.accu[context[-1]], i)
            return list(self.probCounter[context[-1]])[index]
        
    def generateSentence(self):
        result = []
        # limit sentence length to 20
        for i in range(20):
            word = LanguageModel.UNK
            while word == LanguageModel.UNK:
                # make sure word != UNK
                word = self.generateWord(result)
            result.append(word)
            if word == LanguageModel.STOP:
                break
        return result
