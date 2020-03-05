from collections import defaultdict
from languageModel import LanguageModel
import random
import bisect

'''
Tuan Do, Kenneth Lai
'''
class BigramAddK(LanguageModel):

    def __init__(self):
        # P(word2|word1) = self.probCounter[word1][word2]
        self.probCounter = defaultdict(lambda: defaultdict(float))
        self.rand = random.Random()
        self.k = 1
        self.v = 0
        
    '''bigram counts only contain bigrams that have non-zero actual
    counts in the training data.
    add kV to self.total[word1], so that after dividing, the values of
    self.probCounter have formed C(w_(n-1)W_n)/(C_w(n-1) + kV)
    '''
    def train(self, trainingSentences):
        self.accu = defaultdict(list)
        self.total = defaultdict(int)
        
        for sentence in trainingSentences:
            for i in range(1, len(sentence)):
                self.probCounter[sentence[i - 1]][sentence[i]] += 1
                self.total[sentence[i - 1]] += 1
                                
            #start of the context
            self.probCounter[LanguageModel.START][sentence[0]] += 1
            self.total[LanguageModel.START] += 1
                    
            #end of the context                
            self.probCounter[sentence[-1]][LanguageModel.STOP] += 1
            self.total[sentence[-1]] += 1
                
        #add kV (k = 1), V is the number of all possible bigrams
        #subtract 1 for each as start can't apear in the second position 
        #and stop can't appear in the first position
        self.v = len(self.probCounter)

        for word1 in self.total:
            self.total[word1] += self.k * (self.v)
q
        for word1 in self.probCounter: 
            for word2 in self.probCounter[word1]:
                if len(self.accu[word1]) == 0:
                    self.accu[word1] = [self.probCounter[word1][word2]]
                else:
                    self.accu[word1].append(self.accu[word1][-1] + self.probCounter[word1][word2])
                    
                self.probCounter[word1][word2] /= self.total[word1]                               

    
    '''
        add missing k/(C_{n-1} + kV)    
        return the probability of the bigram
        if the bigram is 0, return k/(C_{n-1} + kV)
        unknown words also return k/(C_{n-1} + kV)
    '''
    def getWordProbability(self, sentence, index):
        #vocab = self.getVocabulary(sentence)
        #start of the sentence        
        if index == 0:
            if sentence[index] in self.probCounter[LanguageModel.START]: #bigram
                return self.probCounter[LanguageModel.START][sentence[index]] + self.k/(self.total[LanguageModel.START])
            else:
                return self.k/(self.total[LanguageModel.START])
                  
        #end of the sentence
        if index == len(sentence):
            if sentence[index-1] in self.probCounter:
                if LanguageModel.STOP in self.probCounter[sentence[index-1]]:
                    return self.probCounter[sentence[index-1]][LanguageModel.STOP]+ self.k/(self.total[sentence[index-1]])
                else: #word2 is unknown & known word, not a bigram
                    return self.k/(self.total[sentence[index-1]])
            else: #word1 is unknown
                return self.k/(self.k*(self.v ))

        #non-start, non-end of the sentence (regular words)
        if sentence[index-1] in self.probCounter:
            if sentence[index] in self.probCounter[sentence[index-1]]: #bigram
                return self.probCounter[sentence[index-1]][sentence[index]]+ self.k/(self.total[sentence[index-1]])
            else: #word2 is unknown & known word, not a bigram
                return self.k/(self.total[sentence[index-1]])
        else: #word1 is unknown
            return self.k/(self.k*(self.v ))
            

        
    def getVocabulary(self, context):
        vocab = [word for word in self.probCounter if word != LanguageModel.START]
        vocab.append(LanguageModel.STOP)
        return vocab 


    def generateWord(self, context):
        #generate random integer
        if len(context) == 0:
            i = self.rand.randint(0, self.total[LanguageModel.START] - 1)
            while i > self.accu[LanguageModel.START][-1]:                   
                i = random.choice(self.accu[LanguageModel.START])
            index = bisect.bisect(self.accu[LanguageModel.START], i)-1
            return list(self.probCounter[LanguageModel.START])[index]
        else:
            i = self.rand.randint(0, self.total[context[-1]] - 1)
            while i > self.accu[context[-1]][-1]:                 
                i = random.choice(self.accu[context[-1]]) 
            index = bisect.bisect(self.accu[context[-1]], i) - 1
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
