from collections import defaultdict
from languageModel import LanguageModel
from unigram import Unigram
from bigram import Bigram
import random
import bisect

'''
Tuan Do, Kenneth Lai
'''
class BigramInterpolation(LanguageModel):

    def __init__(self):
        self.unigram = Unigram()
        self.bigram = Bigram()
        self.lambda2 = 0.80
        self.lambda1 = 1 - self.lambda2
    
    def train(self, trainingSentences):
        self.unigram.train(trainingSentences)
        self.bigram.train(trainingSentences)
    
    def getWordProbability(self, sentence, index):
        if index == len(sentence):
            p2 = self.lambda2 * self.bigram.getWordProbability(sentence, index)
            p1 = self.lambda1 * self.unigram.getWordProbability(sentence, index) 
            return p1 + p2
        
        vocab = self.bigram.getVocabulary(sentence[0:index])
            
        if sentence[index] not in vocab:
            p2 = 0
        else:
            p2 = self.lambda2 * self.bigram.getWordProbability(sentence, index)
        p1 = self.lambda1 * self.unigram.getWordProbability(sentence, index) 
        return p1+p2
        
    def getVocabulary(self, context):
        return [word for word in self.unigram.probCounter if word != LanguageModel.START]

    def generateWord(self, context):
        #generate a random real number
        #if the random real number is less than lambda1 then use unigram, otherwise use the bigram
        #this idea is from Deanna Daly
        i = random.uniform(0, 1)
        if i < self.lambda1:
            return self.unigram.generateWord(context)
        else:
            return self.bigram.generateWord(context)

        
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
