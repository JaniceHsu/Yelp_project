# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 20:31:10 2017

@author: Janice
#http://www.nltk.org/book/ch06.html

"""

import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import FreqDist
import csv 
import random
import nltk





def reduceText(text, stops):
    text = text.lower() #assign 1
    text = re.sub('\W', ' ', text) #not in set [a-zA-Z0-9_], general purpose to remove punctuation
    tokens = word_tokenize(text)
    return [w for w in tokens if not w in stops]

def get_word_features(docs):
    posWords = []
    negWords = []
    for t, s in docs:
        if s == '1':
            posWords.extend(t)
        else:
            negWords.extend(t)
    word_features = nltk.FreqDist(posWords+negWords)
    return [w for w, n in word_features.most_common()[:2000]]            

def document_features(document, word_features):
    document_words = set(document) 
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

        
def measureNaiveBayes(fname):
    f = open(fname, 'r', encoding = 'utf8')
    reader = csv.reader(f)
    
    docs = []
    
    stops = set(stopwords.words("english"))                  
    for row in reader:
        break
    for row in reader:
        docs.append((reduceText(row[0], stops), row[1]))
    random.shuffle(docs)
    
    print("collected text and suffled docs")
        
    posWords = []
    negWords = []
    
    for t, s in docs:
        if s == '1':
            posWords.extend(t)
        else:
            negWords.extend(t)
    word_features = nltk.FreqDist(posWords+negWords)
    word_features = [w for w, n in word_features.most_common()[:500]]
    
    featuresets = [(document_features(d, word_features), c) for (d,c) in docs]
    total = len(featuresets) #split numbers
    trainNum = int(total*.8)
    train_set, test_set = featuresets[:trainNum], featuresets[trainNum:]
    print("got feature sets and now classifying")
                     
    wf = open("resultNaiveBayesTestCSV.txt", 'w')
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    wf.write(str(nltk.classify.accuracy(classifier, test_set)))
    wf.write("\n")
    wf.write(str(classifier.show_most_informative_features(20)))
    wf.close()
        
    f.close()
    print("END")
def base(fname):
    f = open(fname, 'r', encoding = 'utf8')
    reader = csv.reader(f)
    pos = 0
    neg = 0
    for row in reader:
        break
    for row in reader:
        if row[1]== '1':
            pos+=1
        else:
            neg+=1
    m = 1 if pos>neg else 0
    print("More :{}".format(m))
    m = pos if pos>neg else neg
    print(m/(pos+neg))
        
    f.close()
    
    
if __name__ == "__main__":
    fname = r"reviews_ChinesePN.csv"
    
    base(fname)