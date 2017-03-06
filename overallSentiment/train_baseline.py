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

        
if __name__ == "__main__":
    fname = r"reviews_ChinesePN_train.csv"
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
    
    word_features = get_word_features(docs)
    
    featuresets = [(document_features(d, word_features), c) for (d,c) in docs]
    total = len(featuresets) #split numbers
    trainNum = int(total*.8)
    train_set, test_set = featuresets[:trainNum], featuresets[trainNum:]
    print("got feature sets and not classifying")
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))    
    classifier.show_most_informative_features(20)     
    
    f.close()
    print("END")