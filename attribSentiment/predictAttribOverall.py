# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 22:41:49 2017

@author: Janice
"""

import csv
import nltk
import random
import numpy as np
from sklearn.cross_validation import KFold


#from nltk import FreqDist
#from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer

def getRows(reader):
    docs = []
    for row in reader:
        attribs = row
        break
    for row in reader:
        docs.append([row[3], row[10], row[11], row[12]])
    return docs, attribs
    

    
def collect_data(fn):
    docs = []
    for i in range(1,6):
        fname = fn+str(i)+".csv"
        f= open(fname, 'r', encoding = 'utf8')
        reader = csv.reader(f)
        data, attribs = getRows(reader)
        docs.extend(data)
        f.close() 
    return np.array(docs), attribs
    

     
def eval_by_attrib(data, attribs, i):
    #y[:,0] to get column 0 for a 2d numpy array
    print(attribs[i])
    #target = data[:,i]
    #docs = data[:,0]
    target = []
    docs = []
    for n in range(len(data)):
        if data[n][i] != '1': #ignore if mentioned and not necessarily positive or negative
            target.append(data[n][i])
            docs.append(data[n][0])

    n = len(docs)
    target = np.array(target)
    docs = np.array(docs)    
    #cross validation
    fold_index = KFold(n, n_folds=5) #stackoverflow
    iteration = 1
    for trainI, testI in fold_index:
        print("Iteration", iteration)
        iteration +=1
        train_docs, test_docs = docs[trainI], docs[testI]
        train_target, test_target = target[trainI], target[testI]

        for i in [1,3,5,10]:
            print("\ntest_classifiers(twenty_train, twenty_test, {} [min_docs])".format(i))
            test_classifiers(train_docs, train_target, test_docs, test_target, i, 3, 5, True) #last paramter -- remove stopwords or not
            test_classifiers(train_docs, train_target, test_docs, test_target, i, 3, 5, False) #last paramter -- remove stopwords or not


'''    for i in range(5):
        train_docs = docs[i*size:(i*size+4*size)%n]
        test_docs = docs[(i*size+4*size)%n:(i*size+5*size)%n]
        print(len(train_docs))
        print(len(test_docs))
        print("__")'''
        
    
    
def test_classifiers(train_docs, train_target, test_docs, test_target,  min_docs, K, K2, removeStopWords):
    """
    Evaluate the accuracy of multiple classifiers by training on the data in 
    "train" and making predictions on the data in "test". The classifiers
    evaluated are: BernoulliNB, MultinomialNB, Logistic, and kNN.
    
    The input train and test data are scikit-learn objects of type "bunch"
    containing both the raw text as well as label information.
    
    The function first calls extract_text_features() to create a common
    vocabulary and feature set for all the classifiers to use.
    
    The classifiers should use tfidf features.
    
    
    Parameters
    ----------
    train: sklearn.datasets.base.Bunch
        Text data with labels for training each classifier
    test: sklearn.datasets.base.Bunch
        Text data with labels for testing each classifier
    min_docs : integer
        Do not include terms in the vocabulary that occur in less than "min_docs" documents    
    K: integer (odd)
        Number of neighbors to use for prediction, e.g., K = 1, 3, 5, ...
 
    """
    #        test_classifiers(train_docs, train_target, test_docs, test_targets, i, 3)
    X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf = extract_text_features(train_docs,  test_docs, min_docs, removeStopWords)
    
    
    num_docs, vocab_size = X_train_counts.shape
    print('Number of (training) documents =',num_docs)
    print('Vocabulary size =',vocab_size)
    

    # Now evaluate the classifiers on the test data
    # Print out the accuracy as a percentage for each classifier.
    # np.mean() can be used to calculate the accuracy. Round the accuracy to 2 decimal places.

    #predict according to different classifier--evaluate results    
    predicted_multNB = fit_and_predict_multinomialNB(X_train_tfidf, train_target, X_test_tfidf)
    predicted_bernNB = fit_and_predict_BernoulliNB(X_train_tfidf, train_target, X_test_tfidf)
    predicted_LR = fit_and_predict_LR(X_train_tfidf, train_target, X_test_tfidf)
    predicted_LR = fit_and_predict_LR(X_train_counts, train_target, X_test_counts)
    predicted_KNN = fit_and_predict_KNN(X_train_tfidf, train_target, X_test_tfidf, K)
    predicted_KNN2 = fit_and_predict_KNN(X_train_tfidf, train_target, X_test_tfidf, K2)
    
    predicted_base = np.array([FreqDist(test_target).most_common(1)[0][0]]*len(test_target))

    # count num of correct predictions / total
    np_test_target = np.array(test_target)
    base = np.sum(predicted_base == np_test_target)/len(np_test_target)
    multNB = np.sum(predicted_multNB ==  np_test_target)/len(np_test_target) 
    bernNB = np.sum(predicted_bernNB ==  np_test_target)/len(np_test_target)
    LR = np.sum(predicted_LR == np_test_target)/len(np_test_target)
    KN = np.sum(predicted_KNN == np_test_target)/len(np_test_target)
    KN2 = np.sum(predicted_KNN2 == np_test_target)/len(np_test_target)

    
    print('Base Accuracy: {:.3f}'.format(base))
    print('Accuracy with multinomial naive Bayes: {:.3f}'.format(multNB))
    print('Accuracy with Bernoulli naive Bayes: {:.3f}'.format(bernNB))
    print('Accuracy with logistic regression: {:.3f}'.format(LR))
    print('Accuracy with kNN, k={} classifier: {:.3f}'.format(K, KN))
    print('Accuracy with kNN, k={} classifier: {:.3f}'.format(K2, KN2))
    
    
if __name__ == "__main__":
    data, attribs = collect_data(r"labeledChineseReview")
    random.shuffle(data)
    print(attribs)
    for i in range(1,4):
        eval_by_attrib(data, attribs, i)
    