# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 22:41:49 2017

@author: Janice
"""

import csv
from variousLearnersOverall import *
import nltk
import random
import numpy
from sklearn.cross_validation import KFold

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
    return numpy.array(docs), attribs
    

     
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
    target = numpy.array(target)
    docs = numpy.array(docs)    
    #cross validation
    fold_index = KFold(n, n_folds=5) #stackoverflow
    for trainI, testI in fold_index:
        train_docs, test_docs = docs[trainI], docs[testI]
        train_target, test_target = target[trainI], target[testI]


'''    for i in range(5):
        train_docs = docs[i*size:(i*size+4*size)%n]
        test_docs = docs[(i*size+4*size)%n:(i*size+5*size)%n]
        print(len(train_docs))
        print(len(test_docs))
        print("__")'''
        
    
    
    
    
if __name__ == "__main__":
    data, attribs = collect_data(r"labeledChineseReview")
    random.shuffle(data)
    print(attribs)
    for i in range(1,4):
        eval_by_attrib(data, attribs, i)
    