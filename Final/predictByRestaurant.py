# -*- coding: utf-8 -*-
"""
Based on predictAttribSort.py results
Use LogReg, and 3 docs for min_docs
            Do not include terms in the vocabulary that occur in less than "min_docs" documents 
            
Provide a summary on a restaurant by providing statistics
on aspects and their sentiments mentioned in Yelp Reviews.

Example outputs:
======= Jugoo Chinese Restaurant =======
    Aspect |     food |    price |  service
mentioned  |   22.54% |    4.93% |   12.68%
positively |   87.50% |   71.43% |  100.00%
negatively |    9.38% |   28.57% |    0.00%

======= Rose Garden Chinese Restaurant =======
    Aspect |     food |    price |  service
mentioned  |   25.71% |    3.57% |    9.29%
positively |   69.44% |   40.00% |   92.31%
negatively |   11.11% |   60.00% |    0.00%

credit: extract_text_features heavily influneced by CS175 assignment2. 
else-- self work

"""


import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from variousLearnersOverall import fit_and_predict_LR
from predictAttribSort import getRows


def collect_data(fn, doc_nums):
    docs = []
    for i in doc_nums:
        fname = fn+str(i)+".csv"
        f= open(fname, 'r', encoding = 'utf8')
        reader = csv.reader(f)
        data, attribs = getRows(reader)
        docs.extend(data)
        f.close() 
    return np.array(docs), attribs

def extract_text_features(train_data, test_data):
    """
    In part, from CS 175 assignment 2.
    Min_df = 3 since prediction accuracy was highest for this value when evaluting with cross validation.
    
    Parameters
    ----------
    train_data : List[str]
    test_data : List[str]   

    Returns two types of training and test data features.
        1) Bags of words (BOWs): X_train_counts, X_test_counts
        2) Term Frequency times Inverse Document Frequency (tf-idf): X_train_tfidf, X_test_tfidf
    """
    # Generate count vectors from the input data, excluding the NLTK stopwords and
    count_vect = CountVectorizer(min_df=3, stop_words ='english')
    
    # Bags of words (BOWs): X_train_counts, X_test_counts
    X_train_counts = count_vect.fit_transform(train_data) 
    X_test_counts = count_vect.transform(test_data)
    
    #Term Frequency times Inverse Document Frequency (tf-idf): X_train_tfidf, X_test_tfidf
    tfidf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    #fit/compute Tfidf weights using X_train_counts
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #apply fitted weights to X_test_counts
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    return (X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf)
    
    
def predict(train_docs, train_target, test_docs):
    # gather predictions and collect stats
    X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf = extract_text_features(train_docs,  test_docs)
    
    num_docs, vocab_size = X_train_counts.shape
    predicted = fit_and_predict_LR(X_train_counts, train_target, X_test_counts)
    
    #collect stats
    pos = 0
    neg = 0
    mentioned = 0
    
    for i in predicted:
        if i != '-':
            mentioned +=1
            if i == '2':
                pos+=1
            if i == '0':
                neg+=1
            # else neutral/ mentioned both positively and negatively
    return (pos, neg, mentioned, num_docs)

    
def eval_by_attrib(train_data, test_data, i):
    # Separate data into review and target
    # Then get predictions/stats
    train_docs = []
    train_target = []

    test_docs = []
    # test_target = [] # Don't care about the target

    for row in train_data:
        train_docs.append(row[0])
        train_target.append(row[i])
    
    for row in test_data:
        test_docs.append(row[0])
        #test_target.append(row[i]) # Don't care about the target
    return predict(train_docs, train_target, test_docs)
    
def eval_restaurant(r):
    # evaluate a particular restaurant r
    # all other restaurants are designated as training data
    train_rs = [1,2,3,4,5]
    train_rs.remove(r)
    predict_r = [r]

    train_data, attribs = collect_data(r"labeledReview", train_rs)
    test_data, attribs = collect_data(r"labeledReview", predict_r)
    #attribs: ['text', 'food', 'price', 'service']
    
    # these are the 5 restaurants that have target labels
    names = ['Jugoo Chinese Restaurant', 'Rose Garden Chinese Restaurant',\
             'Yummy Palace', 'Panda Express', 'Shanghai Lilly'] 
             
    print("======= {:^20} =======".format(names[r-1]))
    stats = []
    for i in range(1,4):
        pos, neg, mentioned, num_docs = (eval_by_attrib(train_data, test_data, i))
        stats.append([mentioned/num_docs*100, pos/mentioned*100, neg/mentioned*100])
    print("{:>10} | {:>8} | {:>8} | {:>8}".format("Aspect", attribs[1], attribs[2], attribs[3]))
    print("{:10} |{:8.2f}% |{:8.2f}% |{:8.2f}%".format("mentioned", stats[0][0], stats[1][0], stats[2][0]))
    print("{:10} |{:8.2f}% |{:8.2f}% |{:8.2f}%".format("positively", stats[0][1], stats[1][1], stats[2][1]))
    print("{:10} |{:8.2f}% |{:8.2f}% |{:8.2f}%".format("negatively", stats[0][2], stats[1][2], stats[2][2]))
    
if __name__ == "__main__":
    eval_restaurant(1)
    eval_restaurant(2)
        
    