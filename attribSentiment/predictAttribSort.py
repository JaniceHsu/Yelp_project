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


from nltk import FreqDist
#from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from variousLearnersOverall import fit_and_predict_multinomialNB, fit_and_predict_BernoulliNB, fit_and_predict_LR, fit_and_predict_KNN



def getRows(reader):
    docs = []
    for row in reader:
        attribs = [row[3], row[10], row[11], row[12]]
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
    #wo -> without stopwords
    
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
    print("Number of documents:", len(docs))
    #cross validation
    fold_index = KFold(n, n_folds=5) #stackoverflow
    
    accur_wo_stopwords = []
    accur_w_stopwords = []  

    md_num = [1,3,5,10]
    for md in md_num:  
        #print("Min docs:", md)
        accur_folds_w = []
        accur_folds_wo = []



        iteration = 1 #to print info
        for trainI, testI in fold_index:
            #print("Iteration", iteration) #visual information
            iteration +=1
            train_docs, test_docs = docs[trainI], docs[testI]
            train_target, test_target = target[trainI], target[testI]

            predicted_base = np.array([FreqDist(test_target).most_common(1)[0][0]]*len(test_target))
            # count num of correct predictions / total
            np_test_target = np.array(test_target)
            base = np.sum(predicted_base == np_test_target)/len(np_test_target)
            
            
            accur_folds_w.append([base] + test_classifiers(train_docs, train_target, test_docs, test_target, md, 3, 5, True)) #last paramter -- remove stopwords or not
            accur_folds_wo.append([base] + test_classifiers(train_docs, train_target, test_docs, test_target, md, 3, 5, False)) #last paramter -- remove stopwords or not
                            
        # shape (6,) --> [base, multNB, bernNB, LR, KN3, KN5]
        # axis = 0 --> mean of each learner over all min_doc values
        accur_wo = np.mean(np.array(accur_folds_wo), axis = 0) 
        accur_w = np.mean(np.array(accur_folds_w), axis = 0) 
                    
        #save to later compare accuracy while taking into account min_df
        accur_wo_stopwords.append(accur_wo)
        accur_w_stopwords.append(accur_w)
        

    
    return np.array(accur_wo_stopwords), np.array(accur_w_stopwords)

def score_overall(accur_wo_stopwords, accur_w_stopwords):

    all_wo = np.array(accur_wo_stopwords)*100
    all_w = np.array(accur_w_stopwords)*100
    # convert to numpy array *100 so percentage format
    all_wo = np.array(accur_wo_stopwords)*100
    all_w = np.array(accur_w_stopwords)*100
    #round to 3 decimal places
    all_wo = np.round(all_wo, decimals = 3)
    all_w = np.round(all_w, decimals = 3)
    
    #compare learner scores overall (axis = 0) --> column
    score_wo = np.round(np.mean(all_wo, axis =0), decimals = 3)
    score_w = np.round(np.mean(all_w, axis = 0), decimals = 3)

    
    print("Compare Learners")
    print("without stopwords")    
    print("{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}".format("base", "multNB", "bernNB", "LogReg", "KN3", "KN5"))
    print("{:8}%{:8}%{:8}%{:8}%{:8}%{:8}%\n".format(score_wo[0], score_wo[1], score_wo[2], score_wo[3], score_wo[4], score_wo[5]))
    
    print("Difference of accuracy without stopwords vs. with stopwords")
    diff_wwo = np.round(score_wo-score_w, decimals = 3)
    print(diff_wwo)
    print("overall benefit to take out stopwords: {:3.2f}%\n".format(np.mean(score_wo-score_w)))
    
    md_num = [1,3,5,10]
    #compare mindoc scores (axis = 1) --> row
    print("Min_docs      [{:7},{:7},{:7},{:7}]".format(md_num[0], md_num[1], md_num[2], md_num[3]))
    md_wo = np.round(np.mean(all_wo, axis = 1), decimals = 3)
    md_w = np.round(np.mean(all_w, axis = 1), decimals = 3)
    print("w/o stopwords", md_wo)
    print("w/ stopwords ", md_w)
    sorted_md_vals = sorted(list(zip(md_num, md_wo)), key = lambda x:x[1])
    print("Num docs | score")
    for mdn, score in sorted_md_vals:
        print("{:6} -> {:3.3f}".format(mdn,score))
    


    
def extract_text_features(train_data, test_data, min_docs, remove_stop_words ):
    """
    Parameters
    ----------
    train_data : List[str]
    test_data : List[str]   
    min_docs : integer
        Do not include terms in the vocabulary that occur in less than "min_docs" documents 

    Returns two types of training and test data features.
        1) Bags of words (BOWs): X_train_counts, X_test_counts
        2) Term Frequency times Inverse Document Frequency (tf-idf): X_train_tfidf, X_test_tfidf

    How to create BOW features:
        CountVectorizer is optimized for creating a sparse matrix representing
        the bag-of-words counts for every document in a corpus of documents all at once.  Both
        objects are useful at different times.

    How to create tf-idf features:
        tf-idf features can be computed using TfidfTransformer with the count matrix (BOWs matrix)
        as an input. The fit method is used to fit a tf-idf estimator to the data, and the
        transform method is used afterwards to transform either the training or test count-matrix
        to a tf-idf representation. The method fit_transform strings these two methods together
        into one.


    Returns
    -------
    Tuple(scipy.sparse.csr.csr_matrix,..)
        Returns X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf as a tuple.

    """
    # Generate count vectors from the input data, excluding the NLTK stopwords and
    if remove_stop_words:
        count_vect = CountVectorizer(min_df=min_docs, stop_words ='english')
    else:
        count_vect = CountVectorizer(min_df=min_docs)        
    
    # Bags of words (BOWs): X_train_counts, X_test_counts
    X_train_counts = count_vect.fit_transform(train_data) #**SLIGHLTLY DIFFERENT DIM (2989, 3966)
    X_test_counts = count_vect.transform(test_data)
    
    #Term Frequency times Inverse Document Frequency (tf-idf): X_train_tfidf, X_test_tfidf
    tfidf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    #fit/compute Tfidf weights using X_train_counts
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #apply fitted weights to X_test_counts
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    return (X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf)
    
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
    #print('Number of (training) documents =',num_docs)
    #print('\tVocabulary size =',vocab_size)
    

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
    
    # count num of correct predictions / total
    np_test_target = np.array(test_target)    
    multNB = np.sum(predicted_multNB ==  np_test_target)/len(np_test_target) 
    bernNB = np.sum(predicted_bernNB ==  np_test_target)/len(np_test_target)
    LR = np.sum(predicted_LR == np_test_target)/len(np_test_target)
    KN = np.sum(predicted_KNN == np_test_target)/len(np_test_target)
    KN2 = np.sum(predicted_KNN2 == np_test_target)/len(np_test_target)

    """
    print('Base Accuracy: {:.3f}'.format(base))
    print('Accuracy with multinomial naive Bayes: {:.3f}'.format(multNB))
    print('Accuracy with Bernoulli naive Bayes: {:.3f}'.format(bernNB))
    print('Accuracy with logistic regression: {:.3f}'.format(LR))
    print('Accuracy with kNN, k={} classifier: {:.3f}'.format(K, KN))
    print('Accuracy with kNN, k={} classifier: {:.3f}'.format(K2, KN2))
    """
    
    return [multNB, bernNB, LR, KN, KN2]
    
    
if __name__ == "__main__":
    data, attribs = collect_data(r"labeledChineseReview")
    random.shuffle(data)
    print(attribs)
    results_w = np.zeros((4,6))
    results_wo = np.zeros((4,6))
    for i in range(1,4):
        w, wo = eval_by_attrib(data, attribs, i)
        print("{:10}------------------------".format(attribs[i].upper()))
        score_overall(w, wo)
        results_w += w
        results_wo +=w
    print("{:10}------------------------".format("OVERALL"))
    results_w = results_w/3
    results_wo = results_wo/3.0
    score_overall(results_w, results_wo)
    

    