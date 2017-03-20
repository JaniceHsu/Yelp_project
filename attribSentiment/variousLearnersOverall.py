# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 21:36:13 2017

@author: Janice
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk import FreqDist
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import csv

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
    print(' Function: extract_text_features()')

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

def fit_and_predict_multinomialNB(X_train, Y_train, X_test):

    """
    Train multinomial naive Bayes model with 'X_train' and 'Y_train' and
    predict the Y values for 'X_test'. (Use 'MultinomialNB' from scikit-learn.)
    Return the predicted Y values.

    Parameters
    ----------
    X_train: scipy sparse matrix
        Data for training (matrix with features, e.g. BOW or tf-idf)
    Y_train: numpy.ndarray
        Labels for training data (target value)
    X_test: scipy sparse matrix
        Test data used for prediction

    Returns
    -------
    numpy.ndarray[int]
        Target values predicted from 'X_test'

 
    """
    # Import the package
    from sklearn.naive_bayes import MultinomialNB 

    ### YOUR SOLUTION STARTS HERE### 
    #used scikit-learn tutorial on training a classifier
    # fit the model... 
    clf = MultinomialNB().fit(X_train, Y_train) #naive bayes
    # make predictions
    predicted_MultinomialnNB = clf.predict(X_test) #predict
    return predicted_MultinomialnNB
    ### END SOLUTION ###  
 


def fit_and_predict_BernoulliNB(X_train, Y_train, X_test):

    """
    Train Bernoulli naive Bayes model with 'X_train' and 'Y_train' and
    predict the Y values for 'X_test'. (Use 'BernoulliNB' from scikit-learn.)
    The 'binarize' threshold should be set to 0.0.
    Return the predicted Y values.


    Parameters
    ----------
    X_train: scipy sparse matrix
        Data for training (matrix with features, e.g. BOW or tf-idf)
    Y_train: numpy.ndarray
        Labels for training data (target value)
    X_test: scipy sparse matrix
        Test data used for prediction

    Returns
    -------
    numpy.ndarray[int]
        Target values predicted from 'X_test'

    """

    # Import the package
    from sklearn.naive_bayes import BernoulliNB 

    ### YOUR SOLUTION STARTS HERE### 
    #referenced to sklearn documentation    
    # fit the model... 
    clf = BernoulliNB(binarize=0.0).fit(X_train, Y_train)  #fit naive bayes to X and Y train data
    # make predictions
    predicted_bernNB = clf.predict(X_test)
    return predicted_bernNB
    ### END SOLUTION ###  


def fit_and_predict_LR(X_train, Y_train, X_test):

    """
    Train logistic regression model with 'X_train' and 'Y_train' and
    predict the Y values for 'X_test'. (Use 'LogisticRegression' from scikit-learn.)
    Return the predicted Y values.


    Parameters
    ----------
    X_train: scipy sparse matrix
        Data for training (matrix with features, e.g. BOW or tf-idf)
    Y_train: numpy.ndarray
        Labels for training data (target value)
    X_test: scipy sparse matrix
        Test data used for prediction

    Returns
    -------
    numpy.ndarray[int]
        Target values predicted from 'X_test'

    """

    # Import the package
    from sklearn.linear_model import LogisticRegression

    ### YOUR SOLUTION STARTS HERE### 
    #referenced to sklearn documentation    
    # Replace FIRSTNAME_LASTNAME with your name

    # fit the model... 
    clf = LogisticRegression().fit(X_train, Y_train)     
    # make predictions 
    predicted_LR = clf.predict(X_test)
    return predicted_LR
    ### END SOLUTION ### 


def fit_and_predict_KNN(X_train, Y_train, X_test, K):

    """
    Train nearest neighbor classifier model with 'X_train' and 'Y_train' and
    predict the Y values for 'X_test'. Use 'KNearestNeighborsClassifier' from 
    scikit-learn with K nearest neighbors (K = 1, 3, 5, ....)
    Return the predicted Y values.


    Parameters
    ----------
    X_train: scipy sparse matrix
        Data for training (matrix with features, e.g. BOW or tf-idf)
    Y_train: numpy.ndarray
        Labels for training data (target value)
    X_test: scipy sparse matrix
        Test data used for prediction
    K: integer (odd)
    	Number of neighbors to use for prediction, e.g., K = 1, 3, 5, ...

    Returns
    -------
    numpy.ndarray[int]
        Target values predicted from 'X_test'

    """
 
    # Import the package
    from sklearn.neighbors import KNeighborsClassifier

    ### YOUR SOLUTION STARTS HERE###
    #referenced to sklearn documentation
    # fit the model (for KNN this is just storing the training data and labels) 
    clf = KNeighborsClassifier(n_neighbors=K).fit(X_train, Y_train)
    # Predict
    predicted_KNN = clf.predict(X_test)
    return predicted_KNN
    
    
def separateTextTarget(reader):
    docs = []
    target = []    
    for row in reader:
        break
    for row in reader:
        docs.append(row[0])
        target.append(row[1])
    return (docs, target)
        
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
    fname = r"reviews_ChinesePN_10000.csv"
    f = open(fname, 'r', encoding = 'utf8')
    reader = csv.reader(f)
    
    docs, target = separateTextTarget(reader)
    train_target = target[:8000]
    test_target = target[8000:]
    train_docs = docs[:8000]
    test_docs = docs[8000:]
    f.close()
    fname = r"reviews_ChinesePN_10000.csv"
    f = open(fname, 'r', encoding = 'utf8')
    reader = csv.reader(f)    
    reader2 = csv.reader(f)
    for row in reader2:
        break
    rows = 0
    pos = 0
    for row in reader2:
        rows +=1
        if row[1] == '1':
            pos +=1
    print("Pos    : {}".format(pos))
    print("NumRows: {}".format(rows))
    print("Base   : {:.3f}".format((pos/rows)*100))
    f.close()    
    
    X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf = extract_text_features(train_docs,  test_docs,3, True)

    predicted_multNB = fit_and_predict_multinomialNB(X_train_tfidf, train_target, X_test_tfidf)
    predicted_bernNB = fit_and_predict_BernoulliNB(X_train_tfidf, train_target, X_test_tfidf)
    predicted_LR = fit_and_predict_LR(X_train_tfidf, train_target, X_test_tfidf)
    predicted_LR = fit_and_predict_LR(X_train_counts, train_target, X_test_counts)
    K=3
    predicted_KNN = fit_and_predict_KNN(X_train_tfidf, train_target, X_test_tfidf, K)

    print('#####Predicted labels with multinomial NB classifier:') 
    for doc, p, i in zip(test_docs, predicted_multNB, range(5)):
        print('{} -> {}, {}'.format(doc[:50], p, test_target[i]))
    print()
    
    # # Bernoulli naive Bayes
    print('#####Predicted labels with Bernoulli NB classifier:') 
    for doc, p, i in zip(test_docs, predicted_bernNB, range(5)):
        print('{} -> {}, {}'.format(doc[:50], p, test_target[i]))
    print()
    
    
    # kNN
    print('####Predicted labels with kNN classifier:')
    for doc, p, i in zip(test_docs, predicted_KNN, range(5)):
        print('{} -> {}, {}'.format(doc[:50], p, test_target[i]))
    print()
    #
    # Logistic Regression 
    print('####Predicted labels with logistic classifier:')
    for doc, p, i in zip(test_docs, predicted_LR, range(5)):
        print('{} -> {}, {}'.format(doc[:50], p, test_target[i]))
    print()

    for i in [1,3,5,10,50, 100]:
        print("\ntest_classifiers(twenty_train, twenty_test, {} [min_docs])".format(i))
        test_classifiers(train_docs, train_target, test_docs, test_target, i, 3, 5, True) #last paramter -- remove stopwords or not
        test_classifiers(train_docs, train_target, test_docs, test_target, i, 3, 5, False) #last paramter -- remove stopwords or not

    