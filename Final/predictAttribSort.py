# -*- coding: utf-8 -*-
"""
Janice Hsu

Uses multinomial naive bayes, bernoulli naive bayes, logistic regression, and kNearestNeighbors (k=3 and k=5) to classify reviews.
Uses cross validation.
Calculates precision and recall.
Compares trends/the effect of min_docs and stopword removal.
Finds scoring per restaurant aspects and all aspects combined.

Conclusion on results:
    Removal of stopwords has minimal improvement.
    Logistic regression makes the best predictions.
    3 docs usually results in the highest scores.
    Models are better than the baseline model.
    
credit: extract_text_features and test_classifiers heavily influneced by CS175 assignment2. 
else-- self work
"""

import csv
import random
import numpy as np
from sklearn.cross_validation import KFold
import warnings
warnings.filterwarnings("ignore")


from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from variousLearnersOverall import fit_and_predict_multinomialNB, fit_and_predict_BernoulliNB, fit_and_predict_LR, fit_and_predict_KNN


def getRows(reader):
    """Extract/organize data from the CSV files containg Yelp review text and 
    aspect-level sentiment labels"""
    docs = []
    for row in reader:
        attribs = [row[3], row[10], row[11], row[12]]
        break
    for row in reader:
        docs.append([row[3], row[10], row[11], row[12]])
    return docs, attribs
    
def collect_data(fn):
    """Extract/combine data from all files"""
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
    """ Predict/evaluate attrib
    wo -> without stopwords
    """
    #Analyze scores specific to attribute i
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
    #get cross validation indexes
    fold_index = KFold(n, n_folds=5) # use of this seen through stackoverflow
    
    accur_wo_stopwords = []
    accur_w_stopwords = []
    prec_wo_all = []
    prec_w_all = []  
    rec_wo_all = []
    rec_w_all = []
    
    md_num = [1,3,5,10] # gather scores for various min_df values
    for md in md_num:  
        accur_folds_w = []
        accur_folds_wo = []

        precision_w = []
        precision_wo = []

        recall_w = []
        recall_wo = []

        for trainI, testI in fold_index: 
            # cross validation: separate into testing/training
            train_docs, test_docs = docs[trainI], docs[testI]
            train_target, test_target = target[trainI], target[testI]

            # target predictions
            np_test_target = np.array(test_target)
            
            # baseline model-- predict most occuring label
            predicted_base = np.array([FreqDist(test_target).most_common(1)[0][0]]*len(test_target))
            
            # performance of baseline model
            base = np.sum(predicted_base == np_test_target)/len(np_test_target)
            base_p, base_r = get_prec_rec(predicted_base, np_test_target)
            
            # last parameter: True->remove stop_words
            # collect perfomrance scores of the many different classifiers
            results, precision, recall = test_classifiers(train_docs, train_target, test_docs, test_target, md, 3, 5, True)
            accur_folds_wo.append([base] + results) 
            precision_wo.append([base_p] + precision)
            recall_wo.append([base_r] +recall)
            
            results, precision, recall = test_classifiers(train_docs, train_target, test_docs, test_target, md, 3, 5, False)
            accur_folds_w.append([base] + results) 
            precision_w.append([base_p] + precision)
            recall_w.append([base_r] + recall)
          
        # shape (6,) --> [base, multNB, bernNB, LR, KN3, KN5]
        # axis = 0 --> mean of each model over all min_doc values 
        # to compare each model's performance on average
        accur_wo = np.mean(np.array(accur_folds_wo), axis = 0)
        prec_wo = np.mean(np.array(precision_wo), axis = 0)
        rec_wo = np.mean(np.array(recall_wo), axis = 0)
        
        accur_w = np.mean(np.array(accur_folds_w), axis = 0) 
        prec_w = np.mean(np.array(precision_w), axis = 0)
        rec_w = np.mean(np.array(recall_w), axis = 0)        
                    
        #save to later compare accuracy while taking into account min_df
        accur_wo_stopwords.append(accur_wo)
        prec_wo_all.append(prec_wo)
        rec_wo_all.append(rec_wo)
        
        accur_w_stopwords.append(accur_w)
        prec_w_all.append(prec_w)
        rec_w_all.append(rec_w)       
        
    return np.array(accur_wo_stopwords), np.array(prec_wo_all), np.array(rec_wo_all),\
         np.array(accur_w_stopwords), np.array(prec_w_all), np.array(rec_w_all)
            

def score_and_compare(accur_wo_stopwords, accur_w_stopwords, topic):
    """
    Using means and sorting, compare the performance of various situations.
        -- different algorithm/model
        -- different min_doc
        -- with and without stopwords
    Organize for printing neatly.
    """
    print("------- Score {}".format(topic))

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
    
    print("Difference -- without stopwords vs. with stopwords")
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
    

    
def get_prec_rec(prediction, np_test_target):
    """ 
    calculate the precision and recall 
    Basic formula found on wikipedia
    """
    positive = 0
    true_positive = 0
    relevant = 0
    for p, t in zip(prediction, np_test_target):
        if p != '-':
            positive +=1
        if p == t and p != '-':
            true_positive +=1
        if t != '-':
            relevant +=1
    try:
        precision = true_positive/float(positive)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = true_positive/float(relevant)
    except ZeroDivisionError:
        recall = 0      
    return precision, recall
    

    
    
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
    """
    # Generate count vectors from the input data
    # include/exclude NLTK stopwords based on parameter
    if remove_stop_words:
        count_vect = CountVectorizer(min_df=min_docs, stop_words ='english')
    else:
        count_vect = CountVectorizer(min_df=min_docs)        
    
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
    
def test_classifiers(train_docs, train_target, test_docs, test_target,  min_docs, K, K2, removeStopWords):
    """
    Predict and evaluate the accuracy of multiple classifiers by training on the data in 
    "train" and making predictions on the data in "test". The classifiers
    evaluated are: BernoulliNB, MultinomialNB, Logistic, and kNN (3 and 5).
    
    Mostly from assignment2. 
    Added baseline model and precision/recall scoring.
    """
    X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf = extract_text_features(train_docs,  test_docs, min_docs, removeStopWords)
    
    num_docs, vocab_size = X_train_counts.shape

    # Evaluate the classifiers on the test data
    #predict according to different classifier--evaluate results    
    predicted_multNB = fit_and_predict_multinomialNB(X_train_tfidf, train_target, X_test_tfidf)
    predicted_bernNB = fit_and_predict_BernoulliNB(X_train_tfidf, train_target, X_test_tfidf)
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
    
    results = [predicted_multNB, predicted_bernNB, predicted_LR, predicted_KNN, predicted_KNN2]
    
    # Calculate precision/recall
    precision = []
    recall = []
    for prediction in results:
        p, r = get_prec_rec(prediction, np_test_target)
        precision.append(p)
        recall.append(r)

    return [multNB, bernNB, LR, KN, KN2], precision, recall
    


if __name__ == "__main__":
    data, attribs = collect_data(r"labeledReview")
    random.shuffle(data)
    print(attribs)
    results_w = np.zeros((4,6))
    results_wo = np.zeros((4,6))
    w_pp = np.zeros((4,6))
    wo_pp = np.zeros((4,6))
    w_rr = np.zeros((4,6))
    wo_rr = np.zeros((4,6))
    
    md_num = [1,3,5,10]

    for i in range(1,4):
        awo, pwo, rwo, aw, pw, rw = eval_by_attrib(data, attribs, i)
        print("====================={:^10}=====================".format(attribs[i].upper()))
        # Compare prediction states for (Accuracy, Precision, Recall) 
        # based on (min number of doc, removal of stop words, modelling algorithm)
        score_and_compare(awo, aw, "Accuracy")
        score_and_compare(pwo, pw, "Precision")
        score_and_compare(rwo, rw, "Recall")
        results_w += aw
        w_pp += pw
        w_rr += rw
        results_wo += awo
        wo_pp += pwo
        wo_rr += rwo
    print("====================={:^10}=====================".format("OVERALL"))
    # Get scores overall (including different min_doc vals, combine scores of different restaurant aspects)
    results_w = results_w/3
    results_wo = results_wo/3.0
    w_pp = w_pp/3.0
    wo_pp = wo_pp/3.0
    w_rr = w_rr/3.0
    wo_rr = wo_rr/3.0
    score_and_compare(results_wo, results_w, "All attrib")
    score_and_compare(wo_pp, w_pp, "All Precision")
    score_and_compare(wo_rr, wo_rr, "All Recall")
    

    