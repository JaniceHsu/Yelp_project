# -*- coding: utf-8 -*-
"""
Displays I/O of predictByRestaurant and predictAttribSort

predictByResataurant:
Provide a summary on a restaurant by providing statistics
on aspects and their sentiments mentioned in Yelp Reviews.

predictAttribSort:
Uses multinomial naive bayes, bernoulli naive bayes, logistic regression, and 
    kNearestNeighbors (k=3 and k=5) to classify reviews.
Uses cross validation.
Calculates precision and recall.
Compares trends/the effect of min_docs and stopword removal.
Finds scoring per restaurant aspects and all aspects combined.


See individual python files for details.
"""

from predictByRestaurant import eval_restaurant
from predictAttribSort import collect_data, eval_by_attrib, score_and_compare
import random
import numpy as np

if __name__ == "__main__":
    print("predictByRestaurant.py")
    eval_restaurant(1)
    print()
    
    print("predictAttribSort.py")
    data, attribs = collect_data(r"labeledReview")
    random.shuffle(data)

    results_w = np.zeros((4,6))
    results_wo = np.zeros((4,6))
    
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
        results_wo += awo
    print("====================={:^10}=====================".format("OVERALL"))
    # Get scores overall (including different min_doc vals, combine scores of different restaurant aspects)
    results_w = results_w/3
    results_wo = results_wo/3.0
    score_and_compare(results_wo, results_w, "All attrib")
    