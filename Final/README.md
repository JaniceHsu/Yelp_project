# yelp-review-dataset-sentiment-analysis

This project will use natural language processing tools and algorithms to perform sentiment analysis and text summarization on yelp restaurant reviews.  The number of stars submitted in the review will define the overall sentiment, the target. Restaurant aspects (food, price, service) labeled humans will be used to predict and summarize yelp restaurant reviews. The evaluation will utilize cross validation. 

Data:
Downloaded from the YelpDataset Challenge (https://www.yelp.com/dataset_challenge):
	yelp_academic_dataset_business.csv (all businesses): 144072 
	yelp_academic_dataset_review.csv (all reviews): 4153150
Filtered/manipulated:
	Number of restaurants in the USA by cuisine (that do not serve other cuisines)
	Chinese: 1776 
	Mexican: 3097
	Japanese: 870
	Italian: 2190
	American (Traditional): 4279
	Indian:  342
	Number of reviews on Chinese restaurants (reviews_ChinesePN.csv): 98105 
	Number of reviews with aspect-level sentiment labels: 176

=== Prepare Data ===
json_to_csv_converter.py
	Convert the Yelp dataset from json format to csv format.
	From https://github.com/Yelp/dataset-examples#samples

filter_business_prep.py
	Get statistics on yelp business reviews to decide how to filter businesses.
	Input:
	    yelp_academic_dataset_business.csv 
	    (contains all yelp businesses in the downloaded dataset and their attributes)
	Example Output:
	    E.g. business categories (value, number_of_occurences)
	        ('Restaurants', 48485)
	        ('Shopping', 22466)
	        ('Food', 21189)
	        ('Beauty & Spas', 13711)
	        ('Home Services', 11241)
	        ...
	        ('Alsatian', 1)
	        ('Agriturismi', 1)

filter_business.py
	Filter business based on given requirements.
	Input: CSV file of Yelp_businesses

	Basic Requirements:
	    1. Categories has 'restaurant' in addition to 'food' (if just 'food', you would get Walgreens)
	    2. It is located in the United STates
	    
	Example of other requirements used:
	    General:
	        Filter businesses by cuisine: Japanese, Italian, Chinese, Mexican, American.
	    For aspect-level sentiment analysis:
	        Extracted 5 businesses to hand label:
	            1. 1 business for each of these star ratings to get variety: ['2.5','3.0','3.5','4.0','4.5']
	            2. 30-50 Yelp reviews.

get_reviews.py
	Get Yelp reviews given business.
	Input:
	    CSV file of Yelp business.
	    CSV file of Yelp reviews.
	Output:
	    CSV file of Yelp reviews on businesses from the input file.
	Outside code/Python libraries used:
		csv

reduceNumRows.py
	Select n reviews randomly and write them in a new CSV file.
	Purpose: smaller CSV files for input to be used int he future but 
	    ensure reviews are selected randomly.
	Outside code/Python libraries used:
		csv
		random	    

review_sentLabels.py
	Adds an attribute 'sentiment' label to the CSV file containing yelp reviews based on the number of stars.
	4-5 stars -> 1 (positive)
	1-2 stars -> 0 (negative)
	Outside code/Python libraries used:
		csv
	Input:
		Reads a CSV file containing yelp reviews and associated information (user_id, date, text, stars, etc.)
	Result:
		Writes a CSV file containg the same information plus the sentiment.
	(Input is large and not included.)

replace_null.py
	Replace blank columns representing the fact that the restaurant aspect is not mentioned with '-'.
	Outside code/Python libraries used:
		CSV
	(Input is large and not included.)

=== Aspect Level Sentiment Analysis ===

Restaurant aspects: food, price, service

Restaurant attrib Labels:
- -> not mentioned
0 -> negative
1 -> neutral or mentioned both positively and negatively
2 -> positive



Milestone 1
variousLearnersOverall.py
	Predicts overall review sentiment. Target sentiment based on the number of stars given.
	    1-2 stars -> negative.
	    4-5 stars -> positive.
	Models used
		multinomial naive Bayes
        Bernoulli naive Bayes
        logistic regression
        kNearestNeighbors, k=3 and k=5
    Outside code/Python libraries used
    	CS175 assignment2.py
    	sklearn.feature_extraction.text
    	sklearn.naive_bayes
		sklearn.linear_model
		skelarn.neighbors
    	nltk
    	numpy
    	csv
	Input:
		CSV file: Yelp reviews and number of stars given for the review.
	Example output:
		test_classifiers(50 min_docs)
		Remove stopwords
		Number of (training) documents = 800
		Vocabulary size = 108
		        Base Accuracy: 72.000
		        Accuracy with multinomial naive Bayes: 81.00
		        Accuracy with Bernoulli naive Bayes: 80.50
		        Accuracy with logistic regression: 79.50
		        Accuracy with kNN, k=3 classifier: 74.000000
		        Accuracy with kNN, k=5 classifier: 72.00

Milestone 2
predictAttribSort.py
	Uses multinomial naive bayes, bernoulli naive bayes, logistic regression, and kNearestNeighbors (k=3 and k=5) to classify reviews.
	Uses cross validation.
	Calculates precision and recall.
	Compares trends/the effect of min_docs and stopword removal.
	Finds scoring per restaurant aspects and all aspects combined.
	
	Outside code/Python libraries used
	    CS175 assignment2.py: extract_text_features and test_classifiers (edited)
	    variousLearnsOverall (written previously)
	    csv
	    random
	    nltk
	    numpy
	    sklearn.cross_validation
	    sklearn.feature_extraction.text
	Input:
		CSV files: Yelp Reviews and labels on restaurant aspects (whether or not mentioned and respective sentiment).
	Output:
		['text', 'food', 'price', 'service']
		Number of documents: 151
		=====================   FOOD   =====================
		------- Score Accuracy
		Compare Learners
		without stopwords
		     base   multNB   bernNB   LogReg      KN3      KN5
		  67.591%  71.726%  75.511%  83.978%  76.699%  77.984%

		Difference -- without stopwords vs. with stopwords
		[ 0.     2.645 -2.129  2.65  -2.93   2.85 ]
		overall benefit to take out stopwords: 0.51%

		Min_docs      [      1,      3,      5,     10]
		w/o stopwords [ 76.312  76.731  74.642  74.642]
		w/ stopwords  [ 75.29   75.839  74.072  75.068]
		Num docs | score
		     5 -> 74.642
		    10 -> 74.642
		     1 -> 76.312
		     3 -> 76.731

		------- Score Precision
		Compare Learners
		without stopwords
		     base   multNB   bernNB   LogReg      KN3      KN5
		  67.591%  71.058%  76.302%  84.213%  79.276%  78.383%

		Difference -- without stopwords vs. with stopwords
		[ 0.     2.012 -1.905  1.823  0.832  4.503]
		overall benefit to take out stopwords: 1.21%

		Min_docs      [      1,      3,      5,     10]
		w/o stopwords [ 76.112  76.283  75.156  76.998]
		w/ stopwords  [ 74.678  75.365  74.388  75.273]
		Num docs | score
		     5 -> 75.156
		     1 -> 76.112
		     3 -> 76.283
		    10 -> 76.998

		------- Score Recall
		Compare Learners
		without stopwords
		     base   multNB   bernNB   LogReg      KN3      KN5
		   81.65%  83.406%   80.76%  90.078%  81.017%  83.548%

		Difference -- without stopwords vs. with stopwords
		[ 0.     0.218 -2.557  2.508 -6.898 -0.638]
		overall benefit to take out stopwords: -1.23%

		Min_docs      [      1,      3,      5,     10]
		w/o stopwords [ 84.827  84.756  82.247  81.809]
		w/ stopwords  [ 85.062  85.687  83.58   84.222]
		Num docs | score
		    10 -> 81.809
		     5 -> 82.247
		     3 -> 84.756
		     1 -> 84.827
		Number of documents: 174
		=====================  PRICE   =====================
		...
		===================== SERVICE  =====================
		...
		===================== OVERALL  =====================
		------- Score All attrib
		Compare Learners
		without stopwords
		     base   multNB   bernNB   LogReg      KN3      KN5
		  66.906%  70.628%  75.995%  82.833%  74.064%  72.195%

		Difference -- without stopwords vs. with stopwords
		[ 0.     3.128 -0.219 -0.555  1.632  2.681]
		overall benefit to take out stopwords: 1.11%

		Min_docs      [      1,      3,      5,     10]
		w/o stopwords [ 73.207  75.06   73.938  72.877]
		w/ stopwords  [ 72.715  73.093  72.378  72.45 ]
		Num docs | score
		    10 -> 72.877
		     1 -> 73.207
		     5 -> 73.938
		     3 -> 75.060
	Conclusion:
	    Removal of stopwords has minimal improvement.
	    Logistic regression makes the best predictions.
	    3 docs usually results in the highest scores.
	    Models are better than the baseline model.

Milestone 3
predictByRestaurant.py 
	Final stage in project.
	Provide a summary on a restaurant by providing statistics
	on aspects and their sentiments mentioned in Yelp Reviews.
	Use LogReg, and  min_docs=3 due to results from milestone 2.
	Outside code/Python libraries used
	    CS175 assignment2.py: extract_text_features (edited)
	    variousLearnsOverall (written previously)
	    predictAttribSort (written previously)
	    csv
	    numpy
	    sklearn.feature_extraction.text
	Input:
		CSV files: Yelp reviews and their respective labels on aspects mentioned.
		Reviews are group by restaurant.
	Example output:
		======= Jugoo Chinese Restaurant =======
		    Aspect |     food |    price |  service
		mentioned  |   22.54% |    4.93% |   12.68%
		positively |   87.50% |   71.43% |  100.00%
		negatively |    9.38% |   28.57% |    0.00%

	            
=== Created but not used in the end ===

WordVecGensim.ipynb
	Goal: perform aspect level sentiment analysis using POS tags, gensim, word_vectors, synonyms, etc. on restaurant reviews. 
	Decided to change the approach.
	Outsdie code/Python libraries used:
		csv
		gensim
		nltk


