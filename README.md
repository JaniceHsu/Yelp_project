# yelp-review-dataset-sentiment-analysis
This project uses natural language processing tools and algorithms to ultimately perform attribute level sentiment analysis on yelp restaurant reviews.  In the first step, the overall sentiment analysis was performed on yelp restaurant reviews using multinomial naive Bayes, Bernoulli naive Bayes, logistic regression, and k-nearest neighbors (k=3 and k=5). Evaluation was performed with 80% of the data being training and 20% being testing. When evaluating the overall sentiment of a yelp review, with the number of stars determined the sentiment level.

Attribute-level sentiment analysis was performed on the reviews of 5 yelp businesses, taking into account food, price, and service. The evaluation utilized training and test data and cross-validation. Since the scores indicated that logistic regression and a parameter requirement that the words appear in a minimum amount of 3 documents made the best predictions, these values were used when providing a summary of restaurants using attribute-level sentiment analysis.

### Accuracy Result:


|	       | base|	multNB	|bernNB	|LogReg|	KNN (3)|	KNN (5)|
| --- | --- | --- | --- | --- | --- | --- |
Food	    |70.736	|78.552|	77.816|	88.701	|80.644	|72.758
Price	   | 60.118|	71.904|	79.353	|89.256|	72.273|	66.429
Service	  |55.597	|79.951|	71.471|	88.093|	72.598|	67.904
Overall	|  62.15	|  76.802|	76.213	|88.684|	75.172|	69.03
 
### Precision & Recall

|	       | base|	multNB	|bernNB	|LogReg|	KNN (3)|	KNN (5)|
| --- | --- | --- | --- | --- | --- | --- |
Precision	|29.42	|83.5	|80.95	|85.488|	66.298|	58.321
Recall	|  38.259|	50.719|	54.167|	79.167|	62.718|	51.062


Related Work:
Sentiment analysis algorithms and applications: A survey, Ain Shams Engineering Journal 
By Walaa Medhat, Ahmed Hassan, Hoda Korashy
http://www.sciencedirect.com/science/article/pii/S2090447914000550
*	The process of sentiment of analysis is described as finding opinions identify the sentiments they express, and then classify their polarity as show in the figure.
*	Naïve Bayes classifier is the simplest and most commonly used.
*	There are two approaches to sentiment classification: machine learning approach and lexicon approach (or a hybrid).
*	Machine learning approaches: Naïve Bayes Classifier, Bayesian Network, Maximum Entropy, Linear, Support Vector machines, Neutral Network, Decision tree, etc.
*	Lexicon-based approach: dictionary, corpus, statistic, semantic

Sentiment Analysis of Twitter Data, Columbia University
By Apoorv Agarwal, Boyi Xie, Ilia Vovsha, Owen Rambow, Rebecca Passonneau
http://dl.acm.org/citation.cfm?id=2021114
*	Uses 100 features and over 10,000 features achieves similar accuracy.
*	Associated emoticons with sentiments like “:)”, “(-:”, etc.
*	Built models using Naïve Bayes, MaxEnt, and Support Vector machines: SVM outperformed other classifiers.
*	Feature spaces unigram, bigrams with parts-of-speech features: unigram model outperforms all others.

My process of sentiment analysis is similar to “Sentiment analysis algorithms and applications “. I used some of machine learning approaches described. I attempted to use a lexicon-based approach but am unable to compelte  it successfully within this project course period. I used unigrams similar to that of “Sentiment Analysis of Twitter Data”, but did not incorporate emoticons into my learning algorithms. 

Data Sets 
I used JSON files on yelp businesses and yelp reviews from the yelp dataset found on https://www.yelp.com/dataset_challenge/dataset. Using a tool provided by the winner of the Yelp dataset challenge, I converted the json files to csv files; thus, I used the following:
*	yelp_academic_dataset_business.csv (all businesses): 144,072  total business
.*	Contains attributes such as: business_id, review_count, categories, etc.
*	yelp_academic_dataset_review.csv (all reviews): 4,153,150 total reviews
.* Contains attributes such as review_text, user_id, business_id, etc.
I filtered the restaurants from the business dataset, removed businesses outside of the U.S, and organized them by cuisine. 
* There are 1,776 Chinese restaurants (that do not serve other cuisines)
* 3,907 Mexican restaurants
* 2,190 Italian restaurants
* 870 Japanese
.* etc.

From the filtered restaurants, I retrieved their respective Yelp reviews.
These were used to perform sentiment analysis on reviews overall.
Reviews with 4-5 stars were labeled with a sentiment 1, representing a positive sentiment.
Reviews with 1-2 stars were labeled with a sentiment 0, representing a negative sentiment.
3 star reviews were ignored.
*	There are 98,105 reviews of Chinese restaurants 
*	72,746 are positive in sentiment
*	25359 are negative in sentiment

To perform attribute-level sentiment analysis, 5 Chinese restaurants of varying star ratings ['2.5','3.0','3.5','4.0','4.5'] and containing 30-50 reviews were randomly chosen. The aspect-level sentiments (regarding the food, price, service) of their reviews were labeled by hand, totaling 176 labelled reviews. In the process, I added the condition that the restaurant is not a buffet because I discovered there would be miniscule comments on the service which would be hard for training. There are a varying of components that would make prediction hard for the models I was using. 
For example:
Misspelled words:
“…its over priced…”
Irregular formatting (requires more data to recognize the pattern):
“(+)
 Great Service
 (-)
 Expensive and mediocre food…”
Indirect statements:
“If you were considering the veggie fried rice, just swing by your local supermarket and buy an onion instead.”
Positive and negative comments towards the same restaurant aspect:
“WOW! I had a rum drink off the drink menu that I swear tasted like a grape jolly rancher. DELICIOUS!! …I had the lemon chicken and the general tso's and it was dry and flavorless…”

Technical Approach:
I utilized multinomial naive Bayes, Bernoulli naive Bayes, logistic regression, and k-nearest neighbors (k=3 and k=5) to perform sentiment analysis and attribute-level sentiment analysis on yelp reviews. When performing attribute-level sentiment analysis, logistic regression and a parameter requirement that the words appear in a minimum amount of 3 documents led to the highest performance results. Thus, I used these parameters when determining restaurant summaries. 





