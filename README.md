# NewsSentimentAnalysis_StockPredictions
TextMining,NLP,SentimentAnalysis


## Abstract
In the finance field, Stock market forecasting is very important in the planning of business activities. Stock market attracts researchers to capture the volatility and predicting its next moves. There are many techniques to predict the stock price variations, but this project is focused on using non-quantifiable data such as New York Times’ news articles headlines and predicting future stock changes depending on the news sentiment, assuming that news articles have impact on stock market.
We are using NY Times Archive API to gather the news website articles data over the span of 20 years. Sentiment analysis of the news headlines is then used for training various MachineLearning models to understand their effect and predict the price of Dow Jones Industrial Average (DJIA) stock indices, collected from Yahoo finance website. We have used DJIA stock indices to predict the overall change in US top companies' stock market, instead of predicting individual company’s stock prices. For integrity throughout the project, we considered Adjusted Close price as everyday stock price.
To analyze the sentiment of the news headlines, we used 2 techniques. In the first technique, we used VADER sentiment Analyzer from NLTK package, which is trained using social media and news data. In the second approach, we trained our own.....The evaluation results of these 2 techniques show that the ... method works better.
## Dataset description  
 ### Data preparation:  
  #### Source  
   ##### News Data:
   The news data was gathered through NY Times Archive API. https://developer.nytimes.com/archive_api.json
   ##### Stock indices:
   We used DJIA stock indices, collected from Yahoo finance website. https://finance.yahoo.com/quote/%5EDJI/history
  #### Data preprocessing steps and explanations
  The credit of data preparation techniques used in this project goes to this article published by Dinesh D: https://software.intel.com/en-us/blogs/2017/07/14/stock-predictions-through-news-sentiment-analysis
   ##### Article Filtering:
   We collected news articles from NY Times Archive API over the span of 20 years, from January 1st of 2000 to December 31st of 2019. Afterwards, we removed categories of articles, which were irrelelevant to stock market. Article sections that are kept at the end for sentiment analysis are as follows: 'Business', 'National', 'World', 'U.S.' , 'Politics', 'Opinion', 'Tech', 'Science',  'Health' and 'Foreign'. Out of 66M articles, approximately 719k articles are filtered out after applying the above filters.
   ##### Merge stock indices with articles:
   We concatenated all the articles headlines for a single day and merged them with appropriate date and Adjusted Close price of Dow Jones Industrial Average (DJIA) stock index value. Composite index prices such as DJIA reflect the overall change in the stock market. In general, the machine will get the output for one individual stock wrong most of the time, but when combined with other stocks, the variance in each stock insight will balance out. Therefore, the machine has a higher probability of getting the output right on average when we draw insights for a combination of stocks. Hence, most researchers prefer to predict stock prices of composite index instead of predicting individual company’s stock prices.  
   As the stock market is closed on weekends and US holidays, there are no open/close prices for any of the stocks on those days. We have used the interpolation method from pandas package to interpolate the prices and fill in the missing values.
 ### Raw Data Statistics: Explain the dataset [10 points]
Describe the important properties of the dataset. How many data points are present? Describe the important features of the data and their basic statistics (range, mean, median, max). Details of ground-truth labels or dataset should be given, if applicable. Some examples are below:
Text data: vocabulary size, number of sentences per text entry, average number of tokens per sentences, and more.
Image data: number of images, categories of images, ground-truth labels
Social media data: how many users are present in the dataset, how much information in total and per user is available, time range of data
Network data: number of nodes, edges, properties of nodes and edges, number of labels/classes
Add at least 5 distinct features [5*2 = 10 Points]
-2 points for each missing important detail
Should contain at least 1 relevant table [-5 if not present]
 ### Data Analysis: Explore your data and talk about findings [15 points]
Discuss relevant features, correlations, cluster visualizations, sentiment statistics, network statistics, centrality distributions from the data.
Add at least 5 insights or interpretations [5 * 2 = 10 Points]
Add at least one visualization (figure or plot) [5 points]
All findings and figures should be quantifiable (-3 per non-quantifiable instance). Example: a word cloud is non-quantifiable.
-3 points for every non-relevant analysis below 5.
Please make sure to add only relevant visualizations and insights. Inserting vague plots and figures, for example, unrelated word cloud or generic network visualization will incur penalty (-5 points penalty).

## Describe the Experimental Settings [10 Points]
 ### Evaluation metrics 
 In order to measure performance in the first baseline, we used a validation technique called the k-fold sequential cross validation (k-SCV). Since stock market data is of the form of time series, other methods such as ordinary k-fold crossvalidation are not applicable.
In this method, we train on all days upto a specific day and test for the next days. For the purpose of our analysis, we use k = 3, 6, and 12. For example for a 6-SCV, we trained our model on data from January first to October 31st of every year and tested it for the remaining of the same year. Technique from-> http://cs229.stanford.edu/proj2011/GoelMittal-StockMarketPredictionUsingTwitterSentimentAnalysis.pdf  

The evaluation metrics used for comparing the performance of our models include RMSE, MAE, and R2.
Our own Repository for this project resides at: https://github.com/nazanin-tabatabaei/NewsSentimentAnalysis_StockPredictions

 ### System settings:
Comodity computer with RAM:16GB, GPU:Intel UHD Graphics 630, CPU:Intel Core i7-8750H CPU @ 2.20GHz

## Baseline results and discussions [30 points]
 ### First BaseLine
  #### Baseline description
  This baseline is taken from an article published by Dinesh D at this link-> https://software.intel.com/en-us/blogs/2017/07/14/stock-predictions-through-news-sentiment-analysis  
The code repository for this baseline is here-> https://github.com/dineshdaultani/StockPredictions  
In this baseline, we have used Vader Sentiment Analyzer, which comes with NLTK package. VADER is trained using social media and news data using a lexicon-based approach. It is used to score single merged strings for articles and gives a positive, negative and neutral score for that string through Natural Language Processing.  Hence, Vader is a suitable package for sentiment analysis of our merged news headlines.  
Output of sentiment analysis is then fed to machine learning models from scikit-learn library to predict the stock prices of DJIA indices. The machine learning models used in this baseline are Random Forest, Logistic Regression and Multi-Layer Perceptron (MLP) Classifiers.  
As the prices of the stocks fluctuate a lot, we have used a technique called smoothing which is used in financial markets to take a moving average of the values, which results in comparatively smooth curves. For moving average implementation, we have used the EWMA method from pandas package.  
As an exploration to this baseline, we updated the VADER lexicon with words+sentiments from other sources/lexicons such as the Loughran-McDonald Financial Sentiment Word Lists, and ran the various models mentioned above on the new lexicon and compared the results.  

  #### Baseline result [10 points]
  Below are the results after applying various classifiers:
   ##### Logistic Regression:  
  <img src="https://www.dropbox.com/s/a8exwr39kx8n6gy/P1Logistic.PNG?raw=1">  
  ###### VADER sentiment analyzer
  <img src="https://www.dropbox.com/s/3q66x5gqqyyl9g6/P2Logistic2.PNG?raw=1">  
  ###### VADER sentiment analyzer with modified Lexicon
   ##### Random Forest Regressor:
  <img src="https://www.dropbox.com/s/qwz5efdzus86coc/P1random.PNG?raw=1">  
  ###### VADER sentiment analyzer
  <img src="https://www.dropbox.com/s/grszklmvngj69ak/P2random.PNG?raw=1">  
  ###### VADER sentiment analyzer with modified Lexicon
  Logistic Regression perform badly on this dataset. This shows that classification (directly predicting trends) is probabaly not the ideal methodology for this problem. Our tests also determined that using the MLP classifier (a.k.a. neural networks) showed better results than logistic regression and random forest trained models.
 
 ### Second BaseLine
  #### Baseline description [7.5*2 = 15]
Describe the baseline. Give a short technical description of the baseline, along with its reference, provide details of the kernel or hyper-parameters used, provide links to code repository used [5 points]
-2 per missing detail, reference, link
Why is this baseline suitable for the problem? [2.5 Points]
  #### Baseline result [10 points]
Results of the baseline on your dataset, presented in a table or figure (e.g., a bar chart) [5*2 = 10]
The baselines should be compared on the same metric [-5 if not]

 ### Result discussion [5 points]
Compare the results of both the baselines. Why does one perform better than the other? If applicable, compare the result to the state-of-the-art reported in literature.

## Next steps [5 Points]
For the first baseline, We will be experimenting with Convolutional Neural Network (CNN) and recurrent neural networks models. Some researchers in this field have also stated that Self Organizing Fuzzy Neural Network performs very good in predicting the DJIA values, which could be a next step for this baseline.  
VADER sentiment Analyzer was used for the first baseline, which is built for social media text. As an exploratory addition to the first baseline, we updated the VADER lexicon with words+sentiments from other sources/lexicons such as the Loughran-McDonald Financial Sentiment Word Lists. A next step could be developing a sentiment analyzer which could work better in news article situations.  
In our project we only considered news article sentiment analysis for prediction but in the real scenarios, stock fluctuations show trends which get repeated over a period of time. So there’s a lot of scope in merging the stock trends with the sentiment analysis to predict the stocks which could probably give better results.  
## Contribution
Abstract:Nazanin
Data collection: Nazanin  
Data Preparation: Nazanin  
Raw Data Statistic:  
Data Analysis:  
Evaluation methods: Nazanin  
First Baseline Implemetation, discussion and next steps: Nazanin  
Second Baseline:  
Writing the Midterm Report: All  
Transfering to ACM format:  
