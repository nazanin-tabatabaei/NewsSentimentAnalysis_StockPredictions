# NewsSentimentAnalysis_StockPredictions
TextMining,NLP,SentimentAnalysis


## Abstract
In the finance field, Stock market forecasting is very important in the planning of business activities. Stock market attracts researchers to capture the volatility and predicting its next moves. There are many techniques to predict the stock price variations, but this project is focused on using non-quantifiable data such as New York Times’ news articles headlines and predicting future stock changes depending on the news sentiment, assuming that news articles have impact on stock market.
We are using NY Times Archive API to gather the news website articles data over the span of 20 years. Sentiment analysis of the news headlines is then used for training various MachineLearning models to understand their effect and predict the price of Dow Jones Industrial Average (DJIA) stock indices, collected from Yahoo finance website. We have used DJIA stock indices to predict the overall change in US top companies' stock market, instead of predicting individual company’s stock prices. For integrity throughout the project, we considered Adjusted Close price as everyday stock price.

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
In order to measure accuracy in the first baseline, we used a validation technique called the k-fold sequential cross validation (k-SCV). Since stock market data is of the form of time series, other methods such as ordinary k-fold crossvalidation are not applicable.
In this method, we train on all days upto a specific day and test for the next days. For the purpose of our analysis, we use k = 5. More specifically, we trained our model on data from January first to October 31st of every year and tested it for the remaining of the same year. Technique from-> http://cs229.stanford.edu/proj2011/GoelMittal-StockMarketPredictionUsingTwitterSentimentAnalysis.pdf  

accuracy,AUC

 ### System settings:
Comodity computer with RAM:16GB, GPU:Intel UHD Graphics 630, CPU:Intel Core i7-8750H CPU @ 2.20GHz

## Baseline results and discussions [30 points]
 ### First BaseLine
  #### Baseline description [7.5*2 = 15]
  This baseline is taken from an article published by Dinesh D at this link-> https://software.intel.com/en-us/blogs/2017/07/14/stock-predictions-through-news-sentiment-analysis  
The code repository for this baseline is here-> https://github.com/dineshdaultani/StockPredictions  
In this baseline, we have used Vader Sentiment Analyzer, which comes with NLTK package. VADER is trained using social media and news data using a lexicon-based approach. It is used to score single merged strings for articles and gives a positive, negative and neutral score for that string through Natural Language Processing.  Hence, Vader is a suitable package for sentiment analysis of our merged news headlines.
Output of sentiment analysis is then fed to machine learning models from scikit-learn library to predict the stock prices of DJIA indices. The machine learning models used in this baseline are Random Forest, Logistic Regression and Multi-Layer Perceptron (MLP) Classifiers.
  #### Baseline result [10 points]
Results of the baseline on your dataset, presented in a table or figure (e.g., a bar chart) [5*2 = 10]
The baselines should be compared on the same metric [-5 if not]
We have investigated the causative relation between public
mood as measured from a large scale collection of tweets
from twitter.com and the DJIA values. 
Thirdly, a Self Organizing Fuzzy Neural
Network performs very good in predicting the actual DJIA
values when trained on the feature set consisting of the DJIA
values, Calm mood values and Happiness dimension over the
past 3 days. The performance measure we have used is kfold sequential cross validation, which is more indicative of
the market movements for financial data

Logistic Regression perform badly on
this dataset, giving the same percentage values for Direction
Accuracy for all mood combinations. This shows that classification (directly predicting trends) is not the ideal methodology for this problem.
 
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
Explain in detail your own proposed approach and what novelty or improvement you are adding over the baselines. [2.5 + 2.5 = 5 Points]
-2 for unclear explanation
For development projects, clearly describe what will be done by the final report and how exactly this will be achieved. Example, if you are creating an app, where do you plan to host it.

## Contribution
Data collection: Nazanin  
Data Preparation: Nazanin  
Raw Data Statistic:  
Data Analysis:  
First Baseline Implemetation and discussion: Nazanin  
Second Baseline:  
Next Steps:Nazanin,  
Writing the Midterm Report: All  
