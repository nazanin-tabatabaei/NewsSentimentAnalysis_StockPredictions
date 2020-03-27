# NewsSentimentAnalysis_StockPredictions
TextMining,NLP,SentimentAnalysis


## Abstract
Add an updated abstract from the proposal version. In addition to a brief summary of the project, add a summary of the dataset description and analysis, and baseline results.
Should not be more than 1 column.

## Dataset description [35 Points]
 ### Data preparation: [10 Points]
  #### Source [2 points]
Describe how you got the dataset (e.g., crawling, API, from a website, etc.), give proper references wherever applicable (-2 for missing references of papers, websites, or APIs)
  #### Data preprocessing steps and explanations [4 + 4 Points]
Explain how the data preprocessing, cleaning, imputation, and other processing was done.
Explain why this dataset is necessary and sufficient to achieve the goals of the project.
  #### Please note: the dataset should be fixed and finalized [-10 if not]
Exception: if you are crawling the dataset yourself, then you can continue to crawl the data beyond the midterm, as long as a big fraction of the dataset has been collected already.
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
Explain the parameters of the experiment: what is the split used, cross-validation setting, evaluation metrics (for example, accuracy, AUC, or precision etc.), system settings (RAM, GPU, or CPU statistics).
-2 per missing relevant detail.

## Baseline results and discussions [30 points]
The report should have results from at least 2 baselines. At least one baseline should be from a published paper or preprint. Creating one reasonable baseline yourself (e.g., using feature engineering and standard ML classifier) is allowed. No additional points will be awarded for having more than 2 baselines.
Baseline description [7.5*2 = 15]
Describe the baseline. Give a short technical description of the baseline, along with its reference, provide details of the kernel or hyper-parameters used, provide links to code repository used [5 points]
-2 per missing detail, reference, link
Why is this baseline suitable for the problem? [2.5 Points]
Baseline result [10 points]
Results of the baseline on your dataset, presented in a table or figure (e.g., a bar chart) [5*2 = 10]
The baselines should be compared on the same metric [-5 if not]
Result discussion [5 points]
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
Next Steps:  
