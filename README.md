# Projects
## Case Study : Deep Deterministic Policy Gradient to solve OpenAI LunarLander
* Actor-Critic combines two main methods in reinforcement learning:  policy based algorithms and value based algorithms. The result is a powerful method that is highly suited for environmentswith continuous action spaces.  The goal of the actor-critic method is to optimize both the policyfunction and the value function via function approximation:  neural networks.  The actor networkis given a state, and it calculates a probability distribution over a set of actions.  i.e;  it outputsthe best actions that can be taken.  This output is then fed into the Critic network that evaluatesthose actions.  The result is an interplay of two networks where the actor decides the best actionsand the critic estimates how good these actions are.
* One of the efficient alternatives in this case is to use actor critic algorithm along with deepfunction approximation.  With this method it is possible to learn high dimensional and continuousaction space
* In DDPG computation can be heavy which is why we learn in mini batches rather than learningentirely on the online environment.  In order to achieve this a Replay Buffer is used.  The oldsamples from replay buffer are deleted when it reaches to the maximum memory.  So, at each timecritic  updates  by  evaluating  at  each  timestamp.   Size  of  replay  buffer  is  fixed  rather  large  thanthe one used in Deep Q network as DDPG is an off-policy algorithm.  This increased size allowsalgorithm to learn from more transitions.
* ![alt text](https://github.com/Tejan4422/LunarLander_ddpg/blob/main/lunarlanderpost.png)
* ![alt text](https://github.com/Tejan4422/LunarLander_ddpg/blob/main/ddpg_set1.png "Model Performance")

## Project 2: Stock Market Predictions
* created a website which can predict following
    * HIGHEST values of a particular company
    * LOWEST values of a particular company
    * CLOSING values of a particular company
* Algorithms used:
    * Decision Tree Regression
    * Random Forest Regression
    * Support Vector Regression
*Data taken from ** https://in.finance.yahoo.com/
* Use pandas-datareader package to connect to yahoo server to fetch dataframe
## Case Studies:
![alt text](https://github.com/Tejan4422/StockAnalysis-/blob/master/Case%20Studies/Fig_AXIS_Low.png "Model Performance")
![alt text](https://github.com/Tejan4422/StockAnalysis-/blob/master/Case%20Studies/Fig_AXIS_High.png "Model Performance")
![alt text](https://github.com/Tejan4422/StockAnalysis-/blob/master/Case%20Studies/Fig_AXIS_Close.png "Model Performance")

## Project 3: Convolutional Network to Detect Pneumonia 
# Overview
  * Building a CNN with TensorFlow backend
  * Test on Single Image
  * Model Production with Flask
* Packages Used : 
  * pandas, numpy , keras, tensorFlow, seaborn, flask, matplotlib
* CNN Structure : 
    1. CONV2D with filter size 3*3, No Padding, No Strides, activation Relu, Convert i/p to 64*64
    2. MaxPooling with Pool size 2*2
    3. CONV2D with filter size 3*3, No Padding, No Strides, activation Relu
    4. MaxPooling with Pool size 2*2
    5. Full Connection Activation Sigmoid, Optimizer = Adam, binary Cross Entropy
    6. Model Accuracy on training Data 94%
    7. Model Accuracy on Testing 88%
    ![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/model_summary.png "Model Summary")
* Case Studies : 
Following is the MRI of a Normal Person Pneumonia -ve
![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/normal.png "Negative Case")
![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/pneumonia.png "Positive Case")
![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/model_summary.png "Model Summary")
![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/traning_val_accuracy.png "Model Summary")
![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/loss.png "Model Summary")
![alt text](https://github.com/Tejan4422/Pneumonia_cnn/blob/master/Model_deployment.png "Final Product")

## Project 4 :Glassdoor Job Analysis
## Overview
* Created a tool that estimates data science salaries (MAE ~ $ 11K) to help data scientists negotiate their income when they get a job.
* Scraped over 1000 job descriptions from glassdoor using python and selenium
* Engineered features from the text of each job description to quantify the value companies put on python, excel, aws, and spark. 
* Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model. 
* Built a client facing API using flask 
## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle  
**Scraper Github:** https://github.com/arapfaik/scraping-glassdoor-selenium  
**Scraper Article:** https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905  
**Flask Productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2
## Web Scraping
Tweaked the web scraper github repo (above) to scrape 1000 job postings from glassdoor.com. With each job, we got the following:
*	Job title
*	Salary Estimate
*	Job Description
*	Rating
*	Company 
*	Location
*	Company Headquarters 
*	Company Size
*	Company Founded Date
*	Type of Ownership 
*	Industry
*	Sector
*	Revenue
*	Competitors 
## Data Cleaning
After scraping the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:
*	Parsed numeric data out of salary 
*	Made columns for employer provided salary and hourly wages 
*	Removed rows without salary 
*	Parsed rating out of company text 
*	Made a new column for company state 
*	Added a column for if the job was at the company’s headquarters 
*	Transformed founded date into age of company 
*	Made columns for if different skills were listed in the job description:
    * Python  
    * R  
    * Excel  
    * AWS  
    * Spark 
*	Column for simplified job title and Seniority 
*	Column for description length 
## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables. 
## Model Building 
First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.   
I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.   
I tried three different models:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression** – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 
## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : MAE = 11.22
*	**Linear Regression**: MAE = 18.86
*	**Ridge Regression**: MAE = 19.67

## Project 5 : FIFA 19 Analysis to help Managers make better signings
## Overview
* Tasks achieved
    * Exploratory data analysis on FIFA 19 dataset 
    * Help managers make better signings
    * Focus on young talents
## Data visualization
Remeber that in this stage our goal is not only to explore our data in order to get better predictions. 
We also want to get better understanding what is in data and explore data in 'normal' way. 
This kind of approch can be useful if we have to do some feature engineering, where good data understanding can really 
help to produce better features.
Following heat map shows the most effective ability of a player which contributes the most on his overall figure is his Composure.
![alt text](https://github.com/Tejan4422/Fifa-19-Prediction-and-analysis/blob/master/heatmap.png "Heatmap")
Some of the important insights from following bar chart
    * There's a huge competition in forward line specifically in between strikers and centre forwards
    * When it comes to sign a goalkeepers there are abundant of options
    * If a player wants to make his way to become a legend there are few positions to consider LF, LAM, RAM
![alt text](https://github.com/Tejan4422/Fifa-19-Prediction-and-analysis/blob/master/positionsbar.png "Positions")
I created a small dataset of youth prodigies who have show  a great potential and are must ins for a manager to build a squad
![alt text](https://github.com/Tejan4422/Fifa-19-Prediction-and-analysis/blob/master/positionsbar_youth.png "Positions")
![alt text](https://github.com/Tejan4422/Fifa-19-Prediction-and-analysis/blob/master/workrate.png "workrate")
![alt text](https://github.com/Tejan4422/Fifa-19-Prediction-and-analysis/blob/master/workrate_youth.png "youth workrate")


## Project 6 : Wine Task(NLP and variety of berries classification)
## Overview
* Tasks achieved
    * Exploratory data analysis on training dataset 
    * NLP applied on customer reviews
    * Prediction of variety of wine
## Data visualization
Remeber that in this stage our goal is not only to explore our data in order to get better predictions. 
We also want to get better understanding what is in data and explore data in 'normal' way. 
This kind of approch can be useful if we have to do some feature engineering, where good data understanding can really 
help to produce better features.
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/EDA/price_distribution_barplot.png "Price Distribution")
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/EDA/price_distribution_barplot_province.png "Price Distribution in Province")
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/EDA/points_by_price.png "Points by countires")
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/EDA/points_distribution_barplot.png "Points by countires")
Top Countries in terms of prices of wine in Descending order
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/EDA/top_countrieswith%20price.png "Price by countires")
Following box plot shows varieties against its points which helps us to derive which varieties secure top points
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/EDA/pointsvsvariety.png "Points by variety")
## Customer review analysis along with Sentiment analysis using NLP
  * Before heading towards NLP part lets try to discover some insights from dataset
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/Top%2020%20wineries.png "Top wineries")
Lengths of customer reviews
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/description%20length%20vs%20points.png "Description length")
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/sentiment%20vs%20points.png "Sentiments")
Wine recommender
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/wine_recommended.png "Recommendations")
WordCloud of customer reviews
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/wordcloud_description.png "wordcloud")
wordcloud of titles
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/wordcloud_title.png "wordcloud")
  * TextBlob is used to get the customer sentiments from their reviews
  In this file customer sentiments can be seen in the last column ** https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/review_sentiment_analysis.csv
## Variety Prediction
  * Algorithms used
    * Random Forest Classifier
    * MLP Classifier
 * Following data shows accuracy achieved in training of these algorithms
 Random Forest Classifier withour Hyperparamete tuning was able to record accuracy of close to 50%
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/variety_predictoin/rf_not_tuned.png "rf")
While after hyperparametr tuning Random Forest classifier achived accuracy of close to 53%
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/variety_predictoin/rf_tuned.png "rf")
Neural Network withour hyper parameter tuning was able to record accuracy of 52%. Unfortunately I could not train this model 
with hyperparameter tuning because of long waitings in training but with training accuracy close to 60% can be achieved with this
neural network

## Project 7 :Testing Best fit Classification Algorithm & Hyperparametertuning
* Dataset used : Titanic
* Objective : Based on the data try to classify who would be saved after the disaster since there are very few life saver boats
* Algorithms Used:
   * Random Forest Classification
   * Gradient Boost
   * Decision Tree
   * MLP Classifier

# Project 8  : Corona Analysis
## Case Studies
   * Exploratory Data Analysis on Covid19 Dataset 
   * Implementation on Indian Subcontinent
   * Time Searies Analysis using facebook Prophet
   * Packages used : pandas, numpy, matplotlib, seaborn, prophet
![alt text](https://github.com/Tejan4422/CoronaData/blob/master/India_crd.png "Confirmed vs Recovered vs Deaths")
![alt text](https://github.com/Tejan4422/CoronaData/blob/master/confirmed%20cases.png "Confirmed")
![alt text](https://github.com/Tejan4422/CoronaData/blob/master/deaths.png "Deaths")
![alt text](https://github.com/Tejan4422/CoronaData/blob/master/recovered%20cases.png "Recovered")
![alt text](https://github.com/Tejan4422/CoronaData/blob/master/prophet_confirmed.png "Prophet confirmed")
![alt text](https://github.com/Tejan4422/CoronaData/blob/master/prophet_confirmed_components.png "Prophet confirmed components")
![alt text](https://github.com/Tejan4422/CoronaData/blob/master/prophet_death_components.png "Prophet Death components")
![alt text](https://github.com/Tejan4422/CoronaData/blob/master/prophet_deaths.png "Prophet Death")
![alt text](https://github.com/Tejan4422/CoronaData/blob/master/prophet_recovered.png "Prophet Recovered")
![alt text](https://github.com/Tejan4422/CoronaData/blob/master/prophet_recovered_components.png "Prophet Recovered components")
![alt text](https://github.com/Tejan4422/CoronaData/blob/master/prophet_recovered.png "Pophet Recovered")

# Project 9  : YouTube Trending Analysis
## Overview
* Scraping data from youtube
  * Youtube API key is important in case you want to connect to youtube Dataset
  * Use this key in your script to get data of trending videos on a particular day.
* Cleaning Data
  * Use lambda function to clean data
* Model Building:
  * Using youtube api key get data about recently added videos
  * train classification model on trending data and recent data to classify whether video can be in trending tab or not
  * need to run the script every day for at least two months to gather significant amount of data to train ML Model
  * recently updated on 26/04/2020
  * In order to predict whether a video can go viral or not create classification model whcih is trained on youtube data
  * Hyper Parameter Tuning is required to selecet best possible algorithm
  * Gradient Boosting, SVR, Random Forest, Decision Tree, MLP Classifier are amongst who performs well.
* Case Studies:
![alt text](https://github.com/Tejan4422/Youtube-Trending/blob/master/output/Norma_distribution.png "Distribution Curve")
![alt text](https://github.com/Tejan4422/Youtube-Trending/blob/master/output/allView.png "Complete data")
![alt text](https://github.com/Tejan4422/Youtube-Trending/blob/master/output/boxplotcatIdlikes.png "Box Plot data")
![alt text](https://github.com/Tejan4422/Youtube-Trending/blob/master/output/heatmap.png "HeatMap")

