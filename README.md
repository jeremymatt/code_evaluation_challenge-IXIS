# 1 Introduction
This project is a coding exercise for the Customer Propensity Modeling Project assigned as part of the IXIS data science interview process.  The dataset used in this project is freely [available](https://archive.ics.uci.edu/dataset/222/bank+marketing) for download and consists of data from a direct marketing campaign conducted by a Portuguese bank.  Metadata describing the dataset and features is provided both in the download zip file and on the linked website.  This challenge allows the use of all available open source resources, I elected not to use either the published works that have used this dataset or the github repositories that appear to be tackling this exact IXIS modeling project (e.g., [AriodanteEs](https://github.com/AriodanteEs/IXIS-Data-Science-Challenge/blob/master/ixis%20data%20science%20challenge%20presentation.pdf), [zhou-george](https://github.com/zhou-george/ixisdatasciencechallenge), [mkdevarney](https://github.com/mkdevarney/IXIS-Data-Science-Challenge-DeVarney/blob/main/IXISDataScienceChallenge_DeVarney.py)).  I made this choice in order to showcase my thought processes and skills rather than my ability to adapt the work of others.


# 2 Methods
## 2.1 Dataset Exploration
My exploratory investigation of the dataset consisted of rudimentary data quality checks, comparisons of the potential input features to the target feature, an assessment of the duplicate records in the database, an assessment to determine if the dataset could be separated in to unique individuals, and identification of common personas in the database. I checked that the target feature (expected to be binary) contains exactly two states. Printing summaries of categorical feature labels allowed me to check for duplicate label spellings. I broke down the categorical feature labels by target feature and generated histograms of numeric features in order to identify patterns or anomalies that could be useful for subsequent analysis.  

To determine assess duplicate records in the database, I used the pandas groupby function on all input features and the count function to aggregate the target feature.  Entries greater than 1 in the target feature column indicated repeated input patterns.  To estimate a theoretical maximum classification performance, I first assumed that all non-duplicated entries and any duplicated entries with consistent responses (i.e., duplicated input patterns where all instances had a target feature value of either "yes" or "no") could be classified perfectly.  For the duplicated patterns with mixed "yes" and "no" responses, I assumed a classifier that would classify an input pattern as "yes" if the number of actual yesses was greater than or equal to the actual number noes.

I identified common personas in the dataset using a method similar to my assessment of duplicate records.  I selected a set of input variables containing personally identifiable information (age, job, education, default, housing, and loan) and used groupby to count the number of times each unique pattern appeared in the dataset.  To evaluate each persona's propensity to purchase, I calculated the ratio of "yes" responses to "no" responses.

## 2.2 Feature Engineering and Preprocessing
The dataset contains a number of categorical features where converting to numeric order may be misleading or may cause discontinuities (e.g., day of the week, month, job).  I converted these features to one-hot encoding with a column header format of <original_feature_name>.<label>.  When reviewing the histograms for the pday variable, I noticed that people who had been previously contacted appeared to have a higher propensity to purchase, but that the length of time since the previous contact did not seem particularly predictive (Figure 1).  I converted this feature to a binary variable indicating whether or not the person was previously contacted.

<INSERT FIG1>
Figure 1: Histogram of time-since-previous-contact (pdays).  A value of -1 indicates that the person was not previously contacted.

For all three models discussed below, I elected to use min/max normalization on all input features.  Due to time constraints I did not investigate other normalization methods (e.g., clipping, z-score, log).  

The dataset is highly imbalanced, with a yes/no ratio of approximately 0.11.  To adjust for this, I used scikit-learn to compute balanced class weights based on the full dataset.  These class weights were used for all three models discussed below.

## 2.3 Backpropagation
I implemented a dense backpropagation neural network in Keras with the ability to flexibly choose the number of hidden layers and the number of neurons in each hidden layer. I trained the network using a random 60/20/20 train/validation/test split. I performed an informal parameter sweep of number of layers, number of hidden neurons, learning rate, and dropout. The final parameters I selected were one hidden layer with five hidden neurons, learning rate equal to 0.0005, and dropout of 0.0. I tried both the Adam and RMSprop optimizers; RMSprop appeared to perform slightly better. Using ReLU vs. a sigmoid activation function for the hidden layer(s) did not appear to significantly affect performance.  I did not perform formal optimizer selection or formal parameter selection due to time constraints and because the informal parameter selection did not have a strong effect on overall performance in terms of accuracy, precision, and recall on the test set.

## 2.4 Support Vector Machine
I implemented a support vector machine (SVM) using scikit-learn.  The out-of-box performance was worse than that of either backpropagation or random forest so I elected not to pursue SVM further.  

## 2.5 Random Forest
I implemented random forest using the built-in scikit-learn package.  Based on an informal parameter sweep, the number of predictors and the max depth of each predictor seemed to cause a precision/recall tradeoff.  I performed a parameter sweep with the number of estimators set to 5, 15, 25, 35, 45, and 55 and with max depths of 1-19 (Figure 2). Based on  this curve, 35 estimators with a max depth of 10 appeared to be a reasonable tradeoff between precision (~40%) and recall (~60%).  The scikit-learn random forest package includes an estimate of feature importance.  Using these settings, I conducted feature selection by repeatedly training a new random forest model and then dropping the least-important feature.  I repeated this until the ratio of most important feature to least important feature was less than 10.  As this progressed, I plotted precision and recall vs the number of features removed. 


<INSERT FIG2>
Figure 2: Random Forest overview

# 3 Results

## 3.1 Dataset Exploration
I did not identify any obvious data quality problems with the dataset.  The most common persona (n=2,276/41,188) in the dataset was a 30 year old university-educated admin. with no default, no personal loan, and either with (n=1,232) or without (n=1,044) a housing loan. This group had a relatively high average propensity to purchase (14.2%).  Conversely 30 year old blue-collar workers (n=1,007/41,188) with no default, no personal loan, and either with (n=522) or without (n=485) a housing loan had a low average propensity to purchase (7.1%).  Age appears to be a strong predictor, with people over 60 having a very high propensity to purchase (40.0%).  Education is also a predictor.  People with less than a highschool education have a low average propensity to purchase (8.7%) compared to those with a university education (13.7%).  These summaries are drawn from the ```/outputs/exploratory/aggregate_summary_data.csv``` file.

During the exploration, I noted that the pattern of contacts-per-month is not constant from month to month (Figure 3).  Since this is a multi-year dataset (2008-2013) the reason for this pattern is unclear.

<INSERT FIG3>
Figure 3: Stacked bar chart of contact successes and failures by month.  Note the uneven distribution of total contacts as well as the varying proportions of success vs. failure.

## 3.2 Backpropagation
Backpropagation showed signs of overfitting regardless of the training parameters chosen.  While the training and validation accuracy curves did not appear to diverge systematically, the validation accuracy showed random fluctuations from one training epoch to the next, suggesting that the algorithm may not be generalizing well (Figure 4).

<INSERT FIG4>
Figure 4: Training and validation accuracy vs. training epoch.  Note the noise in the validation accuracy.  


## 3.3 Support Vector Machine

## 3.4 Random Forest

# 4 Conclusions/Future Work


## Future tasks:
1. Additional feature selection and/or feature engineering  
1. Investigate other prediction models  
1. Investigate the uneven pattern of contacts-per-month  

## Questions:
1. What is the nr.employed feature?  The metadata says it is the "number of employees - quarterly indicator (numeric)".  While this information may be irrelevant to this task, it's unclear to me what this is measuring.  Is it an employment indicator? Perhaps the number of people employed per 10,000 people?  