# 1 Introduction
This project is a coding exercise for the Customer Propensity Modeling Project assigned as part of the IXIS data science interview process.  The dataset used in this project is freely [available](https://archive.ics.uci.edu/dataset/222/bank+marketing) for download and consists of data from a direct marketing campaign conducted by a Portuguese bank.  Metadata describing the dataset and features is provided both in the download zip file and on the linked website.  This challenge allows the use of all available open source resources, I elected not to use either the published works that have used this dataset or the github repositories that appear to be tackling this exact IXIS modeling project (e.g., [AriodanteEs](https://github.com/AriodanteEs/IXIS-Data-Science-Challenge/blob/master/ixis%20data%20science%20challenge%20presentation.pdf), [zhou-george](https://github.com/zhou-george/ixisdatasciencechallenge), [mkdevarney](https://github.com/mkdevarney/IXIS-Data-Science-Challenge-DeVarney/blob/main/IXISDataScienceChallenge_DeVarney.py)).  I made this choice in order to showcase my thought processes and skills rather than my ability to adapt the work of others.


# 2 Methods
## 2.1 Dataset Exploration
My exploratory investigation of the dataset consisted of rudimentary data quality checks, comparisons of the potential input features to the target feature, an assessment of the duplicate records in the database, an assessment to determine if the dataset could be separated in to unique individuals, and identification of common personas in the database. I checked that the target feature (expected to be binary) contains exactly two states. Printing summaries of categorical feature labels allowed me to check for duplicate label spellings. I broke down the categorical feature labels by target feature and generated histograms of numeric features in order to identify patterns or anomalies that could be useful for subsequent analysis.  

To determine assess duplicate records in the database, I used the pandas groupby function on all input features and the count function to aggregate the target feature.  Entries greater than 1 in the target feature column indicated repeated input patterns.  To estimate a theoretical maximum classification performance, I first assumed that all non-duplicated entries and any duplicated entries with consistent responses (i.e., duplicated input patterns where all instances had a target feature value of either "yes" or "no") could be classified perfectly.  For the duplicated patterns with mixed "yes" and "no" responses, I assumed a classifier that would classify an input pattern as "yes" if the number of actual yesses was greater than or equal to the actual number noes.

I identified common personas in the dataset using a method similar to my assessment of duplicate records.  I selected a set of input variables containing personally identifiable information (age, job, education, default, housing, and loan) and used groupby to count the number of times each unique pattern appeared in the dataset.  To evaluate each persona's propensity to purchase, I calculated the ratio of "yes" responses to "no" responses.

## 2.2 Feature Engineering
The dataset contains a number of categorical features where order may be misleading (e.g., 

## 2.3 Backpropagation

## 2.4 Support Vector Machine

## 2.5 Random Forest

# 3 Results

## 3.1 Dataset Exploration


## 3.2 Backpropagation

## 3.3 Support Vector Machine

## 3.4 Random Forest

# 4 Conclusions/Future Work
