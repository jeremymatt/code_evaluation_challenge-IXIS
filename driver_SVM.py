# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 23:03:49 2023

@author: jerem
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import helper_functions as HF
from sklearn import svm
import joblib

#current working directory
cd = os.getcwd()

#Set plot font size
plt.rcParams.update({'font.size': 25})

"""
  This dataset is publicly available for research. The details are described in [Moro et al., 2014]. 
  Please include this citation if you plan to use this database:

  [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, In press, http://dx.doi.org/10.1016/j.dss.2014.03.001

  Available at: [pdf] http://dx.doi.org/10.1016/j.dss.2014.03.001
                [bib] http://www3.dsi.uminho.pt/pcortez/bib/2014-dss.txt
"""
#Build path to the dataset
data_dir = os.path.join(cd,'data','bank+marketing','bank-additional','bank-additional')

#Build paths to store extracted data plots 
#Make directory structure if it doesn't exist
data_output_dir = os.path.join(cd,'outputs','SVM')
if not os.path.isdir(data_output_dir):
    os.makedirs(data_output_dir)

#target feature as identified in the metadata
target_feature = 'y'
#Force translation order
target_feature_dict = {0: 'no', 1: 'yes'}

#Load the dataset
data_df = HF.load_dataframe(os.path.join(data_dir,'bank-additional-full.csv'))
#Convert the target feature to binary with a fixed order
data_df = HF.target_feature_to_binary(data_df,target_feature,target_feature_dict)
#From the exploratory histogram of the pday variable, whether a person was 
#was previously contacted seems predictive of the target variable, but
#the length of time since the previous contact does not seem particularly predictive
#Therefore convert to a binary "previously contacted, yes/no" variable instead
data_df = HF.process_pday(data_df)
#Find categorical columns (columns containing text values)
data_type_dict,categorical_features = HF.identify_categorical_cols(data_df)
#Convert the categorical columns to either one-hot encoding or binary depending
#on the number of values
data_df,binary_dict = HF.gen_onehot_cols(data_df,categorical_features)
#min/max normalize the data
data_df = HF.min_max_norm(data_df)

#All keys that are not the target
data_keys = [key for key in data_df.keys() if not key == target_feature]


#Split into 80/20 train/test sets
train, test = np.split(data_df.sample(frac=1, random_state=42), 
                       [int(.8*len(data_df))])

#Separate the target from the data for 
x_train = train[data_keys].values.astype(float)
y_train = train[target_feature].values.astype(int)

x_test = test[data_keys].values.astype(float)
y_test = test[target_feature].values.astype(int)

"""
Check that proportions of target variables in train and test are similar
to the complete dataset
sum(data_df[target_feature])/data_df.shape[0]
sum(y_train)/len(y_train)
sum(y_test)/len(y_test)
"""

#Unbalanced dataset, so calculate class weights
class_weights = HF.gen_class_weight_dict(data_df,target_feature)

#%%

#create and train SVM model
clf = svm.SVC(class_weight=class_weights)
clf.fit(x_train, y_train)

#Predict the test-sample classes
out = clf.predict(x_test)


#save the trained model
joblib.dump(clf,os.path.join(data_output_dir,'trained_model.pkl'))

results_fn = 'SVM_results.txt'
HF.print_results(y_test,out,target_feature_dict,data_output_dir, results_fn)

# #Ordering for confusion matrix so "yes" is the positive class
# reverse = not binary_dict[target_feature][0] == 'yes'
# #generate confusion matrix
# #NOTE: This is confusion matrix generation code I made for another project
# #because the confusion matrix codes I've found haven't suited my needs
# confusion,true_label_set,pred_label_set = HF.confusion_matrix(y_test,out,labels_dict=binary_dict[target_feature],reverse=reverse)

# print(confusion)

# #Ensure classification report has the labels in the and target names in the correct order
# labels = list(binary_dict[target_feature].keys())
# target_names = [binary_dict[target_feature][key] for key in labels]
# print(classification_report(y_test,out,labels=labels,target_names=target_names))
