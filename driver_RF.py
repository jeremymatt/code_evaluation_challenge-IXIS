# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 09:56:00 2023

@author: jerem
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import calendar
import helper_functions as HF
from keras import layers
from keras import Input
from keras import Model
import sklearn
from sklearn.metrics import classification_report,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier as RF


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
plot_output_dir = os.path.join(data_output_dir,'plots')
if not os.path.isdir(plot_output_dir):
    os.makedirs(plot_output_dir)

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


#Split into 60/20/20 train/validate/test sets
train, validate, test = np.split(data_df.sample(frac=1, random_state=42), 
                       [int(.6*len(data_df)), int(.8*len(data_df))])

x_train = train[data_keys].values.astype(float)
y_train = train[target_feature].values.astype(int)

x_val = validate[data_keys].values.astype(float)
y_val = validate[target_feature].values.astype(int)

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

#Init a dictionary to hold results
results_dict = {}

#Rudimentary parameter optimization to evaluate precision/recall tradeoff
#for number of estimators and max depth of each tree
for n_estimators in np.arange(5,60,10):
    #Init lists in the results dict to hold precision and recall for each 
    #max depth value
    results_dict[n_estimators] = {'precision':[],'recall':[]}
    for max_depth in range(1,20):
        #Update user on progress
        print('\rEstimators: {}, depth: {}'.format(n_estimators,max_depth),end='',flush=True)
        #Init and train a model
        model = RF(
            n_estimators = n_estimators,
            max_depth = max_depth,
            class_weight = class_weights)
        model.fit(x_train, y_train)
        #Predict the validation set variables
        out = model.predict(x_val)
        #Store the precision and recall in the results dict
        results_dict[n_estimators]['precision'].append(precision_score(y_val, out))
        results_dict[n_estimators]['recall'].append(recall_score(y_val, out))
   
print('\n\n')
#Plot a precision/recall ROC curve
fig,ax = plt.subplots(1,1,figsize=[20,15])
for n_estimators in results_dict.keys():
    ax.plot(results_dict[n_estimators]['precision'],results_dict[n_estimators]['recall'],label = 'n_estimators: {}'.format(n_estimators))
ax.set_xlabel('precision')
ax.set_ylabel('recall')
ax.grid()
fig.legend()
     

#Selected values targeting precision around 40% and recall around 60%
n_estimators = 25
max_depth = 10

print('\rSelected Estimators: {}, depth: {}    \n\n'.format(n_estimators,max_depth),end='',flush=True)
#Init the model, train, and predict the test-set labels
model = RF(
    n_estimators = n_estimators,
    max_depth = max_depth,
    class_weight = class_weights)
model.fit(x_train, y_train)
out = model.predict(x_test)
        

HF.print_results(y_test,out,target_feature_dict)

# #Determine label order for the confusion matrix
# reverse = not binary_dict[target_feature][0] == 'yes'
# #generate confusion matrix
# #NOTE: This is confusion matrix generation code I made for another project
# #because the confusion matrix codes I've found haven't suited my needs
# confusion,true_label_set,pred_label_set = HF.confusion_matrix(y_test,out,labels_dict=binary_dict[target_feature],reverse=reverse)

# print(confusion)

# #Ensure classification report has the labels in the and target names in the correct order
# labels = list(binary_dict[target_feature].keys())
# target_names = [binary_dict[target_feature][key] for key in labels]
# #Print the classification report
# print(classification_report(y_test,out,labels=labels,target_names=target_names))
