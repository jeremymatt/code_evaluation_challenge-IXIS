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

#Load the dataset
data_df = HF.load_dataframe(os.path.join(data_dir,'bank-additional-full.csv'))
data_df = HF.process_pday(data_df)
data_type_dict,categorical_features = HF.identify_categorical_cols(data_df)
data_df,binary_dict = HF.gen_onehot_cols(data_df,categorical_features)
data_df = HF.min_max_norm(data_df)

#target feature as identified in the metadata
target_feature = 'y'
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

sum(data_df[target_feature])/data_df.shape[0]
sum(y_train)/len(y_train)
sum(y_test)/len(y_test)

classes = np.unique(data_df[target_feature].astype(int))
class_weights = sklearn.utils.class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes = classes,
    y = data_df[target_feature].astype(int))

class_weights = dict(zip(np.unique(classes), class_weights))

#%%

#Init a dictionary to hold results
results_dict = {}

#Rudimentary parameter optimization to evaluate precision/recall tradeoff
#for number of estimators and max depth of each tree
for n_estimators in np.arange(25,150,25):
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
        results_dict[n_estimators]['precision'].append(precision_score(y_test, out))
        results_dict[n_estimators]['recall'].append(recall_score(y_test, out))
   
print('\n\n')
#Plot a precision/recall ROC curve
fig,ax = plt.subplots(1,1,figsize=[20,15])
for n_estimators in results_dict.keys():
    ax.plot(results_dict[n_estimators]['precision'],results_dict[n_estimators]['recall'],label = 'n_estimators: {}'.format(n_estimators))
    
ax.set_xlabel('precision')
ax.set_ylabel('recall')
fig.legend()
     

#Selected values targeting precision around 40% and recall around 60%
n_estimators = 50
max_depth = 10

print('\rSelected Estimators: {}, depth: {}'.format(n_estimators,max_depth),end='',flush=True)
#Init the model, train, and predict the test-set labels
model = RF(
    n_estimators = n_estimators,
    max_depth = max_depth,
    class_weight = class_weights)
model.fit(x_train, y_train)
out = model.predict(x_test)
        
#Determine label order for the confusion matrix
reverse = not binary_dict[target_feature][0] == 'yes'
#generate confusion matrix
#NOTE: This is confusion matrix generation code I made for another project
#because the confusion matrix codes I've found haven't suited my needs
confusion,true_label_set,pred_label_set = HF.confusion_matrix(y_test,out,labels_dict=binary_dict[target_feature],reverse=reverse)

print(confusion)

#Ensure classification report has the labels in the and target names in the correct order
labels = list(binary_dict[target_feature].keys())
target_names = [binary_dict[target_feature][key] for key in labels]
#Print the classification report
print(classification_report(y_test,out,labels=labels,target_names=target_names))
