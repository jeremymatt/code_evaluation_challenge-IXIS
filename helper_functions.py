# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 19:48:22 2023

@author: jerem
"""

import pandas as pd
import numpy as np
from keras import layers
from keras import Input
from keras import Model
import keras
import os

from sklearn.metrics import classification_report
import sklearn



#Drop the "duration" feature per discussion in the metadata:
"""
    Important note:  this attribute highly affects the output target 
    (e.g., if duration=0 then y="no"). Yet, the duration is not known before a 
    call is performed. Also, after the end of the call y is obviously known. 
    Thus, this input should only be included for benchmark purposes and should 
    be discarded if the intention is to have a realistic predictive model.
"""
features_to_drop = ['duration']

# """
# List of categorical features where order may not matter.  Use one-hot encoding
# for these features.
# """
# onehot_features = [
#     'job',
#     'marital',
#     'education',
#     'default',
#     'housing',
#     'loan',
#     'month',
#     'day_of_week',
#     'poutcome']

# binary_features = [
#     'contact']

def gen_class_weight_dict(df,target_feature):
    #Find the unique classes
    classes = np.unique(df[target_feature].astype(int))
    #Compute balanced class weights
    class_weights = sklearn.utils.class_weight.compute_class_weight(
        class_weight = 'balanced',
        classes = classes,
        y = df[target_feature].astype(int))
    #Convert to a dict
    class_weights = dict(zip(np.unique(classes), class_weights))
    return class_weights

def print_results(y_test,out,target_feature_dict,output_dir, results_fn):
    
    
    #Ordering for confusion matrix so "yes" is the positive class
    reverse = True
    #generate confusion matrix
    #NOTE: This is confusion matrix generation code I made for another project
    #because the confusion matrix codes I've found haven't suited my needs
    confusion,true_label_set,pred_label_set = confusion_matrix(y_test,out,labels_dict=target_feature_dict,reverse=reverse)
    
    confusion.to_csv(os.path.join(output_dir,'confusion_matrix.csv'))
    
    print(confusion)
    print('\nClassification Report:')
    #Ensure classification report has the labels in the and target names in the correct order
    labels = list(target_feature_dict.keys())
    target_names = [target_feature_dict[key] for key in labels]
    classification_results = classification_report(y_test,out,labels=labels,target_names=target_names)
    print(classification_results)
    
    with open(os.path.join(output_dir,results_fn), 'w') as f:
        f.write('Confusion matrix (rows = truth, columns = predictions):\n')
        print(confusion,file=f)
        f.write('\n\nClassification report:\n')
        f.write(classification_results)

def target_feature_to_binary(df,target_feature,target_feature_dict):
    
    reverse_dict = {target_feature_dict[key]:key for key in target_feature_dict.keys()}
    target_feature_vals = set(df[target_feature])
    
    for val in target_feature_vals:
        if not val in reverse_dict.keys():
            raise ValueError('ERROR: "{}" not included in target_feature_dict'.format(val))
        df.loc[df[target_feature] == val,target_feature] = reverse_dict[val]
        
    return df

def load_dataframe(path):
    data_df = pd.read_csv(path,sep=';')
    features_to_keep = [key for key in data_df.keys() if not key in features_to_drop]
    data_df = data_df[features_to_keep]
    
    #Convert no previous contact (pdays=999) to pdays=-1 for visualization purposes
    data_df.loc[data_df.pdays == 999,'pdays'] = -1
    
    return data_df

def identify_categorical_cols(df):
    #Determine data type of each feature
    data_type_dict = {}
    for key in df.keys():
        try:
            #If the data can be converted to a float, it is numeric
            df[key].astype(float)
            data_type_dict[key] = 'numeric'
        except:
            #otherwise consider the data to be categorical
            data_type_dict[key] = 'categorical'
    categorical_features = [feat for feat in data_type_dict.keys() if data_type_dict[feat].startswith('cat')]
    features = categorical_features
    
    return data_type_dict,categorical_features

def process_pday(df):
    #based on the exploratory analysis, whether or not a person was previously
    #contacted seems predictive, but the duration doesn't seem terribly important
    #convert variable to binary variable.
    mask = df.pdays == -1
    df.loc[mask,'pdays'] = 1
    df.loc[~mask,'pdays'] = 0
    return df

def gen_onehot_cols(df,features):
    df = df.copy()
    #Init a dictionary to store the original values of binary features so that 
    #info can be recovered if necessary
    binary_dict = {}
    for feature in features:
        #Find the set of unique values
        unique_vals = set(df[feature])
        if len(unique_vals) == 2:  #binary variable, one column will suffice
            #If the feature is binary, remove it from the list of features that
            #will later be dropped from the dataframe
            features.remove(feature)
            #Init sub-dictionary to translate current feature back to 
            #original values
            binary_dict[feature] = {}
            for ctr,val in enumerate(unique_vals):
                #Add the value to the translation dictionary
                binary_dict[feature][ctr] = val
                #Update the value in the dataframe to 0 or 1
                df.loc[df[feature] == val,feature] = ctr
        #one-hot encoding is required, so make new columns
        elif len(unique_vals) > 2: 
            col_names = [feature]
            for val in unique_vals:
                col_name = '{}.{}'.format(feature,val)
                col_names.append(col_name)
                df[col_name] = 0
                df.loc[df[feature]==val,col_name] = 1
        else:
            #the feature has a single value and therefore has no predictive value
            #Retain feature in the features list and it will be dropped from
            #the dataframe
            DoNothing=True
      
    #Q/A code to check that 
    check_onehot_encoding = False
    if check_onehot_encoding:
        feature = 'job'
        col_names = [key for key in df.keys() if key.startswith(feature)]
        temp = df.loc[df.index[:10],col_names].copy()
     
    #Keep all keys in the dataframe as long as they're not in the features list
    keys_to_keep = [key for key in df.keys() if not key in features]
    
    #Drop the unnecssary keys
    df = df[keys_to_keep]
    
    return df,binary_dict

def min_max_norm(df):
    for key in df.keys():
        key_min = df[key].min()
        key_max = df[key].max()
        if (key_min != 0) or (key_max != 1):
            df.loc[:,key] = (key_max-df.loc[:,key])/(key_max-key_min)
            
    return df
            
            
def build_backprop_model(num_features,layer_neuron_list,output_dir,dropout = 0):
    """
    Generates a dense backprop model

    Parameters
    ----------
    num_features : TYPE int
        DESCRIPTION.
        Number of input features to be passed into the network
    layer_neuron_list : TYPE list of integers
        DESCRIPTION.
        Determines the number of hidden layers from len(layer_neuron_list).
        Each element describes the number of hidden layer neurons to use

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
      
    #Define the input layer
    input_layer = Input(shape = (num_features,))
  
    #Add the first layer
    cur_layer = layers.Dense(layer_neuron_list[0],activation='relu')(input_layer)
    #Store the current layer in a list
    layer_list = [cur_layer]
    #For each subsequent layer in the neuron list, add layer to the model
    for ctr,num_neurons in enumerate(layer_neuron_list[1:]):
        temp = layers.Dense(num_neurons,activation='relu')(layer_list[ctr])
        layer_list.append(temp)
    
    
    #Define the output layer with sigmoid activation
    outputs = layers.Dense(1,activation='sigmoid')(layer_list[-1])
    
    #Add dropout
    do = layers.Dropout(dropout)(outputs)

    #Construct the model
    model = Model(inputs = input_layer,outputs=do)
        
    #Compile the model
    model.compile(
        loss='binary_crossentropy', 
        optimizer=keras.optimizers.RMSprop(learning_rate=0.0005), 
        metrics = ['accuracy'])
    
    #Print summary to screen and to file
    model.summary()
    with open(os.path.join(output_dir,'model_summary.txt'),'w',newline='\n') as f:
        model.summary(print_fn = lambda line: f.write('{}\n'.format(line)))
    
    return model        


def confusion_matrix(labels_true,labels_pred,labels_dict=None,reverse=True):
    """
    Generates confusion matrix from true and predicted labels.  Optionally
    renames classes based on labels_dict.  True values are on rows and predicted
    values are on columns

    Parameters
    ----------
    label_true : TYPE
        DESCRIPTION.
    label_pred : TYPE
        DESCRIPTION.
    labels_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    label_true : TYPE pandas dataframe
        DESCRIPTION.

    """
    
    confusion = pd.DataFrame()
    
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    
    true_label_set = set(labels_true)
    pred_label_set = set(labels_pred)
    all_labels_set = true_label_set.union(pred_label_set)
    
    all_labels_set = list(all_labels_set)
    all_labels_set.sort(reverse = reverse)
    
    true_label_set = list(true_label_set)
    true_label_set.sort(reverse = reverse)
    
    pred_label_set = list(pred_label_set)
    pred_label_set.sort(reverse = reverse)
    
    
    
    
    if not type(labels_dict) == dict:
        labels_dict = {}
        for label in all_labels_set:
            labels_dict[label] = str(label)
    
    for true_label in true_label_set:
        true_index = labels_dict[true_label]
        true_mask = labels_true == true_label
        for pred_label in pred_label_set:
            pred_index = labels_dict[pred_label]
            pred_mask = labels_pred == pred_label
            confusion.loc[true_index,pred_index] = int(sum(true_mask&pred_mask))
            
    confusion = confusion.astype(int)
    
    confusion['n'] = confusion.sum(axis=1)
    
    return confusion,true_label_set,pred_label_set