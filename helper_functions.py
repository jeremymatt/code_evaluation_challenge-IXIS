# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 19:48:22 2023

@author: jerem
"""

import pandas as pd



#Drop the "duration" feature per discussion in the metadata:
"""
    Important note:  this attribute highly affects the output target 
    (e.g., if duration=0 then y="no"). Yet, the duration is not known before a 
    call is performed. Also, after the end of the call y is obviously known. 
    Thus, this input should only be included for benchmark purposes and should 
    be discarded if the intention is to have a realistic predictive model.
"""
features_to_drop = ['duration']

"""
List of categorical features where order may not matter.  Use one-hot encoding
for these features.
"""
onehot_features = [
    'job',
    'marital',
    'education',
    'default',
    'housing',
    'loan',
    'month',
    'day_of_week',
    'poutcome']

binary_features = [
    'contact']

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
    
    return dfd

def min_max_norm(df):
    for key in df.keys():
        key_min = df[key].min()
        key_max = df[key].max()
        if (key_min != 0) or (key_max != 1):
            df.loc[:,key] = (key_max-df.loc[:,key])/(key_max-key_min)
            
    return df
            
            
        