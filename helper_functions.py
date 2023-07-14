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
    data_df = pd.read_csv(path)
    features_to_keep = [key for key in data_df.keys() if not key in features_to_drop]
    data_df = data_df[features_to_keep]
    
    return data_df


def gen_onehot_cols(df,features):
    df = df.copy()
    for feature in features:
        unique_vals = set(df[feature])
        col_names = [feature]
        for val in unique_vals:
            col_name = '{}.{}'.format(feature,val)
            col_names.append(col_name)
            df[col_name] = 0
            df.loc[df[feature]==val,col_name] = 1
            
    keys_to_keep = [key for key in df.keys() if not key in features]
    
    df = df[keys_to_keep]
    
    return df
        
def convert_binary_to_int(df,features):
    