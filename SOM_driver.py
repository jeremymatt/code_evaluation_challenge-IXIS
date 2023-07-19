# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 18:03:14 2023

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

import SOM


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
data_output_dir = os.path.join(cd,'outputs','SOM')
if not os.path.isdir(data_output_dir):
    os.makedirs(data_output_dir)

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


legend_text = binary_dict[target_feature]




#Split into 60/20/20 train/validate/test sets
train, test = np.split(data_df.sample(frac=1, random_state=42), 
                       [int(.8*len(data_df))])

x_train = train[data_keys].values.astype(float)
y_train = train[target_feature].values.astype(int)

x_test = test[data_keys].values.astype(float)
y_test = test[target_feature].values.astype(int)

X = data_df[data_keys]
X = test


##################
#  Dataset Settings
##################
n_bins = 10
weights_fn = 'weights.pkl'
num_runs = 1  #Number of SOM runs to complete
#If loading, we only want to re-plot once
verbose = False
plot_fn_prefix = None

##################
#  SOM Training Settings
##################
# grid_size = [50,50]
grid_size = [10,10]
# grid_size = [4,4]
#Set the starting learning rate and neighborhood size
alpha = 0.9
neighborhood_size = int(grid_size[0]/2)
num_epochs = 50
toroidal = True
distance='euclidean'

##################
#  Visualization Settings
##################
plot_legend = True
legend_text = 'auto'
n_clusters = 2
#Method for displaying samples on the u-matrix and feature planes
#either "symbols" or "labels"
#symbols plots points with different symbols based on the "label_ID" column
#labels prints a text string at each sample location (such as the patient ID)
sample_vis='symbols'  
#background for the feature planes.  Either the raw weights or a u-matrix 
#style of visualization showing weight change between grid cells
plane_vis='weights'  #either "weights" or "u_matrix"
extended_marker_list = False
plot_legend = True


#boolean flag to include D in the u-matrix calculations
include_D = False

SOM_model = SOM.SOM(grid_size,X,target_feature,data_keys,alpha,neighborhood_size,toroidal,distance)

SOM_model.train(num_epochs)

SOM_model.save_weights(data_output_dir,weights_fn)
        
# SOM_model.plot_weight_hist(data_output_dir)

SOM_model.calc_u_matrix()

SOM_model.visualization_settings(data_output_dir,sample_vis,legend_text,include_D,target_feature,plane_vis,plot_legend,plot_fn_prefix)

SOM_model.build_color_marker_lists(extended_marker_list)


plot_fn_prefix = 'num_clusters_{}-'.format(n_clusters)
SOM_model.visualization_settings(data_output_dir,sample_vis,legend_text,include_D,target_feature,plane_vis,plot_legend,plot_fn_prefix)
SOM_model.plot_clusters(n_clusters)
