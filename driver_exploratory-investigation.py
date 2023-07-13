# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:22:28 2023

@author: jerem
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import calendar

cd = os.getcwd()

"""
  This dataset is publicly available for research. The details are described in [Moro et al., 2014]. 
  Please include this citation if you plan to use this database:

  [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, In press, http://dx.doi.org/10.1016/j.dss.2014.03.001

  Available at: [pdf] http://dx.doi.org/10.1016/j.dss.2014.03.001
                [bib] http://www3.dsi.uminho.pt/pcortez/bib/2014-dss.txt
"""
data_dir = os.path.join(cd,'data','bank+marketing','bank-additional','bank-additional')

#Load the dataset
data_df = pd.read_csv(os.path.join(data_dir,'bank-additional-full.csv'),sep=';')

"""
Initial Investigation.  Get a sense of the dataset and perform some preliminary
checks to see if the dataset is clean or if there are potential errors
"""
target_feature = 'y'
#Two target feature values (as expected).  
target_feature_set = sorted(list(set(data_df[target_feature])))
print('Target feature values: {}'.format(target_feature_set))

#Determine data type of each feature
data_type_dict = {}
for key in data_df.keys():
    try:
        #If the data can be converted to a float, it is numeric
        data_df[key].astype(float)
        data_type_dict[key] = 'numeric'
    except:
        #otherwise consider the data to be categorical
        data_type_dict[key] = 'categorical'
        
#Convert no previous contact (pdays=999) to pdays=-1 for visualization purposes
data_df.loc[data_df.pdays == 999,'pdays'] = -1


counts_dict = {}
        
for key in data_df.keys():
    if data_type_dict[key] == 'numeric':
        #Plot a histogram
        plt.figure()
        for target in target_feature_set:
            plt.hist(data_df.loc[data_df[target_feature]==target,key],alpha=0.5,bins=20,label = 'target={}'.format(target))
        plt.xlabel(key)
        plt.legend()
    else:
        """
        For each feature, find the breakdown of number of instances 
        """
        counts_dict[key] = []
        print('\n{}:'.format(key))
        cur_feature_set = set(data_df[key])
        for feature_val in cur_feature_set:
            temp_df = data_df.loc[data_df[key]==feature_val]
            num_instances = temp_df.shape[0]
            counts_dict[key].append((feature_val,num_instances))
            percent = 100*num_instances/data_df.shape[0]
            display_string = '{}: {}/{} ({:0.2f}%). Targ. feat.:'.format(feature_val,num_instances,data_df.shape[0],percent)
            for target_feature_val in target_feature_set:
                temp_num_instances = temp_df.loc[temp_df[target_feature] == target_feature_val].shape[0]
                percent = 100*temp_num_instances/num_instances
                display_string = '{} {}={}/{}({:0.2f}%)'.format(display_string,target_feature_val,temp_num_instances,num_instances,percent)
            print('  {}'.format(display_string))
                
     
"""
Some months have very few associated records (e.g., december only has 182 - 0.44%)
Create plot to investigate

"""
#adapted from: https://stackoverflow.com/questions/3418050/how-to-map-month-name-to-month-number-and-vice-versa
month_num_dict = {month.lower(): index for index, month in enumerate(calendar.month_abbr) if month}
day_num_dict = {month.lower(): index for index, month in enumerate(calendar.day_abbr) if month}

month_counts = counts_dict['month']

month_count_dict = {month_num_dict[tpl[0]]:tpl[1] for tpl in month_counts}

month = []
count = []
for i in range(1,13):
    month.append(i)
    if i in month_count_dict.keys():
        count.append(month_count_dict[i])
    else:
        count.append(0)

plt.figure()        
plt.bar(month,count)
plt.xlabel('Month')
plt.ylabel("num records")

"""
Check for records that might indicate duplicates (e.g., different contacts for different people)
max. value for pdays is 27, so assuming that none of this information changes
in 27 days
"""
identifying_records = [
    'age',
    'job',
    'education',
    'default',
    'housing',
    'loan']

temp = data_df[identifying_records].copy()
temp['counts'] = 1
data_df_agg = temp.groupby(identifying_records,as_index=False).count()
print('\n{} unique combinations of identifying information; not enough to separate dataset by individual'.format(data_df_agg.shape[0]))


identifying_records = [
    'age_range',
    'job',
    'education',
    'default',
    'housing',
    'loan']

#Find the minimum and maximum ages
min_age = data_df.age.min()
max_age = data_df.age.max()

#Define age categories 
group_by = 10
min_age_category = int(np.floor(min_age/group_by)*group_by)
max_age_category = int(np.ceil(max_age/group_by)*group_by)
#+1 to max_age_category because arange is on the range of [min,max)
breaks = np.arange(min_age_category,max_age_category+0.1,group_by).astype(int) 
break_tpls = list(zip(breaks[:-1],breaks[1:]))

#add age-range variable
for start,end in break_tpls:
    mask = (data_df.age >= start) & (data_df.age < end)
    data_df.loc[mask,'age_range'] = '{}-{}'.format(start,end-1)


identifying_tpls = list(zip(data_df.age_range,data_df.job,data_df.education,data_df.default,data_df.housing,data_df.loan))
identifying_tpl_set = set(identifying_tpls)

identifying_tpl_set_counts = []

summary_dict = pd.DataFrame()

tpl = identifying_tpls[0]
"""
Aggregate dataset to see who is in the dataset
"""
#Note: It may be possible to do this using pandas groupby (perhaps with lambdas 
#to calculate the "no" and "yes" columns?) but it's not immediately obvious to me
print('\n')
for ctr,tpl in enumerate(identifying_tpl_set):
    print('\rProcessing unique identifier set: {}/{}'.format(str(ctr).zfill(4),len(identifying_tpl_set)),end='',flush=True)
    mask = np.all(data_df[identifying_records] == tpl,axis=1)
    outcome_counts = []
    for target_feature_val in target_feature_set:
        outcome_counts.append(sum(data_df.loc[mask,target_feature] == target_feature_val))
    count = sum(mask)
    
    summary_dict.loc[ctr,'counts'] = count
    summary_dict.loc[ctr,identifying_records] = tpl
    summary_dict.loc[ctr,target_feature_set] = outcome_counts
    
    
summary_dict['frac_y'] = summary_dict.yes/summary_dict.counts
summary_dict.sort_values('counts',ascending=False,inplace=True)
        
"""
Questions/Notes:
    1. Reasons for the uneven distribution of contact-by-month?
    2. If previous contact was a success (e.g., poutcome='success') why was there another contact?
    3. If groupby age,job,education,default,housing,&loan, there are only ~9k unique records
    
"""