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
import helper_functions as HF

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
data_output_dir = os.path.join(cd,'outputs','exploratory')
plot_output_dir = os.path.join(data_output_dir,'plots')
if not os.path.isdir(plot_output_dir):
    os.makedirs(plot_output_dir)

#Load the dataset
data_df = HF.load_dataframe(os.path.join(data_dir,'bank-additional-full.csv'))

"""
Initial Investigation.  Get a sense of the dataset and perform some preliminary
checks to see if the dataset is clean or if there are potential errors
"""
#target feature as identified in the metadata
target_feature = 'y'
#Two target feature values (as expected).  
target_feature_set = sorted(list(set(data_df[target_feature])))
print('Target feature values: {} <== Expected two'.format(target_feature_set))


#%% Investigate the number of records with duplicated input features

#List of input feature column headings
input_features = [key for key in data_df.keys() if not key==target_feature]
#Groupby the input features and count the number of duplicates
groupby_target = data_df.groupby(input_features,as_index=False).count()
#Keep only input patterns that occur more than once
groupby_target = groupby_target[groupby_target.y>1]

#Calculate the number of input records that are not unique
num_duplicated = groupby_target.y.sum()
print('\n# entries that have at least one duplicate: {}'.format(num_duplicated))

#Make copy of dataframe to retain only unique rows (used to create a manual
#confusion matrix)
data_df_unique = data_df.copy()

#Add a "prediction" column
data_df['prediction'] = data_df.y

#Init a list to hold indices of duplicated input values with inconsistent target variables
mixed_target = []
#counter to track progress
ctr = 1

#Init a dataframe to manually build a confusion matrix
confusion_matrix = pd.DataFrame(columns=['yes','no'],index=['yes','no'],data = 0)

for ind,row in groupby_target.iterrows():
    #Update progress for user
    print('\rDup. group {}\{}    '.format(str(ctr).zfill(4),len(groupby_target)),end='',flush = True)
    ctr += 1
    #Find the rows matching the current set of duplicated input features
    #and extract to a temporary dataframe
    mask = np.all(data_df[input_features] == row[input_features],axis=1)
    temp = data_df[mask]
    #Drop the duplicated rows from the unique dataframe
    data_df_unique.drop(index=temp.index,axis=1,inplace=True)
    #Determine the number of "yes" and "no" responses
    num_yes = sum(temp.y=='yes')
    num_no = sum(temp.y=='no')
    
    #In the case of mixed responses for the same input pattern, the best a 
    #classifier can do is pick either "yes" or "no" for that input pattern.  
    #For this exercise, if there are fewer "no" responses, pick "no".  Otherwise
    #pick "yes"
    if num_yes >= num_no:
        #"predict" as "yes"
        #Update the "prediction" column
        data_df.loc[temp.index,'prediction'] = 'yes'
        #Fill in the manual confusion matrix
        confusion_matrix.loc['yes','yes'] += num_yes #Correctly "classified" as "yes"
        confusion_matrix.loc['no','yes'] += num_no #"yes" incorrectly "classified" as "no"
    else:
        #"predict" as "no"
        #Update the "prediction" column
        data_df.loc[temp.index,'prediction'] = 'no'
        #Fill in the manual confusion matrix
        confusion_matrix.loc['no','no'] += num_no #Correctly "classified" as "no"
        confusion_matrix.loc['yes','no'] += num_yes #"no" incorrectly "classified" as "yes"
    
#Add the unique entries to the manual confusion matrix and add a 'n' column
#containing the sums of true values
confusion_matrix.loc['yes','yes'] += sum(data_df_unique.y == 'yes')
confusion_matrix.loc['no','no'] += sum(data_df_unique.y == 'no')
confusion_matrix['n'] = confusion_matrix[['yes','no']].sum(axis=1)

print('\n\n')
#create a 1:1 feature dict and print the reuslts
target_feature_dict = {'yes':'yes','no':'no'}
results_fn = 'ideal_results.txt'
HF.print_results(data_df.y,data_df.prediction,target_feature_dict,data_output_dir,results_fn)
#Print the manual confusion matrix for comparision purposes
print('manual confusion matrix:')
print(confusion_matrix)

#Drop the faked "prediction" column
data_df.drop('prediction',axis=1,inplace=True)

#%%

#Identify which columns exclusively contain numeric data and which do not
data_type_dict,categorical_features = HF.identify_categorical_cols(data_df)

#Init an empty dictionary to hold counts for later use
counts_dict = {}
        
with open(os.path.join(data_output_dir,'exploratory_counts.txt'),'w') as f:
    for key in data_df.keys():
        if data_type_dict[key] == 'numeric':
            #Plot histograms split by the target feature
            
            fig,ax = plt.subplots(1,1,figsize=[20,15])
            for target in target_feature_set:
                ax.hist(data_df.loc[data_df[target_feature]==target,key],alpha=0.5,bins=20,label = 'target={}'.format(target))
            ax.set_xlabel(key)
            ax.set_yscale('log')
            ax.legend()
            fig.savefig(os.path.join(plot_output_dir,'histogram_{}.png'.format(key)),bbox_inches='tight')
            plt.close(fig)
        else:
            """
            For each feature, find the breakdown of number of instances 
            """
            #Init a list to hold counts for the current feature
            counts_dict[key] = []
            f.write('\n{}:\n'.format(key))
            cur_feature_set = set(data_df[key])
            for feature_val in cur_feature_set:
                #Extract a temporary dataframe of records that match the current
                #feature and value
                temp_df = data_df.loc[data_df[key]==feature_val]
                #Find the number of "yes", "no", and total responses
                num_yes = sum(temp_df.y == 'yes')
                num_no = sum(temp_df.y == 'no')
                num_instances = temp_df.shape[0]
                #Add the counts to the dictionary as a tuple
                counts_dict[key].append((feature_val,num_instances,num_yes,num_no))
                #Find the percent of input patterns exhibiting the current 
                #feature value
                percent = 100*num_instances/data_df.shape[0]
                display_string = '{}: {}/{} ({:0.2f}%). Targ. feat.:'.format(feature_val,num_instances,data_df.shape[0],percent)
                #Find the breakdown by target variable
                for target_feature_val in target_feature_set:
                    temp_num_instances = temp_df.loc[temp_df[target_feature] == target_feature_val].shape[0]
                    percent = 100*temp_num_instances/num_instances
                    display_string = '{} {}={}/{}({:0.2f}%)'.format(display_string,target_feature_val,temp_num_instances,num_instances,percent)
                f.write('  {}\n'.format(display_string))
                

"""
Some months have very few associated records (e.g., december only has 182 - 0.44%)
Create plot to investigate

"""
#adapted from: https://stackoverflow.com/questions/3418050/how-to-map-month-name-to-month-number-and-vice-versa
#Dictionary comprehension to create a translation dict from month/day abbrevations
#to counts
month_num_dict = {month.lower(): index for index, month in enumerate(calendar.month_abbr) if month}
day_num_dict = {month.lower(): index for index, month in enumerate(calendar.day_abbr) if month}

#Pull the per-month counts out of the dictionary
month_counts = counts_dict['month']
#Convert from list of tuples to dict
month_count_dict = {month_num_dict[tpl[0]]:(tpl[1],tpl[2],tpl[3]) for tpl in month_counts}

#For each month 1-12, put the counts in lists
month = range(1,13)
count = []
yes_count = []
no_count = []
for i in month:
    if i in month_count_dict.keys():
        count.append(month_count_dict[i][0])
        yes_count.append(month_count_dict[i][1])
        no_count.append(month_count_dict[i][2])
    else:
        count.append(0)
        yes_count.append(0)
        no_count.append(0)

fig,ax = plt.subplots(1,1,figsize=[20,15])     
ax.bar(month,yes_count,label = 'yes')   
ax.bar(month,no_count,bottom=yes_count,label = 'no')
ax.set_xlabel('Month')
ax.set_ylabel("num records")
ax.legend()
fig.savefig(os.path.join(plot_output_dir,'histogram_counts_by_month.png'),bbox_inches='tight')
plt.close(fig)



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

#Make a temporary dataframe containing only identifying records
temp = data_df[identifying_records].copy()
#make a column to hold the counts after aggregation
temp['counts'] = 1
#groupby on the identifying records
data_df_agg = temp.groupby(identifying_records,as_index=False).count()
print('\n{} unique combinations of identifying information'.format(data_df_agg.shape[0]))

"""
NOTE:
    only ~9k unique combinations means that a unique ID for each person in the 
    dataset cannot be generated
"""

#Find the minimum and maximum ages
min_age = data_df.age.min()
max_age = data_df.age.max()

#Define age categories 
group_by = 10
min_age_category = int(np.floor(min_age/group_by)*group_by)
max_age_category = int(np.ceil(max_age/group_by)*group_by)
#+0.1 to max_age_category because arange is on the range of [min,max)
breaks = np.arange(min_age_category,max_age_category+0.1,group_by).astype(int) 
#Build list of age break tuples
break_tpls = list(zip(breaks[:-1],breaks[1:]))

#add age-range variable
for start,end in break_tpls:
    #Find records within the current age range
    mask = (data_df.age >= start) & (data_df.age < end)
    #Add the age range feature to the dataframe
    #use end-1 to avoid overlap (ages are integers, so the largest age that is
    #less than end is end-1)
    data_df.loc[mask,'age_range'] = '{}-{}'.format(start,end-1)

#Update the identifying records list to use age-range instead of raw age
identifying_records = [
    'age_range',
    'job',
    'education',
    'default',
    'housing',
    'loan']

#Generate a list of tuples containing only identifying information
#The following will perform the same action as list(zip()) without hard-coding
#the data columns, but is *significantly* slower
# identifying_tpls = [tuple(data_df.loc[ind,identifying_records]) for ind in data_df.index]
identifying_tpls = list(zip(data_df.age_range,data_df.job,data_df.education,data_df.default,data_df.housing,data_df.loan))
#Unique identifying tuples; personas
identifying_tpl_set = set(identifying_tpls)

#Init an empty dataframe
summary_df = pd.DataFrame()

tpl = identifying_tpls[0]
"""
Aggregate dataset to see what personas are in the dataset
"""
#Note: It may be possible to do this using pandas groupby (perhaps with lambdas 
#to calculate the "no" and "yes" columns?) but it's not immediately obvious to me
print('\n')
for ctr,tpl in enumerate(identifying_tpl_set):
    #Track progress
    print('\rProcessing persona: {}/{}'.format(str(ctr+1).zfill(4),len(identifying_tpl_set)),end='',flush=True)
    #Boolean mask of all matches to the current persona
    mask = np.all(data_df[identifying_records] == tpl,axis=1)
    #Count the breakdown of outcome feature states
    outcome_counts = []
    for target_feature_val in target_feature_set:
        outcome_counts.append(sum(data_df.loc[mask,target_feature] == target_feature_val))
    #Find the number of matches to the current persona
    count = sum(mask)
    
    #Add the data to the summary dataframe
    summary_df.loc[ctr,'counts'] = count
    summary_df.loc[ctr,identifying_records] = tpl
    summary_df.loc[ctr,target_feature_set] = outcome_counts
    
#Calculate the fraction of positive responses for each unique persona
summary_df['frac_y'] = summary_df.yes/summary_df.counts
#Sort high counts to the top of the dataframe
summary_df.sort_values('counts',ascending=False,inplace=True)
#Export to csv
summary_df.to_csv(os.path.join(data_output_dir,'aggregate_summary_data.csv'))
        
"""
Questions/Notes:
    1. Reasons for the uneven distribution of contact-by-month?
    2. If previous contact was a success (e.g., poutcome='success') why was there another contact?
    3. If groupby age,job,education,default,housing,&loan, there are only ~9k unique records
    4. No missing/nan values in the dataset (all missing are explicitly indicated as such)
    
"""