#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:48:38 2024

@author: kmaigler
"""
# =============================================================================
# Import stuff
# =============================================================================
# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import easygui
import pandas as pd

#Import plotting utilities
import seaborn as sns
import matplotlib.pyplot as plt
# =============================================================================

# =============================================================================
# #Read in file
# =============================================================================
#Get name of directory where the data files and pickle file sits, and change to that directory for processing
dir_name = easygui.diropenbox()
os.chdir(dir_name)

#Look for the ms8 files in the directory
file_list = os.listdir('./')
file_names = []
for files in file_list:
    if files[-2:] == 'df':
        file_names.append(files)

df_files = easygui.multchoicebox(
        msg = 'Which file do you want to work from?',
        choices = ([x for x in file_names]))

#Read in dataframe from CSV
df = pd.read_pickle(df_files[0])

#Capitalize labels
df['Notes'] = df['Notes'].str.capitalize()


# =============================================================================
# #add which odor/SOLUTION was paired for each animal in a new column and drop na
# =============================================================================
#dictionary of what the paired odor is for each animal
paired_dict = {' TG53': 'ch',
               ' TG49': 'cis3hex', 
               ' TG51': 'eb', 
               ' TG54': 'eb',
               ' TG47': 'carvone', 
               ' TG48': 'carvone', 
               ' TG52': 'ch',
               ' TG50': 'cis3hex'
           } 
# Function to check if the ID is in the paired dictionary
def check_paired(row):
    a = row['Animal']
    solution = row['SOLUTION']
    waterlist = np.array(['water', 'w'])
    if a in paired_dict and solution in paired_dict[a]:
        return 'paired'
    elif solution in waterlist:
        return 'water'
    else:
        return 'unpaired'

# Apply the function to create a new column 'Status'
df['Paired'] = df.apply(check_paired, axis=1)

#remove trials with licks less than 2
drop2_df = df.loc[(df['LICKS'] > 2)].reset_index()

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # ##Learning index - postpaired*preunpaired/postunpaired*prepaired 
# =============================================================================
# =============================================================================
# =============================================================================
# Function to calculate the learning index for each group
def calculate_ratio(dataframe):
    condition = dataframe['Condition'].iloc[0]
    post_df = dataframe.loc[dataframe["Notes"].isin(["Test1", 'Test2'])]
    test1_df = dataframe.loc[dataframe["Notes"].isin(["Test1"])]
    test2_df = dataframe.loc[dataframe["Notes"].isin(['Test2'])]
    pre_df = dataframe.loc[dataframe["Notes"].isin(["Pretest"])]
    pre_paired_lickssum = pre_df[pre_df['Paired'] == 'paired']['LICKS'].sum()
    post_paired_lickssum= post_df[post_df['Paired'] == 'paired']['LICKS'].sum()
    test1_paired_lickssum= test1_df[test1_df['Paired'] == 'paired']['LICKS'].sum()
    test2_paired_lickssum= test2_df[test2_df['Paired'] == 'paired']['LICKS'].sum()
    pre_unpaired_lickssum = pre_df[pre_df['Paired'] == 'unpaired']['LICKS'].sum()
    post_unpaired_lickssum = post_df[post_df['Paired'] == 'unpaired']['LICKS'].sum()
    test1_unpaired_lickssum= test1_df[test1_df['Paired'] == 'unpaired']['LICKS'].sum()
    test2_unpaired_lickssum= test2_df[test2_df['Paired'] == 'unpaired']['LICKS'].sum()
    ratio = (post_paired_lickssum*pre_unpaired_lickssum) / (post_unpaired_lickssum*pre_paired_lickssum)
    normalized_ratio = np.tanh(ratio)
    ratio_test1 = (test1_paired_lickssum*pre_unpaired_lickssum) / (test1_unpaired_lickssum*pre_paired_lickssum)
    ratio_test2 = (test2_paired_lickssum*pre_unpaired_lickssum) / (test2_unpaired_lickssum*pre_paired_lickssum)
    norm_test1_ratio = np.tanh(ratio_test1)
    norm_test2_ratio = np.tanh(ratio_test2)
    return pd.Series({'Condition': condition, 'Paired_LICKS': post_paired_lickssum, 
                      'Unpaired_LICKS': post_unpaired_lickssum, 'Ratio': ratio, 
                      'Normalized_ratio': normalized_ratio, 'Test1': ratio_test1, 
                      'Normalized_test1': norm_test1_ratio, 'Test2': ratio_test2, 
                      'Normalized_test2': norm_test2_ratio})


# #ISAACs learning index
# drop3_df = df.drop(df[df.LICKS <= 3].index)
# pre_df = drop3_df.loc[drop3_df["Notes"].isin(["Pretest"])]

# post_df = drop3_df.loc[drop3_df["Notes"].isin(["Test1"])]

# #ISAACs learning index
# def calculate_metric(pre_data, post_data):
#     pre_paired = pre_data[pre_data['Paired'] == 'paired']['LICKS']
#     pre_unpaired = pre_data[pre_data['Paired'] == 'unpaired']['LICKS']
#     post_paired = post_data[post_data['Paired'] == 'paired']['LICKS']
#     post_unpaired = post_data[post_data['Paired'] == 'unpaired']['LICKS']
#     if not pre_paired.empty and not pre_unpaired.empty and not post_paired.empty and not post_unpaired.empty:
#         numerator = post_paired.sum() * pre_unpaired.sum()
#         denominator = pre_paired.sum() * post_unpaired.sum()
#         if denominator == 0:  # Prevent division by zero
#             return None
#         return numerator / denominator
#     return None

# igratio_df = pd.DataFrame()
# for animal in pre_df['Animal'].unique():
#     a_pre = pre_df.loc[pre_df["Animal"]== animal]
#     a_post = post_df.loc[post_df["Animal"]== animal]
#     ratio1 = calculate_metric(a_pre, a_post)
#     a_df = pd.DataFrame({'Animal': [animal], 'Ratio': [ratio1]})
#     igratio_df = pd.concat([igratio_df, a_df], ignore_index=True)

    
    
# =============================================================================
#preference score by group (enriched v unenriched)
# =============================================================================
# Group by Animal apply the ratio calculation
ratio_df = drop2_df.groupby(['Animal']).apply(calculate_ratio).reset_index()
mean_ratio = ratio_df.groupby(['Condition'])['Normalized_ratio'].mean().reset_index()

df.to_pickle('%s_ratio_retro_dataframe'%(date.today().strftime("%d_%m_%Y")))
ratio_df.to_pickle('%s_ratio_retro_dataframe'%(date.today().strftime("%d_%m_%Y"))) 

#ratio_df = pd.read_pickle('18_06_2024_ratio_retro_dataframe')

# Plotting the results
fig, ax = plt.subplots(figsize=(4, 7), dpi = 600)
groups = ['Retro Exposed', 'Retro Unexposed']
ax = sns.scatterplot(x='Condition',y='Normalized_ratio',
                     data=mean_ratio, marker = "_", s=1000,color = 'orange', #label = 'Mean'
                     )
ax = sns.scatterplot(x='Condition', 
            y='Normalized_ratio',data=ratio_df, color = 'blue')

#ax.axhline(1, color='gray', linestyle='dotted')
#ax.set_xticks([1,2])
ax.set_xticklabels(groups)
ax.set_ylabel('Learning Index')
ax.set_xlabel('')
ax.set_ylim(0, 1.1)  # Set y-axis from 0 to 1.1
ax.set_title('Retro Group')

# Adjust column width
ax.set_xlim(-0.5, 1.5)  # Adjust these values to fine-tune the margins
plt.xticks(rotation=0)
plt.show()


#run ttest
t, p = stats.ttest_ind(retro_e, retro_u)

retro_e = ratio_df.loc[ratio_df.Condition=='E']['Normalized_ratio']
retro_u = ratio_df.loc[ratio_df.Condition=='U']['Normalized_ratio']


# p
# Out[114]: 0.1955791881497692
