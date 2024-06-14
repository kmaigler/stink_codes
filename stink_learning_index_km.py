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
#df = pd.read_pickle('05_08_2022_grouped_dframe.df')
#Capitalize labels
df['Notes'] = df['Notes'].str.capitalize()


# =============================================================================
# #add which odor/SOLUTION was paired for each animal in a new column and drop na
# =============================================================================
#dictionary of what the paired odor is for each animal
paired_dict = {' GW05': 'carvone',
               ' GW06': 'carvone', 
               ' SG05': 'cis3hex',
               ' SG06': ['P-EB', 'P'],
               ' SG07': 'cis3hex',
               ' SG09': 'carvone'
               }
# Function to check if the ID is in the paired dictionary
def check_paired(row):
    a = row['Animal']
    solution = row['SOLUTION']
    if a in paired_dict and solution in paired_dict[a]:
        return 'paired'
    elif solution == 'water':
        return 'water'
    else:
        return 'unpaired'

# Apply the function to create a new column 'Status'
df['Paired'] = df.apply(check_paired, axis=1)

#remove trials with licks less than 2
drop2_df = df.drop(df[df.LICKS < 2].index)

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
# =============================================================================
#preference score by group (enriched v unenriched)
# =============================================================================
# Group by Animal apply the ratio calculation
ratio_df = drop2_df.groupby(['Animal']).apply(calculate_ratio).reset_index()
mean_ratio = ratio_df.groupby(['Condition'])['Normalized_ratio'].mean().reset_index()
# Create a scatterplot
fig= plt.figure(figsize = (5.5,7), dpi=600)
ax = sns.scatterplot(x='Condition',y='Normalized_ratio',
                     data=mean_ratio, marker = "_", s=1000,color = 'y', ci=68, label = 'Mean')
ax = sns.scatterplot(x='Condition', 
            y='Normalized_test1',data=ratio_df, color = 'b', ci=68, label = 'Test 1')
ax = sns.scatterplot(x='Condition',
            y='Normalized_test2',data=ratio_df, color = 'lightblue', ci=68, label = 'Test 2')
# Add the horizontal line at y=1
ax.plot([-0.5,3.5],[0,0], 'k--', linewidth=1)
ax.set(ylabel='Learning Index')
ax.set(xlabel='')
#ax.set_xticks(range(len(ratio_df['Condition'].unique())))
#ax.set_xticklabels(['Exposed, GCx', 'Unexposed, GCx', 'Exp, no laser', 'Unexp, no laser'],fontsize = 9)
plt.suptitle("Learning Index by Group")
plt.tight_layout()