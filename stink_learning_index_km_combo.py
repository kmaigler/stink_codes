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
df = pd.read_pickle('20_04_2024_grouped_dframe.df')
#df = pd.read_pickle('17_06_2024_grouped_dframe.df')
#Capitalize labels
df['Notes'] = df['Notes'].str.capitalize()


# =============================================================================
# #add which odor/SOLUTION was paired for each animal in a new column and drop na
# =============================================================================
#dictionary of what the paired odor is for each animal
paired_dict = {' TG37': 'carvone', 
               ' TG38': 'carvone', 
               ' TG31': 'carvone', 
               ' TG32': 'carvone',
               ' TG34': 'carvone',
               ' TG35': 'carvone', 
               ' TG30': 'carvone',
               ' TG33': 'carvone',
               ' TG29': 'carvone',
               ' TG36': 'carvone'
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
    post_df = dataframe.loc[dataframe["Notes"].isin(["Post"])]
    pre_df = dataframe.loc[dataframe["Notes"].isin(["Pre"])]
    pre_paired_lickssum = pre_df[pre_df['Paired'] == 'paired']['LICKS'].sum()
    post_paired_lickssum= post_df[post_df['Paired'] == 'paired']['LICKS'].sum()

    pre_unpaired_lickssum = pre_df[pre_df['Paired'] == 'unpaired']['LICKS'].sum()
    post_unpaired_lickssum = post_df[post_df['Paired'] == 'unpaired']['LICKS'].sum()

    ratio = (post_paired_lickssum*pre_unpaired_lickssum) / (post_unpaired_lickssum*pre_paired_lickssum)
    normalized_ratio = np.tanh(ratio)
    return pd.Series({'Condition': condition, 'Paired_LICKS': post_paired_lickssum, 
                      'Unpaired_LICKS': post_unpaired_lickssum, 'Ratio': ratio, 
                      'Normalized_ratio': normalized_ratio})

post_df = tg36.loc[tg36["Notes"].isin(['Post'])]
post_paired_lickssum= post_df[post_df['Paired'] == 'paired']['LICKS'].sum()
pre_df = tg36.loc[tg36["Notes"].isin(['Pre'])]
pre_paired_lickssum= pre_df[pre_df['Paired'] == 'paired']['LICKS'].sum()

post_up_lickssum= post_df[post_df['Paired'] == 'unpaired']['LICKS'].sum()
pre_up_lickssum= pre_df[pre_df['Paired'] == 'unpaired']['LICKS'].sum()

(post_paired_lickssum*pre_up_lickssum) / (post_up_lickssum*pre_paired_lickssum)
test2 = post_paired_lickssum.groupby(['Animal'])['LICKS'].sum()
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

df.to_pickle('%s_ratio_combo_dataframe'%(date.today().strftime("%d_%m_%Y")))
ratio_df.to_pickle('%s_ratio_combo_dataframe'%(date.today().strftime("%d_%m_%Y"))) 

#ratio_df = pd.read_pickle('18_06_2024_ratio_combo_dataframe')

# Plotting the results
fig, ax = plt.subplots(figsize=(4, 7), dpi = 600)
groups = ['Pre-Exposed', 'Non-exposed']
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
ax.set_ylim(0, 1.1)  # Set y-axis from 0 to 1.0
ax.set_title('Combined Group')

# Adjust column width
ax.set_xlim(-0.5, 1.5)  # Adjust these values to fine-tune the margins
plt.xticks(rotation=0)
plt.show()





combo_e = ratio_df.loc[ratio_df.Condition=='E']['Normalized_ratio']
combo_u = ratio_df.loc[ratio_df.Condition=='U']['Normalized_ratio']
#run ttest
t, p = stats.ttest_ind(combo_e, combo_u)
# t
# Out[222]: 1.4346704871510243

# p
# Out[223]: 0.18929428359348122
