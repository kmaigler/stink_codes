#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:42:05 2024

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
from scipy import stats

#Import plotting utilities
import seaborn as sns
import matplotlib.pyplot as plt

#Get name of directory where the data files and pickle file sits, and change to that directory for processing
dir_name = easygui.diropenbox()
os.chdir(dir_name)

#Look for the ms8 files in the directory
file_list = os.listdir('./')
file_names = []
for files in file_list:
    if files[-3:] == 'csv':
        file_names.append(files)

csv_files = easygui.multchoicebox(
        msg = 'Which file do you want to work from?',
        choices = ([x for x in file_names]))

#Read in csv
virus_df = pd.read_csv(csv_files[0])
avg_df = virus_df.groupby(['Animal','Region','Condition'])['Mean'].mean().reset_index()
# Pivot the dataframe to have 1 and 2 as columns for each animal and condition
pivot_df = avg_df.pivot_table(index=['Animal', 'Condition'], 
                                     columns='Region', values='Mean').reset_index()
# Calculate the difference
pivot_df['Normalized_virus'] = pivot_df[1]/pivot_df[2]

#get the learning index from stink_learning_index_km.py ratio_df
# Merge pivot_df with ratio_df based on 'Animal'
merged_df = pd.merge(pivot_df, ratio_df[['Animal', 'Normalized_ratio']], on='Animal', how='left')
# Update the 'Normalized' column in pivot_df with the 'Normalized_ratio' from ratio_df
pivot_df['Learning_index'] = ratio_df['Normalized_ratio']

#separate control/no laser animals
control_df = pivot_df.loc[pivot_df["Condition"].isin(['Ec', 'Uc'])].reset_index()
laser_df = pivot_df.loc[pivot_df["Condition"].isin(['E', 'U'])].reset_index()

#plot scatter w regression
fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(5,4), dpi=600)
sns.regplot(x='Learning_index', y='Normalized_virus', data=laser_df, ax=ax[0], 
            ci=68, scatter_kws={'s': 50, 'color': 'green'}, line_kws={'color': 'green'})
ax[0].set(ylabel='Normalized Virus Expression', xlabel='Learning Index', title='Laser on')
# Plot scatter with regression for control_df
sns.regplot(x='Learning_index', y='Normalized_virus', data=control_df, ax=ax[1],
            ci=68, scatter_kws={'s': 50, 'color': 'grey'}, line_kws={'color': 'grey'})
ax[1].set(ylabel='', xlabel='Learning Index', title='No laser control')
# Set the main title
fig.suptitle('Viral Expression Relationship to Learning Index')
# Adjust layout for better fit
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
