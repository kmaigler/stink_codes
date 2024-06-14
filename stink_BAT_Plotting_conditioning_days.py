"""
Created on Wed Feb 06 10:37:47 2024

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

#Import statistics packages
import pingouin as pg
from pingouin import pairwise_ttests
import statsmodels.api as sm
from scipy import stats

#Import plotting utilities
import seaborn as sns
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D
# =============================================================================
#this dictiontionary allows the graphs to look at paired vs. unpaired odor instead
#of odor names as the paired odor changes identity between animals
#
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


#remove no lick trials, drop zero lick trials
dropna_df = df
dropna_df['LICKS'] = dropna_df['LICKS'].replace({0:np.nan})
dropna_df = dropna_df.dropna()

dropna_df = dropna_df.loc[:,~dropna_df.columns.duplicated()].copy() #remove duplicated columns

#remove trials with licks less than 2
#drop2_df = df.drop(df[df.LICKS < 2].index)

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # ##PLOTTING TOGETHER
# =============================================================================
# =============================================================================

#===========================================================================
#conditioning days paired v unpaired lineplot with averaging over trials NOT animal
# =============================================================================
#dictionary of conditioning days - only one solution is presented every other day
day_dict = {'Con1': 1,
               'Con2': 1, 
               'Con3': 2,
               'Con4': 2,
               'Con5': 3,
               'Con6': 3
               }

# Function to check if the ID is in the paired dictionary
def apply_con_day(row):
    con_day = row['Notes']
    if con_day in day_dict:
        return day_dict[con_day]

# Apply the function to create a new column 'Conditioning_day'
dropna_df['Conditioning_day'] = dropna_df.apply(apply_con_day, axis=1)

#generate a dataframe for each condition
E_df =  dropna_df.loc[(dropna_df['Condition']=='E')]
E_df = E_df.groupby(['Conditioning_day','SOLUTION','Trial_num'])['LICKS'].mean().reset_index()
Ec_df =  dropna_df.loc[(dropna_df['Condition']=='Ec')]
Ec_df = Ec_df.groupby(['Conditioning_day', 'SOLUTION', 'Trial_num'])['LICKS'].mean().reset_index()
U_df =  dropna_df.loc[(dropna_df['Condition']=='U')]
U_df = U_df.groupby(['Conditioning_day', 'SOLUTION', 'Trial_num'])['LICKS'].mean().reset_index()
Uc_df =  dropna_df.loc[(dropna_df['Condition']=='Uc')]
Uc_df = Uc_df.groupby(['Conditioning_day', 'SOLUTION', 'Trial_num'])['LICKS'].mean().reset_index()

#plot over conditioning day by ocnditoin
fig, ax = plt.subplots(1,4, sharey = True, figsize = (8,4), dpi=500)
plt.suptitle('Licks to odor paired with sucrose or unpaired odor over conditioning days')
ax[0] = sns.lineplot(ax = ax[0], x='Conditioning_day',
                  y='LICKS',hue='SOLUTION',data=E_df,palette = ['#094907','#7BE278'], ci=68)
ax[1] = sns.lineplot(ax = ax[1], x='Conditioning_day', legend = '',
                  y='LICKS',hue='SOLUTION',data=Ec_df,palette = ['#0D275C','#7E9EE0'], ci=68)
ax[2] = sns.lineplot(ax = ax[2], x='Conditioning_day',  legend = '',
                  y='LICKS',hue='SOLUTION',data=U_df,palette = ['#094907','#7BE278'], ci=68)
ax[3] = sns.lineplot(ax = ax[3], x='Conditioning_day',
                  y='LICKS',hue='SOLUTION',data=Uc_df,palette = ['#0D275C', '#7E9EE0'], ci=68)
fig.supxlabel('Conditioning Day (collapsed)', fontsize = 10)
ax[0].set(ylabel = 'Licks')
ax[1].set(xlabel = '')
ax[2].set(xlabel = '')
ax[3].set(xlabel = '')
ax[0].set(xlabel = '')
ax[0].set(title = 'Exposed, GCx')
ax[1].set(title = 'Exposed, no laser')
ax[2].set(title = 'Unexposed, GCx')
ax[3].set(title = 'Unexposed, no laser')
ax[0].set(xlabel = '')
plt.tight_layout()
#===========================================================================
#conditioning days paired v unpaired barplot with averaging over trials NOT animal
# just with all ocnditioning days together
# =============================================================================
#remake the dataframe by condition
E_df =  dropna_df.loc[(dropna_df['Condition']=='E')]
Ec_df =  dropna_df.loc[(dropna_df['Condition']=='Ec')]
U_df =  dropna_df.loc[(dropna_df['Condition']=='U')]
Uc_df =  dropna_df.loc[(dropna_df['Condition']=='Uc')]
#generate mean licks by solutiona dn trial num for each conditoin
E_lickmean = E_df.groupby(['SOLUTION','Trial_num'])['LICKS'].mean().reset_index()
Ec_lickmean = Ec_df.groupby(['SOLUTION', 'Trial_num'])['LICKS'].mean().reset_index()
U_lickmean = U_df.groupby(['SOLUTION', 'Trial_num'])['LICKS'].mean().reset_index()
Uc_lickmean = Uc_df.groupby(['SOLUTION', 'Trial_num'])['LICKS'].mean().reset_index()

#plot over conditioning day by ocnditoin
fig, ax = plt.subplots(1,4, sharey = True, figsize = (8,4), dpi=500)
plt.suptitle('Mean licks to odor paired with sucrose or unpaired odor during conditioning days')
ax[0] = sns.barplot(ax = ax[0], x='SOLUTION',
                  y='LICKS', data=E_lickmean,palette = ['#094907','#7BE278'], ci=68)
ax[1] = sns.barplot(ax = ax[1], x='SOLUTION', 
                  y='LICKS',data=Ec_lickmean,palette = ['#0D275C','#7E9EE0'], ci=68)
ax[2] = sns.barplot(ax = ax[2], x='SOLUTION', 
                  y='LICKS', data=U_lickmean,palette = ['#094907','#7BE278'], ci=68)
ax[3] = sns.barplot(ax = ax[3],x='SOLUTION',
                  y='LICKS',data=Uc_lickmean,palette = ['#0D275C', '#7E9EE0'], ci=68)
ax[0].set(ylabel = 'Licks')
ax[1].set(ylabel = '')
ax[2].set(ylabel = '')
ax[3].set(ylabel = '')
ax[0].set(title = 'Exposed, GCx')
ax[1].set(title = 'Exposed, no laser')
ax[2].set(title = 'Unexposed, GCx')
ax[3].set(title = 'Unexposed, no laser')
plt.tight_layout()

#===========================================================================
#conditioning days barplot with conditioning days collapsed
# =============================================================================
#generate a dataframe for each condition
licksum_df = dropna_df.groupby(['Animal','Condition','SOLUTION'])['LICKS'].sum().reset_index()
#plot catplot
fig, ax = plt.subplots(figsize = (5.5,7), dpi=600)
sns.catplot(ax = ax, kind = 'bar', x='SOLUTION', y='LICKS',col='Condition',
                 data=licksum_df,palette = ['#0D275C','#7E9EE0'], #blues
                 ci=68, aspect = 0.5)

#plot barplot w condition on x axis
fig, ax = plt.subplots(figsize = (5.5,7), dpi=600)
ax = sns.barplot(ax = ax, x='Condition', y='LICKS',hue='SOLUTION',
                 data=licksum_df,palette = ['#0D275C','#7E9EE0'], #blues
                 ci=68)
ax.set(ylabel = 'Licks Sum')
ax.set(title = 'Total Licks to Odor Paired with Sucrose vs. Unpaired Odor \n on Conditioning Days')
ax.set_xticklabels(['Exposed, GCx', 'Unexposed, GCx','Exposed, no laser',
                    'Unexposed, no laser'],fontsize = 7.5)