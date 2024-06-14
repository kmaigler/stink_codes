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


#remove no lick trials, drop zero lick trials
dropna_df = df
dropna_df['LICKS'] = dropna_df['LICKS'].replace({0:np.nan})
dropna_df = dropna_df.dropna()

dropna_df = dropna_df.loc[:,~dropna_df.columns.duplicated()].copy() #remove duplicated columns

#remove trials with licks less than 2
drop2_df = df.drop(df[df.LICKS < 2].index)

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # ##PLOTTING TOGETHER
# =============================================================================
# =============================================================================
# =============================================================================
#for plotting hab days
# =============================================================================
#TODO LOAD DF
df.groupby(['Animal','Notes'])['LICKS'].mean()
sns.barplot(x='Animal',y='LICKS',hue='Notes',data=df,palette = 'Accent', ci=68)

hab45_df= df.loc[~(df['Notes']=='Hab3')]

hab45_df.groupby(['Animal','Notes'])['LICKS'].mean()
sns.barplot(x='Animal',y='LICKS',hue='Notes',data=hab45_df,palette = 'Accent', ci=68)
plt.suptitle("licks to water on habituation days 4 and 5")    

hab45_df.groupby(['Animal','Condition', 'SOLUTION'])['LICKS'].mean()
sns.barplot(x='SOLUTION',y='LICKS',hue='Condition',data=hab45_df,palette = 'Set1', ci=68)
plt.suptitle("Difference in mean licks in Enriched (E) vs. Unenriched (U) group \non habitation days")

# =============================================================================
#conditioning days
# =============================================================================
#TODO LOAD DF
dropna_df.groupby(['Condition','SOLUTION'])['LICKS'].mean()
sns.barplot(x='SOLUTION',y='LICKS',hue='Condition',data=dropna_df,palette = 'Set2', ci=68)
plt.suptitle("Difference in mean licks in Enriched (E) vs. Unenriched (U) group \nduring conditioning (GCx)")

#plot licks for each conditioning day without regard to unrichment group
dropna_df.groupby(['Notes','SOLUTION'], sort=True)['LICKS'].mean().reset_index()
sns.barplot(x='Notes',y='LICKS',hue='SOLUTION',data=df,palette = 'Set1', 
            order=['Con1', 'Con2', 'Con3', 'Con4', 'Con5', 'Con6'], ci=68)

max_value = df.LICKS.max() +2
for animal in enumerate(df.Animal.unique()):
    a_df = df[df.Animal == animal]
    sns.barplot(x='SOLUTION', y='LICKS', data=a_df)
    plt.title(animal)
    plt.ylim(0, max_value)
plt.suptitle("Licks to each solution during conditioning")    
plt.show()
    

#plot over time




# =============================================================================
#Pretest vs. test 1
# =============================================================================
#TODO LOAD DF

test2_df = dropna_df.loc[dropna_df["Notes"].isin(["Test2"])]
test1_df = dropna_df.loc[dropna_df["Notes"].isin(["Test1"])]
pretest_df = dropna_df.loc[dropna_df["Notes"].isin(["Pretest"])]
tests_df = dropna_df.loc[dropna_df["Notes"].isin(["Test1", "Test2"])]

#plot pretest and test1 for each animal
max_value = dropna_df.LICKS.max() + 2
animals = dropna_df.Animal.unique()
num_animals = len(animals)
#plot pretest
fig, axes = plt.subplots(nrows=num_animals, ncols=1, figsize=(2, 2.5 * num_animals), sharex=True)
for i, animal in enumerate(animals):
    a_df = pretest_df[pretest_df.Animal == animal]
    sns.barplot(x='Paired', y='LICKS', data=a_df, ax=axes[i])
    axes[i].set_title(animal)
    axes[i].set_ylim(0, max_value)
fig.suptitle("Pretest licks by animal", fontsize=12)
plt.show()

#plot test1
fig, axes = plt.subplots(nrows=num_animals, ncols=1, figsize=(2, 2.5 * num_animals), sharex=True)
for i, animal in enumerate(animals):
    a_df = test1_df[test1_df.Animal == animal]
    sns.barplot(x='Paired', y='LICKS', data=a_df, ax=axes[i])
    axes[i].set_title(animal)
    axes[i].set_ylim(0, max_value)
fig.suptitle("Test 1 licks by animal", fontsize=12)
plt.show()



#dictionary for legend by condition
mylabels_dict = {'U':'No Enrichment', 
                 'E':'Enriched', 
                 'Ec': 'Encriched without laser', 
                 'Uc': 'No Enrichment, no laser'
    }


fig, axes = plt.subplots(1, 2, sharey = True, figsize = (10,5), dpi=500)
fig.suptitle('Conditioning Aquisition \n(PreTest vs. Test1)')


sns.barplot(ax = axes[0], x='Paired',y='LICKS',hue='Condition', order = ['water', 'paired', 'unpaired'], 
            data=pretest_df,ci=68)
handles, labels = axes[0].get_legend_handles_labels()
new_labels = [mylabels_dict[label] for label in labels]
axes[0].legend(handles=handles, labels=new_labels, fontsize =7, loc = 2)
axes[0].set_title('PreTest')
axes[0].set(xlabel = 'Solution')
axes[0].set(ylabel = 'Mean Licks')

#test1_df.groupby(['Animal','Paired'])['LICKS'].mean()
sns.barplot(ax = axes[1], x='Paired',y='LICKS',hue='Condition', order = ['water', 'paired', 'unpaired'],
            data=test1_df,ci=68)
handles, labels = axes[1].get_legend_handles_labels()
new_labels = [mylabels_dict[label] for label in labels]
axes[1].legend(handles=handles, labels=new_labels, fontsize =7, loc = 2)
axes[1].set_title('Test 1')
axes[1].set(xlabel = 'Solution')
axes[1].set(ylabel = '')

# =============================================================================
#Pretest vs. test 1 vs. test 2
# =============================================================================
fig, axes = plt.subplots(1, 3, sharey = True, figsize = (12,5), dpi=500)
fig.suptitle('Conditioning Aquisition \n(PreTest vs. Test1 vs. Test2)')

sns.barplot(ax = axes[0], x='Paired',y='LICKS',hue='Condition', order = ['water', 'paired', 'unpaired'], 
            data=pretest_df,ci=68)
handles, labels = axes[0].get_legend_handles_labels()
new_labels = [mylabels_dict[label] for label in labels]
axes[0].legend(handles=handles, labels=new_labels, fontsize =7, loc = 2)
axes[0].set_title('PreTest')
axes[0].set(xlabel = 'Solution')
axes[0].set(ylabel = 'Mean Licks')

sns.barplot(ax = axes[1], x='Paired',y='LICKS',hue='Condition', order = ['water', 'paired', 'unpaired'],
            data=test1_df,ci=68)
handles, labels = axes[1].get_legend_handles_labels()
new_labels = [mylabels_dict[label] for label in labels]
axes[1].legend(handles=handles, labels=new_labels, fontsize =7, loc = 2)
axes[1].set_title('Test 1')
axes[1].set(xlabel = 'Solution')
axes[1].set(ylabel = '')

sns.barplot(ax = axes[2], x='Paired',y='LICKS',hue='Condition', order = ['water', 'paired', 'unpaired'],
            data=test2_df,ci=68)
handles, labels = axes[2].get_legend_handles_labels()
new_labels = [mylabels_dict[label] for label in labels]
axes[2].legend(handles=handles, labels=new_labels, fontsize =7, loc = 2)
axes[2].set_title('Test 2')
axes[2].set(xlabel = 'Solution')
axes[2].set(ylabel = '')
# =============================================================================
#Pretest vs. test days combined
# =============================================================================
fig, axes = plt.subplots(1, 2, sharey = True, figsize = (10,5), dpi=500)
fig.suptitle('Conditioning Aquisition \n(PreTest vs. Test)')
sns.barplot(ax = axes[0], x='Paired',y='LICKS',hue='Condition', order = ['water', 'paired', 'unpaired'], 
            data=pretest_df,ci=68)
handles, labels = axes[0].get_legend_handles_labels()
new_labels = [mylabels_dict[label] for label in labels]
axes[0].legend(handles=handles, labels=new_labels, fontsize =7, loc = 2)
axes[0].set_title('PreTest')
axes[0].set(xlabel = 'Solution')
axes[0].set(ylabel = 'Mean Licks')
sns.barplot(ax = axes[1], x='Paired',y='LICKS',hue='Condition', order = ['water', 'paired', 'unpaired'],
            data=tests_df,ci=68)
handles, labels = axes[1].get_legend_handles_labels()
new_labels = [mylabels_dict[label] for label in labels]
axes[1].legend(handles=handles, labels=new_labels, fontsize =7, loc = 2)
axes[1].set(xlabel = 'Solution')
axes[1].set(ylabel = '')
axes[1].set_title('Tests 1 and 2')

# =============================================================================
#Pretest vs. test days combined plotting groups together and solution by hue
# =============================================================================
fig, axes = plt.subplots(1, 2, sharey = True, figsize = (10,5), dpi=500)
fig.suptitle('Conditioning Aquisition \n(PreTest vs. Test)')
sns.barplot(ax = axes[0], x='Condition',y='LICKS',hue='Paired', order = ['Ec', 'E', 'U', 'Uc'], 
            data=pretest_df,ci=68)
# handles, labels = axes[0].get_legend_handles_labels()
# new_labels = [mylabels_dict[label] for label in labels]
# axes[0].legend(handles=handles, labels=new_labels, fontsize =7, loc = 2)
axes[0].set_title('PreTest')
axes[0].set(xlabel = 'Condition')
axes[0].set(ylabel = 'Mean Licks')
sns.barplot(ax = axes[1], x='Condition',y='LICKS',hue='Paired', order = ['Ec', 'E', 'U', 'Uc'],
            data=tests_df,ci=68)
# handles, labels = axes[1].get_legend_handles_labels()
# new_labels = [mylabels_dict[label] for label in labels]
# axes[1].legend(handles=handles, labels=new_labels, fontsize =7, loc = 2)
axes[1].set(xlabel = 'Condition')
axes[1].set(ylabel = '')
axes[1].set_title('Tests 1 and 2')



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


# Create a catplot
fig, axes = plt.subplots(1, 2, figsize = (10,5), dpi=500)
sns.scatterplot(ax = axes[0], x='Condition', y='Ratio', data = ratio_df,
            palette = 'Blues', #laser groups are green, control groups grey
            s=30,legend=False) 
# Add the horizontal line at y=1
# axes[0].plot([-0.5, len(enrich_resultdf['Notes'].unique()) - 0.5], [1, 1], 'k--', linewidth=2)
# axes[0].set_title('Unenriched Groups')
# axes[0].set_xlabel=('Test Day')
# axes[0].set_ylabel=('Paired Licks/Unpaired Licks')
#axes[0].legend(labels=['No laser', 'Laser'])
sns.scatterplot(ax = axes[1], x='Condition', y='Normalized_ratio', 
            data=ratio_df,
            palette = ['#808080', '#006400'], #laser groups are green, control groups grey
            markers=True, linewidth=3) 
# Add the horizontal line at y=1
axes[1].plot([-0.5, len(enrich_resultdf['Notes'].unique()) - 0.5], [1, 1], 'k--', linewidth=2)
axes[1].set_title('Enriched Groups')
axes[1].set_xlabel=('Test Day')
axes[1].legend(labels=['No laser', 'Laser'])
plt.suptitle('Preference Score for Paired Odor Before and After Conditioning', fontsize=16)
plt.show()

# =============================================================================
#Test days preference score 
# =============================================================================

# Group by Animal and Note and condition and apply the ratio calculation
result_df = tests_df.groupby(['Animal', 'Notes', 'Condition']).apply(calculate_ratio).reset_index()

# =============================================================================
#Plot ratio of paired/unpaired licks
# =============================================================================
fig, axes = plt.subplots(1, 1, figsize = (4,3), dpi=500)
fig.suptitle('Preference Score for Paired odor in Test Days')
sns.scatterplot(ax = axes, x='Condition',y='Ratio',hue='Notes', 
            data=result_df,ci=68)
axes.plot([-0.2,0,1,2, 2.2], [1,1,1,1,1], color ='k', linestyle = '--', linewidth=1.5)
axes.set(xlabel = 'Condition')
axes.set(ylabel = 'Paired Licks/Unpaired Licks')


# Create a catplot
plt.figure(figsize=(5,4), dpi=500)
g = sns.barplot(x='Condition', order=['E', 'U', 'Ec', 'Uc'], y='Ratio', data=result_df, 
                ci=68, palette = ['#006400', '#006400', '#808080', '#808080']) #laser groups are green, control groups grey
# Define some hatches so enriched groups are hatched
hatches = ['\\', None, '\\', None]
# Loop over the bars
for i,thisbar in enumerate(g.patches):
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[i])
    thisbar.set_edgecolor('w')
# Add the horizontal line at y=1
g.plot([-0.5, len(result_df['Condition'].unique()) - 0.5], [1, 1], 'k--', linewidth=2)
g.set(ylabel='Paired Licks/Unpaired Licks')
plt.suptitle('Preference Score for Paired Odor in Test Days', fontsize=16)
plt.show()



# =============================================================================
# normalize to water
# =============================================================================
#drop pretest
tests_df= df.loc[df.Notes.isin(['Test1','Test2'])]
#average licks by animal for each solution
cis3hex_df = tests_df.loc[tests_df["SOLUTION"].isin(["cis3hex"])]
carvone_df = tests_df.loc[tests_df["SOLUTION"].isin(["carvone"])]
water_df = tests_df.loc[tests_df["SOLUTION"].isin(["water"])]
mean_cis3hexlicks = cis3hex_df.groupby(['Animal', 'Notes', 'Condition'], sort=True)['Bouts_mean'].mean().reset_index()
mean_carvonelicks = carvone_df.groupby(['Animal', 'Notes', 'Condition'], sort=True)['Bouts_mean'].mean().reset_index()
mean_waterlicks = water_df.groupby(['Animal', 'Notes', 'Condition'], sort=True)['Bouts_mean'].mean().reset_index()
#divide both odors  by water
normal_cis3hex = mean_cis3hexlicks
normal_cis3hex['normalized to water'] = normal_cis3hex['Bouts_mean'] / mean_waterlicks['Bouts_mean']
normal_carvone = mean_carvonelicks
normal_carvone['normalized to water'] = normal_carvone['Bouts_mean'] / mean_waterlicks['Bouts_mean']

#plot normalized licks by solution and group
fig, axes = plt.subplots(1,2, figsize=(12,6), sharey=True)
sns.barplot(ax=axes[0], data = normal_cis3hex, x=normal_cis3hex.Condition,
            y=normal_cis3hex['normalized to water'], 
            hue = normal_cis3hex.Notes, hue_order = ['Test1', 'Test2'], 
            palette = ['#FFAC1C','#C04000'], ci=68)
sns.barplot(ax=axes[1], data = normal_carvone, x=normal_carvone.Condition, 
            y=normal_carvone['normalized to water'],
            hue = normal_carvone.Notes, hue_order = ['Test1', 'Test2'], 
            palette = ['#6495ED','#191970'], ci=68)
axes[0].set_title('Cis-3-Hexanol')
axes[1].set_title('Carvone')
axes[0].set_ylabel('normalized licks to water')
fig.suptitle('Comparison of normalized licks to paired (Carvone) and unpaired odors on test days')


#plot each solution over trial number
g = sns.relplot(x="Trial_num", y="LICKS",
                hue="Animal", col="SOLUTION",col_wrap=2,
                kind="line", data=df.loc[df["Notes"].isin(["Test1", "Test2"])])

