#/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
 # @ Author: Juliana Rhee
 # @ Filename: copulation_curves.py
 # @ Create Time: 2024-10-02 16:46:00
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-01-27
 # @ Description: Script for importing csv files for high-throughput behavior assays and analyzing copulation curves
 '''
#%%
import os
import glob
import numpy as np
import pandas as pd

import seaborn as sns
import pylab as plt

import utils as util
import plotting as putil

#%%

def timestamp_to_seconds(df, behavior_var='copulation'):
    '''
    Convert timestamp to seconds.
    Assumes that '{behavior_var}' column is binary ( 1 or 0 ). 
    Assumes that df has '{behavior_var}_onset' column.
    Assumes that '{behavior_var}_onset' column is in MM:SS format.

    Adds '{behavior_var}_onset_sec' and '{behavior_var}_onset_min' columns.
    
    Arguments:
        df -- dataframe
        behavior_var -- behavior variable

    Returns:
        df -- dataframe with '{behavior_var}_onset_sec' and '{behavior_var}_onset_min' columns
    '''
    if f'{behavior_var}_onset_sec' not in df.columns:
        print("Adding {}_onset_sec column".format(behavior_var))
        df[f'{behavior_var}_onset'] = df[f'{behavior_var}_onset'].astype(str)
        df[f'{behavior_var}_onset_sec'] = 0
        df.loc[df[behavior_var]==1, f'{behavior_var}_onset_sec'] = df.loc[df[behavior_var]==1, 
                                                                    f'{behavior_var}_onset']\
                                                                    .apply(util.convert_time_to_seconds)
    # also convert to minutes                                                      
    if f'{behavior_var}_onset_min' not in df.columns:
        print("Adding {}_onset_min column".format(behavior_var))
        df[f'{behavior_var}_onset_min'] = df[f'{behavior_var}_onset_sec'] / 60

    return df

def cumulative_copulations_by_minute(df_, bins, behavior_var='copulation',
                                     normalize=False, 
                                     norm_value=None, 
                                     unit='min'):
    '''
    Bin the data by min (or sec) and count N copulations up to that time.
    Assumes that df has 'copulation' and 'copulation_onset_sec' (or whatever unit).
    '''
    bins_ = [] 
    cum_ = []
    # For each bin, count the number of copulations that occurred up to that time
    for i, bin in enumerate(bins): #bin_edges):
        n_in_bin = df_[ (df_[behavior_var]==1)
                      & (df_['{}_onset_{}'.format(behavior_var, unit)]<=bin)].shape[0]
        bins_.append(bin)
        cum_.append(n_in_bin)
    cum_ = np.array(cum_)
    bins_ = np.array(bins_)

    # Normalize to N copulations
    norm_value = cum_[-1] if norm_value is None else norm_value
    if normalize:
        cum_ = cum_ / norm_value  if norm_value > 0 else np.zeros_like(cum_)

    return cum_, bins_ 



#%%
plot_style='dark'
min_fontsize=18
putil.set_sns_style(plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

species_palette = {'Dmel': 'plum', 
                   'Dyak': 'mediumseagreen',
                   'Dele': 'aquamarine'}

# %%
# basedir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data'
#basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/MF-winged-wingless-20mm'
#basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data'
#experiment = 'multichamber_20mm_winged_v_wingless'
###basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/free_behavior_analysis/ht_winged_vs_wingless'
#experiment = '20mm_3x3_elegans_winged_v_wingless'

species = 'Dmel'
strain = 'CO4N' 
# -----------------
if species == 'Dmel':
    assert strain in ['all', 'CO4N', 'ZH42', 'canton-s']
#'CO4N' #'ZH42' #'canton-s'

basedir = '/Volumes/Juliana/free_behavior_analysis/winged_vs_wingless'
if species == 'Dele':
    experiment = 'Dele_3x3_20mm'
elif species == 'Dmel':
    experiment = 'Dmel_rufei_20mm'
elif species == 'Dyak':
    experiment = 'Dyak_Dmel_3x3_20mm'
    
    
# get CSV files
experiment_dir = os.path.join(basedir, experiment)
csv_fpaths = glob.glob(os.path.join(experiment_dir, '*.csv'))
print(csv_fpaths)

# %%
with_food = False
# Without food
exp_type = 'food' if with_food else 'nofood'
if with_food:
    csv_fpath = [c for c in csv_fpaths if 'food' in c][0]
else:
    csv_fpath = [c for c in csv_fpaths if 'food' not in c][0]

print(csv_fpath)
df0 = pd.read_csv(csv_fpath)
df0.head()

df0.loc[df0['manipulation_male'].isnull(), 'manipulation_male'] = 'winged'

#%% Set output dirs
#figdir = os.path.join(basedir, 'figures', 'copulation_curves')
figdir = os.path.join(basedir, 'figures', 'copulation_curves')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

figid = csv_fpath
print(figid)

#%%a
#incl_start_ix = 20 - 2 if exp_type=='nofood' else 0 # subtract 2 for 1-indexing + header
if 'include' not in df0.columns:
    df0['include'] = 1
df = df0[(df0['species_male']==species)
       #& (df0['manipulation_male'].isin([np.nan, 'wingless']))
       & (df0['courtship']==1) 
       & (df0['genotype_male']=='WT')
       & (df0['include']==1)].copy()
    #].loc[incl_start_ix:].copy()
print(df.shape)

if strain != 'all':
    df = df[df['strain']==strain]
# %
# df.loc[df['manipulation_male'].isnull(), 'manipulation_male'] = 'winged'

#%%
cop_palette = {0: [0.3]*3, 1: [0.7]*3}

# Plot barplot of counts of copulation success by manipulation_male
fig, ax = plt.subplots(figsize=(4, 4))
sns.countplot(data=df, x='manipulation_male', hue='copulation', ax=ax,
              palette=cop_palette)
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1.1, 1), frameon=False)
ax.set_xlabel('')
ax.set_box_aspect(1)
sns.despine(offset=4, trim=True)

for p in ax.patches:
    height = p.get_height()
    if height <= 0:
        continue
    x_center = p.get_x() + p.get_width()/2
    ax.annotate('{:.0f}'.format(height),
                (x_center, 0.5),  # Fixed y position at bottom
                ha='center', va='bottom')
#                (p.get_x()+0.15, p.get_height()+1))
ax.set_box_aspect(1)

ax.set_title(f"{species}-{strain}: N copulations for courting pairs",
             loc='left', fontsize=12)

putil.label_figure(fig, csv_fpath)
# Save
figname = f'{species}-{strain}_copulation_counts_{exp_type}'
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))

print(figname)

#%%
# Convert copulation_onset from MM:SS to seconds
df = timestamp_to_seconds(df, behavior_var='copulation')
if 'frontal_behavior' in df.columns:
    df = timestamp_to_seconds(df, behavior_var='frontal_behavior')

#%%
#% plot N copulations by minute:
#c1 = 'mediumorchid'
#c2 = 'thistle'
if species == 'Dele':
    c1 = 'green'
    c2 = 'lightgreen'
elif species == 'Dyak':
    c1 = 'mediumorchid'
    c2 = 'thistle'
elif species == 'Dmel':
    # Shades of blue
    c1 = 'tab:blue' #'tab:blue'
    c2 = 'powderblue'
ymax = 0.61 if strain == 'all' else 1.01
# ------
cond1 = 'winged'
cond2 = 'wingless'
wing_cdict = {'winged': c1, 'wingless': c2}
# ------
normalize=True
n_pairs_cond1 = df[df['manipulation_male']==cond1].shape[0]
n_pairs_cond2 = df[df['manipulation_male']==cond2].shape[0]
label1 = 'winged (n={})'.format(n_pairs_cond1)
label2 = 'wingless (n={})'.format(n_pairs_cond2)

behavior_var = 'copulation'
unit = 'min'

bins = np.arange(0, 3600, 60) if unit=='sec' else np.arange(0, 61, 1)
cum1, _ = cumulative_copulations_by_minute(df[df['manipulation_male']==cond1], 
                                           bins, behavior_var=behavior_var,
                                           normalize=normalize, unit=unit,
                                           norm_value=n_pairs_cond1)
cum2, _ = cumulative_copulations_by_minute(df[df['manipulation_male']==cond2], 
                                           bins, behavior_var=behavior_var,
                                           normalize=normalize, unit=unit,
                                           norm_value=n_pairs_cond2)

fig, ax =plt.subplots(figsize=(6, 4))
ax.step(bins, cum1, label=label1, color=c1, lw=2)
ax.step(bins, cum2, label=label2, color=c2, lw=2)
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False, 
          fontsize=min_fontsize-2)
ax.set_box_aspect(1)
ax.set_xlabel('Time ({})'.format(unit))
if normalize:
    ax.set_ylabel('Normalized\n{} count'.format(behavior_var))
else:
    ax.set_ylabel('Cum. {} count'.format(behavior_var))
ax.set_ylim([-0.01, ymax])
ax.set_title(f'{species}-{strain}', loc='left', fontsize=12)
sns.despine(offset=4, trim=True)
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.6)

putil.label_figure(fig, csv_fpath)

# save
figname = f'{species}-{strain}_cumulative_{behavior_var}_by_{unit}_{exp_type}'
##.format(species, behavior_var, unit, exp_type)
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
#plt.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

#%%
# Copulation curves, only those taht copulated
bins = np.arange(0, 3600, 60) if unit=='sec' else np.arange(0, 61, 1)
plotd = df[(df['copulation']==1)
           & ~np.isnan(df[f'copulation_onset_{unit}'])].copy()

n_pairs_cond1 = plotd[plotd['manipulation_male']==cond1].shape[0]
n_pairs_cond2 = plotd[plotd['manipulation_male']==cond2].shape[0]
normalize=True

n_pairs_total = n_pairs_cond1 + n_pairs_cond2
label1 = 'winged (n={})'.format(n_pairs_cond1)
label2 = 'wingless (n={})'.format(n_pairs_cond2)

cum1, _ = cumulative_copulations_by_minute(plotd[plotd['manipulation_male']==cond1], 
                                           bins, behavior_var=behavior_var,
                                           normalize=normalize, unit=unit,
                                           norm_value=n_pairs_total)
cum2, _ = cumulative_copulations_by_minute(plotd[plotd['manipulation_male']==cond2], 
                                           bins, behavior_var=behavior_var,
                                           normalize=normalize, unit=unit,
                                           norm_value=n_pairs_total)

fig, ax =plt.subplots(figsize=(6, 4))
ax.step(bins, cum1, label=label1, color=c1, lw=2)
ax.step(bins, cum2, label=label2, color=c2, lw=2)
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False, 
          fontsize=min_fontsize-2)
ax.set_box_aspect(1)
ax.set_xlabel('Time ({})'.format(unit))
if normalize:
    ax.set_ylabel('Normalized\n{} count)'.format(behavior_var))
else:
    ax.set_ylabel('Cum. {} count)'.format(behavior_var))
#ax.set_ylim([-0.01, 0.65])
ax.set_title(f'{species}-{strain}, only those that copulated',
             fontsize=12, loc='left')
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.6)
ax.set_ylim([-0.01, 1.01])
sns.despine(offset=4, trim=True)

putil.label_figure(fig, csv_fpath)

# save
figname = f'{species}-{strain}_cumulative_copulatedonly_{behavior_var}_by_{unit}_{exp_type}'
#.format(species, behavior_var, unit, exp_type)
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
#plt.savefig(os.path.join(figdir, '{}.svg'.format(figname)))


#%%
# For elegans, check frontal
cop = df[df['copulation']==1].copy()
hue_var = 'frontal_behavior' if species=='Dele' else 'circling'

# Count number of copulations with frontal_behavior_present for winged vs. wingless
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=cop, ax=ax,
              x='manipulation_male', 
              hue=hue_var, 
              palette='viridis')
ax.set_xlabel('')
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1), 
                frameon=False, title='front. behav.')
ax.set_box_aspect(1)
ax.set_title(f'{species}-{strain}', loc='left', fontsize=12)
sns.despine(offset=4, trim=True)

for p in ax.patches:
    height = p.get_height()
    if height <= 0:
        continue
    x_center = p.get_x() + p.get_width()/2
    ax.annotate('{:.0f}'.format(height),
                (x_center, 0.5),  # Fixed y position at bottom
                ha='center', va='bottom')
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.6)

putil.label_figure(fig, figid)

# svae
figname = f'{species}-{strain}_frontal_behavior_counts'
#.format(species)
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
#plt.savefig(os.path.join(figdir, '{}.svg'.format(figname)))


#%%
if species == 'Dele':
    cols = ['file_name', 'fly_num', 'copulation', 'frontal_behavior', 
        'circling', 'bi wing extension', 'front wiggle', 
        'manipulation_male']
else:
    cols = ['file_name', 'fly_num', 'copulation', 'circling', 
            'manipulation_male']

displaydf = cop[cols].copy()
# fill nan for frontal_behavior with 0s
for col in ['frontal_behavior', 'circling', 'bi wing extension', 'front wiggle']:
    if col in displaydf.columns:
        displaydf[col].fillna(0, inplace=True)
#displaydf['circling'].fillna(0, inplace=True)
#displaydf['bi wing extension'].fillna(0, inplace=True)
#displaydf['front wiggle'].fillna(0, inplace=True)

# staack the behaviors, circling, front wiggle, and front display, 
behavior_list = ['circling', 'bi wing extension', 'front wiggle']
id_vars = [c for c in cols if c not in behavior_list]
disp = displaydf.melt(id_vars=id_vars, 
                      var_name='behavior_type', 
                      value_name='present')

n_pairs_cond1 = displaydf[displaydf['manipulation_male']==cond1].shape[0]
n_pairs_cond2 = displaydf[displaydf['manipulation_male']==cond2].shape[0]
label1 = f'winged (n={n_pairs_cond1})'
label2 = f'wingless (n={n_pairs_cond2})'
#disp.loc[disp['manipulation_male']==cond1, 'label'] = label1
#disp.loc[disp['manipulation_male']==cond2, 'label'] = label2

# plot counts of each for winged vs wingless
fig, ax = plt.subplots(figsize=(6, 5))
sns.barplot(data=disp, x='behavior_type', y='present', ax=ax,
            hue='manipulation_male', palette=wing_cdict,
            errorbar='ci', hue_order=[cond1, cond2])
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1), 
                frameon=False, title='', labels=[label1, label2])
ax.set_box_aspect(1)
# set x labels to be rotated
ax.set_xlabel('')
ax.set_ylim([0, 1])
ax.set_box_aspect(1.5)

# rotate tick labels (placed at line 288)
ax.tick_params(axis='x', labelrotation=0)
ax.set_xticklabels(['circling', 'wing\next.', 'wiggle'])
plt.setp(ax.get_xticklabels(), ha='right')
ax.set_title(f'{species}-{strain}', loc='left', fontsize=12)

plt.subplots_adjust(left=0.2, bottom=0.2, right=0.5)
cnts = disp[['file_name', 'fly_num', 'manipulation_male']].drop_duplicates()
print(cnts.groupby('manipulation_male').count())

sns.despine(offset=0, trim=True)

putil.label_figure(fig, figid)
# Save
figname = f'{species}-{strain}_frontal_behavior_types_by_wing'
#.format(species)
plt.savefig(os.path.join(figdir, f'{figname}.png'))
#plt.savefig(os.path.join(figdir, f'{figname}.svg'))

# %%
