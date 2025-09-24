#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 02 16:46:00 2024

Author: Juliana Rhee
Email:  juliana.rhee@gmail.com

This script is used to import csv files for high-throughput behavior assays. 
"""
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
plot_style='white'
putil.set_sns_style(plot_style, min_fontsize=24)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

# %%
# basedir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data'
#basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/MF-winged-wingless-20mm'
#basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data'
#experiment = 'multichamber_20mm_winged_v_wingless'
###basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/free_behavior_analysis/ht_winged_vs_wingless'
#experiment = '20mm_3x3_elegans_winged_v_wingless'

basedir = '/Volumes/Juliana/free_behavior_analysis/winged_vs_wingless'
experiment = '20mm_3x3_Dele'
#experiment = 'winged-wingless'

experiment_dir = os.path.join(basedir, experiment)
csv_fpaths = glob.glob(os.path.join(experiment_dir, '*.csv'))
print(csv_fpaths)


#%% Set output dirs
#figdir = os.path.join(basedir, 'figures', 'copulation_curves')
figdir = os.path.join(basedir, 'figures', 'copulation_curves')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)


# %%
with_food = False
# Without food
exp_type = 'food' if with_food else 'nofood'
if with_food:
    csv_fpath = [c for c in csv_fpaths if 'food' in c][0]
else:
    csv_fpath = [c for c in csv_fpaths if 'food' not in c][0]

figid = csv_fpath

print(csv_fpath)
df0 = pd.read_csv(csv_fpath)
df0.head()

df0.loc[df0['manipulation_male'].isnull(), 'manipulation_male'] = 'winged'

#%%a
incl_start_ix = 20 - 2 if exp_type=='nofood' else 0 # subtract 2 for 1-indexing + header
species = 'Dele'
df = df0[(df0['species_male']==species)
       #& (df0['manipulation_male'].isin([np.nan, 'wingless']))
       & (df0['courtship']==1) 
       & (df0['genotype_male']=='WT')
    ].loc[incl_start_ix:].copy()
print(df.shape)

# %
# df.loc[df['manipulation_male'].isnull(), 'manipulation_male'] = 'winged'

#%%
cop_palette = {0: [0.3]*3, 1: [0.7]*3}
# Plot barplot of counts of copulation success by manipulation_male
fig, ax = plt.subplots()
sns.countplot(data=df, x='manipulation_male', hue='copulation', ax=ax,
              palette=cop_palette)
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1.1, 1), frameon=False)
ax.set_xlabel('')
ax.set_box_aspect(1)
sns.despine(offset=4, trim=True)

for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), 
                (p.get_x() + p.get_width()/2 - 0.12, 5))
#                (p.get_x()+0.15, p.get_height()+1))

ax.set_title("{}: N copulations for courting pairs".format(species),
             loc='left', fontsize=1)

putil.label_figure(fig, csv_fpath)
# Save
figname = '{}_copulation_counts_{}'.format(species, exp_type)
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
plt.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

print(os.path.join(figdir, '{}_copulation_counts{}.png'.format(species, exp_type)))

#%%
# Convert copulation_onset from MM:SS to seconds
def convert_time_to_seconds(time_str):
    h, m, s = time_str.split(':')
    return int(h)*60 + int(m)*60 + int(s)

df['copulation_onset'] = df['copulation_onset'].astype(str)
df['copulation_onset_sec'] = 0
df.loc[df['copulation']==1, 'copulation_onset_sec'] = df.loc[df['copulation']==1, 'copulation_onset'].apply(convert_time_to_seconds)
df['copulation_onset_min'] = df['copulation_onset_sec'] / 60

exclude = ['20250605-0920_fly1-9_Dele-HK_3do_gh']
df = df[~df['file_name'].isin(exclude)]

df['frontal_behavior_onset'] = df['event_onset'].astype(str)
df['frontal_behavior_onset_sec'] = 0
df['frontal_behavior'] = [1 if v=='yes' else 0 for v in df['frontal_behavior_present']]
df.loc[df['frontal_behavior']==1, 'frontal_behavior_onset_sec'] = df.loc[df['frontal_behavior']==1, 
                                                                          'frontal_behavior_onset'].apply(convert_time_to_seconds)
df['frontal_behavior_onset_min'] = df['frontal_behavior_onset_sec'] / 60

# plot experimental cumulative distribution of copulation_onset times
# for wing

#fig, ax = pl.subplots()
#sns.histplot(data=df, hue='manipulation_male', ax=ax, #bins=bins,
#             x='copulation_onset_sec', cumulative=True, stat='probability',
#             fill=False, element='step', common_norm=False)

#%%

def cumulative_copulations_by_minute(df_, bins, behavior_var='copulation',
                                     normalize=False, 
                                     norm_value=None, unit='min'):
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

#% plot N copulations by minute:
#c1 = 'mediumorchid'
#c2 = 'thistle'
c1 = 'green'
c2 = 'lightgreen'
cond1 = 'winged'
cond2 = 'wingless'
# ------
normalize=True
n_winged_pairs = df[df['manipulation_male']=='winged'].shape[0]
n_wingless_pairs = df[df['manipulation_male']=='wingless'].shape[0]
label1 = 'winged (n={})'.format(n_winged_pairs)
label2 = 'wingless (n={})'.format(n_wingless_pairs)

behavior_var = 'frontal_behavior'
unit = 'min'

bins = np.arange(0, 3600, 60) if unit=='sec' else np.arange(0, 60, 1)
cum1, _ = cumulative_copulations_by_minute(df[df['manipulation_male']==cond1], 
                                           bins, behavior_var=behavior_var,
                                           normalize=normalize, unit=unit,
                                           norm_value=n_winged_pairs)
cum2, _ = cumulative_copulations_by_minute(df[df['manipulation_male']==cond2], 
                                           bins, behavior_var=behavior_var,
                                           normalize=normalize, unit=unit,
                                           norm_value=n_wingless_pairs)

fig, ax =plt.subplots()
ax.step(bins, cum1, label=label1, color=c1, lw=2)
ax.step(bins, cum2, label=label2, color=c2, lw=2)
ax.legend(bbox_to_anchor=(1, 0), loc='lower right', frameon=False, fontsize=16)
ax.set_box_aspect(1)
ax.set_xlabel('time ({})'.format(unit))
if normalize:
    ax.set_ylabel('Normalized {} count'.format(behavior_var))
else:
    ax.set_ylabel('Cum. {} count'.format(behavior_var))
ax.set_ylim([-0.01, 0.65])
sns.despine(offset=4, trim=True)

putil.label_figure(fig, csv_fpath)

# save
figname = '{}_cumulative_{}_by_{}_{}'.format(species, behavior_var, unit, exp_type)
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
plt.savefig(os.path.join(figdir, '{}.svg'.format(figname)))


#%%

def get_cumulative_dist(sample, bins, normalize=True):
    hist, bin_edges = np.histogram(sample, bins=bins) #, density=True)
    cumulative = np.cumsum(hist)
    if normalize:
        if cumulative[-1] == cumulative[0]:
            cumulative_normalized = np.zeros_like(cumulative)
        else:
            cumulative_normalized = cumulative / cumulative[-1]
        return cumulative_normalized, bin_edges
    else:
        return cumulative, bin_edges

#%%


# For elegans, check frontal

cop = df[df['copulation']==1].copy()

# Count number of copulations with frontal_behavior_present for winged vs. wingless
fig, ax = plt.subplots()
sns.countplot(data=cop, ax=ax,
              x='manipulation_male', 
              hue='frontal_behavior_present', 
              palette='viridis')
ax.set_xlabel('')
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1), frameon=False)
ax.set_box_aspect(1)
sns.despine(offset=4, trim=True)

for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), 
                (p.get_x() + p.get_width()/2 - 0.12, 5))

putil.label_figure(fig, figid)

# svae
figname = 'frontal_behavior_counts'
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
plt.savefig(os.path.join(figdir, '{}.svg'.format(figname)))


#%%

cols = ['file_name', 'fly_num', 'copulation', 'frontal_behavior', 
        'circling', 'front wing display', 'front wiggle', 
        'manipulation_male']
displaydf = cop[cols].copy()

# staack the behaviors, circling, front wiggle, and front display, 
behavior_list = ['circling', 'front wing display', 'front wiggle']
id_vars = [c for c in cols if c not in behavior_list]
disp = displaydf.melt(id_vars=id_vars, var_name='behavior_type', value_name='present')

# plot counts of each for winged vs wingless
fig, ax = plt.subplots()
sns.barplot(data=disp, x='behavior_type', y='present', 
            hue='manipulation_male', ax=ax)

# %%
