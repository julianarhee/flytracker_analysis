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
import pylab as pl

import utils as util
import plotting as putil

#%%
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=18)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%%


# %%
# basedir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data'
basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/ventral-analysis/MF-winged-wingless-20mm'
#experiment = 'winged-wingless'

csv_fpaths = glob.glob(os.path.join(basedir , '*.csv'))
print(csv_fpaths)

#%% Set output dirs
figdir = os.path.join(basedir, 'figures', 'copulation_curves')
if not os.path.exists(figdir):
    os.makedirs(figdir)

# %%

# Without food
csv_fpath = [c for c in csv_fpaths if 'food' not in c][0]
print(csv_fpath)
df0 = pd.read_csv(csv_fpath)
df0.head()

#%%a
incl_start_ix = 20 - 2 # subtract 2 for 1-indexing + header
df = df0[(df0['species_male']=='Dyak')
       & (df0['manipulation_male'].isin([np.nan, 'wingless']))
       & (df0['courtship']==1) 
       & (df0['genotype_male']=='WT')
    ].loc[incl_start_ix:].copy()

# %
df.loc[df['manipulation_male'].isnull(), 'manipulation_male'] = 'winged'

#%%
cop_palette = {0: [0.3]*3, 1: [0.7]*3}
# Plot barplot of counts of copulation success by manipulation_male
fig, ax = pl.subplots()
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

ax.set_title("Dyak: N copulations for courting pairs",
             loc='left', fontsize=16)

putil.label_figure(fig, csv_fpath)
# Save
pl.savefig(os.path.join(figdir, 'copulation_counts.png'))

#%%
# Convert copulation_onset from MM:SS to seconds
def convert_time_to_seconds(time_str):
    h, m, s = time_str.split(':')
    return int(h)*60 + int(m)*60 + int(s)

df['copulation_onset_sec'] = 0
df.loc[df['copulation']==1, 'copulation_onset_sec'] = df.loc[df['copulation']==1, 'copulation_onset'].apply(convert_time_to_seconds)
df['copulation_onset_min'] = df['copulation_onset_sec'] / 60

# plot experimental cumulative distribution of copulation_onset times
# for wing

#fig, ax = pl.subplots()
#sns.histplot(data=df, hue='manipulation_male', ax=ax, #bins=bins,
#             x='copulation_onset_sec', cumulative=True, stat='probability',
#             fill=False, element='step', common_norm=False)

#%%

def cumulative_copulations_by_minute(df_, bins, normalize=False, 
                                     norm_value=None, unit='min'):
    '''Bin the data by min (or sec) and count N copulations up to that time.'''
    bins_ = [] 
    cum_ = []
    # For each bin, count the number of copulations that occurred up to that time
    for i, bin in enumerate(bins): #bin_edges):
        n_in_bin = df_[ (df_['copulation']==1)
                      & (df_['copulation_onset_{}'.format(unit)]<=bin)].shape[0]
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
c1 = 'mediumorchid'
c2 = 'thistle'
cond1 = 'winged'
cond2 = 'wingless'
# ------
normalize=True
n_winged_pairs = df[df['manipulation_male']=='winged'].shape[0]
n_wingless_pairs = df[df['manipulation_male']=='wingless'].shape[0]
label1 = 'winged (n={})'.format(n_winged_pairs)
label2 = 'wingless (n={})'.format(n_wingless_pairs)

unit = 'min'
bins = np.arange(0, 3600, 60) if unit=='sec' else np.arange(0, 60, 1)
cum1, _ = cumulative_copulations_by_minute(df[df['manipulation_male']==cond1], 
                                           bins, normalize=normalize, unit=unit,
                                           norm_value=n_winged_pairs)
cum2, _ = cumulative_copulations_by_minute(df[df['manipulation_male']==cond2], 
                                           bins, normalize=normalize, unit=unit,
                                           norm_value=n_wingless_pairs)

fig, ax =pl.subplots()
ax.step(bins, cum1, label=label1, color=c1, lw=2)
ax.step(bins, cum2, label=label2, color=c2, lw=2)
ax.legend(bbox_to_anchor=(1, 1), loc='upper right', frameon=False)
ax.set_box_aspect(1)
ax.set_xlabel('time ({})'.format(unit))
if normalize:
    ax.set_ylabel('Normalized copulation count')
else:
    ax.set_ylabel('Cum. copulation count')
ax.set_ylim([-0.01, 0.5])
sns.despine(offset=4, trim=True)

putil.label_figure(fig, csv_fpath)

# save
pl.savefig(os.path.join(figdir, 'cumulative_copulations_by_minute.png'))


#%%




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

#%
from statsmodels.distributions.empirical_distribution import ECDF
#
sample1 = df[df['manipulation_male']=='winged']['copulation_onset_sec'].values
sample2 = df[df['manipulation_male']=='wingless']['copulation_onset_sec'].values
x = np.linspace(0, 3600) #min(sample), max(sample))
#sample = np.hstack((sample1, sample2))
# fit a cdf
ecdf1 = ECDF(sample1)
ecdf2 = ECDF(sample2)
# plot the cdf
pl.plot(ecdf1.x, ecdf1.y)
pl.plot(ecdf2.x, ecdf2.y)

# can also try this:
#fig, ax = pl.subplots()
#sns.ecdfplot(data=df, hue='manipulation_male', ax=ax, 
#             x='copulation_onset_sec', stat='proportion')

# Equivalent to:
#pl.figure()
#pl.plot(np.sort(sample1), np.linspace(0, 1, len(sample1), endpoint=False))
#pl.plot(np.sort(sample2), np.linspace(0, 1, len(sample2), endpoint=False))
 
# %%
