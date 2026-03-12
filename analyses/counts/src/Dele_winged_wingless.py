#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : Dele_winged_wingless.py
Created        : 2025/05/31 18:43:57
Project        : /Users/julianarhee/Repositories/flytracker_analysis
Author         : jyr
Email          : juliana.rhee@gmail.com
Last Modified  : 
'''
#%%
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import libs.utils as util
import libs.plotting as putil
import importlib
# %%
putil.set_sns_style(style='white', min_fontsize=6)
bg_color = 'k'
# %%
basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/free_behavior/Dele_winged_vs_wingless'

fname_ele = 'courtship_free_behavior_data - elegans winged vs. wingless.csv'
fname_yak = 'courtship_free_behavior_data - yakuba raw data food.csv'

csv_fpath = os.path.join(basedir, fname_yak)
df = pd.read_csv(csv_fpath)
df.head()

basedir = os.path.split(csv_fpath)[0]
figdir = os.path.join(basedir, 'figures')
if not os.path.exists(figdir):
    os.makedirs(figdir)

# %%
df['copulation_sec'] = df['frame of copulation'] / 60
df['copulation_onset_min'] = df['copulation_sec'] / 60

court = df[df['courtship']==1]
nframes_total = 60*60*60
court['frame of copulation'].fillna(nframes_total, inplace=True) 
court['frame of copulation'].unique()

# %%
court.groupby('manipulation_male').count()
#%%
copu = court[court['frame of copulation'] < nframes_total]

copu.groupby('manipulation_male').count()

#%% 
import numpy as np

def cumulative_copulations_by_minute(df_, bins, normalize=False, 
                                     norm_value=None, 
                                     copulation_frame_var='copulation_onset_min'): 
    #unit='min'):
    '''Bin the data by min (or sec) and count N copulations up to that time.'''
    bins_ = [] 
    cum_ = []
    # For each bin, count the number of copulations that occurred up to that time
    for i, bin in enumerate(bins): #bin_edges):
        n_in_bin = df_[ (df_['copulation']==1)
                      & (df_[copulation_frame_var]<=bin)].shape[0]
        bins_.append(bin)
        cum_.append(n_in_bin)
    cum_ = np.array(cum_)
    bins_ = np.array(bins_)

    # Normalize to N copulations
    norm_value = cum_[-1] if norm_value is None else norm_value
    if normalize:
        cum_ = cum_ / norm_value  if norm_value > 0 else np.zeros_like(cum_)

    return cum_, bins_ 

# %%
#% plot N copulations by minute:
c1 = 'mediumorchid'
c2 = 'thistle'
cond1 = 'winged'
cond2 = 'wingless'
# ------

unit = 'min'
normalize=True

plotd = court.copy()
n_winged_pairs = plotd[plotd['manipulation_male']=='winged'].shape[0]
n_wingless_pairs = plotd[plotd['manipulation_male']=='wingless'].shape[0]
label1 = 'winged (n={})'.format(n_winged_pairs)
label2 = 'wingless (n={})'.format(n_wingless_pairs)

bins = np.arange(0, 3600, 60) if unit=='sec' else np.arange(0, 60, 1)
cum1, _ = cumulative_copulations_by_minute(plotd[plotd['manipulation_male']==cond1], 
                                           bins, normalize=normalize,
                                           norm_value=n_winged_pairs)
cum2, _ = cumulative_copulations_by_minute(plotd[plotd['manipulation_male']==cond2], 
                                           bins, normalize=normalize,
                                           norm_value=n_wingless_pairs)

fig, ax = plt.subplots()
ax.step(bins, cum1, label=label1, color=c1, lw=2)
ax.step(bins, cum2, label=label2, color=c2, lw=2)
ax.legend(bbox_to_anchor=(1, 1), loc='upper right', frameon=False)
ax.set_box_aspect(1)
ax.set_xlabel('time ({})'.format(unit))
if normalize:
    ax.set_ylabel('Normalized copulation count')
else:
    ax.set_ylabel('Cum. copulation count')
ax.set_ylim([-0.01, 0.9])
sns.despine(offset=4, trim=True)

putil.label_figure(fig, csv_fpath)

figname = 'Dele_winged_vs_wingless_cum_copulations_by_minute'
plt.savefig(os.path.join(figdir, figname + '.png'), bbox_inches='tight')
plt.savefig(os.path.join(figdir, figname + '.svg'), bbox_inches='tight')



# %%
