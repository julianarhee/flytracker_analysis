#!/usr/bin/env python3
# -*- coding: utf-8 -*-``
'''
 # @ Author: Juliana Rhee
 # @ Filename:
 # @ Create Time: 2025-09-24 14:24:12
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-09-24 15:34:46
 # @ Description:
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
plot_style='white'
min_fontsize=24
putil.set_sns_style(plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

species_palette = {'Dmel': 'plum', 
                   'Dyak': 'mediumseagreen',
                   'Dele': 'aquamarine'}

#%%
rootdir = '/Volumes/Juliana/free_behavior_analysis'
experiment = '2M2F/chidera'
experiment_dir = os.path.join(rootdir, experiment)

csv_fpaths = glob.glob(os.path.join(experiment_dir, '*.csv'))
print(experiment_dir)

csv_fpath = csv_fpaths[0]
print(csv_fpath)

figid = csv_fpath
print(figid)

df0 = pd.read_csv(csv_fpath)
df0.head()

#%% Set output directories

figdir = os.path.join(experiment_dir, 'count_switches')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

#%%

df0['courtship_duration'].fillna('60:00', inplace=True)
df = df0[df0['courtship_duration'].notna()].copy()

df['courtship_duration_sec'] = df['courtship_duration'].astype(str).apply(util.convert_time_to_seconds)

df['chaining'] = False
df['note'].fillna('', inplace=True)
df.loc[df['note'].str.contains('chaining'), 'chaining'] = True

plotd = df[(df['chaining']==False)].copy()
meandf = plotd.groupby(['species', 'videoname']).agg({'switch': 'sum',
                                         'aggregate': 'sum',
                                         'courtship_duration_sec': 'mean'})\
                                             .reset_index()
                                             
                                            
meandf['switch_rate'] = meandf['switch'] / meandf['courtship_duration_sec']
meandf['aggregate_rate'] = meandf['aggregate'] / meandf['courtship_duration_sec']
meandf

#%%
markersize = 10
alpha=0.75

fig, axn = plt.subplots(1, 2, figsize=(6.5, 3), sharex=True)
ax=axn[0]
sns.stripplot(data=meandf, x='species', y='switch_rate', ax=ax,
              hue='species', palette=species_palette, s=markersize,
              alpha=alpha, jitter=True, edgecolor=bg_color, linewidth=0.5)
# sns.barplot(data=meandf, x='species', y='switch_rate', ax=ax)
mean_val = meandf.groupby('species')['switch_rate'].mean().reset_index()
ax.plot([0, 1], [mean_val[mean_val['species']=='Dmel']['switch_rate'], 
                 mean_val[mean_val['species']=='Dyak']['switch_rate']], 
        color=bg_color, linewidth=0, marker='_', markersize=30)
ax.set_ylabel('Switch rate')

ax=axn[1]
sns.stripplot(data=meandf, x='species', y='aggregate_rate', ax=ax,
              hue='species', palette=species_palette, s=markersize,
              alpha=alpha, jitter=True, edgecolor=bg_color, linewidth=0.5)
mean_val = meandf.groupby('species')['aggregate_rate'].mean().reset_index()
ax.plot([0, 1], [mean_val[mean_val['species']=='Dmel']['aggregate_rate'], 
                 mean_val[mean_val['species']=='Dyak']['aggregate_rate']], 
        color=bg_color, linewidth=0, marker='_', markersize=30)
ax.set_ylabel('Aggregate rate')

for ax in axn:
    sns.despine(offset=2, ax=ax)
    ax.set_xlabel('')
    ax.set_box_aspect(1.5) 
plt.subplots_adjust(wspace=0.6)

putil.label_figure(fig, figid)

plt.savefig(os.path.join(figdir, 'switch_rate.png'))
plt.savefig(os.path.join(figdir, 'switch_rate.svg'))


# %%
