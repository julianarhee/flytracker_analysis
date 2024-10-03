#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sept 25 16:44:00 2024

Author: Juliana Rhee
Email:  juliana.rhee@gmail.com

This script is used to plot FlyTracker variables for free behavior.
"""

#%%
import os
import glob
import numpy as np
import pandas as pd

import importlib
import matplotlib as mpl
import pylab as pl
import seaborn as sns
#%
import matplotlib.gridspec as gridspec

from relative_metrics import load_processed_data
import utils as util
import plotting as putil

#%%
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=12)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%%
create_new = False

# Set sourcedirs
# srcdir = '/Volumes/Juliana/2d-projector-analysis/FlyTracker/processed_mats' #relative_metrics'
srcdir = '/Volumes/Juliana/free-behavior-analysis/MF/FlyTracker/38mm_dyad/processed'

if plot_style == 'white':
    figdir = os.path.join(os.path.split(srcdir)[0], 'plot_flytracker_vars', 'white')
else:
    figdir = os.path.join(os.path.split(srcdir)[0], 'plot_flytracker_vars')

if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

# LOCAL savedir 
#localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/2d-projector/FlyTracker'
localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/MF/38mm-dyad/FlyTracker'
out_fpath_local = os.path.join(localdir, 'processed.pkl')
print(out_fpath_local)
assert os.path.exists(out_fpath_local)

#print("There are {} processed files".format( len(os.listdir(srcdir))))

#found_ = glob.glob(os.path.join(srcdir, '*{}.pkl'.format('df')))
#len(found_)


#%%
importlib.reload(util)
create_new = False

if not create_new:
    if os.path.exists(out_fpath_local):
        df = pd.read_pickle(out_fpath_local)
        print("Loaded local processed data.")
    else:
        create_new = True

if create_new:
    print("Creating new.")
    df = util.load_aggregate_data_pkl(srcdir, mat_type='df')
    print(df['species'].unique())

    #% save
    out_fpath = os.path.join(os.path.split(figdir)[0], 'processed.pkl')
    df.to_pickle(out_fpath)
    print(out_fpath)

    # save local, too
    df.to_pickle(out_fpath_local)

#print(df[['species', 'acquisition']].drop_duplicates().groupby('species').count())
print(df.groupby('species')['acquisition'].nunique())

# Take subset
curr_species = ['Dmel', 'Dyak']
df = df[df['species'].isin(curr_species)]
df.groupby('species')['acquisition'].nunique()

# %%

jaaba_fpath = glob.glob(os.path.join(os.path.split(localdir)[0], 
                           '*mel_yak*_jaaba.pkl'))[0]
print(jaaba_fpath)

jaaba = pd.read_pickle(jaaba_fpath)
jaaba.head()

#%%
figid = jaaba_fpath.split('.pkl')[0]
print(figid)

dataid = os.path.split(figid)[1]


# %%

#% Check missing
missing = [c for c in jaaba['acquisition'].unique() if c not in df['acquisition'].unique()]
print("Missing {} acquisitions in found jaaba.".format(len(missing)))

missing_jaaba = [c for c in df['acquisition'].unique() if c not in jaaba['acquisition'].unique()]
print("Missing {} acquisitions in found processed files.".format(len(missing_jaaba)))

#%%
# merge jaaba --------------------------------------------------------
print("Merging flydf and jaaba...")
ftjaaba = util.combine_jaaba_and_processed_df(df, jaaba)

print(ftjaaba.groupby('species')['acquisition'].nunique())

#%%

if jaaba['chasing'].max() == 1:
    jaaba_thresh_dict = {'orienting': 0, 
                        'chasing': 0,
                        'singing': 0}
else:
    jaaba_thresh_dict = {'orienting': 10,
                        'chasing': 10,
                        'singing': 5}
# binarize behavs
ftjaaba = util.binarize_behaviors(ftjaaba, jaaba_thresh_dict=jaaba_thresh_dict)

#%%
# Compare velocity during singing and chasing vs. chasing only bouts

plotd = ftjaaba[(ftjaaba['chasing_binary']==1)
                         | (ftjaaba['singing_binary']==1)].copy()
plotd.loc[plotd['chasing_binary']==1, 'behavior'] = 'chasing'
plotd.loc[plotd['singing_binary']==1, 'behavior'] = 'singing'
#%%
cmap = {'chasing': 'royalblue', 
        'singing': 'magenta'}
fig, axn = pl.subplots(1, 2, sharex=True, sharey=True)
sns.histplot(data=plotd[plotd['species']=='Dmel'], x='vel', ax=axn[0], 
             label='Dmel', hue='behavior', palette=cmap,
             common_norm=False, stat='probability')
axn[0].legend_.remove()
axn[0].set_title('Dmel')
axn[0].set_xlim([-5, 40])
axn[0].set_ylim([0, 0.12])

sns.histplot(data=plotd[plotd['species']=='Dyak'], x='vel', ax=axn[1], 
             label='Dyak', hue='behavior', palette=cmap,
             common_norm=False, stat='probability')
axn[1].set_title('Dyak')
axn[1].set_xlabel('velocity (mm/s)')

sns.move_legend(axn[1], bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
sns.despine(offset=4)

for ax in axn:
    ax.set_box_aspect(1)

putil.label_figure(fig, figid)

figname = 'vel_sing_vs_chase_hist__{}'.format(dataid)
os.path.join(figdir, '{}.png'.format(figname))
print(figdir, figname)


# %%
#% Split into bouts of courtship
d_list = []
for acq, df_ in ftjaaba.groupby('acquisition'):
    df_ = df_.reset_index(drop=True)
    df_ = util.mat_split_courtship_bouts(df_, bout_marker='courtship')
    dur_ = util.get_bout_durs(df_, bout_varname='boutnum', return_as_df=True,
                    timevar='sec')
    d_list.append(df_.merge(dur_, on=['boutnum']))
ftjaaba = pd.concat(d_list)

# Subdivide into mini bouts
subbout_dur = 0.20
ftjaaba = util.subdivide_into_subbouts(ftjaaba, bout_dur=subbout_dur)

#%%

if 'fpath' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['fpath'])
if 'filename' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['filename'])
if 'name' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['name'])
if 'strain' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['strain'])

#%%
#plotd = ftjaaba[(ftjaaba['chasing_binary']==1)
#                         | (ftjaaba['singing_binary']==1)].copy()
plotd = ftjaaba[ftjaaba['courtship']==1].copy()
plotd.loc[plotd['chasing_binary']==1, 'behavior'] = 'chasing'
plotd.loc[plotd['singing_binary']==1, 'behavior'] = 'singing'

#%%
# average over subbout

bout_type = 'subboutnum'
meanbouts = plotd.groupby(['species', 'acquisition', 'behavior', 
                           bout_type]).mean().reset_index()
meanbouts.head()

if bout_type == 'subboutnum':
    bout_type = 'subboutnum-{}'.format(subbout_dur)

#%%
# Bin dist_to_other during chasing and singing
species_palette = {'Dmel': 'lavender', 
                   'Dyak': 'mediumorchid'}
error_type = 'ci'

max_dist = np.ceil(meanbouts['dist_to_other'].max())
bin_size=5
bins = np.arange(0, max_dist+bin_size, bin_size)
meanbouts['binned_dist_to_other'] = pd.cut(meanbouts['dist_to_other'], 
                                       bins=bins, labels=bins[:-1])  

fig, axn = pl.subplots(1, 2, sharex=True, sharey=True)
sns.barplot(data=meanbouts,
             x='binned_dist_to_other', 
             y='chasing_binary', ax=axn[0], 
             errorbar=error_type, errcolor=bg_color,
             hue='species', palette=species_palette, edgecolor=bg_color)
axn[0].legend_.remove()
axn[0].set_ylabel("p(chasing|courtship)")

sns.barplot(data=meanbouts,
             x='binned_dist_to_other', 
             y='singing_binary', ax=axn[1], 
             errorbar=error_type, errcolor=bg_color,
             hue='species', palette=species_palette, edgecolor=bg_color)
for ax in axn:
    ax.set_box_aspect(1)
    ax.set_xlabel('distance to other (mm)')
# format xticks to single digit numbers:
bin_edges = [str(int(x)) for x in bins[:-1]]
bin_edges[0] = '<{}'.format(bin_size)
bin_edges[-1] = '>{}'.format(int(bins[-2]))
axn[0].set_xticklabels([str(x) for x in bin_edges], rotation=0)

axn[1].set_ylabel("p(singing|courtship)")

sns.move_legend(axn[1], bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
sns.despine(offset=4)

fig.text(0.1, 0.8, 'Behaviors split by consecutive courtship bouts ({})'.format(bout_type))
fig.text(0.1, 0.85, 'Dmel (n={}), Dyak (n={})'.format(\
                    plotd[plotd['species']=='Dmel']['acquisition'].nunique(),
                    plotd[plotd['species']=='Dyak']['acquisition'].nunique()))

putil.label_figure(fig, figid)

figname = 'dist_to_other_frac-{}_sing_vs_chase_hist__{}'.format(bout_type, dataid)
os.path.join(figdir, '{}.png'.format(figname))
print(figdir, figname)

# %%