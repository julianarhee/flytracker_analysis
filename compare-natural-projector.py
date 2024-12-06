#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
import sys
import glob
import importlib

import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import pylab as pl
import matplotlib as mpl

import matplotlib.gridspec as gridspec


from relative_metrics import load_processed_data
import utils as util
import plotting as putil

#%%
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=24)
bg_color = [0.7]*3 if plot_style=='dark' else 'w'

#%%
minerva_base = '/Volumes/Julie'

for assay in ['38mm-dyad', '2d-projector']:
    if assay == '2d-projector':
        # Set sourcedirs
        srcdir = os.path.join(minerva_base, '2d-projector-analysis/FlyTracker/processed_mats') #relative_metrics'
        # LOCAL savedir 
        localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/2d-projector/FlyTracker'
    elif assay == '38mm-dyad':
        # src dir of processed .dfs from feat/trk.mat files (from relative_metrics.py)
        srcdir = os.path.join(minerva_base, 'free-behavior-analysis/FlyTracker/38mm_dyad/processed')
        # local savedir for giant pkl
        localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/38mm-dyad/FlyTracker'

    # get local file for aggregated data
    out_fpath_local = os.path.join(localdir, 'processed.pkl')


    if assay == '38mm-dyad':
        nat = pd.read_pickle(out_fpath_local)
    else:
        proj = pd.read_pickle(out_fpath_local)

# %%

assay = '38mm-dyad'
fname = 'free_behavior_data_mel_yak_20240403'
nat = util.load_jaaba(assay, fname=fname)

fname = 'projector_data_all'
proj = util.load_jaaba(assay, fname=fname)

#%%

nat = nat.drop(columns=['filename', 'name', 'strain'])
proj = proj.drop(columns=['name'])
proj = proj.rename(columns={'filename': 'acquisition'})

#%%

jaaba_thresh_dict = {'orienting': 10,
                    'chasing': 10,
                    'singing': 5}

nat = util.binarize_behaviors(nat, jaaba_thresh_dict=jaaba_thresh_dict)
proj = util.binarize_behaviors(proj, jaaba_thresh_dict=jaaba_thresh_dict)

#%%
# subdivide into smaller boutsa
bout_dur = 1.0
fps = 60.

nat['sec'] = nat['frame'] / fps
proj['sec'] = proj['frame'] / fps
nat = util.subdivide_into_bouts(nat, bout_dur=bout_dur)
proj = util.subdivide_into_bouts(proj, bout_dur=bout_dur)

#%% Get mean value of small bouts

nbins = nat.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()
pbins = proj.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()

#%%

nbins['assay'] = 'natural'
pbins['assay'] = 'projector'

pbins['species'] = ['Dyak' if 'yak' in v else 'Dmel' for v in pbins['species']]

#%%
behav = 'singing'
min_thr = 0.2

fig, axn = pl.subplots(1, 2, sharex=True, sharey=True)

n_ = nbins[(nbins['{}_binary'.format(behav)]>min_thr)]
p_ = pbins[(pbins['{}_binary'.format(behav)]>min_thr)]


d1 = n_[['species', 'boutnum', 'assay', 'dovas']].copy()
d2 = p_[['species', 'boutnum', 'assay', 'dovas']].copy()

plotdf = pd.concat([d1, d2], axis=0).reset_index(drop=True)

for i, (sp, d_) in enumerate(plotdf.groupby('species')):
    ax=axn[i]
    sns.histplot(data=d_, x='dovas', hue='assay', legend=0, ax=ax,
             stat='probability', common_norm=False) #(n_['dovas'], bins=100, alpha=0.5, 


# %%

n_['dovas_deg'] = np.rad2deg(n_['dovas'])
p_['dovas_deg'] = np.rad2deg(p_['dovas'])

joint_kws={'bw_adjust': 0.75, 'levels': 30}
g = sns.jointplot(data=n_, x='dovas_deg', y='abs_rel_vel',
              hue='species', kind='kde',  common_norm=False,
              joint_kws=joint_kws)
g.fig.suptitle('natural')

#%%

g = sns.jointplot(data=p_, x='dovas_deg', y='abs_rel_vel',
              hue='species', kind='kde',  common_norm=False,
              joint_jws=joint_kws)

g.fig.suptitle('projector')
# %%
