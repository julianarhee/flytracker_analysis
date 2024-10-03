#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 18:51:00 2024

Author: Juliana Rhee
Email:  juliana.rhee@gmail.com

This script is used to plot FlyTracker variables from videos captured from ventral view.
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

from relative_metrics import load_processed_data
import utils as util
import plotting as putil

#%%
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=12)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

# %%
srcdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/ventral imaging'

# Select acquisition
acq = '20240917-1052_fly2_Dyak-WT_5do_gh'

#%%
# Get action paths
found_actions_paths = glob.glob(os.path.join(srcdir, acq, '*', '*-actions.mat'))
actions_df = util.load_ft_actions(found_actions_paths)

# %%
# Load FlyTracker data
acqdir = os.path.join(srcdir, acq)
fps=60

calib_, trk_, feat_ = util.load_flytracker_data(acqdir, fps=fps, 
                                                    calib_is_upstream=False,
                                                    subfolder=acq,
                                                    filter_ori=True)

# %%
action_type = 'VPO'
actions_ = actions_df[actions_df['action']==action_type].head()

#%%
yvar = 'vel'

nsec_win = 1
nframes_win = nsec_win * fps
longest_dur = (actions_['end'] - actions_['start']).max()
m_list = []
for i, (start, end) in enumerate(actions_[['start', 'end']].values):

    pad_end = longest_dur - (end - start)
    pad_start = nframes_win
    # Get the relevant data
    #curr_frames = np.arange(start-pad_start, end+pad_end)
    curr_frames = np.arange(start - nframes_win, start + nframes_win)
    v_ = []
    for id, f_ in feat_.groupby('id'):
        curr_vel = pd.DataFrame(f_[(f_['frame'].isin(curr_frames))][yvar])
        curr_vel['id'] = id
        curr_vel['framenum'] = np.arange(0, len(curr_vel))
        v_.append(curr_vel)
    vel_ = pd.concat(v_)

    vel_['bout'] = i
    m_list.append(vel_)

vel = pd.concat(m_list)
vel.head()
# %%

fig, ax = pl.subplots()
sns.lineplot(data=vel, x='framenum', y=yvar, hue='id', ax=ax) #, style='bout')
ax.axvline(x=nframes_win, color=bg_color, linestyle=':')

# %%
