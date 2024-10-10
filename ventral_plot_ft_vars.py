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
basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee'# imaging'
video_srcdir = os.path.join(basedir, 'caitlin_data', 'ventral imaging')

csv_fpaths = glob.glob(os.path.join(basedir , 'ventral-analysis', '*.csv'))
print(csv_fpaths)

#%% Set output dirs
figdir = os.path.join(basedir, 'ventral-analysis', 'figures', 'ventral_plot_ft_vars')
if not os.path.exists(figdir):
    os.makedirs(figdir)

#%% Load metadata

csv_fpath = csv_fpaths[0]
print(csv_fpath)
df0 = pd.read_csv(csv_fpath)
df0.head()

df = df0[(df0['annotated']=='yes')].copy() # & ~(df0['active'].isnull())].copy()
df.groupby('species_male')['file name'].count()

dataid = csv_fpath.split('RU ')[-1]
print(dataid)
#%%
a_list = []
mat_error = []; index_error = []
for acq, acq_df in df.groupby('file name'):
    if acq_df['species_male'].iloc[0]=='Dmel':
        found_actions_paths = glob.glob(os.path.join(video_srcdir, 'Rufei',
                                '{}'.format(acq), '*', '*-actions.mat'))
    else:
        acq = acq.replace(' ', 'do')
        found_actions_paths = glob.glob(os.path.join(video_srcdir, 
                                '{}*'.format(acq), '*', '*-actions.mat'))
    try:
       if len(found_actions_paths)>1:
           print('Multiple actions found for {}'.format(acq))
           for i in found_actions_paths:
               print(i)

       _actions = util.load_ft_actions([found_actions_paths[-1]])
       _actions['acquisition'] = acq
       _actions['species'] = acq_df['species_male'].iloc[0]
       a_list.append(_actions)
    except Exception as MatReadError:
        mat_error.append(acq)
    except IndexError:
        index_error.append(acq)

if len(mat_error) > 0:
    print("Unable to load .mat for:")
    for i in mat_error:
        print(' .   {}'.format(i))
if len(index_error) > 0:
    print("Index error for:")
    for i in index_error:
        print(' .   {}'.format(i))
actions = pd.concat(a_list)
actions.head() 

#%%

action_counts = actions[(actions['action'].isin(['VPO', 'OVE'])) 
                        & (actions['likelihood']>1)]\
                        .groupby(['acquisition', 'action']).count().reset_index()

action_counts['species'] = action_counts['acquisition'].str.extract(r'(D\w{3})')
action_counts.loc[action_counts['species'].isnull(), 'species'] = 'Dmel'

total_action_counts = action_counts.groupby(['species','action'])['boutnum'].sum()
# total_action_counts[('Dmel', 'OVE')]


#%%
# Select acquisition
#acq = '20240917-1052_fly2_Dyak-WT_5do_gh'
# acq = '20240904-1033_fly4_Dyak-WT_5do_gh' #ove=16, vpo=7
#acq = '20240912-0956_fly2_Dyak-WT_3do_gh' # ove=21, vpo=13
#acq = '20240912-1120_fly5_Dyak-WT_2do_gh' # ove=8, vpo=15

acq = '20240916-1108_fly1_Dyak-WT_5do_gh' # ove=10, vpo=6
#acq = '20240920-0955_fly1_Dyak-WT_2do_gh' # ove=10, vpo=5

#acq = '030722_Canton-S_age5_3'
#%
# Get action paths

importlib.reload(util)
if 'Dyak' in acq:
    found_actions_paths = glob.glob(os.path.join(video_srcdir, 
                        '{}*'.format(acq), '*', '*-actions.mat'))
else:
    found_actions_paths = glob.glob(os.path.join(video_srcdir, 'Rufei', 
                        '{}*'.format(acq), '*', '*-actions.mat'))

actions_df = util.load_ft_actions([found_actions_paths[-1]])

#actions_df = actions[actions['acquisition']==acq].copy()


# %%
fps = 60
# Load FlyTracker data
acqdir = os.path.join(video_srcdir, acq)
calib_, trk_, feat_ = util.load_flytracker_data(acqdir, fps=fps, 
                                                    calib_is_upstream=False,
                                                    subfolder='*',
                                                    filter_ori=True)

#%%

def get_df_centered_on_actions(actions_, feat_, trk_, nframes_win):

    longest_dur = (actions_['end'] - actions_['start']).max()
    m_list = []
    for i, (start, end) in enumerate(actions_[['start', 'end']].values):
        # Get the relevant data
        if nframes_win==0:
            curr_frames = np.arange(start, end+1)
        else:
            curr_frames = np.arange(start - nframes_win, start + nframes_win)

        f_ = feat_[(feat_['frame'].isin(curr_frames))].copy()
        t_ = trk_[(trk_['frame'].isin(curr_frames))].copy()
        f_['bout'] = i
        rel_frame_nums = curr_frames - start
        #    v_ = []
        for id, ff_ in f_.groupby('id'):
            if len(ff_) < len(curr_frames):
                f_.loc[ff_.index, 'framenum'] = np.arange(0, len(ff_))
                f_.loc[ff_.index, 'rel_time'] = rel_frame_nums[0:len(ff_)] / fps 
            else:
                f_.loc[ff_.index, 'framenum'] = np.arange(0, len(curr_frames))
                f_.loc[ff_.index, 'rel_time'] = rel_frame_nums / fps 
        c_ = pd.merge(f_, t_, left_index=True, right_index=True, suffixes=[None,'_drop']) 
        drop_cols = [c for c in c_.columns if 'drop' in c]
        c_.drop(columns=drop_cols, inplace=True)
        c_['acquisition'] = actions_['acquisition'].unique()[0]
        #c_ = pd.merge(f_, t_, on=['id', 'fpath', 'frame', 'sec'], how='outer') 
        if 'species' not in c_.columns or ('species' in c_.columns and c_['species'].isnull().all()):
            c_['species'] = actions_['species'].unique()[0] #c_['acquisition'].str.extract(r'(D\w{3})')
        m_list.append(c_)

    if len(m_list)==0:
        return None
    
    vpo = pd.concat(m_list)
    return vpo

#%% Get VPOs
nsec_win = 1
nframes_win = nsec_win * fps

vpo_actions = actions_df[(actions_df['action']=='VPO')
                      & (actions_df['likelihood']>1)].copy()
vpo = get_df_centered_on_actions(vpo_actions, feat_, trk_, nframes_win)
vpo.head()

#%% Get OVEs
ove_actions = actions_df[(actions_df['action']=='OVE')
                      & (actions_df['likelihood']>1)].copy()
ove = get_df_centered_on_actions(ove_actions, feat_, trk_, nframes_win)
ove.head()

#%% PLOT --- 
yvar = 'vel'

fig, axn = pl.subplots(1, 2, sharey=True, sharex=True)
ax=axn[0]
sns.lineplot(data=vpo, x='rel_time', y=yvar, hue='id', ax=ax) #, style='bout')
ax.axvline(x=0, color=bg_color, linestyle=':')
ax.set_title('VPO (n={})'.format(vpo['bout'].nunique()))
ax.legend_.remove()

ax=axn[1]
sns.lineplot(data=ove, x='rel_time', y=yvar, hue='id', ax=ax) #, style='bout')
ax.axvline(x=0, color=bg_color, linestyle=':')
ax.set_title('OVE (n={})'.format(ove['bout'].nunique()))

sns.move_legend(ax, loc='center left', bbox_to_anchor=(1, 0.5), title='ID', frameon=False)

for ax in axn:
    ax.set_box_aspect(1)

putil.label_figure(fig, acq)

figname = 'male_{}_vpo_ove__{}'.format(yvar, acq)
fig_fpath = os.path.join(figdir, '{}.png'.format(figname))
pl.savefig(fig_fpath)
print(fig_fpath)

#%%
vpo_frames = get_df_centered_on_actions(vpo_actions, feat_, trk_, 0)
ove_frames = get_df_centered_on_actions(ove_actions, feat_, trk_, 0)
vpo_frames['action'] = 'vpo'
ove_frames['action'] = 'ove'


evs = pd.concat([vpo_frames, ove_frames], axis=0)

plot_yvars = ['ang_vel', 'max_wing_ang', 'vel']# 'major_axis_len'
#yvar = 'minor_axis_len'
fig, axn = pl.subplots(1, len(plot_yvars)) #, sharex=True, sharey=True)

for ax, yvar in zip(axn.flat, plot_yvars):
    sns.histplot(data=evs[evs['id']==0], x=yvar, ax=ax, hue='action', 
             fill=None)
#ax=axn[1]
#sns.histplot(data=ove_frames, x=yvar, ax=ax)


# %%

# AGGGREGATE ACROSS ACQUISITIONS

nsec_win = 2
nframes_win = nsec_win * fps
min_likelihood = 2

v_ =[]
o_ = []
for acq, actions_df in actions.groupby('acquisition'):
    print(acq)

    if 'Dyak' in acq:
        acqdir = os.path.join(video_srcdir, acq)
    else:
        acqdir = os.path.join(video_srcdir, 'Rufei', acq)
    try:
        calib_, trk_, feat_ = util.load_flytracker_data(acqdir, fps=fps, 
                                                    calib_is_upstream=False,
                                                    subfolder='*',
                                                    filter_ori=True)
    except Exception as MatReadError:
        continue

    # Get VPOs
    vpo_actions = actions_df[(actions_df['action']=='VPO')
                        & (actions_df['likelihood']>=min_likelihood)].copy()
    vpo = get_df_centered_on_actions(vpo_actions, feat_, trk_, nframes_win)
    if vpo is not None:
        v_.append(vpo)
    else: 
        print('No VPOs found for {}'.format(acq))

    # Get OVEs
    ove_actions = actions_df[(actions_df['action']=='OVE')
                        & (actions_df['likelihood']>=min_likelihood)].copy()
    ove = get_df_centered_on_actions(ove_actions, feat_, trk_, nframes_win)
    if ove is not None:
        o_.append(ove)
    else:
        print('No OVEs found for {}'.format(acq))

all_vpo = pd.concat(v_)
all_ove = pd.concat(o_)

#all_vpo.loc[all_vpo['species'].isnull(), 'species'] = 'Dmel'
#all_ove.loc[all_ove['species'].isnull(), 'species'] = 'Dmel'

all_ove['action'] = 'ove'
all_vpo['action'] = 'vpo'
all_evs = pd.concat([all_vpo, all_ove], axis=0)

#%%
# Print counts
cnts = all_evs.groupby(['species', 'action', 'acquisition'])['bout'].nunique().reset_index()
print(cnts.groupby(['species', 'action'])['bout'].sum())

final_action_counts = cnts.groupby(['species', 'action'])['bout'].sum()

# %%
# Compare VEL for OVEs. vs VPOs

yvar = 'vel'
sex_palette = {0: 'cornflowerblue',
               1: 'red'}
# Use regexp to find the pattern 'Dxxx' where 'xxx' is any 3 combination of letters:
#df['species'] = df['acquisition'].str.extract(r'(D\w{3})')

fig, axn = pl.subplots(2, 2, sharey=False, sharex=True)
for ri, (species, df_) in enumerate(all_evs.groupby('species')):
    for ci, (action, ddf_) in enumerate(df_.groupby('action')):
        #ax=axn[0]
        ax=axn[ri, ci]
        sns.lineplot(data=ddf_, x='rel_time', y=yvar, 
                     hue='id', ax=ax, palette=sex_palette) #, style='bout')
        ax.axvline(x=0, color=bg_color, linestyle=':')
        curr_counts = ddf_[['acquisition', 'bout']].drop_duplicates().shape[0] ##final_action_counts[(species, action)] 
        ax.set_title('{}, {} (n={})'.format(species, action, 
                    curr_counts))
        ax.legend_.remove()
        if species=='Dmel':
            ax.set_ylim([0, 18])
        else:
            ax.set_ylim([0, 8])
pl.subplots_adjust(hspace=0.5, wspace=0.5)

putil.label_figure(fig, dataid)
figname = 'vpo_ove_male-female_min-likelihood-{}_{}'.format(min_likelihood, yvar)
fig_fpath = os.path.join(figdir, '{}.png'.format(figname))
pl.savefig(fig_fpath)

#%%

# COMBINE ove/vpo, compare MEL vs YAK with w.e.

col_m = 'cornflowerblue'
col_f = 'red'
sex_palette = {0: col_m, 1: col_f}
twinx = False
nr = 1 if twinx else 2
plot_type = 'twinx' if twinx else 'sep'

subsample = False
sampling = 'resampled' if subsample else 'true'

yvar1 = 'vel'
yvar2 = 'max_wing_ang'
fig, axn = pl.subplots(nr, 2, sharey=False, sharex=True, 
                       figsize=(10, 5))
for ri, (species, df_) in enumerate(all_evs.groupby('species')):
    #ax=axn[0]
    ax=axn[ri] if twinx else axn[0, ri]
    if species == 'Dyak' and subsample:
        # randomly sample
        curr_bouts = df_[['acquisition', 'bout']].drop_duplicates()
        sampled_bouts = curr_bouts.sample(n=70, replace=False)
        plotd = pd.concat([df_[(df_['acquisition']==a) & (df_['bout']==b)] for (a, b) in sampled_bouts[['acquisition', 'bout']].values])
    else:
        plotd = df_.copy()
    print(plotd.shape)
    sns.lineplot(data=plotd, x='rel_time', y=yvar1, 
                    hue='id', ax=ax, palette=sex_palette) #, style='bout')
    # annotations
    ax.axvline(x=0, color=bg_color, linestyle=':')
    # title
    curr_count = plotd[['acquisition', 'action', 'bout']].drop_duplicates().shape[0]  
    ax.set_title('{} (n={})'.format(species, curr_count))
    # legend
    if ri==0:
        ax.legend_.remove()
    else:
        legh = putil.custom_legend(['male', 'female'], [col_m, col_f] )
        ax.legend(handles=legh, loc='upper left', 
                  bbox_to_anchor=(1.2, 1), frameon=False)
    
    ax.set_ylim([0, 10])
    #ax.set_box_aspect(1)
    #if plot_wings:
    if twinx:
        ax2 = ax.twinx()
    else:
        ax2=axn[1, ri]
    yvar2 = 'max_wing_ang'
    sns.lineplot(data=plotd[plotd['id']==0], x='rel_time', y=yvar2, ax=ax2,
                color=bg_color)
    # annotations
    ax2.axvline(x=0, color=bg_color, linestyle=':')
    ax2.set_ylim([0., 1.5])

pl.subplots_adjust(hspace=0.1, wspace=0.3, right=0.9)

putil.label_figure(fig, dataid)
figname = 'abext-wingang-{}_{}-data_male-female_min-likelihood-{}_{}'.format(plot_type, sampling, min_likelihood, yvar)
fig_fpath = os.path.join(figdir, '{}.png'.format(figname))
pl.savefig(fig_fpath)

#%%

yvar = 'max_wing_ang'

# Use regexp to find the pattern 'Dxxx' where 'xxx' is any 3 combination of letters:
#df['species'] = df['acquisition'].str.extract(r'(D\w{3})')

fig, axn = pl.subplots(1, 2, sharey=True, sharex=True)
ax=axn[0]
sns.lineplot(data=all_vpo[all_vpo['id']==0], x='rel_time', y=yvar, ax=ax) #, style='bout')
ax.axvline(x=0, color=bg_color, linestyle=':')
ax.set_title('VPO (n={})'.format(vpo['bout'].nunique()))

ax=axn[1]
sns.lineplot(data=all_ove[all_ove['id']==0], x='rel_time', y=yvar, ax=ax) #, style='bout')
ax.axvline(x=0, color=bg_color, linestyle=':')
ax.set_title('OVE (n={})'.format(ove['bout'].nunique()))

#sns.move_legend(ax, loc='center left', bbox_to_anchor=(1, 0.5), title='ID', frameon=False)

# %%

plot_yvars = ['ang_vel', 'max_wing_ang', 'vel']# 'major_axis_len'
#yvar = 'minor_axis_len'
fig, axn = pl.subplots(1, len(plot_yvars)) #, sharex=True, sharey=True)

for ax, yvar in zip(axn.flat, plot_yvars):
    sns.histplot(data=all_evs[all_evs['id']==0], x=yvar, ax=ax, hue='action', 
             fill=None, stat='probability', common_norm=False)

# %%
