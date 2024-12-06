#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import os
import glob

import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns

import importlib
#%%
import theta_error as the
import plotting as putil
import utils as util

# %%
# Set plotting
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=18)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

# %%

basedir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/2d-projector'
fname = 'df_P1aCS-LC10aGT.pkl'

fpath = os.path.join(basedir, 'florian-data', fname)
assert os.path.exists(fpath), fpath

with open(fpath, 'rb') as f:
    ftjaaba = pd.read_pickle(f)
ftjaaba.head()
# %%

ftjaaba['stim_hz'] = ftjaaba['speed_rps'].copy()
ftjaaba['acquisition'] = ftjaaba['exp_id'].copy()
ftjaaba['sec'] = ftjaaba['time'].copy()
ftjaaba['id'] = 0

ftjaaba['dist_to_other'] = ftjaaba['dist_to_dot'].copy()
ftjaaba['species'] = 'Dmel'

#%%

excl_cols = ['gtype', 'food', 'obj', 'housing', 'comments', 
             'exp_id', 'exp', 'sex']
incl_cols = [c for c in ftjaaba.columns if c not in excl_cols]
ftjaaba = ftjaaba[incl_cols]

    #%% =====================================================
    # subdivide 
    # --------------------------------------------------------
    # split into small bouts
    # --------------------------------------------------------
    #%
    # subdivide into smaller boutsa
    bout_dur = 0.20
    ftjaaba = util.subdivide_into_subbouts(ftjaaba, bout_dur=bout_dur)

    #% FILTER
    #min_boutdur = 0.05
    min_dist_to_other = 2
    #%
    filtdf = ftjaaba[(ftjaaba['id']==0)
                    #& (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                    #& (ftjaaba['targ_pos_theta']<=max_pos_theta)
                    & (ftjaaba['dist_to_other']>=min_dist_to_other)
                    #& (ftjaaba['boutdur']>=min_boutdur)
                    #& (ftjaaba['good_frames']==1)
                    #& (ftjaaba['led_level']>0)
                    ].copy() #.reset_index(drop=True)

    # drop rows with only 1 instance of a given subboutnum
    #min_nframes = min_boutdur * 60
    filtdf = filtdf[filtdf.groupby(['species', 'acquisition', 'subboutnum'])['subboutnum'].transform('count')>min_nframes]
    #%%
    # Get mean value of small bouts
    if 'strain' in filtdf.columns:
        filtdf = filtdf.drop(columns=['strain'])
    #%
    meanbouts = filtdf.groupby(['species', 'acquisition', 'subboutnum']).mean().reset_index()
    meanbouts.head()

    cmap='viridis'
    stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)

#%%
    # find the closest matching value to one of the keys in stimhz_palette:
    meanbouts['stim_hz'] = meanbouts['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))   

# %%
    #%% ------------------------------------------------
    # ANG_VEL vs. THETA_ERROR
    # -------------------------------------------------
    #%
    xvar ='theta_error'
    yvar = 'ang_vel_fly_shifted' #'ang_vel_fly_shifted' #'ang_vel' #'ang_vel_fly'
    plot_hue= True
    plot_grid = True
    nframes_lag = 2

    shift_str = 'SHIFT-{}frames_'.format(nframes_lag) if 'shifted' in yvar else ''
    hue_str = 'stimhz' if plot_hue else 'no-hue'

    # Set palettes
    cmap='viridis'
    #stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)

    # Get CHASING bouts 
    behav_var = 'pursuit_jaaba'
    min_frac_bout = 0.9
    chase_ = meanbouts[ (meanbouts['{}'.format(behav_var)]>min_frac_bout) ].copy()
                    #    & (meanbouts['ang_vel_fly_shifted']< -25)].copy()
    #chase_ = filtdf[filtdf['{}_binary'.format(behav)]>0].copy()

#%%
    # Check florian's variables
    fig = pl.figure()
    ax = fig.add_subplot(1,2, 1) #, ax = pl.subplots()
    ax.scatter(chase_['relx'], chase_['rely'])
    # set aspect equal:
    ax.set_aspect(1)
    ax.plot(0, 0, 'r*')

    abs_ang = np.arctan2(chase_['rely'], chase_['relx'])
    chase_['theta_error'] = abs_ang
    #fig, ax = pl.subplots(subplot_kw={'projection': 'polar'})
    ax = fig.add_subplot(1,2,2, projection='polar') 
    ax.scatter(chase_['theta_error'], chase_['dist_to_dot'])
    ax.plot(0, 0, 'r*')

#%%
    if 'shifted' in yvar:
        figtitle = '{} bouts, where min fract of bout >= {:.2f}\nshifted {} frames'.format(behav, min_frac_bout, nframes_lag)
    else:
        figtitle = '{} bouts, where min fract of bout >= {:.2f}'.format(behav, min_frac_bout)

    species_str = '-'.join(chase_['species'].unique())

    xlabel = r'$\theta_{E}$ at $\Delta t$ (rad)'
    ylabel = '$\omega_{f}$ (rad/s)'

    # SCATTERPLOT:  ANG_VEL vs. THETA_ERROR -- color coded by STIM_HZ
    fig = the.plot_regr_by_species(chase_, xvar, yvar, hue_var='stim_hz', 
                            plot_hue=plot_hue, plot_grid=plot_grid,
                            xlabel=xlabel, ylabel=ylabel, bg_color=bg_color,
                            stimhz_palette=stimhz_palette)
    fig.suptitle(figtitle, fontsize=12)
    pl.subplots_adjust(wspace=0.25)

    #for ax in fig.axes:
    #    #ax.invert_yaxis()
    #    ax.invert_xaxis()

    #putil.label_figure(fig, figid)
    #
# %%
