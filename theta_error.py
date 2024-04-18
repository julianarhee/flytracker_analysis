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
import scipy.signal as signal
def cross_correlation_lag(x, y, fps=60):
    correlation = signal.correlate(x-np.mean(x), y - np.mean(y), mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    lag_frames = lags[np.argmax(correlation)] #lags[np.argmax(abs(correlation))]
    t_lag = lag_frames / fps 
    return correlation, lags, lag_frames, t_lag

def get_window_centered_bouts(turn_start_frames, flydf, nframes_win):
    '''
    For each provided frame, get a window of nframes_win before and after the frame.

    Arguments:
        turn_start_frames -- _description_
        flydf -- _description_
        nframes_win -- _description_

    Returns:
        turnbouts (pd.DataFrame): columns include turn_bout_num and rel_sec.
    '''
    d_list = []
    for i, ix in enumerate(turn_start_frames):
        start_ = ix - nframes_win
        stop_ = ix + nframes_win
        t_onset = flydf.loc[ix]['sec']
        d_  = flydf.loc[start_:stop_].copy()
        d_['turn_bout_num'] = i
        d_['rel_sec'] = (d_['sec'].astype(float) - t_onset).round(2)
        d_['turn_start_frame'] = ix
        if len(d_['rel_sec'].values) < 13:
            print("bad len")
            break
        d_list.append(d_)

    turnbouts = pd.concat(d_list)

    return turnbouts


def get_turn_bouts(flydf, min_acc=100, min_dist_to_other=15, min_facing_angle=np.deg2rad(90), min_vel=20):
    #min_ang_vel = 4.36
    turndf = flydf[ (flydf['stim_hz']>0) 
                & (flydf['ang_acc']>=min_acc)
                #& (flydf['chasing']>8)
                #& (flydf['chasing']>10)
                & (flydf['good_frames']==1)
                & (flydf['dist_to_other']<=min_dist_to_other)
                & (flydf['facing_angle'] <= min_facing_angle)
                & (flydf['vel']>min_vel)
                ]

    # get start/stop indices of consecutive rows
    turn_bout_ixs = util.get_indices_of_consecutive_rows(turndf)

    # find turn starts
    turn_start_frames = [c[0] for c in turn_bout_ixs] #incl_bouts] #high_ang_vel_bouts]
    turnbouts = get_window_centered_bouts(turn_start_frames, flydf, nframes_win)

    return turnbouts


def plot_individual_turns(turnbouts, v1='facing_angle', v2='ang_vel'):
    fig, axn = pl.subplots(1, 2, sharex=True, figsize=(10,5))
    ax=axn[0]
    sns.lineplot(data=turnbouts, x='rel_sec', y=xvar1, ax=ax, hue='turn_bout_num', palette='Reds', legend=0,
                lw=0.5, alpha=0.5)
    ax.set_ylabel(r'$\theta_{E}$')
    ax=axn[1]
    sns.lineplot(data=turnbouts, x='rel_sec', y=xvar2, ax=ax, hue='turn_bout_num', palette='Blues', legend=0,
                lw=0.5, alpha=0.5)
    ax.set_ylabel(r'$\omega_{f}$')
    pl.subplots_adjust(wspace=0.5, top=0.8)
    fig.text(0.1, 0.95,  '{}, all turns ang_acc > {:.2f}'.format(acq, min_acc), 
                    fontsize=8)
    return fig

import statsmodels.api as sm

import scipy.stats as spstats
def annotate_regr(data, ax, x='facing_angle', y='ang_vel', fontsize=8, **kws):
    r, p = spstats.pearsonr(data[x], data[y])
    ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes, fontsize=fontsize)


def cross_corr_each_bout(turnbouts, v1='facing_angle', v2='ang_vel'):
    # Get all turn bouts, windowed +/-
    xcorr = []
    t_lags = []
    all_lags = []
    for i, d_ in turnbouts.groupby('turn_bout_num'):
        # cross corr
        correlation, lags, lag_frames, t_lag = cross_correlation_lag(d_[v2], d_[v1], fps=60)
        xcorr.append(correlation)
        all_lags.append(lags)
        t_lags.append(t_lag)

    return np.array(xcorr), np.array(all_lags), np.array(t_lags)

def plot_cross_corr_results(turnbouts, xcorr, lags, t_lags, v1='facing_angle', v2='ang_vel', 
                            col1='r', col2='dodgerblue', bg_color=[0.7]*3):

    # PLOT MEAN + SEM of aligned turn bout traces
    fig, axn =pl.subplots(1, 3, figsize=(15, 4))
    ax1 = axn[0]
    ax2 = ax1.twinx()
    sns.lineplot(data=turnbouts, x='rel_sec', y=v1, ax=ax1, lw=0.5, color=col1)
    sns.lineplot(data=turnbouts, x='rel_sec', y=v2, ax=ax2, lw=0.5, color=col2)
    for ax, sp, lb, col in zip([ax1, ax2], ['left', 'right'], [r'$\theta_{E}$', r'$\omega_{f}$'], [col1, col2]):
        ax.set_ylabel(lb)
        putil.change_spine_color(ax, col, sp)
    ax1.axvline(x=0, color=bg_color, linestyle='--')

    #% PLOT cross-correlation (mean + sem)
    ax=axn[1]
    xcorr_mean = np.array(xcorr).mean(axis=0)
    xcorr_sem = np.array(xcorr).std(axis=0) / np.sqrt(len(xcorr))
    lags_mean = np.array(lags).mean(axis=0)
    ax.plot(lags_mean, xcorr_mean, color=bg_color)
    ax.fill_between(lags_mean, xcorr_mean-xcorr_sem, xcorr_mean+xcorr_sem, color=bg_color, alpha=0.5)
    ax.set_ylabel('cross corr.')

    # PLOT distribution of TIME LAGS
    ax=axn[2]
    ax.hist(t_lags, bins=20, color=bg_color)
    med_lag = np.median(np.array(t_lags))
    ax.axvline(x=med_lag, color='w', linestyle='--')
    ax.set_title('Median lag: {:.2f}ms'.format(med_lag*1000))
    ax.set_xlabel("lags (sec)")

    pl.subplots_adjust(wspace=0.6)

    return fig



def compare_regr_pre_post_shift(flydf, shifted, col=[0.7]*3, markersize=5):
    turn_start_frames = shifted['turn_start_frame'].values

    scatter_kws={'s': markersize}
    fig, axn = pl.subplots(1, 3, figsize=(15, 4))
    ax=axn[0]
    sns.regplot(data=flydf.loc[turn_start_frames], x='facing_angle', y='ang_vel', ax=ax, 
                color=col, scatter_kws=scatter_kws) 
    ax.set_title('same frame')
    annotate_regr(flydf.loc[turn_start_frames], ax, x='facing_angle', y='ang_vel', fontsize=12)
    ax.set_xlabel(r'$\theta_{E}$')
    ax.set_ylabel(r'$\omega_{f}$')

    ax=axn[1]
    sns.regplot(data=shifted, x='facing_angle', y='ang_vel', ax=ax, 
                color=col, scatter_kws=scatter_kws) #markersize=markersize)
    ax.set_title('lag {} frames'.format(lag_frames))
    annotate_regr(shifted, ax, x='facing_angle', y='ang_vel', fontsize=12)
    ax.set_xlabel(r'$\theta_{E}$')
    ax.set_ylabel(r'$\omega_{f}$')

    ax=axn[2]
    sns.regplot(data=shifted.dropna(), x='facing_angle_vel_abs', y='ang_vel', ax=ax, 
                color=col, scatter_kws=scatter_kws) #markersize=markersize)
    annotate_regr(shifted.dropna(), ax, x='facing_angle_vel_abs', y='ang_vel', fontsize=12)
    ax.set_xlabel(r'$\omega_{\theta_{E}}$')
    ax.set_ylabel(r'$\omega_{f}$')

    for ax in axn:
        ax.set_box_aspect(1)

    return fig



#%%
def circular_distance(ang1, ang2):
    # efficiently computes the circular distance between two angles (Tom/Rufei)
    # should be in radians
    circdist = np.angle(np.exp(1j * ang1) / np.exp(1j * ang2))

    return circdist


def calculate_theta_error(f1, f2, xvar='pos_x', yvar='pos_y'):
    vec_between = f2[[xvar, yvar]] - f1[[xvar, yvar]]
    abs_ang = np.arctan2(vec_between[yvar], vec_between[xvar])
    th_err = circular_distance(abs_ang, f1['ori']) # already bw -np.pi, pi
    #th_err = [util.set_angle_range_to_neg_pos_pi(v) for v in th_err]
    #th_err[0] = th_err[1]
    f1['theta_error'] = th_err
    f1['theta_error_dt'] = pd.Series(np.unwrap(f1['theta_error'].interpolate().ffill().bfill())).diff() / f1['sec_diff'].mean()
    f1['theta_error_deg'] = np.rad2deg(f1['theta_error'])

    return f1


#%%
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=24)
bg_color = [0.7]*3 if plot_style=='dark' else 'w'

#%% LOAD ALL THE DATA
#savedir = '/Volumes/Julie/free-behavior-analysis/FlyTracker/38mm_dyad/processed'
#figdir = os.path.join(os.path.split(savedir)[0], 'figures', 'relative_metrics')
importlib.reload(util)

assay = '2d-projector' # '38mm-dyad'
create_new = False

minerva_base = '/Volumes/Julie'

#%%
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

# set figdir
figdir = os.path.join(os.path.split(srcdir)[0], 'predictive_coding', 'theta_error')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)
#%%
create_new = False

# get local file for aggregated data
out_fpath_local = os.path.join(localdir, 'processed.pkl')
print(out_fpath_local)

# try reading if we don't want to create a new one
if not create_new:
    if os.path.exists(out_fpath_local):
        df = pd.read_pickle(out_fpath_local)
        print("Loaded local processed data.")
    else:
        create_new = True
print(create_new)

#%%
# cycle over all the acquisition dfs in srcdir and make an aggregated df
if create_new:
    df = util.load_aggregate_data_pkl(srcdir, mat_type='df')
    print(df['species'].unique())

    #% save
    out_fpath = os.path.join(os.path.split(figdir)[0], 'processed.pkl')
    df.to_pickle(out_fpath)
    print(out_fpath)

    # save local, too
    df.to_pickle(out_fpath_local)

df['acquisition'] = ['_'.join([f for f in f.split('_')[0:-1] if 'led' not in f]) for f in df['acquisition']]
# summary of what we've got
print(df[['species', 'acquisition']].drop_duplicates().groupby('species').count())


#%% Manually calculate THETA_ERROR

# df['ori'] = -1 * df['ori']

for acq, df_ in df.groupby('acquisition'):
    f1 = df_[df_['id']==0].copy().reset_index(drop=True)
    f2 = df_[df_['id']==1].copy().reset_index(drop=True)
    f1 = calculate_theta_error(f1, f2)
    f2 = calculate_theta_error(f2, f1)
    df.loc[ (df['acquisition']==acq) & (df['id']==0), 'theta_error'] = f1['theta_error'].values
    df.loc[ (df['acquisition']==acq) & (df['id']==1), 'theta_error'] = f2['theta_error'].values


#%% plotting settings
curr_species = ['Dele', 'Dmau', 'Dmel', 'Dsant', 'Dyak']
species_cmap = sns.color_palette('colorblind', n_colors=len(curr_species))
print(curr_species)
species_palette = dict((sp, col) for sp, col in zip(curr_species, species_cmap))

#%% Set fig id
figid = srcdir  

#%% Load jaaba data
importlib.reload(util)

fname = 'free_behavior_data_mel_yak_20240403' if assay=='38mm-dyad' else None
jaaba = util.load_jaaba(assay, fname=fname)
#%
if 'filename' in jaaba.columns and 'acquisition' not in jaaba.columns:
    jaaba = jaaba.rename(columns={'filename': 'acquisition'})

print(jaaba[['species', 'acquisition']].drop_duplicates().groupby('species').count())

#%% 
# merge jaaba --------------------------------------------------------
# Merge jaaba and processed data
# --------------------------------------------------------
c_list = []
no_dfs = []
for acq, ja_ in jaaba.groupby('acquisition'):
    df_ = df[(df['acquisition']==acq) & (df['id']==0)].reset_index(drop=True)
    try:
        if len(df_)>0:
            if ja_.shape[0] < df_.shape[0]:
                last_frame = ja_['frame'].max()
                df_ = df_[df_['frame']<=last_frame]
            else:
                assert ja_.shape[0] == df_.shape[0], "Mismatch in number of flies between jaaba {} and processed data {}.".format(ja_.shape, df_.shape) 
            drop_cols = [c for c in ja_.columns if c in df_.columns]
            combined_ = pd.concat([df_, ja_.drop(columns=drop_cols)], axis=1)
            assert combined_.shape[0] == df_.shape[0], "Bad merge: {}".format(acq)
            c_list.append(combined_)
        else:
            no_dfs.append(acq)
    except Exception as e:
        print(acq)
        print(e)
        continue

ftjaaba = pd.concat(c_list, axis=0).reset_index(drop=True)
# summarize what we got
ftjaaba[['species', 'acquisition']].drop_duplicates().groupby('species').count()

#% Check missing
missing = [c for c in jaaba['acquisition'].unique() if c not in df['acquisition'].unique()]
print(len(missing))

#% Rename variables
if 'courtship' in ftjaaba.columns:
    ftjaaba = ftjaaba.rename(columns={'courtship': 'courting'})

#% Add bouts
if 'fpath' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['fpath'])
if 'name' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['name'])

#% Split into bouts of courtship
d_list = []
for acq, df_ in ftjaaba.groupby('acquisition'):
    df_ = df_.reset_index(drop=True)
    df_ = util.mat_split_courtship_bouts(df_, bout_marker='courting')
    dur_ = util.get_bout_durs(df_, bout_varname='boutnum', return_as_df=True,
                    timevar='sec')
    d_list.append(df_.merge(dur_, on=['boutnum']))
ftjaaba = pd.concat(d_list)

#%%
# Calculate a bunch of additional params
winsize=5
#print(acq)
importlib.reload(util)

#df_ = ftjaaba[ftjaaba['acquisition']==acq]
ftjaaba['targ_pos_theta_deg'] = np.rad2deg(ftjaaba['targ_pos_theta'])
ftjaaba['facing_angle_deg'] = np.rad2deg(ftjaaba['facing_angle'])
ftjaaba['rel_vel'] = np.nan

importlib.reload(util)

d_list = []
for acq, df_ in ftjaaba.groupby('acquisition'):
    # Calculate target angular vel
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='targ_pos_theta', vel_var='targ_ang_vel',
                                  time_var='sec', winsize=winsize)
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='targ_pos_theta_deg', vel_var='targ_ang_vel_deg',
                                  time_var='sec', winsize=winsize)

    # Calculate facing angle vel
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='facing_angle', vel_var='facing_angle_vel',)
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='facing_angle_deg', vel_var='facing_angle_vel_deg',)

    # Calculate "theta-error"
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='theta_error', vel_var='theta_error_vel', 
                                                     time_var='sec', winsize=winsize)
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='ori', vel_var='ang_vel_fly', 
                                                     time_var='sec', winsize=winsize)

    # Calculate relative vel
    #tmp_d = []
    #for i, d_ in df_.groupby('id'):    
    df_['rel_vel'] = df_['dist_to_other'].interpolate().diff() / df_['sec'].diff().mean()
    #    tmp_d.append(df_)
    #tmp_ = pd.concat(tmp_d)


    d_list.append(df_)
    # update ftjaaba
    #ftjaaba.loc[ftjaaba['acquisition']==acq, 'rel_vel'] = df_['rel_vel']
    #ftjaaba.loc[ftjaaba['acquisition']==acq, 'targ_ang_vel'] = df_['targ_ang_vel'].values
    #ftjaaba.loc[ftjaaba['acquisition']==acq, 'targ_ang_vel_deg'] = df_['targ_ang_vel_deg']
    #ftjaaba.loc[ftjaaba['acquisition']==acq, 'facing_angle_vel'] = df_['facing_angle_vel']
    #ftjaaba.loc[ftjaaba['acquisition']==acq, 'facing_angle_vel_deg'] = df_['facing_angle_vel_deg']  
    #ftjaaba.loc[ftjaaba['acquisition']==acq, 'theta_error_vel'] = df_['theta_error_vel']

ftjaaba = pd.concat(d_list)

    # 
#% and get abs values
ftjaaba['rel_vel_abs'] = np.abs(ftjaaba['rel_vel']) 
ftjaaba['targ_ang_vel_abs'] = np.abs(ftjaaba['targ_ang_vel'])
ftjaaba['targ_pos_theta_abs'] = np.abs(ftjaaba['targ_pos_theta'])
ftjaaba['targ_ang_size_deg'] = np.rad2deg(ftjaaba['targ_ang_size'])
ftjaaba['targ_ang_vel_deg_abs'] = np.abs(ftjaaba['targ_ang_vel_deg'])

ftjaaba['facing_angle_deg'] = np.rad2deg(ftjaaba['facing_angle'])
ftjaaba['ang_vel_abs'] = np.abs(ftjaaba['ang_vel'])
if 'dovas' in ftjaaba.columns:
    ftjaaba['dovas_deg'] = np.rad2deg(ftjaaba['dovas'])

ftjaaba['facing_angle_vel_abs'] = np.abs(ftjaaba['facing_angle_vel'])
ftjaaba['facing_angle_vel_deg_abs'] = np.abs(ftjaaba['facing_angle_vel_deg'])

ftjaaba['theta_error_deg'] = np.rad2deg(ftjaaba['theta_error'])
ftjaaba['theta_error_vel_deg'] = np.rad2deg(ftjaaba['theta_error_vel'])
ftjaaba['theta_error_abs'] = np.abs(ftjaaba['theta_error'])

if 'good_frames' not in ftjaaba.columns:
    ftjaaba['good_frames'] = 1

#%%
if 'filename' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['filename'])

#%% get means by BOUT
groupcols = [ 'species', 'acquisition', 'boutnum']

min_pos_theta = np.deg2rad(-160)
max_pos_theta = np.deg2rad(160)
min_dist_to_other = 1

#%% ------------------------------
# JAABA THRESHOLDS
# ------------------------------
if ftjaaba['chasing'].max() == 1:
    jaaba_thresh_dict = {'orienting': 0, 
                        'chasing': 0,
                        'singing': 0}
else:
    jaaba_thresh_dict = {'orienting': 10,
                        'chasing': 10,
                        'singing': 5}
print(jaaba_thresh_dict)

#%%

ftjaaba['ang_vel_deg'] = np.rad2deg(ftjaaba['ang_vel'])
ftjaaba['ang_vel_abs_shifted'] = ftjaaba.groupby('acquisition')['ang_vel_abs'].shift(-2)
ftjaaba['ang_vel_fly_shifted'] = ftjaaba.groupby('acquisition')['ang_vel_fly'].shift(-2)

# binarize behavs
ftjaaba = util.binarize_behaviors(ftjaaba, jaaba_thresh_dict=jaaba_thresh_dict)

#%%


#%% =====================================================
# subdivide --------------------------------------------------------
# split into small bouts
# --------------------------------------------------------
bout_dur = 0.25 
min_boutdur = 0.25
min_dist_to_other = 2
#%
filtdf = ftjaaba[(ftjaaba['id']==0)
                #& (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                #& (ftjaaba['targ_pos_theta']<=max_pos_theta)
                & (ftjaaba['dist_to_other']>=min_dist_to_other)
                & (ftjaaba['boutdur']>=min_boutdur)
                & (ftjaaba['good_frames']==1)
                & (ftjaaba['led_level']>0)
                ].copy() #.reset_index(drop=True)

# subdivide into smaller boutsa
# bout_dur = 0.5
filtdf = util.subdivide_into_bouts(filtdf, bout_dur=bout_dur)

#%%

# Get mean value of small bouts
if 'strain' in filtdf.columns:
    filtdf = filtdf.drop(columns=['strain'])
#%
meanbouts = filtdf.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()
meanbouts.head()

cmap='viridis'
stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)

# find the closest matching value to one of the keys in stimhz_palette:
meanbouts['stim_hz'] = meanbouts['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))   

#%% ------------------------------------------------
# ANG_VEL vs. THETA_ERROR
# -------------------------------------------------
#import copy
#
#def regplot(
#    *args,
#    ax=None,
#    line_kws=None,
#    marker=None,
#    scatter_kws=None,
#    **kwargs
#):
#    # this is the class that `sns.regplot` uses
#    plotter = sns.regression._RegressionPlotter(*args, **kwargs)
#
#    # this is essentially the code from `sns.regplot`
#    #ax = kwargs.get("ax", None)
#    if ax is None:
#        ax = pl.gca()
#
#    scatter_kws = {} if scatter_kws is None else copy.copy(scatter_kws)
#    scatter_kws["marker"] = marker
#    line_kws = {} if line_kws is None else copy.copy(line_kws)
#
#    plotter.plot(ax, scatter_kws, line_kws)
#
#    # unfortunately the regression results aren't stored, so we rerun
#    grid, yhat, err_bands = plotter.fit_regression(pl.gca())
#
#    # also unfortunately, this doesn't return the parameters, so we infer them
#    slope = (yhat[-1] - yhat[0]) / (grid[-1] - grid[0])
#    intercept = yhat[0] - slope * grid[0]
#    return slope, intercept
import regplot as rpl


#%%

xvar = 'theta_error'
yvar = 'ang_vel_fly'

# Set palettes
cmap='viridis'
stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)
vmin = min(list(stimhz_palette.keys()))
vmax = max(list(stimhz_palette.keys()))

# Get CHASING bouts 
behav = 'chasing'
min_frac_bout = 0.5
chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>min_frac_bout].copy()

# SCATTERPLOT:  ANG_VEL vs. THETA_ERROR -- color coded by STIM_HZ
fig, axn = pl.subplots(1, 2, sharex=True, sharey=True)
for ai, (sp, df_) in enumerate(chase_[chase_['stim_hz']>0].groupby('species')):
    ax = axn[ai]
    # plot scatter
    sns.scatterplot(data=df_, x=xvar, y=yvar, ax=ax,
                 hue='stim_hz', palette=stimhz_palette, legend=0, edgecolor='none', alpha=0.7)
    ax.set_title(sp)
    # do fit
    res = rpl.regplot(data=df_, ax=ax, x=xvar, y=yvar,
                color='w', scatter=False) #, ax=ax)
    # res.params: [intercept, slope]
    ax.set_box_aspect(1)
    fit_str = 'OLS: y = {:.2f}x + {:.2f}'.format(res.params[1], res.params[0])
    print(fit_str) #lope, intercept)
    ax.text(0.05, 0.85, fit_str, fontsize=8, transform=ax.transAxes)

    # set xlabel to be theta subscript E
    ax.set_xlabel(r'$\theta_{E}$')
    ax.set_ylabel('$\omega_{f}$ (deg/s)')
    annotate_regr(df_.dropna(), ax, x=xvar, y=yvar, fontsize=8)

putil.colorbar_from_mappable(ax, cmap=cmap, norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                             axes=[0.92, 0.3, 0.01, 0.4], hue_title='stim. freq. (Hz)', fontsize=18)

fig.suptitle('{} bouts, where min fract of bout >= {:.2f}'.format(behav, min_frac_bout))

putil.label_figure(fig, figid)
figname = 'sct_{}_v_{}_stimhz_mel-v-yak_min-frac-bout-{}'.format(yvar, xvar, min_frac_bout)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir, figname)

#%% quick test -- compare to delay, 2 frames


xvar = 'theta_error'
yvar = 'ang_vel_fly_shifted'
nframes_lag = 2

chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>min_frac_bout].copy()

# shift ang_vel back 2 frames
# chase_['ang_vel_abs_shifted'] = chase_.groupby('acquisition')['ang_vel_abs'].shift(-2)
fig, axn = pl.subplots(1, 2, sharex=True, sharey=True)
for ai, (sp, df_) in enumerate(chase_[chase_['stim_hz']>0].groupby('species')):
    ax = axn[ai]
    # plot scatter
    sns.scatterplot(data=df_, x=xvar, y=yvar, ax=ax,
                 hue='stim_hz', palette=stimhz_palette, legend=0, edgecolor='none', alpha=0.7)
    ax.set_title(sp)
    # do fit
    res = rpl.regplot(data=df_, ax=ax, x=xvar, y=yvar, 
                color='w', scatter=False) #, ax=ax)

    # res.params: [intercept, slope]
    fit_str = 'OLS: y = {:.2f}x + {:.2f}'.format(res.params[1], res.params[0])
    #print() #lope, intercept)
    ax.text(0.05, 0.85, fit_str, fontsize=8, transform=ax.transAxes)
    ax.set_box_aspect(1)
    
    # set xlabel to be theta subscript E
    ax.set_xlabel(r'$\theta_{E}$')
    ax.set_ylabel('$\omega_{f}$ (deg/s)')
    annotate_regr(df_.dropna(), ax, x=xvar, y=yvar, fontsize=8)

fig.suptitle('{} bouts, where min fract of bout >= {:.2f}, shifted'.format(behav, min_frac_bout))

putil.label_figure(fig, figid)
figname = 'sct_{}_v_{}_SHIFT-{}frames_stimhz_mel-v-yak_min-frac-bout-{}'.format(yvar, xvar, nframes_lag, min_frac_bout)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir, figname)



#%% Fit REGR to each stim_hz level

xvar = 'theta_error' #'facing_angle_vel_deg_abs'
yvar = 'ang_vel_fly_shifted' #'ang_vel_abs'

show_scatter=True

behav = 'chasing'
min_frac_bout = 0.25
chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>min_frac_bout].copy()


if 'vel' in xvar:
    xlabel = r'$\omega_{\theta}$'
else:
    xlabel = r'$\theta_{E}$'
ylabel = '$\omega_{f}$'


plot_type = 'regr-sct' if show_scatter else 'regr'
for ai, (sp, df_) in enumerate(chase_[chase_['stim_hz']>0].groupby('species')):
    g = sns.lmplot(data=df_, x=xvar, y=yvar, 
                 hue='stim_hz', palette=stimhz_palette, legend=0, scatter=show_scatter)
    g.fig.axes[0].set_title( '{}. min_frac={:.2f}'.format(sp, min_frac_bout) , fontsize=8)

    g.fig.axes[0].set_box_aspect(1)
    
    # set xlabel to be theta subscript E
    g.fig.axes[0].set_xlabel(xlabel) #r'$\theta_{E}$')
    g.fig.axes[0].set_ylabel(ylabel) #'$\omega_{f}$ (deg/s)')

    if xvar == 'theta_error':
        if show_scatter is False:
            pl.xlim([-2, 2])
            pl.ylim([-10, 10])
        else:
            g.fig.axes[0].set_xlim([-2, 2])
        g.fig.axes[0].set_ylim([-15, 15])
        #pl.xlim([-2, 2])
       # pl.ylim([15, 15])
    else:
        pl.xlim([0, 80])
        pl.ylim([0, 10])

    putil.colorbar_from_mappable(g.fig.axes[0], cmap=cmap, norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                             axes=[0.96, 0.3, 0.01, 0.4])
    pl.subplots_adjust(top=0.9)

    putil.label_figure(g.fig, figid)
    figname = '{}_{}_v_{}_stimhz_{}_min-frac-bout-{}'.format(plot_type, yvar, xvar, sp, min_frac_bout)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    print(figdir, figname)


#%% IS THIS FAKE? look at 1 animal
# facing angle vs. ang vel for FRAMES v BOUTS


acqs = meanbouts['acquisition'].unique()
#acq = acqs[3] #'20240212-1215_fly3_Dmel_sP1-ChR_3do_sh_8x8'a
acq = '20240212-1215_fly3_Dmel_sP1-ChR_3do_sh_8x8'


xvar = 'theta_error'
yvar = 'ang_vel_fly_shifted'


behav = 'chasing'
# Compare FRAMES vs. MINI-BOUTS
frames_ = filtdf[ (filtdf['acquisition']==acq) & (filtdf['{}_binary'.format(behav)]>0)].copy()
mean_ = frames_.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()
mean_['stim_hz'] = mean_['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))
mean1 = meanbouts[ (meanbouts['acquisition']==acq) & (meanbouts['{}_binary'.format(behav)]>0)].copy()

# SCATTER for frames or for bouts
fig, axn =pl.subplots(1, 2, figsize=(10,4))
ax=axn[0]
sns.scatterplot(data=mean_, x=xvar, y=yvar, ax=ax, hue='stim_hz', palette=stimhz_palette,
                legend=0)
ax.set_title('{:.2f}s bouts'.format(bout_dur))
ax=axn[1]
sns.scatterplot(data=frames_, x=xvar, y=yvar, ax=ax, hue='stim_hz', palette=stimhz_palette,
                legend=0)
ax.set_title('frames')

pl.subplots_adjust(top=0.8)
fig.suptitle(acq, fontsize=18)

pl.subplots_adjust(wspace=0.5)

#%% same, but split by stim_Hz

n_stim = filtdf['stim_hz'].nunique() # no zero

do_bouts = True 
xvar = 'theta_error'
yvar = 'ang_vel_fly_shifted'


plotd_ = mean1.copy() if do_bouts else frames_.copy()
data_type = 'BOUTS' if do_bouts else 'FRAMES'

if 'vel' in xvar:
    xlabel = r'$\omega_{\theta}$'
else:
    xlabel = r'$\theta_{E}$'
ylabel = '$\omega_{f}$'

all_stims = sorted(list(stimhz_palette.keys()))
start_i = all_stims.index(plotd_['stim_hz'].min())

fig, axn = pl.subplots(n_stim, 1, sharex=True, sharey=True, figsize=(4, n_stim*2.5))
for i, (stim, sd_) in enumerate(plotd_.groupby('stim_hz')):  
    ax=axn[i+start_i]
    ax.set_title(stim, loc='left', fontsize=12)
    sns.scatterplot(data=sd_, x=xvar, y=yvar, ax=ax, hue='stim_hz', palette=stimhz_palette,
                legend=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

axn[-1].set_xlabel(xlabel)

pl.subplots_adjust(hspace=0.5, bottom=0.2)

putil.label_figure(fig, acq)

fig.suptitle('{:.2f}s {}'.format(bout_dur, data_type), fontsize=24)



#%% 
# --------------------------------------------------------
# Look at time-courses, 1 fly
# --------------------------------------------------------

all_acqs = ftjaaba[ftjaaba['led_level']>0]['acquisition'].unique()

acq = '20240216-1434_fly3_Dmel_sP1-ChR_2do_sh_4x4'
#acq = '20240222-1611_fly7_Dmel_sP1-ChR_2do_sh_8x8'

#yaks = [a for a in all_acqs if 'Dyak' in a]
#print(len(yaks))
#acq = yaks[50]
print(acq)

#%%
behav = 'chasing'
flydf = ftjaaba[(ftjaaba['acquisition']==acq) & (ftjaaba['id']==0)
              & (ftjaaba['good_frames']==1)].copy()
print(flydf.shape)

# get frames
frames_ = filtdf[ (filtdf['acquisition']==acq) & (filtdf['{}_binary'.format(behav)]>0)].copy()
# get means
mean_ = frames_.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()
mean_['stim_hz'] = mean_['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))


#%% ----------------------------
# Does angular vel. INCREASE with stim_hz? with theta-error?
# ----------------------------
flydf['ang_acc'] = flydf['ang_vel'].diff() / flydf['sec_diff'].mean()
flydf['facing_angle_acc'] = flydf['facing_angle_vel_deg_abs'].diff() / flydf['sec_diff'].mean()

flydf['ang_acc_fly'] = flydf['ang_vel_fly'].diff() / flydf['sec_diff'].mean()
flydf['theta_error_acc'] = flydf['theta_error_vel'].diff() / flydf['sec_diff'].mean()
flydf['ang_vel_fly_abs'] = np.abs(flydf['ang_vel_fly'])
flydf['ang_vel_fly_deg'] = np.rad2deg(flydf['ang_vel_fly'])
flydf['ang_vel_fly_abs_deg'] = np.rad2deg(flydf['ang_vel_fly_abs'])

#%
fig, axn = pl.subplots(1, 2)
ax=axn[0]
flydf['ang_vel'].plot(ax=ax)
ax.set_title('ang_vel')
ax=axn[1]
flydf['ang_vel_fly_abs_deg'].plot(ax=ax)
ax.set_title('ang_vel_fly')

#%% Why ang_vel_fly so crazy

fig, ax =pl.subplots()
sns.scatterplot(data=flydf, x='ang_vel', y='ang_vel_fly', ax=ax)



#%%
# Look at moments of high ang vel.
#jmin_ang_vel = 4.36
min_ang_vel = 5
min_vel = 15
min_dist_to_other = 15
min_facing_angle = np.deg2rad(90)
passdf = flydf[ (flydf['stim_hz']>0) 
               #& (flydf['ang_vel']>=min_ang_vel)
               & (flydf['ang_vel_fly_abs']>=min_ang_vel)
               #& (flydf['chasing']>8)
               #& (flydf['chasing']>10)
               & (flydf['good_frames']==1)
               & (flydf['dist_to_other']<=min_dist_to_other)
               & (flydf['facing_angle']<min_facing_angle)
               & (flydf['vel']>min_vel)
               ]

# get start/stop indices of consecutive rows
high_ang_vel_bouts = util.get_indices_of_consecutive_rows(passdf)
print(len(high_ang_vel_bouts))

# filter duration?
min_bout_len = 4/60 #min 2 frames
print(min_bout_len)
fps = 60.
incl_bouts = util.filter_bouts_by_frame_duration(high_ang_vel_bouts, min_bout_len, fps)
print("{} of {} bouts pass min dur {}sec".format(len(incl_bouts), 
                                        len(high_ang_vel_bouts), min_bout_len))

# find turn starts
turn_start_frames = [c[0] for c in incl_bouts] #high_ang_vel_bouts]

#%% 
importlib.reload(putil)
# PLOT TIME COURSES
nframes_win = 2*fps

# parse bout
start_ix = turn_start_frames[-1] - nframes_win #12049 #18320 #turn_start_frames[4] #- nframes_win
stop_ix = turn_start_frames[-1] + nframes_win-60 #*2

plotdf = flydf.loc[start_ix:stop_ix]

# threshold for angular acceleration 
ang_acc_thr = 100
ang_peaks = plotdf[plotdf['ang_acc'] > ang_acc_thr ]
high_ang_bouts = util.get_indices_of_consecutive_rows(ang_peaks)

# find turn starts
high_ang_start_frames = [c[0] for c in high_ang_bouts] #onsec_bouts]
print(len(high_ang_start_frames))

#% PLOT.
targ_color = 'r'
fly_color = 'cornflowerblue'
accel_color = [0.6]*3

xvar = 'sec'
var1 = 'theta_error'
vel_var1 = 'theta_error_vel'
var2 = 'ang_vel_fly'
acc_var2 = 'ang_acc_fly'

fig, axn = pl.subplots(3, 1, figsize=(8,6), sharex=True)

ax=axn[0]
ax.plot(plotdf[xvar], plotdf[var1], targ_color)
# color y-axis spine and ticks red
ax.set_ylabel(r'$\theta_{E}$') #, color=targ_color) #r'$\theta_{E}$'
putil.change_spine_color(ax, targ_color, 'left')
ax.axhline(y=0, color=targ_color, linestyle='--', lw=0.5)

ax2 = ax.twinx()
ax2.plot(plotdf[xvar], plotdf[var2], fly_color)
# color y-axis spine and ticks blue
#ax2.spines['right'].set_color(fly_color)
ax2.set_ylabel(r'$\omega_{f}$') #, color=fly_color)
putil.change_spine_color(ax2, fly_color, 'right')
ax2.axhline(y=min_ang_vel, color='w', linestyle='--', lw=0.5)
ax2.axhline(y=-1*min_ang_vel, color='w', linestyle='--', lw=0.5)

ax=axn[1]
ax.plot(plotdf[xvar], plotdf[vel_var1], targ_color)
ax.set_ylabel(r'$\omega_{\theta_{E}}$') # color=targ_color)
putil.change_spine_color(ax, targ_color, 'left')
ax2 = ax.twinx()
ax2.plot(plotdf[xvar], plotdf[var2], fly_color)
ax2.set_ylabel(r'$\omega_{f}$') # color=fly_color)
putil.change_spine_color(ax2, fly_color, 'right')

ax=axn[2]
ax.plot(plotdf[xvar], plotdf[acc_var2], fly_color)
ax.set_ylabel('ang_acc') #color=fly_color)

for i, f in enumerate(high_ang_start_frames):
    print(f)
    ax.plot(plotdf.loc[plotdf['frame']==f][xvar], plotdf.loc[plotdf['frame']==f]['ang_acc'], 
            color = 'w', marker='o', markersize=5)

axn[0].set_title('{}, frames: {}-{}'.format(acq, start_ix, stop_ix), loc='left', 
                 fontsize=8)

#%% PSTH of turn bouts:  FACING_ANGLE and ANG_VEL

plotdf = flydf.copy()
nframes_win = 0.1*fps

xvar1 = 'theta_error' #'facing_angle'
col1 = 'r'

xvar2 = 'ang_vel_fly' #'ang_vel'
col2 = 'cornflowerblue'

secs = []
v1 = []
v2 = []
t_lags = []
fig, ax1 = pl.subplots()
ax2 = ax1.twinx()
for i, ix in enumerate(high_ang_start_frames):
    start_ = ix - nframes_win
    stop_ = ix + nframes_win
    t_onset = plotdf.loc[ix]['sec']
    d_  = plotdf.loc[start_:stop_].copy()

    d_['rel_sec'] = d_['sec'] - t_onset
    for xv, col, ax in zip([xvar1, xvar2], [col1, col2], [ax1, ax2]):
        sns.lineplot(data=d_, x='rel_sec', y=xv, ax=ax,
                     color=col, lw=1, alpha=0.75)

    secs.append(d_['rel_sec'].values)
    v1.append(d_[xvar1].values)
    v2.append(d_[xvar2].values) 
    # cross corr
    correlation, lags, lag_frames, t_lag = cross_correlation_lag(d_[xvar2], d_[xvar1], fps=60)
 
    t_lags.append(t_lag)
ax.axvline(x=0, color=bg_color, linestyle='--')

mean_v1 = np.array(v1).mean(axis=0)
mean_v2 = np.array(v2).mean(axis=0)
mean_sec = np.array(secs).mean(axis=0)
for xv, col, ax, mean_ in zip([xvar1, xvar2], [col1, col2], [ax1, ax2], [mean_v1, mean_v2]):
    #mean_ = plotdf.groupby('rel_sec')[xv].mean().reset_index() #.plot(x='rel_sec', y=xvar1, ax=ax, legend=0)
    ax.plot(mean_sec, mean_, color=col, lw=3)
    #sns.lineplot (data=mean_, x='rel_sec', y=xv, ax=ax, color=col, lw=3)
#ax.set_box_aspect(1)
ax1.set_ylabel(r'$\theta_{E}$')
ax2.set_ylabel(r'$\omega_{f}$')
putil.change_spine_color(ax1, col1, 'left')
putil.change_spine_color(ax2, col2, 'right')
ax.set_title('{}, frames: {}-{}'.format(acq, start_ix, stop_ix), loc='left', 
                 fontsize=8)

#%%  Cross Correlate vars

#x = mean_v1
#y = mean_v2

correlation, lags, lag_frames, t_lag = cross_correlation_lag(mean_v2, mean_v1, fps=60)
#correlation = signal.correlate(x-np.mean(x), y - np.mean(y), mode="full")
#lags = signal.correlation_lags(len(x), len(y), mode="full")
#lag_frames = lags[np.argmax(correlation)] #lags[np.argmax(abs(correlation))]
#t_lag = lag_frames / fps 

# plot means
fig, axn =pl.subplots(1, 3, figsize=(16,4))
ax1 = axn[0]
ax2 = ax1.twinx()
for xv, col, mean_, ax in zip([xvar1, xvar2], [col1, col2], [mean_v1, mean_v2], [ax1, ax2]):
    ax.plot(mean_sec, mean_, color=col, lw=3)
ax1.set_ylabel(r'$\theta_{E}$')
ax2.set_ylabel(r'$\omega_{f}$')
putil.change_spine_color(ax1, col1, 'left')
putil.change_spine_color(ax2, col2, 'right')
ax1.set_xlabel('time (sec)')
ax1.set_title('mean bout vals')

# plot cross-corr
ax=axn[1]
ax.plot(lags, correlation, c='w')
max1 = np.argmax(mean_v1)
max2 = np.argmax(mean_v2)
print(max1, max2)
peak_diff_sec = mean_sec[max2] - mean_sec[max1]
ax.set_ylabel('cross correlation')
ax.set_xlabel('lag (frames)')
ax.set_title('Peak diff: {:.2f}sec\nx-corr peak: {:.2f}msec'.format(peak_diff_sec, t_lag*1E3))

# plot distribution of time lags
ax=axn[2]
ax.hist(t_lags, color=bg_color)
ax.set_title('Median lag: {:.2f}msec'.format(np.median(t_lags)*1E3))
ax.set_xlabel("lags (sec)")

pl.subplots_adjust(wspace=0.7)

#%%

# Just use ANG_ACC

flydf = ftjaaba[(ftjaaba['acquisition']==acq) & (ftjaaba['id']==0)].copy()
flydf['ang_acc'] = flydf['ang_vel'].diff() / flydf['sec_diff'].mean()
flydf['facing_angle_acc'] = flydf['facing_angle_vel_deg_abs'].diff() / flydf['sec_diff'].mean()

#%% Look at moments of high ang acc.


#%%
min_vel = 10
min_dist_to_other = 20 #15
min_facing_angle = np.deg2rad(90)
min_acc = 150

turnbouts = get_turn_bouts(flydf, min_acc=min_acc, min_vel=min_vel, min_dist_to_other=min_dist_to_other,
                           min_facing_angle=min_facing_angle)

#%% 
v1 = 'facing_angle'
v2 = 'ang_vel'
xcorr, lags, t_lags = cross_corr_each_bout(turnbouts, v1=v1, v2=v2)

#%% 
# Trio: Average aligned turn bouts, cross-correlation, distribution of time lags
fig = plot_cross_corr_results(turnbouts, xcorr, lags, t_lags, v1=v1, v2=v2, col1=col1, col2=col2)

fig.text(0.1, 0.95,  '{}, all turns ang_acc > {:.2f}'.format(acq, min_acc), 
                    fontsize=8)
# save
figname = 'mean-turn-bouts-xcorr-tlags_acc-thr-{}_{}'.format(min_acc, acq)
putil.label_figure(fig, acq)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir, figname)


#%% Aligned INDIVIUAL TURNS, 2 subplots of theta_error and ang_vel
fig = plot_individual_turns(turnbouts, v1='facing_angle', v2='ang_vel')
# save
figname = 'turn-bouts_acc-thr-{}_{}'.format(min_acc, acq)
putil.label_figure(fig, acq)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir, figname)

#%% SHIFT frames
med_lag = np.median(np.array(t_lags))
lag_frames = med_lag * fps
print(lag_frames)

d_list = []
for f in turn_start_frames:
    fly_ = flydf.loc[f][['ang_vel', 'ang_acc']]
    targ_ = flydf.loc[f-lag_frames][['facing_angle', 'facing_angle_vel', 'facing_angle_vel_abs', 'facing_angle_acc']]
    d_ = pd.concat([fly_, targ_], axis=0)
    d_['turn_start_frame'] = f
    d_list.append(d_)
shifted = pd.concat(d_list, axis=1).T
shifted = shifted.astype(float)

#%% Plot correlations between theta_error and ang_vel for same frame and lagged frame
col = bg_color
markersize=5
fig = compare_regr_pre_post_shift(flydf, shifted, col=col, markersize=markersize)
# save
putil.label_figure(fig, acq)
figname = 'lag-shifted-regr_acc-thr-{}_{}'.format(min_acc, acq)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir, figname)

#%% DO PAIRPLOT
cols = [c for c in shifted.columns if c != 'turn_start_frame']
with sns.plotting_context(rc={"axes.labelsize":20}):
    g = sns.pairplot(data=shifted[cols], diag_kws={'color': bg_color}, plot_kws={'color': bg_color, 's': 5})

putil.label_figure(g.fig, acq)
figname = 'pairplot_lag-shifted_acc-thr-{}_{}'.format(min_acc, acq)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

#%% check model fits
y = 'ang_acc'
x = 'facing_angle'
scatter_kws = {'s': 5}

fig, axn = pl.subplots(1,2, figsize=(10,4))
for ax, x, y in zip(axn.flat, ['facing_angle', 'facing_angle_vel'], ['ang_vel', 'ang_acc']):
    sns.regplot(data=shifted.dropna(), x=x, y=y, ax=ax, 
            color=col, scatter_kws=scatter_kws) #markersize=markersize)

    model = sm.OLS(shifted.dropna()[y], sm.add_constant(shifted.dropna()[x]))
    results = model.fit()

    print("REGR. RESULTS: x={}, y={} ----------------------- ".format(x, y))
    print(results.summary())
pl.subplots_adjust(wspace=0.5)

putil.label_figure(fig, acq)
figname = 'regr_vel-v-acc_lag-shifted_acc-thr-{}_{}'.format(min_acc, acq)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

#%% Check for SMALL theta errors

small_theta_error = np.deg2rad(15)
small_errors = shifted[shifted['facing_angle'] < small_theta_error ].copy()

fig, axn = pl.subplots(1, 2, figsize=(15, 4))
ax=axn[0]
sns.regplot(data=small_errors, x='facing_angle', y='ang_vel', ax=ax)
ax.set_title('lag {} frames'.format(lag_frames))
ax=axn[1]
sns.regplot(data=small_errors, x='facing_angle_vel_abs', y='ang_vel', ax=ax)


#%% Steady state conditions
min_ang_vel = 4.36
min_vel = 20 # 20
min_dist_to_other = 15
min_facing_angle = np.deg2rad(90)
min_acc = 100
chasedf = flydf[ (flydf['stim_hz']>0) 
               #& (flydf['ang_acc']>=min_acc)
               #& (flydf['chasing']>8)
               #& (flydf['chasing']>10)
               & (flydf['good_frames']==1)
               & (flydf['dist_to_other']<=min_dist_to_other)
               & (flydf['facing_angle'] <= min_facing_angle)
               & (flydf['vel']>min_vel)
               ].copy()

# get start/stop indices of consecutive rows
chase_bout_ixs = util.get_indices_of_consecutive_rows(chasedf)

# filter by duration
min_bout_len = 0.1
fps = 60.
incl_bouts = util.filter_bouts_by_frame_duration(chase_bout_ixs, min_bout_len, fps)
print("{} of {} bouts pass min dur {}sec".format(len(incl_bouts), len(chase_bout_ixs), min_bout_len))

min_theta_diff = np.deg2rad(5)
min_stable_frames = 5
stable_bouts = []
for (start, stop) in incl_bouts:
    thetas = flydf[flydf['facing_angle']>=0.05].loc[start:stop]['facing_angle'].diff().abs()

    stable_ixs = pd.DataFrame( thetas[thetas <= min_theta_diff]) #.index.tolist()
    if len(stable_ixs) == 0:
        continue
    # get start indices of consecutive frames
    consec_bouts = util.get_indices_of_consecutive_rows(stable_ixs)
    pass_ = [c for i, c in enumerate(consec_bouts) if c[1]-c[0]>=min_stable_frames]
    stable_bouts.extend(pass_)
print("Found {} stable bouts".format(len(stable_bouts)))

#%% Get average values during steady state conditions
d_list = []
for i, (start, stop) in enumerate(stable_bouts):
    avg_theta = flydf.loc[start:stop]['facing_angle'].mean()
    avg_ang = flydf.loc[start+2:stop+2]['ang_vel'].mean()
    d_list.append(pd.DataFrame({'facing_angle': avg_theta, 'ang_vel': avg_ang}, index=[i]))
    #d_list.append(flydf.loc[start:stop])

stabledf = pd.concat(d_list)

fig, ax =pl.subplots()
ax.plot(flydf.loc[start:stop]['facing_angle'].diff())
ax.axhline(y=min_theta_diff, color='w', linestyle='--')

fig, ax =pl.subplots()
sns.regplot(data=stabledf, x='facing_angle', y='ang_vel', ax=ax)

#%%
# PLOT MEAN + SEM of aligned turn bout traces
fig, axn =pl.subplots(1, 3, figsize=(15, 4))

v1 = 'facing_angle'
v2 = 'ang_vel'
ax1 = axn[0]
ax2 = ax1.twinx()
sns.lineplot(data=turnbouts, x='rel_sec', y=v1, ax=ax1, lw=0.5, color=col1)
sns.lineplot(data=turnbouts, x='rel_sec', y=v2, ax=ax2, lw=0.5, color=col2)
for ax, sp, lb, col in zip([ax1, ax2], ['left', 'right'], [r'$\theta_{E}$', r'$\omega_{f}$'], [col1, col2]):
    ax.set_ylabel(lb)
    putil.change_spine_color(ax, col, sp)
ax1.axvline(x=0, color=bg_color, linestyle='--')

v1 = 'facing_angle'
v2 = 'vel'
ax1 = axn[1]
ax2 = ax1.twinx()
sns.lineplot(data=turnbouts, x='rel_sec', y=v1, ax=ax1, lw=0.5, color=col1)
sns.lineplot(data=turnbouts, x='rel_sec', y=v2, ax=ax2, lw=0.5, color=col2)
for ax, sp, lb, col in zip([ax1, ax2], ['left', 'right'], [r'$\theta_{E}$', r'$v_{f}$'], [col1, col2]):
    ax.set_ylabel(lb)
    putil.change_spine_color(ax, col, sp)
ax1.axvline(x=0, color=bg_color, linestyle='--')


ax=axn[2]
sns.scatterplot(data=flydf[flydf['chasing']>10], x='ang_vel', y='vel', ax=ax, color=bg_color, s=5)

pl.subplots_adjust(wspace=0.7)






#%% ---------------------------------------------
# PLOT ALLO vs. EGO

import parallel_pursuit as pp
importlib.reload(pp)

behav = 'chasing'
min_frac_bout = 0.8
do_bouts=False

markersize=5
huevar='stim_hz'
cmap='viridis'
plot_com=True

xvar= 'facing_angle'
yvar = 'targ_pos_radius'

is_flytracker=True
sign = -1 if is_flytracker else 1

chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>=min_frac_bout].copy()

data_type = 'BOUTS' if do_bouts else 'FRAMES' 
if not do_bouts:
    # plot frames
    palette_dict = putil.get_palette_dict(filtdf[filtdf['stim_hz']>0], huevar, cmap='viridis')
    plotdf = filtdf[( filtdf['chasing']>jaaba_thresh_dict['chasing'])
              & ( filtdf['stim_hz']>0 )].copy()
else:
    palette_dict = stimhz_palette
    plotdf = chase_[chase_['stim_hz']>0].copy()

plotdf['facing_angle'] = sign * plotdf['facing_angle']


vmin = min(list(palette_dict.keys()))
vmax = max(list(palette_dict.keys()))
hue_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

for sp, p_ in plotdf.groupby('species'):
    fig, axn = pl.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True,
                                subplot_kw={'projection': 'polar'})
    pp.plot_allo_vs_egocentric_pos(p_, axn=axn, xvar=xvar, yvar=yvar, huevar=huevar,
                                palette_dict=palette_dict, hue_norm=hue_norm, markersize=5,
                                com_markersize=40, com_lw=1)
    for ax in axn:
        yl = ax.get_yticklabels()
        ax.set_yticklabels([v if i%2==0 else '' for i, v in enumerate(yl)])
    pl.subplots_adjust(wspace=0.6, top=0.8, right=0.8)

    putil.label_figure(fig, figid)

    fig.suptitle('{}: {} {}, where min fract of bout >= {:.2f}'.format(sp, behav, data_type, min_frac_bout))

    figname = 'allo-v-ego-{}-{}_{}-v-{}_min-frac-bout-{}_{}'.format(behav, data_type, xvar, yvar, min_frac_bout, sp)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

    print(figdir, figname)



#%% Plot as 2D hists

xvar = 'facing_angle_deg'
yvar = 'ang_vel_abs' #'ang_vel_abs'

plot_behavs = ['orienting', 'chasing', 'singing']
fig, axn =pl.subplots(2, len(plot_behavs), figsize=(10,8), sharex=True, sharey=True)

nbins=25
x_bins = np.linspace(0, meanbouts[xvar].max(), nbins, endpoint=False)
y_bins = np.linspace(0, meanbouts[yvar].max(), nbins, endpoint=False)

meanbouts['x_bin'] = pd.cut(meanbouts[xvar], bins=x_bins, labels=x_bins[0:-1])
meanbouts['y_bin'] = pd.cut(meanbouts[yvar], bins=y_bins, labels=y_bins[0:-1])


vmin = 0
vmax = 1.0
cmap='magma'
for ri, (sp, means_) in enumerate(meanbouts.groupby('species')):

    #xmin, xmax = means_[xvar].min(), means_[xvar].max()
    #ymin, ymax = means_[yvar].min(), means_[yvar].max()
     
    for ci, behav in enumerate(plot_behavs):
        ax = axn[ri, ci]
        ax.set_title(behav)
        
        probs = means_.groupby(['x_bin', 'y_bin'])['{}_binary'.format(behav)].mean().reset_index()
        arr_ = pd.pivot_table(probs, columns='x_bin', index='y_bin').values

        ax.imshow(arr_, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        print(np.nanmax(arr_))        
        
        ax.set_xticks(np.arange(len(x_bins)))
        ax.set_xticklabels([np.round(v) if i%10==0 else '' for i, v in enumerate(x_bins)]) #probs['x_bin'])
        ax.set_yticks(np.arange(len(y_bins)))
        ax.set_yticklabels([np.round(v) if i%10==0 else '' for i, v in enumerate(y_bins)]) #probs['x_bin'])

        # Create 2D histogram using np.histogram2d()
        #hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=25)

        # Normalize
        #total_counts = np.sum(hist)
        #hist_normalized = hist / total_counts

        # Create meshgrid from edges
        #x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)
        #pcm = ax.pcolormesh(x_mesh, y_mesh, hist_normalized.T, cmap='magma') #, vmin=0, vmax=0.05) #, vmin=0, vmax=0.05)

        if ci==0:
            ax.set_title('{}: {}'.format(sp, behav))
        # ax.set_xlim([xmin, xmax])
        #ax.set_ylim([ymin, 300])

        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)
        ax.set_box_aspect(1)

putil.colorbar_from_mappable(ax, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                             cmap=cmap, axes=[0.92, 0.3, 0.01, 0.4])

fig.suptitle('Prob of each behavior, bin x and y values (n={} bins)'.format(nbins))

putil.label_figure(fig, figid)


#%%


#%%



#%%

nbins=20
#bins = np.arange(0, nbins)
binlabels = np.arange(0, nbins)
plotdf['targ_rel_pos_x_binned'] = pd.cut(plotdf['targ_rel_pos_x'], bins=nbins)
plotdf['targ_rel_pos_y_binned'] = pd.cut(plotdf['targ_rel_pos_y'], bins=nbins)
plotdf['targ_rel_pos_x_left'] = [v.left if isinstance(v, pd.Interval) else v for v in plotdf['targ_rel_pos_x_binned']]
plotdf['targ_rel_pos_y_left'] = [v.left if isinstance(v, pd.Interval) else v for v in plotdf['targ_rel_pos_y_binned']]



#%%


#%%
import SeabornFig2Grid as sfg
import matplotlib.gridspec as gridspec

gdict = dict((sp, []) for sp in plotdf['species'].unique())
for (sp, acq), df_ in plotdf.groupby(['species', 'acquisition']):
    g1 = sns.jointplot(data=df_, x=xvar, y=yvar, hue='species', kind='kde',
                    palette=species_palette)
    gdict[sp].append(g1)

max_n = int(plotdf[['species', 'acquisition']].drop_duplicates().groupby('species').count().max())
fig = pl.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, max_n)

for ri, sp in enumerate(gdict.keys()):
    for ci, g1 in enumerate(gdict[sp]):
        mg0 = sfg.SeabornFig2Grid(g1, fig, gs[ri, ci])




# %%
