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

import statsmodels.api as sm
import regplot as rpl
import scipy.stats as spstats


#%% FUNCTIONS 

def load_and_process_relative_metrics_df(srcdir): #, out_fpath):
    df0 = util.load_aggregate_data_pkl(srcdir, mat_type='df')
    #print(df0['species'].unique())
    #% save
    #out_fpath = os.path.join(os.path.split(figdir)[0], 'relative_metrics.pkl')
    #df0.to_pickle(out_fpath)
    #print(out_fpath)

    # save local, too
    #df0.to_pickle(out_fpath_local)
    #df0['acquisition'] = ['_'.join([f for f in f.split('_')[0:-1] if 'led' not in f]) for f in df0['acquisition']]
    #%
    # Manually calculate THETA_ERROR
    # df['ori'] = -1 * df['ori']
    d_list = []
    for acq, df_ in df0.groupby('acquisition'):
        f1 = df_[df_['id']==0].copy().reset_index(drop=True)
        f2 = df_[df_['id']==1].copy().reset_index(drop=True)
        # calculate theta error
        f1 = calculate_theta_error(f1, f2)
        f1 = calculate_theta_error_from_heading(f1, f2)
        f2 = calculate_theta_error(f2, f1)
        f2 = calculate_theta_error_from_heading(f2, f1)
        # add
        d_list.append(f1)
        d_list.append(f2)
    df0 = pd.concat(d_list)
    return df0

def process_flydf_add_jaaba(df0, jaaba_fname='projector_data_mel_yak_20240330'):
    #jaaba_fname = 'free_behavior_data_mel_yak_20240403' if assay=='38mm-dyad' else None

    winsize = 5
    df = calculate_additional_angle_metrics(df0, winsize=5)

    #% ------------------------------
    # JAABA THRESHOLDS
    # ------------------------------
    jaaba = util.load_jaaba(assay, fname=jaaba_fname)

    if jaaba['chasing'].max() == 1:
        jaaba_thresh_dict = {'orienting': 0, 
                            'chasing': 0,
                            'singing': 0}
    else:
        jaaba_thresh_dict = {'orienting': 10,
                            'chasing': 10,
                            'singing': 5}
    print(jaaba_thresh_dict)

    #%
    if 'filename' in jaaba.columns and 'acquisition' not in jaaba.columns:
        jaaba = jaaba.rename(columns={'filename': 'acquisition'})
    #print(jaaba[['species', 'acquisition']].drop_duplicates().groupby('species').count())

    #% Check missing
    missing = [c for c in jaaba['acquisition'].unique() if c not in df['acquisition'].unique()]
    print("Missing {} acquisitions in found jaaba.".format(len(missing)))

    #%
    # merge jaaba --------------------------------------------------------
    print("Merging flydf and jaaba...")
    ftjaaba = util.combine_jaaba_and_processed_df(df, jaaba)
    #%
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

    #% Clean up a few things
    if 'good_frames' not in ftjaaba.columns:
        ftjaaba['good_frames'] = 1
    #%
    if 'filename' in ftjaaba.columns:
        ftjaaba = ftjaaba.drop(columns=['filename'])

    if 'dovas' in ftjaaba.columns:
        ftjaaba['dovas_deg'] = np.rad2deg(ftjaaba['dovas'])

    #% TEMP THINGS:
    ftjaaba = shift_variables_by_lag(ftjaaba, lag=2)
    ftjaaba['ang_vel_fly_shifted_abs'] = np.abs(ftjaaba['ang_vel_fly_shifted'])

    # binarize behavs
    ftjaaba = util.binarize_behaviors(ftjaaba, jaaba_thresh_dict=jaaba_thresh_dict)

    return ftjaaba



#%
def cross_correlation_lag(x, y, fps=60):
    '''
    Cross-correlation between two signals x and y, with a given fps.
    Returns the cross-correlation, lags, lag in frames, and lag in seconds.
    Lag is defined as how much Y lags relative to X.

    Arguments:
        x -- _description_
        y -- _description_

    Keyword Arguments:
        fps -- _description_ (default: {60})

    Returns:
        correlation (n.array): cross-correlation between x and y
        lags (np.array): lag / displacement indices array for 1D cross-correlation 
        lag_frames (int): lag/displacement in frames
        t_lag (flat): lag in seconds

    '''
    import scipy.signal as signal
    correlation = signal.correlate(x-np.mean(x), y - np.mean(y), mode="full")
    npoints = len(x)
    ccorr = correlation / (npoints * y.std() * x.std()) 

    lags = signal.correlation_lags(len(x), len(y), mode="full")
    lag_frames = lags[np.argmax(correlation)] #lags[np.argmax(abs(correlation))]
    t_lag = lag_frames / fps 
    return ccorr, lags, lag_frames, t_lag


def get_window_centered_bouts(turn_start_frames, flydf, nframes_win):
    '''
    For each provided frame, get a window of nframes_win before and after the frame.

    Arguments:
        turn_start_frames (list/n.array): list of frame indices (int) to center windows around
        flydf (pd.DataFrame): dataframe with focal fly data
        nframes_win (int): N frames to include before and after the turn start frame

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
        d_['rel_sec'] = (d_['sec'].interpolate().ffill().bfill().astype(float) - t_onset).round(2)
        d_['turn_start_frame'] = ix
        if len(d_['rel_sec'].values) < 13:
            print("bad len")
            break
        d_list.append(d_)

    turnbouts = pd.concat(d_list)

    return turnbouts


def get_turn_bouts(flydf, min_dist_to_other=15, min_ang_acc=100, #min_ang_vel=5, 
                   min_facing_angle=np.deg2rad(90), 
                   min_vel=15, nframes_win=0.1*60, filter_dur=False):
    # prev used min_acc = 100, but it's weird

    #min_ang_vel = 4.36
#    turndf = flydf[ (flydf['stim_hz']>0) 
#                & (flydf['ang_acc']>=min_acc)
#                #& (flydf['chasing']>8)
#                #& (flydf['chasing']>10)
#                & (flydf['good_frames']==1)
#                & (flydf['dist_to_other']<=min_dist_to_other)
#                & (flydf['facing_angle'] <= min_facing_angle)
#                & (flydf['vel']>min_vel)
#                ]

    passdf = flydf[ (flydf['stim_hz']>0) 
                #& (flydf['ang_vel']>=min_ang_vel)
                #& (flydf['ang_vel_fly_abs']>=min_ang_vel)
                & (flydf['ang_acc']>=min_ang_acc)
                #& (flydf['chasing']>8)
                & (flydf['chasing_binary']>0)
                & (flydf['good_frames']==1)
                & (flydf['dist_to_other']<=min_dist_to_other)
                & (flydf['facing_angle']<min_facing_angle)
                & (flydf['vel']>min_vel)
                ]
    # get start/stop indices of consecutive rows
    high_ang_vel_bouts = util.get_indices_of_consecutive_rows(passdf)
    if len(high_ang_vel_bouts) == 0:
        print("No turn bouts! - {}".format(flydf['acquisition'].unique()))
        return None
    
    #print(len(high_ang_vel_bouts))

    # filter duration?
    if filter_dur:
        min_bout_len = 2/60 #min 2 frames
        print(min_bout_len)
        fps = 60.
        incl_bouts = util.filter_bouts_by_frame_duration(high_ang_vel_bouts, min_bout_len, fps)
        print("{} of {} bouts pass min dur {}sec".format(len(incl_bouts), 
                                                len(high_ang_vel_bouts), min_bout_len))

        turn_start_frames = [c[0] for c in incl_bouts]
    else:
        # find turn starts
        turn_start_frames = [c[0] for c in high_ang_vel_bouts] #incl_bouts]

    turnbouts = get_window_centered_bouts(turn_start_frames, flydf, nframes_win)

    return turnbouts

#% Calculation functions

def calculate_theta_error(f1, f2, xvar='pos_x', yvar='pos_y'):
    vec_between = f2[[xvar, yvar]] - f1[[xvar, yvar]]
    abs_ang = np.arctan2(vec_between[yvar], vec_between[xvar])
    th_err = util.circular_distance(abs_ang, f1['ori']) # already bw -np.pi, pi
    #th_err = [util.set_angle_range_to_neg_pos_pi(v) for v in th_err]
    #th_err[0] = th_err[1]
    f1['abs_ang_between'] = abs_ang # this is the line-of-sigh, line btween pursuer and target
    f1['theta_error'] = th_err
    f1['theta_error_dt'] = pd.Series(np.unwrap(f1['theta_error'].interpolate().ffill().bfill())).diff() / f1['sec_diff'].mean()
    f1['theta_error_deg'] = np.rad2deg(f1['theta_error'])

    return f1

def calculate_theta_error_from_heading(f1, f2, xvar='pos_x', yvar='pos_y'):
    if 'heading' not in f1.columns:
        f1 = calculate_heading(f1)
        f2 = calculate_heading(f2)

    vec_between = f2[[xvar, yvar]] - f1[[xvar, yvar]]
    abs_ang = np.arctan2(vec_between[yvar], vec_between[xvar])
    th_err = util.circular_distance(abs_ang, f1['heading']) # already bw -np.pi, pi
    #th_err = [util.set_angle_range_to_neg_pos_pi(v) for v in th_err]
    #th_err[0] = th_err[1]
    f1['theta_error_heading'] = th_err
    f1['theta_error_heading_dt'] = pd.Series(np.unwrap(f1['theta_error'].interpolate().ffill().bfill())).diff() / f1['sec_diff'].mean()
    f1['theta_error_heading_deg'] = np.rad2deg(f1['theta_error'])

    return f1

def calculate_heading(df_, winsize=5):
    #% smooth x, y, 
    df_['pos_x_smoothed'] = df_.groupby('id')['pos_x'].transform(lambda x: x.rolling(winsize, 1).mean())
    df_['pos_y_smoothed'] = 1*df_.groupby('id')['pos_y'].transform(lambda x: x.rolling(winsize, 1).mean())  

    # calculate heading
    df_['heading'] = np.arctan2(df_['pos_y_smoothed'].diff(), df_['pos_x_smoothed'].diff())
    df_['heading_deg'] = np.rad2deg(df_['heading']) #np.rad2deg(np.arctan2(df_['pos_y_smoothed'].diff(), df_['pos_x_smoothed'].diff())) 

    return df_

def calculate_additional_angle_metrics(df0, winsize=5):

    df = df0[df0['id']==0].copy()

    d_list = []
    for acq, df_ in df.groupby('acquisition'):
        # Calculate target angular vel
        df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='targ_pos_theta', vel_var='targ_ang_vel',
                                    time_var='sec', winsize=winsize)

        # Calculate facing angle vel
        df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='facing_angle', vel_var='facing_angle_vel',)

        # Calculate "theta-error"
        df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='theta_error', vel_var='theta_error_dt', 
                                                         time_var='sec', winsize=winsize)
        df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='ori', vel_var='ang_vel_fly', 
                                                        time_var='sec', winsize=winsize)

        # Calculate difference in ori between consecutive rows 
        df_['turn_size'] = df_.groupby('id')['ori'].transform(lambda x: x.diff())

        # Calculate relative vel
        #tmp_d = []
        #for i, d_ in df_.groupby('id'):    
        df_['rel_vel'] = df_['dist_to_other'].interpolate().diff() / df_['sec'].diff().mean()
        #    tmp_d.append(df_)
        #tmp_ = pd.concat(tmp_d)

        df_['ang_acc'] = df_['ang_vel'].diff() / df_['sec_diff'].mean()
        df_['ang_acc_smoothed'] = util.smooth_orientations_pandas(df_['ang_acc'], winsize=3) 

        df_['facing_angle_acc'] = df_['facing_angle_vel'].diff() / df_['sec_diff'].mean()

        df_['ang_acc_fly'] = df_['ang_vel_fly'].diff() / df_['sec_diff'].mean()
        df_['ang_acc_fly_smoothed'] = df_['ang_vel_fly_smoothed'].diff() / df_['sec_diff'].mean()

        df_['theta_error_acc'] = df_['theta_error_dt'].diff() / df_['sec_diff'].mean()

        d_list.append(df_)

    df = pd.concat(d_list)

    #%
    df.loc[(df['ang_vel_fly']>80) | (df['ang_vel_fly']<-80), 'ang_vel_fly'] = np.nan

    #% and get abs values
    df['targ_pos_theta_deg'] = np.rad2deg(df['targ_pos_theta'])
    df['facing_angle_deg'] = np.rad2deg(df['facing_angle'])
    df['rel_vel'] = np.nan

    df['rel_vel_abs'] = np.abs(df['rel_vel']) 
    df['targ_ang_vel_abs'] = np.abs(df['targ_ang_vel'])
    df['targ_pos_theta_abs'] = np.abs(df['targ_pos_theta'])
    df['targ_ang_size_deg'] = np.rad2deg(df['targ_ang_size'])
    df['targ_ang_vel_deg'] = np.rad2deg(df['targ_ang_vel'])
    df['targ_ang_vel_deg_abs'] = np.abs(df['targ_ang_vel_deg'])
    df['facing_angle_deg'] = np.rad2deg(df['facing_angle'])
    df['ang_vel_fly_abs'] = np.abs(df['ang_vel_fly'])

    df['facing_angle_vel_abs'] = np.abs(df['facing_angle_vel'])
    df['facing_angle_vel_deg'] = np.rad2deg(df['facing_angle_vel'])
    df['facing_angle_vel_deg_abs'] = np.abs(df['facing_angle_vel_deg'])

    df['theta_error_deg'] = np.rad2deg(df['theta_error'])
    df['theta_error_dt_deg'] = np.rad2deg(df['theta_error_dt'])
    df['theta_error_abs'] = np.abs(df['theta_error'])

    df['ang_vel_abs'] = np.abs(df['ang_vel'])
    df['ang_vel_deg'] = np.rad2deg(df['ang_vel'])

    # ftjaaba['ang_vel_abs'] = np.abs(ftjaaba['ang_vel'])
    df['ang_vel_deg'] = np.rad2deg(df['ang_vel'])

    df['ang_vel_fly_abs'] = np.abs(df['ang_vel_fly'])
    df['ang_vel_fly_deg'] = np.rad2deg(df['ang_vel_fly'])
    df['ang_vel_fly_abs_deg'] = np.rad2deg(df['ang_vel_fly_abs'])

    return df

def shift_variables_by_lag(df, lag=2):
    '''
    shift fly response variables BACK by lag 

    Arguments:
        df -- _description_

    Keyword Arguments:
        lag -- _description_ (default: {2})

    Returns:
        _description_
    '''
    df['ang_vel_abs_shifted'] = df.groupby('acquisition')['ang_vel_abs'].shift(-lag)
    df['ang_vel_fly_shifted'] = df.groupby('acquisition')['ang_vel_fly'].shift(-lag)
    df['vel_shifted'] = df.groupby('acquisition')['vel'].shift(-lag)
    df['vel_shifted_abs'] = np.abs(df['vel_shifted']) 
    df['ang_vel_shifted'] = df.groupby('acquisition')['ang_vel'].shift(-lag)

    return df

def cross_corr_each_bout(turnbouts, v1='facing_angle', v2='ang_vel', fps=60):
    '''
    Cycle through all turns specified in turnbouts dataframe and do cross-correlation
    between v1 and v2. Calls cross_correlation_lag()

    Arguments:
        turnbouts -- _description_

    Keyword Arguments:
        v1 -- _description_ (default: {'facing_angle'})
        v2 -- _description_ (default: {'ang_vel'})
        fps -- _description_ (default: {60})

    Returns:
        xcorr (np.array) -- M x N array, M = num of turn bouts, N = length of windowed cross-correlation 
        all_lags -- lag indices
        t_lags -- calculate lag (in sec) for each turn bout 
    '''
    # Get all turn bouts, windowed +/-
    xcorr = []
    t_lags = []
    all_lags = []
    for i, d_ in turnbouts.groupby('turn_bout_num'):
        # cross corr
        correlation, lags, lag_frames, t_lag = cross_correlation_lag(d_[v2].interpolate().ffill().bfill(), 
                                                                     d_[v1].interpolate().ffill().bfill(), fps=fps)
        xcorr.append(correlation)
        all_lags.append(lags)
        t_lags.append(t_lag)

    return np.array(xcorr), np.array(all_lags), np.array(t_lags)


def split_theta_error(chase_, theta_error_small=np.deg2rad(10), theta_error_large=np.deg2rad(25)):
    chase_['error_size'] = None
    chase_.loc[(chase_['theta_error'] < theta_error_small) \
            & (chase_['theta_error'] > -theta_error_small), 'error_size'] = 'small'
    chase_.loc[(chase_['theta_error'] > theta_error_large) \
            | (chase_['theta_error'] < -theta_error_large), 'error_size'] = 'large'

    return chase_

#% PLOTTING

def plot_regr_by_species(chase_, xvar, yvar, hue_var=None, plot_hue=False, plot_grid=True, 
                         cmap='viridis', stimhz_palette=None, bg_color=[0.7]*3, xlabel=None, ylabel=None):
    '''
    Do linear regression of yvar on xvar. Plot scatter of xvar vs. yvar, color by hue_var if specified.

    Arguments:
        chase_ -- _description_
        xvar -- _description_
        yvar -- _description_

    Keyword Arguments:
        hue_var -- _description_ (default: {None})
        plot_hue -- _description_ (default: {False})
        plot_grid -- _description_ (default: {True})
        cmap -- _description_ (default: {'viridis'})
        stimhz_palette -- _description_ (default: {None})
        bg_color -- _description_ (default: {[0.7]*3})
        xlabel -- _description_ (default: {None})
        ylabel -- _description_ (default: {None})

    Returns:
       fig  
    '''
    if plot_hue and stimhz_palette is None:
        # create palette
        stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba[hue_var]>=0], hue_var, cmap=cmap)

    if stimhz_palette is not None:
        vmin = min(list(stimhz_palette.keys()))
        vmax = max(list(stimhz_palette.keys()))
 
    # For each species, plot linear regr with fits in a subplot
    n_species = chase_['species'].nunique()
    fig, axn = pl.subplots(1, n_species, sharex=True, sharey=True)
    for ai, (sp, df_) in enumerate(chase_[chase_['stim_hz']>0].groupby('species')):
        ax = axn[ai]
        # plot scatter
        if plot_hue:
            sns.scatterplot(data=df_, x=xvar, y=yvar, ax=ax,
                        hue='stim_hz', palette=stimhz_palette, 
                        legend=0, edgecolor='none', alpha=0.7)
        else:
            sns.scatterplot(data=df_, x=xvar, y=yvar, ax=ax, color=bg_color,
                        legend=0, edgecolor='none', alpha=0.7, s=2)

        if plot_grid:
            ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
            ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)

        ax.set_title(sp)

        # do fit
        res = rpl.regplot(data=df_, ax=ax, x=xvar, y=yvar,
                    color=bg_color, scatter=False) #, ax=ax)
        # res.params: [intercept, slope]
        ax.set_box_aspect(1)
        fit_str = 'OLS: y = {:.2f}x + {:.2f}'.format(res.params[1], res.params[0])
        print(fit_str) #lope, intercept)
        ax.text(0.05, 0.85, fit_str, fontsize=8, transform=ax.transAxes)

        # set xlabel to be theta subscript E
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        putil.annotate_regr(df_, ax, x=xvar, y=yvar, fontsize=8)

    if plot_hue:
        putil.colorbar_from_mappable(ax, cmap=cmap, norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                                axes=[0.92, 0.3, 0.01, 0.4], hue_title='stim. freq. (Hz)', fontsize=18)

    return fig

def plot_ang_v_fwd_vel_by_theta_error_size(chase_, var1='vel_shifted', var2='ang_vel_shifted', 
                                        lw=1, err_palette={'small': 'r', 'large': 'b'}):

    '''
    3-subplot figure: (1) spatial distribution of small vs. large theta errors, (2) velocity histograms, (3) angular vel histograms

    Returns:
        fig
    ''' 
    fig = pl.figure(figsize=(12,5)) #, axn = pl.subplots(1, 3, figsize=(7,4))

    # plot theta error relative to focal male
    ax = fig.add_subplot(1, 3, 1)
    sns.scatterplot(data=chase_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
                    hue='error_size', palette=err_palette, s=2)
    ax.set_aspect(1)
    ax.plot(0, 0, 'wo', markersize=3)
    ax.axis('off')
    ax.legend_.remove()
    #sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', frameon=False)

    #% plot HISTS ------------------------------------------------
    ax1 = fig.add_subplot(1, 3, 2)
    sns.histplot(data=chase_, x=var1,  ax=ax1, bins=50, linewidth=lw,
                stat='probability', cumulative=False, element='step', fill=False,
                hue='error_size', palette=err_palette, common_norm=False, legend=0)
    ax.set_xlabel('forward vel')

    ax2 = fig.add_subplot(1, 3, 3, sharey=ax1)
    sns.histplot(data=chase_, x=var2, ax=ax2, color='r', bins=50, 
                stat='probability', cumulative=False, element='step', fill=False,
                hue='error_size', palette=err_palette, common_norm=False)
    ax2.set_xlabel('angular vel')

    sns.move_legend(ax2, bbox_to_anchor=(1,1), loc='upper left', frameon=False)

    # plot median values
    curr_ylim = np.round(ax1.get_ylim()[-1], 2)*1.15
    for v, ax in zip([var1, var2], [ax1, ax2]):
        med_ = chase_.groupby('error_size')[v].median()
        for mval, cval in err_palette.items():
            ax.plot(med_[mval], curr_ylim, color=cval, marker='v', markersize=10) #ax.axvline(x=m, color=c, linestyle='--')
        #print(med_)
        ax.set_box_aspect(1)

    return fig

def plot_regr_by_hue(chase_, xvar, yvar, hue_var='stim_hz', stimhz_palette=None, cmap='viridis', show_scatter=False):
    '''
    Plot linear regression of yvar on xvar, color by hue_var.
    '''
    if stimhz_palette is None:
        # create palette
        stimhz_palette = putil.get_palette_dict(chase_[chase_[hue_var]>0], hue_var, cmap=cmap)
    vmin = min(list(stimhz_palette.keys()))
    vmax = max(list(stimhz_palette.keys()))

    g = sns.lmplot(data=df_, x=xvar, y=yvar, 
                hue='stim_hz', palette=stimhz_palette, legend=0, scatter=show_scatter, 
                scatter_kws={'alpha':0.5})

    g.fig.axes[0].set_box_aspect(1)
    
    if xvar == 'theta_error':
        if show_scatter is False:
            pl.xlim([-2, 2])
            pl.ylim([-10, 10])
        else:
            g.fig.axes[0].set_xlim([-2, 2])
        g.fig.axes[0].set_ylim([-15, 15])
    else:
        pl.xlim([0, 80])
        pl.ylim([0, 10])

    putil.colorbar_from_mappable(g.fig.axes[0], cmap=cmap, norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                            axes=[0.96, 0.3, 0.01, 0.4])
    pl.subplots_adjust(top=0.9)

    return g



def select_data_subset(filtdf, meanbouts, do_bouts=False, behav='chasing', min_frac_bout=0.5, is_flytracker=True):

    sign = -1 if is_flytracker else 1
    #chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>=min_frac_bout].copy()

    if not do_bouts:
        # plot frames
        plotdf = filtdf[( filtdf['{}_binary'.format(behav)]>0)
                & ( filtdf['stim_hz']>0 )].copy()
    else:
        plotdf = meanbouts[ (meanbouts['{}_binary'.format(behav)]>min_frac_bout)
                          & (meanbouts['stim_hz']>0) ].copy()

    plotdf['facing_angle'] = sign * plotdf['facing_angle']

    return plotdf

def plot_allo_ego_frames_by_species(plotdf, xvar='facing_angle', yvar='targ_pos_radius',
                                    markersize=5, huevar='stim_hz', cmap='viridis', plot_com=True,
                                    stimhz_palette=None):
    if stimhz_palette is None:
        stimhz_palette = putil.get_palette_dict(plotdf[plotdf['stim_hz']>0], huevar, cmap=cmap)

    # plot
    vmin = min(list(stimhz_palette.keys()))
    vmax = max(list(stimhz_palette.keys()))
    hue_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axn = pl.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True,
                                subplot_kw={'projection': 'polar'})
    pp.plot_allo_vs_egocentric_pos(plotdf, axn=axn, xvar=xvar, yvar=yvar, huevar=huevar,
                                palette_dict=stimhz_palette, hue_norm=hue_norm, markersize=5,
                                com_markersize=40, com_lw=1)
    for ax in axn:
        yl = ax.get_yticklabels()
        ax.set_yticklabels([v if i%2==0 else '' for i, v in enumerate(yl)])
    pl.subplots_adjust(wspace=0.6, top=0.8, right=0.8)

    return fig


# ---- turn analysis -----

def plot_individual_turns(turnbouts, v1='facing_angle', v2='ang_vel'):
    
    fig, axn = pl.subplots(1, 2, sharex=True, figsize=(10,5))
    ax=axn[0]
    sns.lineplot(data=turnbouts, x='rel_sec', y=v1, ax=ax, hue='turn_bout_num', palette='Reds', legend=0,
                lw=0.5, alpha=0.5)
    ax.set_ylabel(r'$\theta_{E}$')
    ax=axn[1]
    sns.lineplot(data=turnbouts, x='rel_sec', y=v2, ax=ax, hue='turn_bout_num', palette='Blues', legend=0,
                lw=0.5, alpha=0.5)
    ax.set_ylabel(r'$\omega_{f}$')
    pl.subplots_adjust(wspace=0.5, top=0.8)

    return fig


def plot_cross_corr_results(turnbouts, xcorr, lags, t_lags, v1='facing_angle', v2='ang_vel',
                            v1_label = None, v2_label = None,
                            col1='r', col2='dodgerblue', bg_color=[0.7]*3):

    if v1_label is None:
        v1_label = r'$\theta_{E}$' + '\n{}'.format(v1)
    if v2_label is None:
        v2_label = r'$\omega_{f}$' + '\n{}'.format(v2)

    # PLOT MEAN + SEM of aligned turn bout traces
    fig, axn =pl.subplots(1, 3, figsize=(17, 4))
    ax1 = axn[0]
    ax2 = ax1.twinx()
    sns.lineplot(data=turnbouts, x='rel_sec', y=v1, ax=ax1, lw=0.5, color=col1)
    sns.lineplot(data=turnbouts, x='rel_sec', y=v2, ax=ax2, lw=0.5, color=col2)
    # Change spine colors
    for ax, sp, lb, col in zip([ax1, ax2], ['left', 'right'], [v1_label, v2_label], [col1, col2]):
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
    ax.set_xlabel('lag of {} relative to {}'.format(v2, v1))

    # PLOT distribution of TIME LAGS
    ax=axn[2]
    ax.hist(t_lags, bins=20, color=bg_color)
    med_lag = np.median(np.array(t_lags))
    ax.axvline(x=med_lag, color='w', linestyle='--')
    ax.set_title('Median lag: {:.2f}ms'.format(med_lag*1000))
    ax.set_xlabel("lags (sec)")

    for ax in fig.axes:
        ax.set_box_aspect(1)

    pl.subplots_adjust(wspace=1)

    return fig

def compare_regr_pre_post_shift(flydf, shifted, x='facing_angle', y='ang_vel', y1='turn_size',
                                x1='facing_angle_vel', lag_frames=2, col=[0.7]*3, markersize=5):
    '''
    Regression plots of y on x, before and after shifting by lag_frames.

    Returns:
        _description_
    '''
    turn_start_frames = shifted['turn_start_frame'].values
    unshifted = flydf.loc[turn_start_frames].copy() #.dropna()

    scatter_kws={'s': markersize}
    fig, axn = pl.subplots(2, 3, figsize=(15, 10))

    # unshifted
    plotdf_shifted = shifted[(shifted['turn_size']<=np.pi) 
                    & (shifted['turn_size']>=-np.pi)]
    plotdf_unshifted = unshifted[(unshifted['turn_size']<=np.pi) 
                    & (unshifted['turn_size']>=-np.pi)]

    plotd_ = plotdf_unshifted.copy()
    title_ = 'unshifted'
    for ai, (x_, y_) in enumerate(zip([x, x, x1], [y, y1, y])): 
        ax=axn[0, ai]
        # Plot X vs. Y
        sns.regplot(data=plotd_, x=x_, y=y_, ax=ax, color=col, scatter_kws=scatter_kws) 
        ax.set_title(title_)
        # Annotate
        putil.annotate_regr(plotd_, ax, x=x_, y=y_, fontsize=12)

        model_ = sm.OLS(plotd_[y_], sm.add_constant(plotd_[x_]))
        res_ = model_.fit()
        fit_str = 'OLS: y = {:.2f}x + {:.2f}, R2={:.2f}'\
                    .format(res_.params[1], res_.params[0], res_.rsquared)
        ax.text(0.1, 0.9, fit_str, transform=ax.transAxes)
        if 'dt' in x_ or 'vel' in x_:
            ax.set_xlabel(r'$\omega_{\theta_{E}}$' + '\n{}'.format(x_))
        else:
            ax.set_xlabel(r'$\theta_{E}$' + '\n{}'.format(x_))
        ax.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(y_))

    # SHFITED
#    ax=axn[0, 1]
#    plotd_ = shifted.dropna()
#
#    sns.regplot(data=plotd_, x=x, y=y, ax=ax, 
#                color=col, scatter_kws=scatter_kws) #markersize=markersize)
#    ax.set_title('lag {} frames'.format(lag_frames))
#    putil.annotate_regr(plotd_, ax, x=x, y=y, fontsize=12)
#    # Annotate
#    model_ = sm.OLS(plotd_[y], sm.add_constant(plotd_[x]))
#    res_ = model_.fit()
#    fit_str = 'OLS: y = {:.2f}x + {:.2f}, R2={:.2f}'\
#                .format(res_.params[1], res_.params[0],
#                        res_.rsquared)
#    ax.text(0.1, 0.9, fit_str, transform=ax.transAxes)
#
#    ax.set_xlabel(r'$\theta_{E}$' + '\n{}'.format(x))
#    ax.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(y))

#    ax=axn[0, 2] # Time derivatives
#    # Shoudl this be facing_angle??
#    plotd_ = shifted.dropna()
#    sns.regplot(data=plotd_, x=x1, y=y, ax=ax, color=col, scatter_kws=scatter_kws) #markersize=markersize)
#    ax.set_title('lag {} frames'.format(lag_frames))
#    putil.annotate_regr(plotd_, ax, x=x1, y=y, fontsize=12)
#    # Annotate
#    model_ = sm.OLS(plotd_[y], sm.add_constant(plotd_[x1]))
#    res_ = model_.fit()
#    fit_str = 'OLS: y = {:.2f}x + {:.2f}, R2={:.2f}'\
#                .format(res_.params[1], res_.params[0], res_.rsquared)
#    ax.text(0.1, 0.9, fit_str, transform=ax.transAxes)
#
#    ax.set_xlabel(r'$\omega_{\theta_{E}}$' + '\n{}'.format(x1))
#    ax.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(y))

    plotd_ = plotdf_shifted.copy()
    ax=axn[1, 0] # Time derivatives
    # Shoudl this be facing_angle??
    title_ = 'lag {}'.format(lag_frames)
    for ai, (x_, y_) in enumerate(zip([x, x, x1], [y, y1, y])): 
        ax=axn[1, ai]
        sns.regplot(data=plotd_, x=x_, y=y_, ax=ax, color=col, scatter_kws=scatter_kws) #markersize=markersize)
        ax.set_title(title_)
        putil.annotate_regr(plotd_, ax, x=x_, y=y_, fontsize=12)
        # Annotate
        model_ = sm.OLS(plotd_[y_], sm.add_constant(plotd_[x_]))
        res_ = model_.fit()
        fit_str = 'OLS: y = {:.2f}x + {:.2f}, R2={:.2f}'\
                    .format(res_.params[1], res_.params[0], res_.rsquared)
        ax.text(0.1, 0.9, fit_str, transform=ax.transAxes)
        if 'dt' in x_ or 'vel' in x_:
            ax.set_xlabel(r'$\omega_{\theta_{E}}$' + '\n{}'.format(x_))
        else:
            ax.set_xlabel(r'$\theta_{E}$' + '\n{}'.format(x_))
        if 'turn_size' in y_:
            ax.set_ylabel(y_)
        else:
            ax.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(y_))

    for ax in axn.flat:
        ax.set_box_aspect(1)

    return fig



def count_n_turns_in_window(flydf, turn_bout_starts, high_ang_start_frames, fps=60.):
    '''
    For longer chunks entered around turn bout (2 seconds), count the number of actual turns (high_ang_start_frames)
    Assumes that high_ang_vel_bouts were filtered to be min. 2 frames or longer (turn_bout_starts).

    Arguments:
        flydf -- _description_
        turn_bout_starts -- _description_
        high_ang_start_frames -- _description_

    Keyword Arguments:
        fps -- _description_ (default: {60.})
    '''
    nframes_win = 2*fps
    t_list = []
    for turn_ix, frame_num in enumerate(turn_bout_starts):
        start_ix = turn_bout_starts[turn_ix] - nframes_win #12049 #18320 #turn_start_frames[4] #- nframes_win
        stop_ix = turn_bout_starts[turn_ix] + nframes_win #-60 #*2
        plotdf = flydf.loc[start_ix:stop_ix]

        n_turns = len(plotdf[plotdf['frame'].isin(high_ang_start_frames)])
        t_list.append(pd.DataFrame({'turn_ix': turn_ix, 'frame_num': frame_num, 
                                    'n_turns': n_turns, 'start_ix': start_ix, 'stop_ix': stop_ix},
                                index=[turn_ix]))
    turn_counts = pd.concat(t_list)

    return turn_counts


def plot_timecourses_for_turn_bouts(plotdf, high_ang_start_frames, xvar = 'sec', varset='varset2_smoothed', 
                                    targ_color='r', fly_color='cornflowerblue', accel_color=[0.7]*3):
    '''
    Plot timecourses of theta_error, facing_angle, facing_angle_vel, ang_vel, ang_acc, ang_acc_fly
    Three subplots, first 2 have twin axes with ang vel of fly

    varset: 'varset2' or 'varset2_smoothed'
        - use varset1 to use FlyTracker's direct output (facing_angle and ang_vel/ang_acc)
        - use varset2 to use theta_error and ang_vel_fly (calculated from ori/etc.)
        - use varset2_smoothed to more closely match FT's output, but with own calculated metrics.
        
    ''' 
    if 'varset2' in varset:
        smooth_sfx = '_smoothed' if 'smoothed' in varset else ''
        var1 = 'theta_error{}'.format(smooth_sfx) #'facing_angle'
        vel_var1 = 'theta_error_dt{}'.format(smooth_sfx) #'facing_angle_vel'
        var2 = 'ang_vel_fly{}'.format(smooth_sfx) #'ang_vel'
        acc_var2 = 'ang_acc_fly{}'.format(smooth_sfx) #'ang_acc'
        center_yaxis = True
    else:
        var1 = 'facing_angle'
        vel_var1 = 'facing_angle_vel'
        var2 = 'ang_vel'
        acc_var2 = 'ang_acc'
        center_yaxis=False

    fig, axn = pl.subplots(3, 1, figsize=(8,6), sharex=True)

    # Plot theta_error and fly's angular velocity
    ax=axn[0]
    ax.plot(plotdf[xvar], plotdf[var1], targ_color)
    # color y-axis spine and ticks red
    ax.set_ylabel(r'$\theta_{E}$' + '\n{}'.format(var1)) #, color=targ_color) #r'$\theta_{E}$'
    putil.change_spine_color(ax, targ_color, 'left')
    ax.axhline(y=0, color=targ_color, linestyle='--', lw=0.5)

    ax2 = ax.twinx()
    ax2.plot(plotdf[xvar], plotdf[var2], fly_color)
    # Color y-axis spine and ticks blue
    ax2.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(var2)) #, color=fly_color)
    putil.change_spine_color(ax2, fly_color, 'right')
    # Center around 0
    if center_yaxis:
        curr_ylim = np.round(plotdf[var2].abs().max(), 0)
        ax2.set_ylim(-curr_ylim, curr_ylim)
        
    mean_ang_vel = np.mean(plotdf[plotdf['frame'].isin(high_ang_start_frames)][var2]) #'ang_vel']) 
    ax2.axhline(y=mean_ang_vel, color='w', linestyle='--', lw=0.5)
    ax2.axhline(y=-1*mean_ang_vel, color='w', linestyle='--', lw=0.5)

    # plot time derivative of ax0's 1st axis
    ax=axn[1]
    ax.plot(plotdf[xvar], plotdf[vel_var1], targ_color)
    ax.set_ylabel(r'$\omega_{\theta_{E}}$' + '\n{}'.format(vel_var1)) # color=targ_color)
    putil.change_spine_color(ax, targ_color, 'left')
    ax2 = ax.twinx()
    ax2.plot(plotdf[xvar], plotdf[var2], fly_color)
    ax2.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(var2)) # color=fly_color)
    putil.change_spine_color(ax2, fly_color, 'right')

    # plot acceleration (used to find peaks for turn bouts to align)
    ax=axn[2]
    ax.plot(plotdf[xvar], plotdf[acc_var2], fly_color)
    ax.set_ylabel(acc_var2) #color=fly_color)

    angf = []
    for i, f in enumerate(high_ang_start_frames):
        #print(f)
        ax.plot(plotdf.loc[plotdf['frame']==f][xvar], 
                plotdf.loc[plotdf['frame']==f][acc_var2], 
                color = 'w', marker='o', markersize=5)
        angf.append(f)


    return fig

def select_turn_bouts_for_plotting(flydf, min_ang_acc=100, min_dist_to_other=25, 
                                   min_facing_angle=np.deg2rad(90), min_vel=15, fps=60.):
    '''
    Filter flydf (assumes good_frames==1) for turn bouts that meet criteria for high angular acceleration,
    but return longer bouts that have multiple turns in them (in contrast to `get_turn_bouts()` which returns 
    the turn-centered bout.

    Arguments:
        flydf -- _description_

    Keyword Arguments:
        min_ang_acc -- _description_ (default: {100})
        min_dist_to_other -- _description_ (default: {25})
        min_facing_angle -- _description_ (default: {np.deg2rad(90)})
        min_vel -- _description_ (default: {15})
        fps -- _description_ (default: {60.})

    Returns:
        _description_
    '''
    passdf = flydf[ (flydf['stim_hz']>0) 
                & (flydf['ang_acc']>min_ang_acc)
                & (flydf['dist_to_other']<=min_dist_to_other)
                & (flydf['facing_angle']<min_facing_angle)
                & (flydf['vel']>min_vel)
                ]
    # get start/stop indices of consecutive rows
    high_ang_vel_bouts = util.get_indices_of_consecutive_rows(passdf)
    print(len(high_ang_vel_bouts))

    # For plotting, find epochs where turn bouts were at leat 2 frames
    # filter duration?
    min_bout_len = 2/fps #min 2 frames
    incl_bouts = util.filter_bouts_by_frame_duration(high_ang_vel_bouts, min_bout_len, fps)
    print("{} of {} bouts pass min dur {}sec".format(len(incl_bouts), 
                                            len(high_ang_vel_bouts), min_bout_len))

    # find turn starts
    turn_bout_starts = [c[0] for c in incl_bouts]

    # find turn starts
    high_ang_start_frames = [c[0] for c in high_ang_vel_bouts] #onsec_bouts]
    print(len(high_ang_start_frames))

    return turn_bout_starts, high_ang_start_frames


def get_turn_psth_values(plotdf, high_ang_start_frames, interval=10,
                                 yvar1='facing_angle', yvar2='ang_vel', nframes_win=0.1*60, fps=60):

    '''
    For EACH turn (specified by high_ang_start_frames), calculate cross-correlation between yvar1 and yvar2.
    Returns a dataframe with the values of yvar1 and yvar2, relative to the start of the turn.
    Also returns the time lag (in seconds) calculated for each turn.

    Returns:
        _description_
    '''
    d_list = []
    t_lags = []
    for i, ix in enumerate(high_ang_start_frames[0::interval]):
        start_ = ix - nframes_win
        stop_ = ix + nframes_win
        t_onset = plotdf.loc[ix]['sec']
        d_  = plotdf.loc[start_:stop_].copy()
        # Skip too short bouts
        if d_.shape[0] < (nframes_win*2 + 1):
            continue
        # Shift time to 0
        d_['rel_sec'] = d_['sec'] - t_onset
        # cross corr
        correlation, lags, lag_frames, t_lag = cross_correlation_lag(d_[yvar2].interpolate(), 
                                                                     d_[yvar1].interpolate(), fps=fps)
        # Append
        d_[[yvar1, yvar2, 'rel_sec']] = d_[[yvar1, yvar2, 'rel_sec']].interpolate().ffill().bfill()
        d_['rel_sec'] = d_['rel_sec'].round(3) # round so we can use these to average later
        d_['turn_ix'] = i
        d_['turn_frame'] = ix
        d_list.append(d_[[yvar1, yvar2, 'rel_sec', 'turn_ix', 'turn_frame']])
        t_lags.append(t_lag)
    t_lags = np.array(t_lags)
    turns_ = pd.concat(d_list).reset_index(drop=True)

    return turns_, t_lags


def plot_psth_all_turns(turns_, yvar1='theta_error', yvar2='ang_vel_fly', col1='r', col2='cornflowerblue',
                        bg_color=[0.7]*3, ax1=None, ax2=None, lw_all=0.5, lw_mean=3):
    '''
    Plot all turns (thin lines), and mean (thick) vals for theta_error (yvar1) and ang_vel_fly (yvar2).
    Plots on twin y axes.
    ''' 
    if ax1 is None or ax2 is None:
        fig, ax1 = pl.subplots()
        ax2 = ax1.twinx()

    #% PLOT each turn
    for i_, d_ in turns_.groupby('turn_ix'):
        sns.lineplot(data=d_, x='rel_sec', y=yvar1, ax=ax1, color=col1, lw=lw_all, alpha=0.5)
        sns.lineplot(data=d_, x='rel_sec', y=yvar2, ax=ax2, color=col2, lw=lw_all, alpha=0.5)

    # Center y axes
    if turns_[yvar2].min() < -1*turns_[yvar2].max():
        ylim2 = turns_[yvar2].abs().max()
        ax2.set_ylim([-ylim2, ylim2])

    if turns_[yvar1].min() < -1*turns_[yvar1].max():
        ylim1 = turns_[yvar1].abs().max()
        ax1.set_ylim([-ylim1, ylim1])

    # Plot MEAN turns
    mean_turns_ = turns_.groupby('rel_sec').mean().reset_index()
    for xv, col, ax, var_ in zip([yvar1, yvar2], [col1, col2], [ax1, ax2], [yvar1, yvar2] ):
        ax.plot(mean_turns_['rel_sec'], mean_turns_[var_], color=col, lw=lw_mean)

    #ax.set_box_aspect(1)
    ax1.set_ylabel(r'$\theta_{E}$' + '\n{}'.format(yvar1))
    ax2.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(yvar2))
    putil.change_spine_color(ax1, col1, 'left')
    putil.change_spine_color(ax2, col2, 'right')

    # Plot vertical line at 0
    ax1.axvline(x=0, color=bg_color, linestyle='--')
    for ax in fig.axes:
        ax.set_box_aspect(1)

    return fig

def plot_mean_cross_corr_results(mean_turns_, correlation, lags, t_lags, t_lag=2, yvar1='theta_error', yvar2='ang_vel_fly', 
                         col1='r', col2='cornflowerblue', bg_color=[0.7]*3): 
    '''
    Plot the meant turns and cross-correlation between yvar1 and yvar2 for a set of turns shown in a selected chunk of time.
    For visualizing example set (does not include ALL turns).

    Arguments:
        mean_turns_ -- _description_
        correlation -- _description_
        lags -- _description_
        t_lags -- _description_

    Keyword Arguments:
        yvar1 -- _description_ (default: {'theta_error'})
        yvar2 -- _description_ (default: {'ang_vel_fly'})
        col1 -- _description_ (default: {'r'})
        col2 -- _description_ (default: {'cornflowerblue'})
        bg_color -- _description_ (default: {[0.7]*3})

    Returns:
        _description_
    '''
    # plot means
    fig, axn =pl.subplots(1, 3, figsize=(16,4))
    ax1 = axn[0]
    ax2 = ax1.twinx()
    for xv, col, ax in zip([yvar1, yvar2], [col1, col2], [ax1, ax2]):
        sns.lineplot(data=mean_turns_, x='rel_sec', y=xv, ax=ax, color=col, lw=3)

    #for xv, col, mean_, ax in zip([yvar1, yvar2], [col1, col2], [mean_v1, mean_v2], [ax1, ax2]):
    #    ax.plot(mean_sec, mean_, color=col, lw=3)
    ax1.set_ylabel(r'$\theta_{E}$' + '\n{}'.format(yvar1))
    ax2.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(yvar2))
    putil.change_spine_color(ax1, col1, 'left')
    putil.change_spine_color(ax2, col2, 'right')
    ax1.set_xlabel('time (sec)')
    ax1.set_title('mean bout vals')

    # plot cross-corr
    ax=axn[1]
    ax.plot(lags, correlation, c=bg_color)
    max1 = np.argmax(mean_turns_[yvar1])
    max2 = np.argmax(mean_turns_[yvar2])
    #print(max1, max2)
    peak_diff_sec = mean_turns_['rel_sec'].iloc[max2] - mean_turns_['rel_sec'].iloc[max1]
    ax.set_ylabel('cross correlation')
    ax.set_xlabel('lag of {} rel. to {} (frames)'.format(yvar2, yvar1))
    ax.set_title('Peak diff: {:.2f}sec\nx-corr peak: {:.2f}msec'.format(peak_diff_sec, t_lag*1E3))

    # plot distribution of time lags
    ax=axn[2]
    ax.hist(t_lags, color=bg_color)
    ax.set_title('Median lag: {:.2f}msec'.format(np.median(t_lags)*1E3))
    ax.set_xlabel("lags (sec)")

    for ax in fig.axes:
        ax.set_box_aspect(1)

    pl.subplots_adjust(wspace=1)

    return fig

#% 

def shift_vars_by_lag(flydf, high_ang_start_frames, med_lag, fps=60):
    '''
    Shift a subset of variables in flydf by a median lag (med_lag) to align with high_ang_start_frames.

    Arguments:
        flydf -- _description_
        high_ang_start_frames -- _description_
        med_lag -- _description_

    Keyword Arguments:
        fps -- _description_ (default: {60})

    Returns:
        _description_
    '''
    lag_frames = np.round(med_lag * fps)

    orig_list = []
    d_list = []

    unshifted_vars = ['ang_vel', 'ang_acc', 'turn_size', 'ang_vel_fly', 'ang_acc_fly', 
                      'ang_vel_fly_smoothed', 'ang_acc_fly_smoothed']
    shifted_vars = ['theta_error', 'theta_error_dt', 'facing_angle', 'facing_angle_vel', 
                    'facing_angle_vel_abs', 'facing_angle_acc', 
                    'theta_error_smoothed', 'theta_error_dt_smoothed']
    all_vars =  unshifted_vars + shifted_vars

    for f in high_ang_start_frames: #turn_start_frames:
        fly_ = flydf.loc[f][unshifted_vars]
        targ_ = flydf.loc[f-lag_frames][shifted_vars].interpolate().ffill().bfill()
        d_ = pd.concat([fly_, targ_], axis=0)
        d_['turn_start_frame'] = f

        d2_ = flydf.loc[f][all_vars]        
        d2_['turn_start_frame'] = f
        orig_list.append(d2_)
        d_list.append(d_)
    shifted = pd.concat(d_list, axis=1).T
    shifted = shifted.astype(float)

    unshifted = pd.concat(orig_list, axis=1).T
    unshifted = unshifted.astype(float)

    return shifted, unshifted

#%
def aggregate_turns_across_flies(ftjaaba, v1='theta_error', v2='ang_vel_fly', min_n_turns=5,
                                 min_ang_acc=100, min_vel=15, min_dist_to_other=25,
                                 min_facing_angle=np.deg2rad(90), fps=60, nframes_win=0.1*60):
    no_turns = []
    few_turns = []

    t_list = []
    f_list = []
    for acq, flydf in ftjaaba.groupby('acquisition'):
        # Get turn bouts for current acquisition 
        turns_ = get_turn_bouts(flydf, min_ang_acc=min_ang_acc, #min_ang_vel=min_ang_vel, 
                                min_vel=min_vel, min_dist_to_other=min_dist_to_other,
                                min_facing_angle=min_facing_angle, 
                                nframes_win=nframes_win)
        # Identify funky files
        if turns_ is None:
            no_turns.append(acq)
            continue

        if turns_['turn_bout_num'].nunique() < min_n_turns:
            few_turns.append(acq)
            continue

        # Do cross-correlation
        turns_ = turns_.reset_index(drop=True)
        xcorr, lags, t_lags = cross_corr_each_bout(turns_, v1=v1, v2=v2)

        # Save delta_t_lag
        for ((turn_ix, t), l) in zip(turns_.groupby('turn_bout_num'), t_lags):
            turns_.loc[t.index, 'delta_t_lag'] = l

        # Append
        t_list.append(turns_)

    # Aggregate
    aggr_turns = pd.concat(t_list)

    return aggr_turns 

def get_theta_errors_before_turns(aggr_turns, fps=60):
    # For each turn_bout_num in each acquisition, get the mean delta_t_lag for that turn_bout_num
    # Then, shift theta_error by that mean delta_t_lag and call it 'previous_theta_error'
    # Finally, add a column 'previous_theta_error' to the turn_bouts dataframe

    t_list = []
    for acq, d_ in aggr_turns.groupby('acquisition'):
        for ti, t_ in d_.groupby('turn_bout_num'):
            mean_delta_t = d_['delta_t_lag'].mean()
            t_['previous_theta_error'] = t_['theta_error'].shift(int(mean_delta_t*fps))
            t_list.append(t_[t_['frame']==int(t_['turn_start_frame'].unique())])
    #%
    turn_starts = pd.concat(t_list).reset_index(drop=True)

    return turn_starts



#%%

if __name__ == '__main__':
    #%% 
    # Set plotting
    plot_style='white'
    putil.set_sns_style(plot_style, min_fontsize=18)
    bg_color = [0.7]*3 if plot_style=='dark' else 'k'

    #% plotting settings
    curr_species = ['Dele', 'Dmau', 'Dmel', 'Dsant', 'Dyak']
    species_cmap = sns.color_palette('colorblind', n_colors=len(curr_species))
    print(curr_species)
    species_palette = dict((sp, col) for sp, col in zip(curr_species, species_cmap))

    #%% 
    # LOAD ALL THE DATA
    #savedir = '/Volumes/Julie/free-behavior-analysis/FlyTracker/38mm_dyad/processed'
    #figdir = os.path.join(os.path.split(savedir)[0], 'figures', 'relative_metrics')
    importlib.reload(util)

    assay = '2d-projector' # '38mm-dyad'
    experiment = 'circle_diffspeeds'
    create_new = False

    #%%
    minerva_base = '/Volumes/Juliana'
    local_basedir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data'
    localdir = os.path.join(local_basedir, assay, experiment, 'FlyTracker')

    if assay == '2d-projector':
        # Set sourcedirs
        srcdir = os.path.join(minerva_base, '2d-projector-analysis/FlyTracker/processed_mats') #relative_metrics'
    elif assay == '38mm-dyad':
        # src dir of processed .dfs from feat/trk.mat files (from relative_metrics.py)
        srcdir = os.path.join(minerva_base, 'free-behavior-analysis/FlyTracker/38mm_dyad/processed')

    # server loc for aggregated pkl
    out_fpath = os.path.join(os.path.split(srcdir)[0], 'relative_metrics.pkl')

    # get local file for aggregated data
    out_fpath_local = os.path.join(localdir, 'relative_metrics.pkl')
    print("Loading processed data from:", out_fpath_local)
    assert os.path.exists(out_fpath_local), "Local aggr. file does not exist:\n{}".format(out_fpath_local)

    # set figdir
    if plot_style=='white':
        figdir =os.path.join(minerva_base, '2d-projector-analysis', experiment, 'FlyTracker', 'theta_error', 'white') 
    else:
        figdir = os.path.join(minerva_base, '2d-projector-analysis', experiment, 'FlyTracker', 'theta_error')
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    print("Saving figs to: ", figdir)

    #% Set fig id
    figid = srcdir  

    #%%
    create_new = False

    # try reading if we don't want to create a new one
    if not create_new:
        if os.path.exists(out_fpath_local):
            df0 = pd.read_pickle(out_fpath_local)
            print("Loaded local processed data.")
        else:
            create_new = True
    print(create_new)

    #%%
    # cycle over all the acquisition dfs in srcdir and make an aggregated df
    if create_new:

        df0 = load_and_process_relative_metrics_df(srcdir)
        #% Save processed data
        df0.to_pickle(out_fpath)
        df0.to_pickle(out_fpath_local)

    # summary of what we've got
    print(df0[['species', 'acquisition']].drop_duplicates().groupby('species').count())

    #%%
    # =====================================================
    # Single fly dataframes
    # =====================================================
    new_ftjaaba = False
    jaaba_fname = 'mel_yak_20240330' # should be basename contained in ./JAABA/projector_data_{}.pkl 
 
    processed_ftjaaba_fpath = os.path.join(localdir, 'ftjaaba_{}.pkl'.format(jaaba_fname))
    if not new_ftjaaba:
        try:
            ftjaaba = pd.read_pickle(processed_ftjaaba_fpath)
            print(ftjaaba['species'].unique())
            ftjaaba.head()
        except Exception as e:
            print(e)
            new_ftjaaba = True
    print(new_ftjaaba)
    #%%
    if new_ftjaaba:
        #% Process
        ftjaaba = process_flydf_add_jaaba(df0, jaaba_fname=jaaba_fname)    
        #% SAVE
        ftjaaba.to_pickle(processed_ftjaaba_fpath)

#%%
    print(figid)
    print(figdir)

    # summarize what we got
    ftjaaba[['species', 'acquisition']].drop_duplicates().groupby('species').count()

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
    min_boutdur = 0.05
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

    # drop rows with only 1 instance of a given subboutnum
    min_nframes = min_boutdur * 60
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

    # find the closest matching value to one of the keys in stimhz_palette:
    meanbouts['stim_hz'] = meanbouts['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))   

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
    stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)

    # Get CHASING bouts 
    behav = 'chasing'
    min_frac_bout = 0.9
    chase_ = meanbouts[ (meanbouts['{}_binary'.format(behav)]>min_frac_bout) ].copy()
                    #    & (meanbouts['ang_vel_fly_shifted']< -25)].copy()
    #chase_ = filtdf[filtdf['{}_binary'.format(behav)]>0].copy()

    if 'shifted' in yvar:
        figtitle = '{} bouts, where min fract of bout >= {:.2f}\nshifted {} frames'.format(behav, min_frac_bout, nframes_lag)
    else:
        figtitle = '{} bouts, where min fract of bout >= {:.2f}'.format(behav, min_frac_bout)

    species_str = '-'.join(chase_['species'].unique())

    xlabel = r'$\theta_{E}$ at $\Delta t$ (rad)'
    ylabel = '$\omega_{f}$ (rad/s)'

    # SCATTERPLOT:  ANG_VEL vs. THETA_ERROR -- color coded by STIM_HZ
    fig = plot_regr_by_species(chase_, xvar, yvar, hue_var='stim_hz', 
                               plot_hue=plot_hue, plot_grid=plot_grid,
                            xlabel=xlabel, ylabel=ylabel, bg_color=bg_color)
    fig.suptitle(figtitle, fontsize=12)
    pl.subplots_adjust(wspace=0.25)

    for ax in fig.axes:
        #ax.invert_yaxis()
        ax.invert_xaxis()


    putil.label_figure(fig, figid)
    #figname = 'sct_{}_v_{}_stimhz_{}_min-frac-bout-{}'.format(yvar, xvar, species_str, min_frac_bout)
    figname = 'sct_{}_v_{}_{}{}_{}_min-frac-bout-{}'.format(yvar, xvar, shift_str, hue_str, species_str, min_frac_bout)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

    print(figdir, figname)

    #%% 
    # ------------------------------------------------
    # Compare small vs. large theta -error
    # -------------------------------------------------
    var1 = 'vel_shifted'
    var2 = 'ang_vel_shifted'
    err_palette={'small': 'r', 'large': 'cornflowerblue'}
    min_frac_bout = 0.
    use_bouts = True

    theta_error_small = np.deg2rad(10)
    theta_error_large = np.deg2rad(25)

    for curr_species in ftjaaba['species'].unique():

        nframes_lag_plot = 2 if 'shifted' in var1 or 'shifted' in var2 else 0

        if use_bouts:
            chase_ = meanbouts[(meanbouts['{}_binary'.format(behav)]>min_frac_bout)
                        & (meanbouts['species']==curr_species)].copy().reset_index(drop=True)
        else:
            chase_ = filtdf[(filtdf['{}_binary'.format(behav)]>min_frac_bout)
                        & (filtdf['species']==curr_species)].copy().reset_index(drop=True)
        chase_ = split_theta_error(chase_, theta_error_small=theta_error_small, theta_error_large=theta_error_large)

        # plot ------------------------------------------------
        fig = plot_ang_v_fwd_vel_by_theta_error_size(chase_, 
                            var1=var1, var2=var2, err_palette=err_palette, lw=2)
        fig.text(0.1, 0.9, '{} bouts, frac. of bout > {:.1f}, lag {} frames'.format(behav, min_frac_bout, nframes_lag_plot), fontsize=12)

        fig.text(0.1, 0.85, curr_species, fontsize=24)
        pl.subplots_adjust(wspace=0.6, left=0.1, right=0.9, top=0.9)

        putil.label_figure(fig, figid)
        figname = 'big-v-small-theta-error_ang-v-fwd-vel_{}_{}_{}_min-frac-bout-{}'.format(curr_species, theta_error_small, theta_error_large, min_frac_bout)
        pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
        pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

        print(figdir, figname)

    #
    #%% Fit REGR to each stim_hz level

    xvar = 'theta_error' #'facing_angle_vel_deg_abs'
    yvar = 'ang_vel_fly_shifted' #'ang_vel_abs'

    show_scatter = True

    behav = 'chasing'
    min_frac_bout = 0.5
    chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>min_frac_bout].copy()

    if 'vel' in xvar:
        xlabel = r'$\omega_{\theta}$'
    else:
        xlabel = r'$\theta_{E}$'
    ylabel = '$\omega_{f}$'

    plot_type = 'regr-sct' if show_scatter else 'regr'

    for ai, (sp, df_) in enumerate(chase_[chase_['stim_hz']>0].groupby('species')):
        if 'shifted' in yvar:
            figtitle = '{}: {} bouts, where min fract of bout >= {:.2f}\nshifted {} frames'.format(sp, behav, min_frac_bout, nframes_lag)
        else:
            figtitle = '{}: {} bouts, where min fract of bout >= {:.2f}'.format(sp, behav, min_frac_bout)

        g = plot_regr_by_hue(chase_, xvar, yvar, hue_var='stim_hz', stimhz_palette=stimhz_palette, show_scatter=show_scatter)
        g.fig.text(0.1, 0.93, figtitle)
        # set xlabel to be theta subscript E
        g.fig.axes[0].set_xlabel(xlabel) #r'$\theta_{E}$')
        g.fig.axes[0].set_ylabel(ylabel) #'$\omega_{f}$ (deg/s)')

        putil.label_figure(g.fig, figid)
        figname = '{}_{}_v_{}_stimhz_{}_min-frac-bout-{}'.format(plot_type, yvar, xvar, sp, min_frac_bout)
        pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
        pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

        print(figdir, figname)

    #%% ---------------------------------------------
    # PLOT ALLO vs. EGO

    # Test LONG bouts?? -- no.

    # subdivide into smaller boutsa
    bout_dur = 0.5
    ftjaaba_longbouts = util.subdivide_into_subbouts(ftjaaba, bout_dur=bout_dur)

    #% FILTER
    min_boutdur = 0.1
    min_dist_to_other = 2
    #%
    filtdf_longbouts = ftjaaba_longbouts[(ftjaaba_longbouts['id']==0)
                    #& (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                    #& (ftjaaba['targ_pos_theta']<=max_pos_theta)
                    & (ftjaaba_longbouts['dist_to_other']>=min_dist_to_other)
                    & (ftjaaba_longbouts['boutdur']>=min_boutdur)
                    & (ftjaaba_longbouts['good_frames']==1)
                    & (ftjaaba_longbouts['led_level']>0)
                    ].copy() #.reset_index(drop=True)

    # drop rows with only 1 instance of a given subboutnum
    min_nframes = min_boutdur * 60
    filtdf_longbouts = filtdf_longbouts[filtdf_longbouts.groupby(['species', 'acquisition', 'subboutnum'])['subboutnum'].transform('count')>min_nframes]
    #%%
    # Get mean value of small bouts
    if 'strain' in filtdf_longbouts.columns:
        filtdf_longbouts = filtdf_longbouts.drop(columns=['strain'])
    #%
    meanbouts_long = filtdf_longbouts.groupby(['species', 'acquisition', 'subboutnum']).mean().reset_index()
    meanbouts_long.head()

    # find the closest matching value to one of the keys in stimhz_palette:
    meanbouts_long['stim_hz'] = meanbouts_long['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))   


#%%


    import parallel_pursuit as pp
    importlib.reload(pp)

    behav = 'chasing'
    min_frac_bout = 0.5
    do_bouts = True

    markersize=5
    huevar='stim_hz'
    cmap='viridis'
    plot_com=True

    xvar= 'facing_angle'
    yvar = 'targ_pos_radius'

    is_flytracker=True
    data_type = 'BOUTS' if do_bouts else 'FRAMES' 

    plotdf = select_data_subset(filtdf, meanbouts, behav=behav, min_frac_bout=min_frac_bout, 
                                do_bouts=do_bouts, is_flytracker=is_flytracker)
    
    for sp, p_ in plotdf.groupby('species'):

        fig = plot_allo_ego_frames_by_species(p_, xvar=xvar, yvar=yvar,
                                          markersize=markersize, huevar=huevar, cmap=cmap, plot_com=plot_com,
                                          stimhz_palette=stimhz_palette)
        putil.label_figure(fig, figid)
        fig.suptitle('{}: {} {}, where min fract of bout >= {:.2f}'.format(sp, behav, data_type, min_frac_bout))

        for ax in fig.axes:
            #ax.set_ylim([0, 700])
            curr_ticks = ax.get_yticklabels()
            ax.set_yticklabels(['' for i, v in enumerate(curr_ticks) 
                                if i==len(curr_ticks)], fontsize=12)
            ax.set_xticklabels('')
        pl.subplots_adjust(right=0.9)

        figname = 'allo-v-ego-{}-{}_{}-v-{}_min-frac-bout-{}_{}'.format(behav, data_type, xvar, yvar, min_frac_bout, sp)
        pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
        pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

        print(figdir, figname)


#%%  Single animal

    curr_acq = '20240222-1611_fly7_Dmel_sP1-ChR_2do_sh_8x8'

    behav = 'chasing'
    min_frac_bout = 0.1
    do_bouts = False
    data_type = 'bouts' if do_bouts else 'frames'
    plotdf_ = select_data_subset(filtdf, meanbouts, behav=behav, min_frac_bout=min_frac_bout, 
                                do_bouts=do_bouts, is_flytracker=is_flytracker)
    

    currplotdf = plotdf_[plotdf_['acquisition'] == curr_acq].copy()
    print(currplotdf.shape)


    fig = plot_allo_ego_frames_by_species(currplotdf, xvar=xvar, yvar=yvar,
                                        markersize=markersize, huevar=huevar, cmap=cmap, plot_com=plot_com,
                                        stimhz_palette=stimhz_palette)
    
    for ax in fig.axes:
        ax.set_ylim([0, 500])
        curr_ticks = ax.get_yticklabels()
        ax.set_yticklabels(['' for i, v in enumerate(curr_ticks) 
                            if i==len(curr_ticks)], fontsize=12)
        ax.set_xticklabels('')

    putil.label_figure(fig, figid)
    fig.suptitle('{}\n{} {}, where min fract of bout >= {:.2f}'.format(curr_acq, behav, data_type, min_frac_bout))

    figname = 'allo-v-ego-{}-{}_{}-v-{}_min-frac-bout-{}_{}'.format(behav, data_type, xvar, yvar, min_frac_bout, curr_acq)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

    print(figdir, figname)



#%% =======================================================
# AGGREGATE all turns across flies
# =======================================================
    #min_ang_acc = 100
    min_vel = 15
    min_dist_to_other = 25
    min_facing_angle = np.deg2rad(90)
    min_ang_acc = 120
    fps = 60
    nframes_win = 0.1*fps
    v1 = 'theta_error' #'theta_error'
    v2 = 'ang_vel_fly' #'ang_vel_fly'
    min_n_turns = 5

    aggr_turns = aggregate_turns_across_flies(ftjaaba, v1=v1, v2=v2, min_n_turns=min_n_turns,
                                              min_ang_acc=min_ang_acc, min_vel=min_vel, min_dist_to_other=min_dist_to_other,
                                              min_facing_angle=min_facing_angle, fps=fps, nframes_win=nframes_win
                                              )
    #%% Box plot -- distribution of delta-Ts by stim_hz
    plotdf = aggr_turns[['stim_hz', 'turn_bout_num', 'delta_t_lag', 'species', 'acquisition']].drop_duplicates()

    fig, axn =pl.subplots(1, 2, figsize=(12,4), sharex=True, sharey=True)
    for ai, (sp, df_) in enumerate(plotdf.groupby('species')):
        ax=axn[ai]
        sns.boxplot(data=df_, x='stim_hz', y='delta_t_lag', ax=ax, palette=stimhz_palette, fliersize=0)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
        ax.axhline(y=0, color=bg_color, linestyle=':')
        #ax.set_box_aspect(1)
        ax.set_title(sp, loc='left')

    pl.subplots_adjust(left=0.1, right=0.9, wspace=0.3)
    putil.label_figure(fig, figid)

    figname = 'delta_t_lag_vs_stim_hz__by_species'
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    print(figdir, figname)

    #%% CHECK for negative delta_t
    neg_delta_t = []
    for acq, turns_ in aggr_turns.groupby('acquisition'):
        if turns_['delta_t_lag'].median() < 0:
            neg_delta_t.append(acq)
    but_pass_filt = [c for c in neg_delta_t if c in filtdf['acquisition'].unique()]
    but_pass_filt

#%%
    # Get theta_errors before turns
    turn_starts = get_theta_errors_before_turns(aggr_turns, fps=fps)

#%%
    # Plot distribution of theta-errors prior to each turn
    xvar = 'previous_theta_error'
    fig, ax= pl.subplots()
    sns.histplot(data=turn_starts, x=xvar, bins=50, alpha=0.5, ax=ax,
                hue='species', palette={'Dmel': 'magenta', 'Dyak': 'cyan'},
                common_norm=False, stat='probability', cumulative=False,
                fill=False, element='step')
    ax.set_box_aspect(1)
    sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1,1), frameon=False)

    spstats.mannwhitneyu(turn_starts[turn_starts['species']=='Dmel'][xvar].dropna(),
                     turn_starts[turn_starts['species']=='Dyak'][xvar].dropna(),
                     use_continuity=False)

 #%%
    # Plot polar distribution of theta-errors prior to each turn
    fig, ax = pl.subplots(subplot_kw={'projection': 'polar'}, figsize=(5,5))
    # For each acquisition, create a new variable "theta_error_shifted" that shifts theta_error forward by delta_t_lag*fps
    # Then, plot the distribution of theta_error_shifted
    mean_deg = {}      
    colors = ['magenta', 'cyan']
    for si, (sp, d_) in enumerate(turn_starts.groupby('species')):
        #ax=axn[si]
        putil.circular_hist(ax, d_['previous_theta_error'].dropna(), bins=50,
                            theta_zero_location='E', theta_direction=1, 
                            facecolor=colors[si], alpha=0.5)
        ylim = ax.get_ylim()[-1]
        mean_ = spstats.circmean(d_['previous_theta_error'].dropna(), high=np.pi, low=-np.pi, nan_policy='omit')

        # Plot means
        ax.plot([0, mean_], [0, ylim], color=colors[si], linestyle='-', label=sp)
        mean_deg.update({sp: np.rad2deg(mean_)})

    ax.set_title(r'$\theta_{E}$' + ' prior to turns (ang_acc>{})\n{}: {:.2f}, {}: {:.2f}'\
                        .format(min_ang_acc, 'Dmel', mean_deg['Dmel'], 'Dyal', mean_deg['Dyak']), fontsize=16)
    legh = putil.custom_legend(['Dmel', 'Dyak'], colors) #, loc='upper right')
    ax.legend(handles=legh, bbox_to_anchor=(1.4, 1), loc='upper right', frameon=False, fontsize=16)
        #ax.set_title(sp)
    pl.subplots_adjust(top=0.8)

    putil.label_figure(fig, figid)
    figname = 'polarhist_theta_error_shifted_before_turn_acc-thr-{}'.format(min_ang_acc)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
      
#%%