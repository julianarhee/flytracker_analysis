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

from transform_data.relative_metrics import load_processed_data, calculate_angle_metrics_focal_fly 
import utils as util
import plotting as putil

import statsmodels.api as sm
import regplot as rpl
import scipy.stats as spstats


#%% FUNCTIONS 

#def load_and_process_relative_metrics_df(srcdir): #, out_fpath):
#    df0 = util.load_aggregate_data_pkl(srcdir, mat_type='df')
#    #print(df0['species'].unique())
#    #% save
#    #out_fpath = os.path.join(os.path.split(figdir)[0], 'relative_metrics.pkl')
#    #df0.to_pickle(out_fpath)
#    #print(out_fpath)
#
#    # save local, too
#    #df0.to_pickle(out_fpath_local)
#    #df0['acquisition'] = ['_'.join([f for f in f.split('_')[0:-1] if 'led' not in f]) for f in df0['acquisition']]
#    #%
#    # Manually calculate THETA_ERROR
#    # df['ori'] = -1 * df['ori']
#    d_list = []
#    for acq, df_ in df0.groupby('acquisition'):
#        f1 = df_[df_['id']==0].copy().reset_index(drop=True)
#        f2 = df_[df_['id']==1].copy().reset_index(drop=True)
#        # calculate theta error
#        f1 = calculate_theta_error(f1, f2)
#        f1 = calculate_theta_error_from_heading(f1, f2)
#        f2 = calculate_theta_error(f2, f1)
#        f2 = calculate_theta_error_from_heading(f2, f1)
#        # add
#        d_list.append(f1)
#        d_list.append(f2)
#    df0 = pd.concat(d_list)
#    return df0

# def addtional_angle_metrics_single(df_, winsize=5, fps=60, filter_funky=True):
#     '''
#     Calculate additional angle metrics for a single dataframe of a focal fly.
#     Arguments:
#         df_ -- dataframe with focal fly data
#         winsize -- smoothing window size (default: 5)
#         fps -- frames per second (default: 60)
#         filter_funky -- whether to filter out funky values (default: True)
#     '''
#     # Calculate target angular vel
#     df_ = util.smooth_and_calculate_velocity_circvar(df_, 
#                             smooth_var='targ_pos_theta', vel_var='targ_ang_vel',
#                             time_var='sec', winsize=winsize)
# 
#     # Calculate facing angle vel
#     df_ = util.smooth_and_calculate_velocity_circvar(df_, 
#                             smooth_var='facing_angle', vel_var='facing_angle_vel',)
# 
#     # Calculate "theta-error"
#     try:
#         df_tmp = util.smooth_and_calculate_velocity_circvar(df_, 
#                                 smooth_var='theta_error', vel_var='theta_error_dt', 
#                                 time_var='sec', winsize=winsize)
#         assert all(np.isnan(df_tmp['theta_error']))==False
#     except AssertionError as e:
#         print("AssertionError: {}".format(e))
#         df_['theta_error_dt'] = np.nan
#     except Exception as e:
#         print("Error calculating theta error: {}".format(e))
#         df_['theta_error_dt'] = np.nan
# 
#     # TMP splitiom to get vel into mm/s
#     df_['vel_fly'] = np.concatenate(
#                         (np.zeros(1), 
#                         np.sqrt(np.sum(np.square(np.diff(df_[['pos_x_smoothed', 'pos_y_smoothed']], axis=0)), 
#                         axis=1)))) / (winsize*4/fps)# /4 
#     df_.loc[df_['vel_fly']>50, 'vel_fly'] = np.nan 
#     df_['vel_fly'] = df_['vel_fly'].interpolate().ffill().bfill()
#         
#     df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='ori', 
#                                                         vel_var='ang_vel_fly', 
#                                                         time_var='sec', winsize=winsize)            
#     # Calculate difference in ori between consecutive rows 
#     df_['turn_size'] = df_.groupby('id')['ori'].transform(lambda x: x.diff())
# 
#     # Calculate relative vel
#     df_['rel_vel'] = df_['dist_to_other'].interpolate().diff() / df_['sec'].diff().mean()
# 
#     df_['ang_acc'] = df_['ang_vel'].diff() / df_['sec_diff'].mean()
#     df_['ang_acc_smoothed'] = util.smooth_orientations_pandas(df_['ang_acc'], winsize=3) 
# 
#     df_['facing_angle_acc'] = df_['facing_angle_vel'].diff() / df_['sec_diff'].mean()
# 
#     df_['ang_acc_fly'] = df_['ang_vel_fly'].diff() / df_['sec_diff'].mean()
#     df_['ang_acc_fly_smoothed'] = df_['ang_vel_fly_smoothed'].diff() / df_['sec_diff'].mean()
# 
#     df_['theta_error_acc'] = df_['theta_error_dt'].diff() / df_['sec_diff'].mean()
# 
#     if filter_funky:    
#         df_.loc[(df_['ang_vel_fly']>80) | (df_['ang_vel_fly']<-80), 'ang_vel_fly'] = np.nan
# 
#     return df_
  
# def calculate_angle_metrics_focal_fly(df0, winsize=5, grouper='acquisition',
#                                       fps=60, filter_funky=True):
#     '''
#     Loop through all the acquisitions (or specified by grouper) to calculate additional angle metrics for a focal fly.
#     Arguments:
#         df0 -- dataframe with focal fly data
#         winsize -- smoothing window size (default: 5)
#         grouper -- column to group by (default: 'acquisition')
#         fps -- frames per second (default: 60)
#         filter_funky -- whether to filter out funky values of ang vel (default: True)
#     '''
#     assert df0['id'].nunique()==1, "Too many fly IDs, specify 1"
# 
#     # df_ = df[df['acquisition']==acq].copy()
#     d_list = []
#     for acq, df_ in df0.groupby(grouper):
#         #print(acq)
#        df_ = additional_angle_metrics_single(df_, winsize=winsize, fps=fps, filter_funky=filter_funky) 
#        d_list.append(df_)
# 
#     df = pd.concat(d_list)
#     #%
#     #df.loc[(df['ang_vel_fly']>80) | (df['ang_vel_fly']<-80), 'ang_vel_fly'] = np.nan
# 
#     #% and get abs values
#     df['targ_pos_theta_deg'] = np.rad2deg(df['targ_pos_theta'])
#     df['facing_angle_deg'] = np.rad2deg(df['facing_angle'])
#     #df['rel_vel'] = np.nan
# 
#     df['rel_vel_abs'] = np.abs(df['rel_vel']) 
#     df['targ_ang_vel_abs'] = np.abs(df['targ_ang_vel'])
#     df['targ_pos_theta_abs'] = np.abs(df['targ_pos_theta'])
#     if 'targ_ang_size' in df.columns: #has_size:
#         df['targ_ang_size_deg'] = np.rad2deg(df['targ_ang_size'])
#     df['targ_ang_vel_deg'] = np.rad2deg(df['targ_ang_vel'])
#     df['targ_ang_vel_deg_abs'] = np.abs(df['targ_ang_vel_deg'])
#     df['facing_angle_deg'] = np.rad2deg(df['facing_angle'])
#     df['ang_vel_fly_abs'] = np.abs(df['ang_vel_fly'])
# 
#     df['facing_angle_vel_abs'] = np.abs(df['facing_angle_vel'])
#     df['facing_angle_vel_deg'] = np.rad2deg(df['facing_angle_vel'])
#     df['facing_angle_vel_deg_abs'] = np.abs(df['facing_angle_vel_deg'])
# 
#     df['theta_error_deg'] = np.rad2deg(df['theta_error'])
#     df['theta_error_dt_deg'] = np.rad2deg(df['theta_error_dt'])
#     df['theta_error_abs'] = np.abs(df['theta_error'])
# 
#     df['ang_vel_abs'] = np.abs(df['ang_vel'])
#     df['ang_vel_deg'] = np.rad2deg(df['ang_vel'])
# 
#     # ftjaaba['ang_vel_abs'] = np.abs(ftjaaba['ang_vel'])
#     df['ang_vel_deg'] = np.rad2deg(df['ang_vel'])
# 
#     df['ang_vel_fly_abs'] = np.abs(df['ang_vel_fly'])
#     df['ang_vel_fly_deg'] = np.rad2deg(df['ang_vel_fly'])
#     df['ang_vel_fly_abs_deg'] = np.rad2deg(df['ang_vel_fly_abs'])
# 
#     return df
# 
 
def add_jaaba_to_flydf(df, jaaba): #$jaaba_fpath) #jaaba_fname='projector_data_mel_yak_20240330'):
    #jaaba_fname = 'free_behavior_data_mel_yak_20240403' if assay=='38mm-dyad' else None
    '''
    Combine male fly dataframe with jaaba (1 occurrence of each frame).
    Split into bouts of courtship and add to dataframe, binarize JAABA events.

    Returns:
        _description_
    '''
    #% ------------------------------
    # JAABA THRESHOLDS
    # ------------------------------
    #jaaba = util.load_jaaba(assay, experiment, fname=jaaba_fname)

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

    # binarize behavs
    ftjaaba = util.binarize_behaviors(ftjaaba, jaaba_thresh_dict=jaaba_thresh_dict)

    #% TEMP THINGS:
    ftjaaba = shift_variables_by_lag(ftjaaba, lag=2)
    ftjaaba['ang_vel_fly_shifted_abs'] = np.abs(ftjaaba['ang_vel_fly_shifted'])

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


def shift_variables_by_lag(df, lag=2):
    '''
    shift fly response variables BACK by lag, so that the value at time t=0 corresponds to variable's value at lag N frames (future)
    x 0, y 2 (instead of 0).
    
    Arguments:
        df -- _description_

    Keyword Arguments:
        lag -- _description_ (default: {2})

    Returns:
        _description_
    '''
    df['ang_vel_abs_shifted'] = df.groupby('acquisition')['ang_vel_abs'].shift(-lag)
    df['ang_vel_fly_shifted'] = df.groupby('acquisition')['ang_vel_fly'].shift(-lag)
    df['vel_fly_shifted'] = df.groupby('acquisition')['vel_fly'].shift(-lag)

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

def plot_regr_by_species(chase_, xvar, yvar, hue_var=None, plot_hue=False, 
                         plot_grid=True, grid_lw=0.25,
                         cmap='viridis', stimhz_palette=None, bg_color=[0.7]*3, 
                         xlabel=None, ylabel=None, fitstr_xloc=0.1, fitstr_yloc=0.9, annot_fontsize=7,
                         regr_lw=1, scatter_color=None, marker_size=5, figsize=(10, 4),
                         pearsons_xloc=0.1, pearsons_yloc=0.8):
                        
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
    if plot_hue:
        if stimhz_palette is None:
            # create palette
            stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba[hue_var]>=0], hue_var, cmap=cmap)

    if stimhz_palette is not None:
        vmin = min(list(stimhz_palette.keys()))
        vmax = max(list(stimhz_palette.keys()))
    if scatter_color is None:
        scatter_color = bg_color
    # For each species, plot linear regr with fits in a subplot
    n_species = chase_['species'].nunique()
    fig, axn = pl.subplots(1, n_species, sharex=True, sharey=True, figsize=figsize)
    for ai, (sp, df_) in enumerate(chase_.groupby('species')):
        if n_species>1:
            ax = axn[ai]
        else:
            ax=axn
        # plot scatter
        if plot_hue:
            sns.scatterplot(data=df_, x=xvar, y=yvar, ax=ax,
                        hue='stim_hz', palette=stimhz_palette, 
                        legend=0, edgecolor='none', alpha=0.7, s=marker_size)
        else:
            sns.scatterplot(data=df_, x=xvar, y=yvar, ax=ax, color=scatter_color, #bg_color,
                        legend=0, edgecolor='none', alpha=0.7, s=marker_size)

        if plot_grid:
            ax.axvline(x=0, color=bg_color, linestyle='--', lw=grid_lw)
            ax.axhline(y=0, color=bg_color, linestyle='--', lw=grid_lw)

        ax.set_title(sp)

        # do fit
        res = rpl.regplot(data=df_, ax=ax, x=xvar, y=yvar,
                    color=bg_color, scatter=False, line_kws={'lw': regr_lw}) #, ax=ax)
        # res.params: [intercept, slope]
        ax.set_box_aspect(1)
        lr, r2 = get_R2_ols(df_, xvar, yvar)
        r2_str = 'OLS: y = {:.2f}x + {:.2f}\nR2={:.2f}'.format(lr.coef_[0], 
                                                                    lr.intercept_,
                                                                    r2)
        ax.text(fitstr_xloc, fitstr_yloc, r2_str, fontsize=annot_fontsize, transform=ax.transAxes) 
        #fit_str = 'OLS: y = {:.2f}x + {:.2f}'.format(res.params[1], res.params[0])
        #print(fit_str) #lope, intercept)
        #ax.text(0.05, 0.85, fit_str, fontsize=8, transform=ax.transAxes)

        # set xlabel to be theta subscript E
        pl.tick_params(axis='both', which='both', pad=0)
        ax.set_xlabel(xlabel, labelpad=2)
        ax.set_ylabel(ylabel, labelpad=2)
        putil.annotate_regr(df_, ax, x=xvar, y=yvar, fontsize=annot_fontsize,
                            xloc=pearsons_xloc, yloc=pearsons_yloc)

    if plot_hue:
        putil.colorbar_from_mappable(ax, cmap=cmap, norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                                axes=[0.92, 0.3, 0.01, 0.4], hue_title='stim. freq. (Hz)', fontsize=18)

    return fig

def plot_ang_v_fwd_vel_by_theta_error_size(chase_, var1='vel_shifted', var2='ang_vel_shifted', 
                        lw=1, err_palette={'small': 'r', 'large': 'b'}, 
                        figsize=(10, 4), fly_marker='o', fly_marker_size=5, fly_color='gray',
                        median_marker_size=3, scatter_size=3, scatter_alpha=0.5,
                        axis_off=True, plot_dir='E', use_mm=True,
                        plot_scatter_axes=True, x_scale=5,
                        scatter_xlim=None, scatter_ylim=None, scatter_int=1):

    '''
    3-subplot figure: (1) spatial distribution of small vs. large theta errors, (2) velocity histograms, (3) angular vel histograms

    Returns:
        fig
    ''' 
    fig = pl.figure(figsize=figsize, dpi=300) #, axn = pl.subplots(1, 3, figsize=(7,4))

    # plot theta error relative to focal male
    ax = fig.add_subplot(1, 3, 1)
    if plot_dir == 'E':
        xvar = 'targ_rel_pos_x_mm' if use_mm else 'targ_rel_pos_x'
        yvar = 'targ_rel_pos_y_mm' if use_mm else 'targ_rel_pos_y'
    elif plot_dir == 'N':
        xvar = 'targ_rel_pos_y_mm' if use_mm else 'targ_rel_pos_y'
        yvar = 'targ_rel_pos_x_mm' if use_mm else 'targ_rel_pos_x'
    # Scatter plot of relative position
    sns.scatterplot(data=chase_.iloc[0::scatter_int], x=xvar, y=yvar, ax=ax, 
                    hue='error_size', palette=err_palette, s=scatter_size,
                    alpha=scatter_alpha)
    if plot_scatter_axes:
        ax.set_xlabel('Relative x (mm)' if use_mm else 'Relative x (px)', labelpad=2)
        ax.set_ylabel('Relative y (mm)' if use_mm else 'Relative y (px)', labelpad=2)
        pl.tick_params(axis='both', which='both', pad=0)    
    else:
        # only plot scale bar
        ax.set_xticks([0, x_scale])
        ax.set_xticklabels(['', '{} mm'.format(x_scale) if use_mm else '{} px'.format(x_scale)])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticks([])
        sns.despine(ax=ax, bottom=True, left=True, trim=True)
        ax.spines['left'].set_visible(False)
    if scatter_xlim is not None:
        if plot_dir == 'E':
            ax.set_xlim([0, scatter_xlim])
            ax.set_ylim([-scatter_ylim, scatter_ylim])
        else:
            ax.set_xlim([-scatter_xlim, scatter_xlim])
            ax.set_ylim([0, scatter_ylim])
           
    ax.set_aspect(1)
    ax.plot(0, 0, marker=fly_marker, color=fly_color, markersize=fly_marker_size)
    if axis_off==True:
        ax.axis('off')
    ax.legend_.remove()
    #sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', frameon=False)

    #% plot HISTS ------------------------------------------------
    ax1 = fig.add_subplot(1, 3, 2)
    sns.histplot(data=chase_, x=var1,  ax=ax1, bins=50, linewidth=lw,
                stat='probability', cumulative=False, element='step', fill=False,
                hue='error_size', palette=err_palette, common_norm=False, legend=0)
    ax1.set_xlabel('Forward vel', labelpad=2)
    pl.tick_params(axis='both', which='both', pad=0)    

    ax2 = fig.add_subplot(1, 3, 3, sharey=ax1)
    sns.histplot(data=chase_, x=var2, ax=ax2, color='r', bins=50, 
                stat='probability', cumulative=False, element='step', fill=False,
                hue='error_size', palette=err_palette, common_norm=False)
    ax2.set_xlabel('Angular vel', labelpad=2)
    pl.tick_params(axis='both', which='both', pad=0)    

    sns.move_legend(ax2, bbox_to_anchor=(1,1), loc='upper left', frameon=False)

    # plot median values
    curr_ylim = np.round(ax1.get_ylim()[-1], 2)*1.15
    for v, ax in zip([var1, var2], [ax1, ax2]):
        med_ = chase_.groupby('error_size')[v].median()
        for mval, cval in err_palette.items():
           ax.plot(med_[mval], curr_ylim, color=cval, marker='v', markersize=median_marker_size) #ax.axvline(x=m, color=c, linestyle='--')
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

    '''
    Select subset of the dataframe where <behav> is True (JAABA-classified behavior, for ex.)
    '''
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
                            col1='r', col2='dodgerblue', bg_color=[0.7]*3,
                            fig_w=10, fig_h=5):

    if v1_label is None:
        v1_label = r'$\theta_{E}$' + '\n{}'.format(v1)
    if v2_label is None:
        v2_label = r'$\omega_{f}$' + '\n{}'.format(v2)

    # PLOT MEAN + SEM of aligned turn bout traces
    fig, axn =pl.subplots(1, 3, figsize=(fig_w, fig_h))
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


# some stats
def mixed_anova_stats(stimhz_means, yvar, within='stim_hz', between='species', 
                        subject='acquisition',
                        between1='Dmel', between2='Dyak'): 
    '''
    Do mixed anova, with between & within factors. Expect 2 bewteen groups (usually species),
    and the within factor is some measure within pair (like testing multiple stim_hz).
    Uses pingouin's mixed_anova function and the Mann-Whitney U test to compare.
    
    Returns:
        results_df -- DataFrame with Mann-Whitney U test results for each frequency
        aov -- ANOVA results
    '''
    aov = pg.mixed_anova(data=stimhz_means, dv=yvar,
                        within=within, between=between, 
                        subject=subject)
    # Perform the Mann-Whitney U test for each frequency
    # Store results
    results = []
    # Loop over each frequency and run the Mann-Whitney U test
    for freq, subset in stimhz_means.groupby(within):#['frequency'].unique():        
        group1 = subset[subset['species'] == between1][yvar]
        group2 = subset[subset['species'] == between2][yvar]
        
        stat, pval = spstats.mannwhitneyu(group1, group2, 
                                            alternative='two-sided') 
        results.append({
            'frequency': freq,
            'U_statistic': stat,
            'p_value': pval
        })

    # Convert to DataFrame for display
    results_df = pd.DataFrame(results).sort_values('frequency')
    
    return results_df, aov

def add_multiple_comparisons(results_df):
    '''
    For each comparison, has a p-value -- returns if significant after multiple comparisons
    using FDR and Bonferroni correction.
    
    Returns:
    - results_df: DataFrame with p-values and significance columns
    '''
    from statsmodels.stats.multitest import multipletests
    # Extract p-values
    pvals = results_df['p_value'].values
    
    # Apply corrections
    reject_fdr, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    reject_bonf, pvals_bonf, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')

    # Add to DataFrame
    results_df['p_fdr'] = pvals_fdr
    results_df['sig_fdr'] = reject_fdr
    results_df['p_bonf'] = pvals_bonf
    results_df['sig_bonf'] = reject_bonf

    return results_df

def get_R2_ols(plotd, xvar, yvar):
    from sklearn import linear_model
    X = plotd[xvar].interpolate().ffill().bfill().values
    y = plotd[yvar].interpolate().ffill().bfill().values
    lr = linear_model.LinearRegression()
    lr.fit(X.reshape(len(X), 1), y)
    r2 = lr.score(X.reshape(len(X), 1), y)
    return lr, r2

def annotate_axis(ax, annot_str, fontsize=6, color='k'):
    ax.annotate(annot_str, xy=(0.5, 0.95), xycoords='axes fraction', 
                fontsize=fontsize,
                ha='center', va='center', color=color)


def count_chasing_frames(ftjaaba, grouper=['species', 'acquisition'],
                         chase_var = 'chasing_binary'):
    # N frames of chasing each out of all frames by acquisition in ftjaaba 
    n_frames_chasing = ftjaaba[ftjaaba[chase_var]>0].groupby(grouper)['frame'].count().reset_index()
    n_frames_chasing.rename(columns={'frame': 'n_frames_chasing'}, inplace=True)
    
    n_frames_total = ftjaaba.groupby(grouper)['frame'].count().reset_index()
    n_frames_total.rename(columns={'frame': 'n_frames_total'}, inplace=True)
    # Merge the two dataframes
    chase_counts = n_frames_chasing.merge(n_frames_total, on=grouper, how='left')  
    chase_counts['frac_frames_chasing'] = chase_counts['n_frames_chasing'] / chase_counts['n_frames_total']
   
    return chase_counts


#%%

if __name__ == '__main__':
    #%% 
    # Set plotting
    plot_style='dark'
    min_fontsize=18
    putil.set_sns_style(plot_style, min_fontsize=min_fontsize)
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

    create_new = False

    assay = '2d_projector' # '38mm-dyad'
    experiment = 'circle_diffspeeds'

    #assay = '38mm_dyad' 
    #experiment = 'MF'

    #%%
    minerva_base = '/Volumes/Juliana'

    # server loc for aggregated pkl
    #out_fpath = os.path.join(srcdir, 'relative_metrics.pkl')

    # Specify local dirs
    #local_basedir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data'
    local_basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/free_behavior_analysis' 
    localdir = os.path.join(local_basedir, assay, experiment, 'FlyTracker')

    if assay == '2d_projector':
        # Set sourcedirs
        srcpath = '2d_projector_analysis/circle_diffspeeds/FlyTracker'       
        # local jaaba_file
        jaaba_fname = 'ftjaaba_mel_yak_20240330'

    elif assay == '38mm_dyad':
        # Set src
        srcpath= 'free_behavior_analysis/38mm_dyad/MF/FlyTracker'
        # local jaaba_file
        jaaba_fname = 'jaaba_20240403' #'jaaba_free_behavior_data_mel_yak_20240403'
        # Meta fpath
        meta_fpath = '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/meta/courtship-free-behavior-GG (Responses) - Form Responses 1.csv'
        
    srcdir =  os.path.join(minerva_base, srcpath)

    # get local file for aggregated data
    out_fpath_local = os.path.join(localdir, 'relative_metrics.pkl')
    assert os.path.exists(out_fpath_local), "Local aggr. file does not exist:\n{}".format(out_fpath_local)

    # set figdir
    if plot_style=='white':
        figdir =os.path.join(srcdir, 'theta_error', 'white') 
    else:
        figdir = os.path.join(srcdir, 'theta_error')

    if not os.path.exists(figdir):
        os.makedirs(figdir)
    print("Saving figs to: ", figdir)

    #% Set fig id
    figid = srcdir  


    #%% Load aggregate relative metrics 
    print("Loading processed data from:", out_fpath_local)
    df0 = pd.read_pickle(out_fpath_local)

    if assay == '2d-projector':
        # Only select subset of data (GG):
        #df0 = df0.dropna()
        df0 = util.split_condition_from_acquisition_name(df0)
        #ft_acq = df0['acquisition'].unique()
    #%
    # summary of what we've got
    print(df0[['species', 'acquisition']].drop_duplicates().groupby('species').count())

    #%%
    # =====================================================
    # Single fly dataframes
    # =====================================================
    new_ftjaaba = False

    processed_ftjaaba_fpath = os.path.join(localdir, 'ftjaaba.pkl')
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
        #% Load JAABA (matlab output, from GG)
        jaaba_fpath = os.path.join(localdir, '{}.pkl'.format(jaaba_fname))
        assert os.path.exists(jaaba_fpath), "JAABA path not found"
        jaaba = pd.read_pickle(jaaba_fpath)

        #% Subselect acquisitions that have JAABA
        df0 = df0[df0['acquisition'].isin(jaaba['acquisition'])]

        #%
        # Process df0 for male fly
        winsize = 5
        if assay == '2d_projector':
            grouper = ['species', 'acquisition'] 
        else:
            grouper = ['species', 'acquisition', 'fly_pair']
        f1 = df0[df0['id']==0].copy()
        f1 = rel.calculate_angle_metrics_focal_fly(f1, winsize=5, grouper=grouper,
                                               has_size=True)

    #%% Add JAABA to flydf
    if new_ftjaaba:
        #% Process
        ftjaaba = add_jaaba_to_flydf(f1, jaaba) #jaaba_fpath)    
        #% SAVE
        ftjaaba.to_pickle(processed_ftjaaba_fpath)

        print(processed_ftjaaba_fpath)

#%%
    print(figid)
    print(figdir)

    # summarize what we got
    ftjaaba[['species', 'acquisition']].drop_duplicates().groupby('species').count()

    #%% 
    if assay == '38mm_dyad':
        # Strain info
        # Load GG metadata
        meta = pd.read_csv(meta_fpath)
        meta.head() 
        incl_strains = ['WT', 'SD105N']
        incl_acqs = meta[meta['genotype_male'].isin(incl_strains)]['logfile'].unique()
        print(len(incl_acqs), "acquisitions with strains:", incl_strains) 
        
        ftjaaba = ftjaaba[ftjaaba['acquisition'].isin(incl_acqs)].copy()
        # ftjaaba[ftjaaba['acquisition'].isin(incl_acqs)][['species', 'acquisition']].drop_duplicates().groupby('species').count()     
    
    #%%    
    # Convert pixels to mm?
    # --------------------------------------------
    if assay == '2d_projector':
        video_srcdir = '/Volumes/Giacomo/JAABA_classifiers/projector/changing_dot_size_speed' 
    elif assay == '38mm_dyad':
        video_srcdir = '/Volumes/Giacomo/JAABA_classifiers/free_behavior'
        
    ftjaaba['PPM'] = np.nan
    no_calib = []
    for acq, df_ in ftjaaba.groupby('acquisition'):
        calib_fpath = glob.glob(os.path.join(video_srcdir, '{}*'.format(acq), 'calibration.mat'))
        if len(calib_fpath)>0: #os.path.exists(calib_fpath):
            acq_dir = os.path.split(calib_fpath[0])[0] 
            calib = util.load_calibration(acq_dir, calib_is_upstream=False)
            ftjaaba.loc[ftjaaba['acquisition']==acq, 'PPM'] = calib['PPM']
        else:
            print("No calibration found for acquisition: {}".format(acq))
            no_calib.append(acq)
    # Get average PPM across acquisitions to fill NaNs
    avg_ppm = ftjaaba['PPM'].mean()
    print("No calib for {} of {} acquisition. Filling with avg PPM: {:.2f}mm/pixel".format(
        len(no_calib), ftjaaba['acquisition'].nunique(), avg_ppm))
    ftjaaba['PPM'] = ftjaaba['PPM'].fillna(avg_ppm)
    # Convert to mm
    ftjaaba['targ_rel_pos_x_mm'] = ftjaaba['targ_rel_pos_x'] / ftjaaba['PPM']
    ftjaaba['targ_rel_pos_y_mm'] = ftjaaba['targ_rel_pos_y'] / ftjaaba['PPM'] 

    #%% =====================================================
    # subdivide 
    #% Split into bouts of courtship
    get_courtship_bouts = False
    if get_courtship_bouts:
        d_list = []
        for acq, df_ in ftjaaba.groupby('acquisition'):
            df_ = df_.reset_index(drop=True)
            df_ = util.mat_split_courtship_bouts(df_, bout_marker='courtship')
            dur_ = util.get_bout_durs(df_, bout_varname='boutnum', return_as_df=True,
                            timevar='sec')
            d_list.append(df_.merge(dur_, on=['boutnum']))
        ftjaaba = pd.concat(d_list)

    # --------------------------------------------------------
    # split into small bouts
    # --------------------------------------------------------
    #%
    # subdivide into smaller boutsa
    bout_dur = 0.10
    ftjaaba = util.subdivide_into_subbouts(ftjaaba, bout_dur=bout_dur)

    #% FILTER
    min_boutdur = 0.05
    min_dist_to_other = 2
    min_pos_theta = np.deg2rad(-90)
    max_pos_theta = np.deg2rad(90)
    #%
    if assay == '2d_projector':
        filtdf = ftjaaba[(ftjaaba['id']==0)
                        & (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                        & (ftjaaba['targ_pos_theta']<=max_pos_theta)
                        & (ftjaaba['dist_to_other']>=min_dist_to_other)
                        #& (ftjaaba['boutdur']>=min_boutdur)
                        & (ftjaaba['good_frames']==1) # only for PROJECTOR
                        & (ftjaaba['led_level']>0)    # only for PROJECTOR
                        ].copy() #.reset_index(drop=True)
    else:
        filtdf = ftjaaba[(ftjaaba['id']==0)].copy() 
        
#                         & (ftjaaba['targ_pos_theta']>=min_pos_theta) 
#                         & (ftjaaba['targ_pos_theta']<=max_pos_theta)
#                         & (ftjaaba['dist_to_other']>=min_dist_to_other)
#                         ].copy() #.reset_index(drop=True)
        
    # drop rows with only 1 instance of a given subboutnum
    min_nframes = min_boutdur * 60
    #filtdf = filtdf[filtdf.groupby(['species', 'acquisition', 'subboutnum'])['subboutnum'].transform('count')>min_nframes]
    
    #%%
    # Get mean value of small bouts
#     if 'strain' in filtdf.columns:
#         filtdf = filtdf.drop(columns=['strain'])
#     if 'fpath' in filtdf.columns:
#         filtdf = filtdf.drop(columns=['fpath'])
#     if 'filename' in filtdf.columns:
#         filtdf = filtdf.drop(columns=['filename'])
    #%
    if 'stimsize' in filtdf.columns:
        grouper = ['species', 'acquisition', 'stimsize', 'acq_fly', 'subboutnum']
    else:
        grouper = ['species', 'acquisition', 'subboutnum'] 
    # Apply custom aggregation
    meanbouts = filtdf.groupby(grouper, as_index=False).agg(
                    {col: 'mean' if pd.api.types.is_numeric_dtype(dtype)
                    else util.only_if_unique
                    for col, dtype in filtdf.dtypes.items()}
                ).reset_index()
    meanbouts.head()

    # Make stimulus palette for changing size/speed - PROJECTOR only
    cmap='viridis'
    if assay == '2d_projector':
        stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)

        # find the closest matching value to one of the keys in stimhz_palette:
        meanbouts['stim_hz'] = meanbouts['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))   

    #%% ------------------------------------------------
    # SCATTER PLOT: ANG_VEL vs. THETA_ERROR
    # ------------------------------------------------- 
    pl.rcParams['axes.linewidth'] = 0.25
    xvar = 'theta_error'
    yvar = 'ang_vel_fly_shifted' #_shifted' #'ang_vel_fly_shifted' #'ang_vel' #'ang_vel_fly'
    plot_hue= False #True
    plot_grid = True
    plot_frames = False
    plot_example_animal = True
    min_frac_bout = 0.0
    # ---------------------------------------------------
    species_colors = ['plum', 'lightgreen']

    data_type_str = 'FRAMES' if plot_frames else 'BOUTS-min-frac-bout-{}'.format(min_frac_bout)
    nframes_lag = 2
    shift_str = 'SHIFT-{}frames_'.format(nframes_lag) if 'shifted' in yvar else ''
    hue_str = 'stimhz' if plot_hue else 'no-hue'
    hue_var = 'stim_hz' if plot_hue else None
    
    # Set palettes
    cmap='viridis'

    # Get CHASING bouts 
    behav = 'chasing'
    if plot_frames:
        chase_ = filtdf[filtdf['{}_binary'.format(behav)]>0].copy().reset_index(drop=True)
    else:
        chase_ = meanbouts[ (meanbouts['{}_binary'.format(behav)]>min_frac_bout) ].copy().reset_index(drop=True)
        
    if assay == '2d-projector':
        chase_ = chase_[chase_['stim_hz']>0]
    # annotations
    if 'shifted' in yvar:
        figtitle = '{}: {}\nshifted {} frames'.format(behav, data_type_str, nframes_lag)
    else:
        figtitle = '{}: {}'.format(behav, data_type_str)

    species_str = '-'.join(chase_['species'].unique())
    if 'shifted' in yvar:
        xlabel = r'$\theta_{E}$ at $\Delta t$ (rad)'
    else:
        xlabel = r'$\theta_{E}$ (rad)'
    ylabel = '$\omega_{f}$ (rad/s)'

    if min_fontsize==6:
        figsize=(2.1, 3)
    else:
        figsize=(6, 4)
    # SCATTERPLOT:  ANG_VEL vs. THETA_ERROR -- color coded by STIM_HZ
    fig = plot_regr_by_species(chase_, xvar, yvar, hue_var=hue_var, 
                            plot_hue=plot_hue, plot_grid=plot_grid, grid_lw=0.5,
                            xlabel=xlabel, ylabel=ylabel, bg_color=bg_color,
                            fitstr_xloc=0.01, fitstr_yloc=1.25,  
                            pearsons_xloc=0.01, pearsons_yloc=1.2, 
                            figsize=figsize, annot_fontsize=6, #in_fontsize, 
                            marker_size=0.5, regr_lw=0.5, scatter_color='w')
    
    for ax in fig.axes:
        ax.tick_params(pad=-0.1)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-10, 10])
    sns.despine(offset=2)
     
    if plot_example_animal:    
        # Plot 1 animal overlaid?
        yak_acq = chase_[chase_['species']=='Dyak']['acquisition'].unique()[1]
        mel_acq = chase_[chase_['species']=='Dmel']['acquisition'].unique()[2]
        example_yak = chase_[chase_['acquisition']==yak_acq].copy()
        example_mel = chase_[chase_['acquisition']==mel_acq].copy()
         
        for ai, (sp, sp_df, col) in enumerate(zip(['Dmel', 'Dyak'], [example_mel, example_yak], species_colors)):
            ax=fig.axes[ai]
            ax.scatter(sp_df[xvar], sp_df[yvar], s=0.1, color=col, alpha=0.75)
            #ax.set_title(sp, fontsize=4)
    pl.subplots_adjust(wspace=0.25, left=0.2)
    fig.text(0.01, 0.9, figtitle, fontsize=6, ha='left')

    #for ax in fig.axes:
        #ax.invert_yaxis()
    #    ax.invert_xaxis()
    putil.label_figure(fig, figid)
    #figname = 'sct_{}_v_{}_stimhz_{}_min-frac-bout-{}'.format(yvar, xvar, species_str, min_frac_bout)
    figname = 'scatter-with-ex_{}_v_{}_{}{}_{}_{}'.format(\
                yvar, xvar, shift_str, hue_str, species_str, data_type_str)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))
    print(figdir, figname)

    #%% 
    # ANG_VEL vs. ERROR: Fit EACH fly pair
    # -------------------------------------------------
    xvar = 'theta_error'
    yvar = 'ang_vel_fly_shifted' #_shifted' #'ang
    fits_ = []
    for acq, df_ in chase_.groupby('acquisition'):    
        # Fit linear regression for each acquisition
        lr, r2 = get_R2_ols(df_, xvar, yvar)
        fit_tmp = pd.DataFrame({'species': df_['species'].unique()[0],
                                'acquisition': acq,
                                'coef': lr.coef_[0],
                                'intercept': lr.intercept_,
                                'r2': r2}, index=[0])
        fits_.append(fit_tmp)        

    gain_fits = pd.concat(fits_).reset_index(drop=True)
    gain_fits['species'] = gain_fits['species'].astype('category')
    gain_fits['acquisition'] = gain_fits['acquisition'].astype('category')
    gain_fits['coef'] = gain_fits['coef'].astype(float)     
     
    # Plot
    norm = pl.Normalize( np.floor(gain_fits['r2'].min()), 
                         np.ceil(gain_fits['r2'].max()) )

    fig, axn = pl.subplots(1, 2, figsize=(5,3), sharex=True) #, dpi=300)
    #sns.boxplot(data=gain_fits, x='species', y='coef', ax=ax, 
    #             palette=species_palette, linewidth=0.5, fliersize=0.5)
    for ai, yvar in enumerate(['coef', 'intercept']):
        ax=axn[ai]
        sns.stripplot(data=gain_fits, x='species', y=yvar, ax=ax, 
                    hue='r2', hue_norm=norm, palette='viridis', alpha=1, size=8, 
                    linewidth=0.5, edgecolor=bg_color, jitter=True, legend=0)
        ax.set_xlabel('')
        ax.set_box_aspect(1)
        sns.despine(ax=ax, offset=2)
        if assay == '38mm_dyad':
            if yvar=='coef':
                ax.set_ylim([0, 10.5]) 
            elif yvar == 'intercept':
                ax.set_ylim([-0.6, 0.6])

        v1 = gain_fits[gain_fits['species']=='Dyak'][yvar].values
        v2 = gain_fits[gain_fits['species']=='Dmel'][yvar].values 
        # Test if the two distributions are different
        # Use the Kolmogorov-Smirnov test
        # Use the t-test:
        stat, pval = spstats.mannwhitneyu(v1, v2, alternative='two-sided')
        print(yvar, stat, pval)
        # Annotate stats with stars
        if pval < 0.001:
            annot_str = '***'
        elif pval < 0.01:
            annot_str = '**'
        elif pval < 0.05:
            annot_str = '*'
        else:
            annot_str = 'ns'
        annotate_axis(ax, annot_str, fontsize=min_fontsize-2, color=bg_color)
    # Plot continuous colorbar for hue variable
    # Plot colorbar, keep subplot axes the same size
    ax = fig.add_axes([0.92, 0.15, 0.1, 0.7])  # [left, bottom, width, height]
    ax.axis('off')
    sm = pl.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # Only needed for older versions of matplotlib
    cbar = pl.colorbar(sm, ax=ax)#, fontsize=min_fontsize-4) #, pad=0.01) #shrink=1)
    cbar.set_label('R2', fontsize=min_fontsize-2)
    cbar.ax.tick_params(labelsize=min_fontsize-2) 
    pl.subplots_adjust(wspace=0.7)      
   
    # Save 
    putil.label_figure(fig, figid) 
    figname = 'gain-linfits_{}_{}_{}{}_{}'.format(yvar, xvar, shift_str, hue_str, data_type_str)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    print(figdir, figname) 
     
#%%
    # Compare steering gain for FAST vs SLOW fwd vel
    # -------------------------------------------------
    # See Collie et al., 2024, Figure 4c
    plot_frames = True
    frame_subset = 'chase'
    min_frac_bout = 0
    # ---------------------------------------------------
    data_type = 'FRAMES' if plot_frames else 'BOUTS-min-frac-bout-{}'.format(min_frac_bout)

    if plot_frames: 
        if frame_subset == 'all':
            chase_ = ftjaaba.copy().reset_index(drop=True)
        else:
            frame_subset = 'chase'
            #chase_ = filtdf[filtdf['{}_binary'.format(behav)]>0].copy().reset_index(drop=True)
            chase_ = ftjaaba[ftjaaba['{}_binary'.format(behav)]>0].copy().reset_index(drop=True)

    else:
        chase_ = meanbouts[ (meanbouts['{}_binary'.format(behav)]>min_frac_bout) ].copy().reset_index(drop=True)
    if 'stim_hz' in chase_.columns: #assay == '2d_projector':
        chase_ = chase_[chase_['stim_hz']>0]

    #%%
    # PLOT: HISTOGRAM of velocity for each species 
    # and get upper and lower quartiles for each species
    fwd_vel_var = 'vel'

    # First, get top and bottom quartile of vel_shifted for each species separately
    def get_upper_lower_quartiles(mel_vel, lower=0.25, upper=0.75):
        mel_q_lower = np.quantile(mel_vel, lower)
        mel_q_upper = np.quantile(mel_vel, upper)
        #print(mel_q_lower, mel_q_upper)
        return mel_q_lower, mel_q_upper
  
    lower_q = 0.3
    upper_q = 0.7 
    mel_vel = chase_[chase_['species']=='Dmel'][fwd_vel_var].interpolate().ffill().bfill().values
    mel_lower, mel_upper = get_upper_lower_quartiles(mel_vel, 
                                        lower=lower_q, upper=upper_q)
      
    yak_vel = chase_[chase_['species']=='Dyak'][fwd_vel_var].interpolate().ffill().bfill().values 
    yak_lower, yak_upper = get_upper_lower_quartiles(yak_vel,
                                        lower=lower_q, upper=upper_q) 

    fig, ax = pl.subplots()
    sns.histplot(data=chase_, x=fwd_vel_var, alpha=0.6,
                 hue='species', bins=50, ax=ax, palette=species_palette,
                 )
    ax.axvline(mel_lower, color=species_palette['Dmel'], linestyle='--')
    ax.axvline(mel_upper, color=species_palette['Dmel'], linestyle='--')
    ax.axvline(yak_lower, color=species_palette['Dyak'], linestyle='--')
    ax.axvline(yak_upper, color=species_palette['Dyak'], linestyle='--') 
    sns.move_legend(ax, 'upper left', bbox_to_anchor=(1,1), 
                    title='species', fontsize=6, frameon=False)     
    vel_ranges = {'Dmel': (mel_lower, mel_upper), 
                  'Dyak': (yak_lower, yak_upper)}
    
    ax.set_box_aspect(1)
    putil.label_figure(fig, figid)
    
    figname = 'vel_hist_by_species_{}-{}'.format(data_type, frame_subset) 
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    #%%
    
    chase_counts = count_chasing_frames(ftjaaba)
    # Plot results
    fig, ax = pl.subplots(figsize=(2, 2), dpi=300)
    sns.barplot(data=chase_counts, x='species', y='frac_frames_chasing', 
                 palette=species_palette, ax=ax)
    ax.set_ylabel('Frac. chasing')
    ax.set_xlabel('')
    
    putil.label_figure(fig, figid)
    figname = 'frac_frames_chasing' 
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    print(figdir, figname)
    
     
    #%%  
    # PLOT: Linear regression of steering gain:  ang_vel @ lag vs. theta_error
    # Compare slopes for FAST vs. SLOW forward velocity in each species
    # ------------------------------------------------
    vir_colors = sns.color_palette('viridis', n_colors=5)
    forward_vel_palette = {'slow': vir_colors[1], 'fast': vir_colors[-2]}

    yvar = 'ang_vel_fly_shifted'
    chase_['forward_vel_type'] = None
    for sp, sp_chase in chase_.groupby('species'): 
        fwd_vel_slow, fwd_vel_fast = vel_ranges[sp]
        chase_.loc[((chase_['species']==sp)
                   & (chase_[fwd_vel_var]<fwd_vel_slow)), 'forward_vel_type'] = 'slow'
        chase_.loc[((chase_['species']==sp)
                     & (chase_[fwd_vel_var]>fwd_vel_fast)), 'forward_vel_type'] = 'fast' 
                 
    # Plot steering gain by forward velocity (linear regression)    
    if min_fontsize==6:
        figsize=(4,2)
    else:
        figsize=(6,4)
    fig, axn =pl.subplots(1, 2, sharex=True, sharey=True,
                          figsize=figsize, dpi=300)
    for si, (sp, sp_chase) in enumerate(chase_.groupby('species')):

        ax = axn[si] 
        for fwd_vel, vel_df in sp_chase.groupby('forward_vel_type'):
            sns.regplot(data=vel_df, ax=ax,
                        x='theta_error', y=yvar, 
                        color=forward_vel_palette[fwd_vel],
                        scatter=False, label=fwd_vel,
                        scatter_kws={'s': 0.5, 'alpha': 0.2,
                                     'color': forward_vel_palette[fwd_vel]},
                        truncate=True)
        fwd_vel_slow, fwd_vel_fast = vel_ranges[sp]
        ax.set_title('{}: slow={:.2f}, fast={:.2f}'.format(sp, fwd_vel_slow, fwd_vel_fast), fontsize=6) 
        if si==1:
            ax.legend(title='forward vel', fontsize=6, frameon=False,
                  bbox_to_anchor=(1,1), loc='upper left')
        ax.set_box_aspect(1)
        ax.set_xlabel('error (object pos., rad)')
        ax.set_ylabel(yvar)
    fig.text(0.1, 0.92, 'Steering gain by vel ({}, frame subset: {})'.format(data_type, frame_subset), 
             fontsize=6) 
    pl.subplots_adjust(wspace=0.5, top=0.85)
     
    putil.label_figure(fig, figid)
    figname = 'regr_gain_fast_v_slow_{}-{}'.format(data_type, frame_subset) 
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    
    #%% 
    # Fit linear regression to get steering gain for SLOW vs. FAST vel per fly
    from sklearn import linear_model
    import sklearn.metrics as skmetrics
    
    # PER FLY: Compare steering gain for FAST vs SLOW fwd vel
    d_list = [] 
    for (sp, acq), df_ in chase_.groupby(['species', 'acquisition']):
        # Fit linear regression between theta error and angular velocity for slow and fast forward_vel
        # Fit the model for each forward velocity type
        for fwd_vel, vel_df in df_.groupby('forward_vel_type'):
            lr = linear_model.LinearRegression()

            X = vel_df['theta_error'].interpolate().ffill().bfill().values.reshape(-1, 1)
            y = vel_df['ang_vel_fly_shifted'].interpolate().ffill().bfill().values
            lr.fit(X, y)
            y_pred = lr.predict(X)
            # Get the slope (gain) and intercept
            gain = lr.coef_[0]
            intercept = lr.intercept_
            # Get R2 scopre
            r2 = skmetrics.r2_score(y.flatten(), y_pred.flatten())
            tmpdf = pd.DataFrame({'species': sp, 'acquisition': acq,
                                    'forward_vel_type': fwd_vel, 
                                    'gain': gain, 
                                    'intercept': intercept,
                                    'R2': r2}, index=[0])
            # Append to the list
            d_list.append(tmpdf)
            
            #print(f"Species: {sp}, Acquisition: {acq}, Forward Velocity: {fwd_vel}, Gain: {gain}, Intercept: {intercept}")
    gain_df = pd.concat(d_list, ignore_index=True).reset_index(drop=True)
     
    #%% 
    # PLOT:  For each fly, plot pairwise gain for slow vs. fast velocity 
    # --------------------------------------
    fig, ax = pl.subplots()     
    sns.stripplot(data=gain_df, x='species', y='gain', 
                    hue='forward_vel_type',
                    palette=forward_vel_palette, dodge=True,
                    jitter=False, alpha=0.5, s=8, ax=ax, legend=False)
    for sp, sp_df in gain_df.groupby('species'):
        dodge = 0.25
        x_pos = ax.get_xticks()[list(gain_df['species'].unique()).index(sp)]

        for acq, acq_df in sp_df.groupby('acquisition'):
            print(sp)
            # Find the slow and fast values
            slow_data = acq_df[acq_df['forward_vel_type'] == 'slow']
            fast_data = acq_df[acq_df['forward_vel_type'] == 'fast']
            
            if not slow_data.empty and not fast_data.empty:
                # Draw line between points
                x1 = x_pos - dodge
                x2 = x_pos + dodge
                y2 = slow_data['gain'].values[0]
                y1 = fast_data['gain'].values[0]
                ax.plot([x1, x2], [y1, y2], 
                        color=bg_color, alpha=0.5, linewidth=1)
                
   # Add pointplot to show group statistics (mean and CI)
    sns.pointplot(data=gain_df, x='species', y='gain', hue='forward_vel_type',
              palette=forward_vel_palette, dodge=dodge, join=False,
              markers='o', scale=1.2, ci=95, ax=ax)

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title='forward vel',
                   frameon=False)
    ax.set_xlabel('')
    ax.set_box_aspect(1)
    ax.set_title('Steering gain, {}, frame subset: {}'.format(data_type, frame_subset), 
                 fontsize=6)
    
    putil.label_figure(fig, figid)
    figname = 'gain_by_forward_vel_by_fly_{}-{}'.format(data_type, frame_subset)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    #pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))
    print(figname)
    
    #%%
    # PLOT: TURNS in response to OBJECT POSITION
    # ------------------------------------------------
    # plot ang_vel_fly vs. binned theta_error
    #chase_ = ftjaaba.copy()
    
    #chase_['ang_vel_fly_shifted'] = chase_['ang_vel_fly_shifted'].interpolate().ffill().bfill() 

    yvar = 'ang_vel_fly_shifted'
    chase_['theta_error_deg'] = np.rad2deg(chase_['theta_error'])
    start_bin = -180
    end_bin = 180
    bin_size = 20
    chase_['binned_theta_error'] = pd.cut(chase_['theta_error_deg'],
                                    bins=np.arange(start_bin, end_bin, bin_size),
                                    labels=np.arange(start_bin+bin_size/2,
                                                        end_bin-bin_size/2, bin_size))    
    # Get average ang vel across bins
    avg_ang_vel = chase_.groupby([
                        'species', 'acquisition', 'binned_theta_error', 
                        ])[yvar].mean().reset_index()
    #avg_ang_vel = avg_ang_vel.dropna() 
    fig, ax = pl.subplots(figsize=(5, 4))
    sns.lineplot(data=avg_ang_vel, x='binned_theta_error', y=yvar,
                    hue='species', palette=species_palette, ax=ax,
                    errorbar='se', marker='o') #errwidth=0.5)
    ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
    ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
    ax.set_xticks(np.linspace(start_bin, end_bin, 7))
    #ax.set_xticklabels(np.arange(start_bin+bin_size/2, end_bin-bin_size/2+1, bin_size))
    #ax.set_box_aspect(1)
    sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1),
                    frameon=False, title='species', fontsize=min_fontsize-2)
    ax.set_title('Avg. ang vel. by object pos ({}, frame_subset: {})'.format(data_type, frame_subset),
                 fontsize=8)
   
    putil.label_figure(fig, figid) 
    figname = 'turns-{}_by_objectpos_{}'.format(yvar, data_type_str)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname))) 
     
    #%%
    # is ang_vel < 0 leftward or rightward turn?
    hue_var = 'ang_vel_fly'
    fig, ax = pl.subplots()
    sns.scatterplot(data=chase_.iloc[0::5], x='targ_rel_pos_x', y='targ_rel_pos_y',
                    hue='ang_vel_fly', ax=ax, palette='coolwarm',
                    edgecolor='none', s=5, alpha=0.9)
    sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1),
                    frameon=False, title=hue_var, fontsize=min_fontsize-2) 
    ax.set_aspect(1)
    
    #%%
    # plot log_gain vs. forward velocity:
    # Is there a difference in gain for near vs. far theta?
    # ------------------------------------------------
    chase_['gain'] = chase_['ang_vel_fly_shifted'] / chase_['theta_error']
    chase_['log_gain'] = np.log1p(chase_['gain'])  # Apply log transformation to compress the range
    fig, ax = pl.subplots()
    sns.scatterplot(data=chase_, x=fwd_vel_var, y='log_gain', ax=ax, 
                    hue='species', palette=species_palette,
                    s=0.5, alpha=0.5)
     
    putil.label_figure(fig, figid)
    figname = '{}_v_{}_by_forward_vel_{}'.format('log_gain',
                                                     'vel_shifted', data_type)
    
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

    #%% 
    # PLOT: individuals, Check that ang_vel vs. theta_error is linear for EACH fly pair
    # ang vel vs. theta error - 1 fly
    # ------------------------------------------------ 
    chase_ = filtdf[filtdf['{}_binary'.format(behav)]>0].copy().reset_index(drop=True)
  
    #chase_.groupby('species')['acquisition'].nunique()
    xvar= 'theta_error'
    yvar = 'ang_vel_fly_shifted' 
    curr_species = 'Dyak'

    #acq = chase_['acquisition'].unique()[0]  
    for sp, sp_df in chase_.groupby('species'):
        if sp=='Dyak':
            nr=2
            nc = 5
        elif sp=='Dmel':
            nr=3
            nc=5
        fig, axn = pl.subplots(nr, nc, figsize=(nr*1.5, nc*0.5), 
                                                sharex=True, sharey=True) 
        for ai, (acq, df_) in enumerate(sp_df.groupby('acquisition')): 
            ax=axn.flat[ai]
            sns.regplot(data=df_, x=xvar, y=yvar, ax=ax, color=bg_color,
                            scatter_kws={'s': 0.5, 'color': bg_color},
                            truncate=False, line_kws={'color': bg_color, 'lw': 0.5})
            ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
            ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
            #ax.set_title(acq, fontsize=4)

            # do fit
            ax.set_box_aspect(1)
            lr, r2 = get_R2_ols(df_, xvar, yvar)
            r2_str = '{}. OLS: y = {:.2f}x + {:.2f}\nR2={:.2f}'.format(ai, lr.coef_[0], 
                                                                        lr.intercept_,
                                                                        r2)
            ax.text(0.01, 1.2, r2_str, fontsize=4, transform=ax.transAxes) 
            putil.annotate_regr(df_, ax, x=xvar, y=yvar, fontsize=4,
                                xloc=0.01, yloc=1.15)
            # set xlabel to be theta subscript E
        figname = '{}_{}_{}_by_acquisition'.format(curr_species, yvar, xvar) 
        pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    
    #%% 
    # Big vs. Small theta error
    # ------------------------------------------------- 
    #%%
    # PLOT: Compare ANG vs. FWD velocity:  small vs. large theta -error
    # -------------------------------------------------
    behav = 'chasing'
    chase_ = ftjaaba[ftjaaba['{}_binary'.format(behav)]>0].copy().reset_index(drop=True)

    var1 = 'vel_shifted'
    var2 = 'ang_vel_shifted' #'ang_vel_fly_shifted'
    err_palette={'small': 'r', 'large': 'cornflowerblue'}
    min_frac_bout = 0.
    use_bouts = True
    data_type = 'BOUTS' if use_bouts else 'FRAMES'

    theta_error_small_deg = 10
    theta_error_large_deg = 25 #35
    theta_error_small = np.deg2rad(theta_error_small_deg)
    theta_error_large = np.deg2rad(theta_error_large_deg)
    
    nframes_lag_plot = 2 if 'shifted' in var1 or 'shifted' in var2 else 0

    if min_fontsize < 10:
        figsize = (5.5, 2.5)
    else:
        figsize = (8, 4)
    for curr_species in ftjaaba['species'].unique():
        if use_bouts:
            chase_ = meanbouts[(meanbouts['{}_binary'.format(behav)]>min_frac_bout)
                        & (meanbouts['species']==curr_species)].copy().reset_index(drop=True)
        else:
            chase_ = filtdf[(filtdf['{}_binary'.format(behav)]>min_frac_bout)
                        & (filtdf['species']==curr_species)].copy().reset_index(drop=True)
        chase_ = split_theta_error(chase_, theta_error_small=theta_error_small, 
                                   theta_error_large=theta_error_large)
        if 'stim_hz' in chase_.columns: #assay == '2d_projector':
            chase_ = chase_[chase_['stim_hz']>0]
                    #    & (meanbouts['ang_vel_fly_shifted']< -25)].copy()
        # plot ------------------------------------------------
        if assay == '2d_projector':
            scatter_int = 2
            scatter_size = 0.5
            scatter_alpha=0.3
            plot_dir = 'N'
            fly_marker = '^'
        else:
            scatter_int = 1
            scatter_size = 0.5
            scatter_alpha=0.5
            plot_dir = 'N'
            fly_marker = '^'
        fig = plot_ang_v_fwd_vel_by_theta_error_size(chase_, figsize=figsize,
                            var1=var1, var2=var2, err_palette=err_palette, lw=1, 
                            scatter_size=scatter_size, scatter_alpha=scatter_alpha, 
                            fly_marker=fly_marker, fly_marker_size=1, 
                            median_marker_size=5, 
                            plot_dir=plot_dir, axis_off=False, use_mm=True,
                            plot_scatter_axes=False, x_scale=5,
                            scatter_ylim=30, scatter_xlim=30, scatter_int=scatter_int)
        for ax in fig.axes:
            sns.despine(ax=ax, offset=2)
        fig.text(0.1, 0.92, 
                 '{} bouts, frac. of {} > {:.1f}, lag {} frames (small=+/-{}, large=+/-{})'.format(
                behav, data_type, min_frac_bout, nframes_lag_plot, theta_error_small_deg, theta_error_large_deg), fontsize=8)

        fig.text(0.1, 0.8, curr_species, fontsize=min_fontsize)
        pl.subplots_adjust(wspace=0.5, left=0.1, right=0.9, top=0.9)

        putil.label_figure(fig, figid)
        figname = 'big-v-small-theta-error_ang-v-fwd-vel_{}_thetaS{}_thetaL{}_{}'\
                    .format(curr_species, theta_error_small_deg, theta_error_large_deg, data_type)
        pl.savefig(os.path.join(figdir, '{}.png'.format(figname)), dpi=300)
        pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

        print(figdir, figname)
        
    #%%
 
    #%%
    # PLOT: ANG vs. FWD VEL: compare BIG v SMALL error -- AVERAGE BY FLY.
    # -------------------------------------------------
    theta_error_small_deg = 10
    theta_error_large_deg = 25 #35
    theta_error_small = np.deg2rad(theta_error_small_deg)
    theta_error_large = np.deg2rad(theta_error_large_deg)
    
    nframes_lag_plot = 2 if 'shifted' in var1 or 'shifted' in var2 else 0

    use_bouts = True
    set_axes = True 
    axes_str  = '_sharey' if set_axes else ''
    data_type = 'BOUTS' if use_bouts else 'FRAMES'
    min_frac_bout = 0
    if use_bouts:
        chase_ = meanbouts[(meanbouts['{}_binary'.format(behav)]>min_frac_bout)].copy().reset_index(drop=True)
    else:
        chase_ = ftjaaba[(ftjaaba['{}_binary'.format(behav)]>min_frac_bout)].copy().reset_index(drop=True)
    chase_ = split_theta_error(chase_, theta_error_small=theta_error_small, 
                                       theta_error_large=theta_error_large)
    if 'stim_hz' in chase_.columns: #assay == '2d_projector':
        chase_ = chase_[chase_['stim_hz']>0]

    var1 = 'vel_shifted'
    var2 = 'ang_vel_shifted'
    # Compare small vs. large theta error by fly
    # get mean from groupby, ignore non-numeric columns for mean 
    mean_by_fly = chase_.groupby(['species', 'acquisition', 'error_size'], as_index=False).agg(
                    {col: 'median' if pd.api.types.is_numeric_dtype(dtype)
                    else util.only_if_unique
                    for col, dtype in filtdf.dtypes.items()}
                ).reset_index()

    # stack mean_by_fly so that ang_vel_shifted and vel_shifted are in the same column
    stacked_mean_by_fly = pd.melt(mean_by_fly, 
                        id_vars=['species', 'acquisition', 'error_size'],
                        value_vars=[var1, var2], 
                        var_name='vel_type', value_name='vel_value')
   
    ylim_var1 = 35 if assay == '2d_projector' else 25 
    ylim_var2 = 6 if assay == '2d_projector' else 4 
    #%plot 
    stat_test = 'mwtest' # 'ttest' # 'kstest'
    stat_fontsize = 8
    if min_fontsize < 10:
        figsize==(2, 1.1)
    else:
        figsize=(5, 3)
    err_palette={'small': 'r', 'large': 'cornflowerblue'}
    for si, (sp, sp_df) in enumerate(mean_by_fly.groupby('species')):
        fig, axn = pl.subplots(1, 2, figsize=figsize, sharex=True)
        for ai, (plot_var, plot_name) in enumerate(zip([var1, var2], ['Forward vel (mm/s)', 'Angular vel (rad/s)'])):
            ax=axn[ai] 
            sns.stripplot(data=sp_df, y=plot_var, ax=ax, 
                    hue='error_size', palette=err_palette, dodge=True, 
                    legend=0, linewidth=0.25, s=3) 
            sns.barplot(data=sp_df, y=plot_var, ax=ax,
                        hue='error_size', dodge=True, palette=err_palette, 
                        alpha=1.0, legend=ai==1, errcolor=bg_color)
            
            if set_axes:
                if plot_var == var1:
                    ax.set_ylim([0, ylim_var1]) 
                if plot_var == var2:
                    ax.set_ylim([0, ylim_var2])
            ax.set_ylabel(plot_name, labelpad=2)
            ax.set_xlabel(sp, labelpad=0)
            sns.despine(offset=2, ax=ax, bottom=True)
            if ai==1:
                sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1), 
                                frameon=False, title='error size', fontsize=5)
           
            # stats
            v1 = sp_df[sp_df['error_size']=='large'][plot_var].values
            v2 = sp_df[sp_df['error_size']=='small'][plot_var].values 
            # Test if the two distributions are different
            # Use the Kolmogorov-Smirnov test
            # Use the t-test:
            if stat_test == 'ttest':
                stat, pval = spstats.ttest_ind(v1, v2, equal_var=False)
            elif stat_test == 'kstest':
                stat, pval = spstats.ks_2samp(v1, v2) 
            elif stat_test == 'mwtest':
                stat, pval = spstats.mannwhitneyu(v1, v2, alternative='two-sided')
            print('K-S test statistic:', stat)
            print('p-value:', pval)
            # Draw significance stars over the bars
            if pval < 0.01:
                annotate_axis(ax, '**', color=bg_color, fontsize=10)
            elif pval < 0.05:
                annotate_axis(ax, '*', color=bg_color, fontsize=10)
            else:
                annotate_axis(ax, 'ns', color=bg_color, fontsize=10)
            # Check if the distributions are significantly different
            if pval < 0.05:
                print('The distributions are significantly different (reject H0)')
            else:
                print('The distributions are not significantly different (fail to reject H0)')

        pl.subplots_adjust(wspace=0.6, left=0.15, right=0.9, top=0.85)
        fig.text(0.1, 0.92, 
                 '{} bouts, frac. of {} > {:.1f}, lag {} frames (small=+/-{}, large=+/-{})'.format(
                behav, data_type, min_frac_bout, nframes_lag_plot, theta_error_small_deg, theta_error_large_deg), fontsize=8)

        axn[0].tick_params(axis='y', which='both', pad=0)
        axn[1].tick_params(axis='y', which='both', pad=0)
        
        putil.label_figure(fig, figid, fontsize=4)
        figname = 'error-size_{}_BY-FLY_{}_{}_{}{}'.format(sp, var1, var2, data_type, axes_str)
        pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
        pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))
        print(figdir, figname)
                                       
    #%%
    # Is there a DISTANCE difference for small theta in yak? 
    plot_per_fly = True
    plot_strip = False 
    plot_alpha=1 if assay == '2d_projector' else 1.
    
    binned_var = 'targ_ang_size_deg' #'dist_to_other'
    # -------------------------------------------------
    # Get CHASING bouts
    chase_ = filtdf[(filtdf['{}_binary'.format(behav)]>min_frac_bout)].copy().reset_index(drop=True)
    #chase_ = meanbouts[(meanbouts['{}_binary'.format(behav)]>min_frac_bout)].copy().reset_index(drop=True)

    chase_ = split_theta_error(chase_, theta_error_small=theta_error_small, 
                                       theta_error_large=theta_error_large)
    if 'stim_hz' in chase_.columns: #assay == '2d_projector':
        chase_ = chase_[chase_['stim_hz']>0]
        
    # bin distane 
    bin_size = 5 if binned_var=='targ_ang_size_deg' else 5
    max_dist = 40 if binned_var=='targt_ang_size_deg' else 30 #np.ceil(ftjaaba['dist_to_other'].max())
    dist_bins = np.arange(0, max_dist+bin_size, bin_size)

    # Cut dist_to_other into bins and assign label to new columns:
    chase_['binned_{}'.format(binned_var)] = pd.cut(chase_[binned_var], 
                                        bins=dist_bins, 
                                        labels=dist_bins[:-1])   
    chase_['binned_{}'.format(binned_var)].fillna(0, inplace=True)
    chase_['binned_{}'.format(binned_var)] = chase_['binned_{}'.format(binned_var)].astype(int)

    #mean_by_bin = chase_.groupby(['species', 'acquisition', 'binned_dist_to_other', 'error_size']).mean().reset_index()    
    mean_by_bin = chase_.groupby(['species', 'acquisition', 'binned_{}'.format(binned_var), 'error_size'], as_index=False).agg(
                    {col: 'mean' if pd.api.types.is_numeric_dtype(dtype)
                    else util.only_if_unique
                    for col, dtype in filtdf.dtypes.items()}
                ).reset_index()
    # Plot
    if min_fontsize==6:
        figsize=(3, 2)
    else:
        figsize=(6, 4)
    #curr_df = chase_[chase_['species']=='Dmel'].copy() 
    for curr_species, curr_df in chase_.groupby('species'):
        fig, axn = pl.subplots(2, 2, figsize=figsize)
        for ai, (plot_var) in enumerate([var1, var2]):
            ax=axn[ai, 0]
            sns.histplot(data=curr_df, x=plot_var, hue='error_size', ax=ax,
                            palette=err_palette, alpha=plot_alpha, bins=50, 
                            common_norm=False, fill=False, 
                            stat='probability', element='step', legend=0) 
            # plot median value of histograms
            medians = curr_df.groupby(['species', 'error_size'])[plot_var].mean().reset_index()
            # plot median value for each 
            ylim = ax.get_ylim()[-1] + 0.05
            for err_size, err_df in medians.groupby('error_size'):
                ax.plot(err_df[plot_var], ylim, markersize=3,
                        color=err_palette[err_size], marker='v') 
            if plot_var=='vel_shifted':
                ax.set_xlabel('forward vel (mm/s)')
            elif plot_var=='ang_vel_shifted':
                ax.set_xlabel('angular vel (rad/s)')
           
            ax=axn[ai, 1]
            # bin dist_to_other
            if plot_per_fly:
                plotd = curr_df.groupby(['species', 'acquisition', 'binned_{}'.format(binned_var), 'error_size'], as_index=False).agg(  
                            {col: 'mean' if pd.api.types.is_numeric_dtype(dtype)
                            else util.only_if_unique
                            for col, dtype in filtdf.dtypes.items()}
                        ).reset_index() 
            else:
                plotd = curr_df.copy()
            sns.barplot(data=plotd, #[curr_df['error_size']=='small'], 
                            y=plot_var, x='binned_{}'.format(binned_var), ax=ax,
                            hue='error_size', palette=err_palette, alpha=plot_alpha,
                            legend=1, errorbar='se', errwidth=0.5, errcolor=bg_color,
                            hue_order=['small', 'large'], edgecolor=bg_color)
            if plot_strip:
                sns.stripplot(data=plotd, y=plot_var, x='binned_{}'.format(binned_var), ax=ax,
                            hue='error_size', palette=err_palette, dodge=True, legend=0,
                            linewidth=0.25, alpha=plot_alpha, s=5)
            sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1,1),
                            frameon=False, title='error size')

            if plot_var=='vel_shifted':
                ax.set_ylabel('forward vel (mm/s)')
            elif plot_var=='ang_vel_shifted':
                ax.set_ylabel('angular vel (rad/s)')
            if binned_var == 'dist_to_other':
                ax.set_xlabel('distance to other (mm)')
            elif binned_var == 'targ_ang_size_deg':
                ax.set_xlabel('target size (deg)') 
             
        for ax in axn.flat:
            sns.despine(offset=2, ax=ax)
        pl.subplots_adjust(wspace=0.5, hspace=0.6, 
                           left=0.1, right=0.9, top=0.9)
        fig.text(0.1, 0.93, '{}: {}'.format(curr_species, assay), 
                 fontsize=6)
       
        putil.label_figure(fig, figid) 
        figname = 'error-size_v_binned_{}_{}_frames'.format(curr_species, binned_var)
        pl.savefig(os.path.join(figdir, '{}.png'.format(figname)), dpi=300)
        pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)), dpi=300)
               
    #%% 
    # Compare # of small-error and # of large-error frames
    # -------------------------------------------------
    import pingouin as pg 

    species_palette = {'Dmel': 'plum', 
                       'Dyak': 'mediumseagreen'} 
   
    chase_ = filtdf[(filtdf['{}_binary'.format(behav)]>0)].copy()

    theta_error_small_deg = 10
    theta_error_large_deg = 25 #35
    theta_error_small = np.deg2rad(theta_error_small_deg)
    theta_error_large = np.deg2rad(theta_error_large_deg) 
    chase_ = split_theta_error(chase_, theta_error_small=theta_error_small, 
                                       theta_error_large=theta_error_large)
    counts_by_fly = chase_.groupby(['species', 'acquisition', 'error_size'])['frame'].count().reset_index()
    # append courtship frame count to counts_by_fly
    courtship_frames = filtdf[filtdf['courting']==1].groupby(['species', 'acquisition'])['frame'].count().reset_index()
    
    counts_by_fly = counts_by_fly.merge(courtship_frames[['species', 'acquisition', 'frame']],
                                        on=['species', 'acquisition'], 
                                        suffixes=('', '_courtship'), how='left')
       
    # Convert to SEC from frames 
    fps = 60.
    counts_by_fly['sec'] = counts_by_fly['frame'] / fps
    counts_by_fly['frac'] = counts_by_fly['frame'] / counts_by_fly['frame_courtship']
    
    count_var = 'frac' 
    if count_var == 'frac':
        ylabel = "Fraction of time"
    if min_fontsize < 10:
        figheight = 1.5
        markersize=3
    else:
        figheight = 3
        markersize = 5 
    g = sns.catplot(data=counts_by_fly, x='species', y=count_var, 
                    col='error_size', hue='species', 
                    #col_order=['small', 'large'],
                    kind='bar', height=figheight, aspect=0.8, sharey=False,
                    palette=species_palette, legend=False,
                    errorbar='se', errwidth=0.5, errcolor=bg_color)
    fig = g.figure
    for i, (error_size, err_df) in enumerate(counts_by_fly.groupby('error_size')):
        ai = 0 if error_size=='large' else 1
        ax = g.axes[0, ai]
        sns.stripplot(data=err_df, x='species', y=count_var, 
                    hue='species', palette=species_palette, 
                    dodge=True, jitter=True, s=markersize,
                    linewidth=0.25, ax=ax, alpha=1)
        ax.set_xlabel('')
        sns.despine(ax=ax, offset=2, bottom=True, trim=False)
    for ax in fig.axes:
        ax.tick_params(which='both', pad=0)
        ax.set_ylabel(ylabel, labelpad=2)
    # Rename facetgrid columns to not have the variable name
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    
    #sns.histplot(data=n_frames_per_cond, x='frame', hue='error_size', ax=ax)
    # Stats
    results_df, aov = mixed_anova_stats(counts_by_fly, yvar='frame',
                                        within='error_size', between='species', 
                                        subject='acquisition', between1='Dmel', between2='Dyak')
    results_df = add_multiple_comparisons(results_df)
    print(results_df)
    mc = 'p_fdr'
    large_p = results_df[results_df['frequency']=='large'][mc].values[0]  
    small_p = results_df[results_df['frequency']=='small'][mc].values[0]
    if large_p < 0.01:
        annotate_axis(g.axes[0, 0], '**', color=bg_color, fontsize=10)
    elif large_p < 0.05:
        annotate_axis(g.axes[0, 0], '*',  color=bg_color, fontsize=10)        
    else:
        annotate_axis(g.axes[0, 0], 'ns', color=bg_color, fontsize=10)
    if small_p < 0.01:
        annotate_axis(g.axes[0, 1], '**', color=bg_color, fontsize=10)
    elif small_p < 0.05: 
        annotate_axis(g.axes[0, 1], '*', color=bg_color, fontsize=10)
    else:
        annotate_axis(g.axes[0, 1], 'ns', color=bg_color, fontsize=10)
    pl.subplots_adjust(wspace=1)
    fig.text(0.01, 0.95, 
             'Proportion of courtship ({}) with large({})/small({}) err (mixed anova, {})'.format(count_var, theta_error_large_deg, theta_error_small_deg, mc), 
             fontsize=4)
    putil.label_figure(fig, figid, fontsize=4)
    figname = 'nframes_by_error-size'
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))
    print(figdir, figname)


    #%%
    import statsmodels.api as sm
    from sklearn import linear_model
    
    #%%
    # REGR: Compare theta_error vs. theta_error_dt for small vs. large errors
    # ------------------------------------------------
    #min_frac_bout = 0.
    curr_species = 'Dmel' 
    plot_frames = True
    data_type_str = 'FRAMES' if plot_frames else 'BOUTS-min-frac-bout-{}'.format(min_frac_bout)
   
    if plot_frames:
        chase_ = filtdf[(filtdf['{}_binary'.format(behav)]>min_frac_bout)].copy()
    else: 
        min_frac_bout = 0.9
        chase_ = meanbouts[ (meanbouts['{}_binary'.format(behav)]>min_frac_bout) ].copy()
    if 'stim_hz' in chase_.columns: #assay == '2d_projector':
        chase_ = chase_[chase_['stim_hz']>0]
        
    small_error_deg = 10
    large_error_deg = 25 
    theta_error_small = np.deg2rad(small_error_deg)
    theta_error_large = np.deg2rad(large_error_deg) #np.rad2deg(theta_error_small), np.rad2deg(theta_error_large)
    
    chase_ = split_theta_error(chase_, theta_error_small=theta_error_small, 
                               theta_error_large=theta_error_large)
    chase_['theta_error_dt_flipped'] = chase_['theta_error_dt'] * -1
    
    # Check small theta vs. vel
    large = chase_[chase_['error_size']=='large']
    small = chase_[chase_['error_size']=='small']
 
    xvar1 = 'theta_error'
    xvar2 = 'theta_error_dt_flipped'
    yvar = 'ang_vel_fly_shifted' #'ang_vel_fly_shifted' #'ang_vel' #'ang_vel_fly' 
    
    fig, axn = pl.subplots(2, 2)
    for ri, (error_size, plotd) in enumerate(zip(['large', 'small'], 
                                                 [chase_[chase_['species']==curr_species], 
                                small[small['species']==curr_species]])):
        for ai, xvar in enumerate([xvar1, xvar2]):
            ax=axn[ri, ai] 
            sns.regplot(data=plotd, x=xvar, y=yvar, ax=ax,
                        scatter_kws={'s': 0.5, 'alpha': 0.5}, color='k') 
            # do fit
            res = rpl.regplot(data=plotd, ax=ax, x=xvar, y=yvar,
                        color=bg_color, scatter=False) #, ax=ax)
            # res.params: [intercept, slope]
            ax.set_box_aspect(1)
            pearsons = putil.annotate_regr(plotd, ax, x=xvar, y=yvar, fontsize=8)
            print(pearsons)
           
            # OLS
            lr, r2 = get_R2_ols(plotd, xvar, yvar)
            r2_str = 'OLS: y = {:.2f}x + {:.2f}\nR2={:.2f}'.format(lr.coef_[0], 
                                                                     lr.intercept_,
                                                                     r2)
            print(r2_str)
            ax.text(0.05, 0.9, r2_str, fontsize=8, transform=ax.transAxes)
            ax.set_title('{} error, {}'.format(error_size, xvar), fontsize=6) 

    fig.text(0.1, 0.9, 'Large {} v Small {} errors, plot {}'.format(large_error_deg, small_error_deg, data_type))    
    pl.subplots_adjust(wspace=0.5, hspace=0.5, top=0.85)


     
   #%%
   
   # END OF 38mm_DYAD, FREE BEHAVIOR
    
#%%
    hue_var = 'stim_hz'
    cmap='viridis'
    stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba[hue_var]>=0], 
                                            hue_var, cmap=cmap)

    #
    #%% Fit REGR to each stim_hz level

    if assay == '2d_projector':
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
    # ALLO vs EGO 
    import parallel_pursuit as pp
    importlib.reload(pp)

    behav = 'chasing'
    min_frac_bout = 0.5
    do_bouts = False

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

        fig = plot_allo_ego_frames_by_species(p_.iloc[0::4], xvar=xvar, yvar=yvar,
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


#%%  
    # Single animal -- allo vs ego
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

    #%%
    # summary plot of theta error vs. stim_hz for 1 example
    currplotdf['facing_angle_abs'] = np.abs(currplotdf['facing_angle'])
    fig, axn = pl.subplots(1, 2, sharex=True, figsize=(6, 3))
    ax=axn[0]
    sns.pointplot(data=currplotdf, x='stim_hz', y='facing_angle_abs', ax=ax, 
                  errorbar=('ci', 95))
    ax=axn[1]
    sns.pointplot(data=currplotdf, x='stim_hz', y='targ_pos_radius', ax=ax)

    pl.subplots_adjust(wspace=0.5)
     
#%%
    # FIGURE:
    # 1) allocentric frames for 1 example fly
    # 2) Dmel: all bouts, male-centered
    # 3) Dyak: all bouts, male-centered

    min_frac_bout = 0

    frames_ = select_data_subset(filtdf, meanbouts, behav=behav, 
                                 min_frac_bout=min_frac_bout, 
                                do_bouts=False, is_flytracker=is_flytracker)
    bouts_ = select_data_subset(filtdf, meanbouts, behav=behav, 
                                min_frac_bout=min_frac_bout, 
                                do_bouts=True, is_flytracker=is_flytracker)
    

    single_acq_df = frames_[frames_['acquisition'] == curr_acq].copy()

    huevals = np.array(list(stimhz_palette.keys()))
    hue_norm = mpl.colors.Normalize(vmin=huevals.min(), vmax=huevals.max())
    alpha=0.75

    fig, axn = pl.subplots(1,3, figsize=(10,5), sharex=True, sharey=False,
                               subplot_kw={'projection': 'polar'})

    #  plot allocentric - get polar coords from transformed
    # ctr_x, _y are the transformed coordinates of the TARGET
    rad, th = util.cart2pol(single_acq_df['ctr_x'].values, single_acq_df['ctr_y'].values)
    single_acq_df['pos_radius'] = rad
    single_acq_df['pos_theta'] = th
    ax = axn[0] #fig.add_subplot(121,projection='polar')
    ax.set_title('Example fly (allocentric)')
    sns.scatterplot(data=single_acq_df, ax=ax,
                    x='pos_theta', y='pos_radius', s=markersize,
                    hue=huevar, palette=stimhz_palette,
                    hue_norm=hue_norm,
                    edgecolor='none', legend=0, alpha=alpha) 

    # plot egocentric
    #xvar='targ_pos_theta'
    #yvar='targ_pos_radius'
    xvar= 'facing_angle'
    yvar = 'targ_pos_radius'
    for ai, (sp, spec_df) in enumerate(bouts_.groupby('species')): 
        ax=axn[ai+1]
        ax.set_title('{}: egocentric'.format(sp))
        sns.scatterplot(data=spec_df, ax=ax,
                        x=xvar, y=yvar, s=markersize,
                        hue=huevar, palette=stimhz_palette, 
                        hue_norm=hue_norm,
                        edgecolor='none', legend=0, alpha=alpha)
        if plot_com:
            # plot Center of Mass
            for hueval, f_ in spec_df.groupby(huevar):
                cm_theta = pd.Series(np.unwrap(f_[xvar])).mean()
                cm_radius = f_[yvar].mean()
                ax.scatter(cm_theta, cm_radius, s=30, c=stimhz_palette[hueval],
                        marker='o', edgecolor='k', lw=0.5,
                        label='COM: {:.2f}'.format(hueval))

    for ai, ax in enumerate(fig.axes):
        if ai>0:
            ax.set_ylim([0, 800])
        curr_ticks = ax.get_yticklabels()
        ax.set_yticklabels(['' for i, v in enumerate(curr_ticks) 
                            if i==len(curr_ticks)], fontsize=12)
        ax.set_xticklabels('')

    putil.label_figure(fig, figid)
    fig.text(0.1, 0.9, 'ex: {}\n{} {}, where min fract of bout >= {:.2f}'.format(curr_acq, behav, data_type, min_frac_bout))

    figname = 'allo-ego_examplefly_Dyak_Dmel_minfracbout={}'.format(min_frac_bout)
    print(figname)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

#%%
    # Pointplots:  Do some sumamry stats: ERROR / DISTANCE vs. STIMULUS SPEED 
    # ------------------------------------------------------------------------
    import re 
    import pingouin as pg
    average_fly = True 
    is_flytracker = True
    do_bouts = False

    stat_type = 'avg-fly' if average_fly else 'avg-trial'
    species_colors = ['plum', 'mediumseagreen']

    col_mel = species_colors[0] # plum #mediumorchid'
    col_yak = species_colors[1] # lightgreen #'darkcyan'
    species_palette = {'Dmel': col_mel, 'Dyak': col_yak}
    min_frac_bout = 0.2
    data_type = 'BOUTS-min-{}'.format(min_frac_bout) if do_bouts else 'FRAMES' 
    frames_ = select_data_subset(filtdf[(filtdf['courting']==1)
                                      & (filtdf['stim_hz']>0.)], 
                                 meanbouts[(meanbouts['stim_hz']>0.)], behav=behav, 
                                 min_frac_bout=min_frac_bout, 
                                do_bouts=do_bouts, is_flytracker=is_flytracker)
    if 'strain' in frames_.columns:
        frames_ = frames_.drop(columns=['strain'])
          
    # convert to degrees
    frames_['facing_angle_abs'] = np.abs(frames_['facing_angle']) 
    frames_['facing_angle_abs_deg'] = np.rad2deg(frames_['facing_angle_abs'])
    # Convert pixels to mm
    #ppm = 23 #(mostly 25.7, some 22.2)
    frames_['targ_pos_radius_mm'] = frames_['targ_pos_radius'] / frames_['PPM']
    
    # get mean 
    counter = 'acq_fly' if average_fly else 'acquisition'
    if average_fly:
        stimhz_means_multi_per = frames_.groupby(['species', 'acquisition', 'stim_hz']).mean().reset_index()
        stimhz_means_multi_per['acq_fly'] = [f"{m.group(1)}_{m.group(2)}" for s in \
                                                stimhz_means_multi_per['acquisition'] if \
                                                (m := re.match(r"(\d{8})-\d{4}_(fly\d)", s))]
        stimhz_means_multi_per = stimhz_means_multi_per.drop(columns=['acquisition'])
        stimhz_means = stimhz_means_multi_per.groupby(['species', 'acq_fly', 'stim_hz']).mean().reset_index()
        stimhz_means.head() 
        
    else:
        stimhz_means = frames_.groupby(['species', 'acquisition', 'stim_hz']).mean().reset_index()       
    #%
    # count conditions
    cnts = stimhz_means.groupby(['stim_hz', 'species'])[counter].count().reset_index()
    #if not average_fly:
    exclude_levels = cnts[cnts[counter] < 6]['stim_hz'].unique()
    stimhz_means = stimhz_means[~stimhz_means['stim_hz'].isin(exclude_levels)]
     
    errorbar = 'se'
    if min_fontsize < 10:
        figsize = (2, 0.9)
    else:
        figsize = (8, 4)
    fig, axn = pl.subplots(1, 2, sharex=True, figsize=figsize)
       
    #% stats?       
    for ai, yvar in enumerate(['facing_angle_abs_deg', 'targ_pos_radius_mm']):
        ax=axn[ai]
        # plot
        sns.pointplot(data=stimhz_means, x='stim_hz', y=yvar, ax=ax, 
                    errorbar=errorbar, hue='species', palette=species_palette, scale=0.5,
                    legend=ai==1)
        if yvar=='facing_angle_abs_deg':
            ax.set_ylabel('Eror (deg)', labelpad=2)
            ylim = 32 #29
        else:
            ax.set_ylabel('Distance (mm)', labelpad=2)
            ylim = 15 #16
        if ai==1:
            sns.move_legend(axn[1], loc='upper left', bbox_to_anchor=(1,1), 
                    frameon=False, title='')

        # Stats
        results_df, aov = mixed_anova_stats(stimhz_means, yvar=yvar,
                                            within='stim_hz', between='species', 
                                            subject=counter, between1='Dmel', between2='Dyak')
        results_df = add_multiple_comparisons(results_df)
        print(results_df)
       
        # Map frequency to x-tick positions and add asterisks above max point
        xticks = ax.get_xticks()  # Numerical x positions (e.g., [0, 1, 2, ...])
        xticklabels = [float(label.get_text()) for label in ax.get_xticklabels()]  # e.g., [0.025, 0.05, ...]
        x_pos_map = dict(zip(xticklabels, xticks))
        significant_frequencies = results_df[results_df['sig_fdr']]['frequency'].values
        for freq in significant_frequencies:
            xpos = x_pos_map[freq]
            #value_dmel = stimhz_means[(stimhz_means['stim_hz'] == freq) & (stimhz_means['species'] == 'Dmel')]['targ_pos_radius_mm'].values[0]
            #value_dyak = stimhz_means[(stimhz_means['stim_hz'] == freq) & (stimhz_means['species'] == 'Dyak')]['targ_pos_radius_mm'].values[0]   
            #ymax = 16 #max(value_dmel, value_dyak) + 2
            ax.text(xpos, ylim + 1.5, '*', ha='center', va='bottom')
     
    for ax in axn:
        ax.set_xlabel('Stimulus speed (Hz)')    
        ax.set_box_aspect(1)
        # only label every other x-tick, plus first and last
        curr_ticks = ax.get_xticks()
        sub_ticks = curr_ticks[0::2]
        if curr_ticks[-1] not in sub_ticks:
            sub_ticks = np.append(sub_ticks, curr_ticks[-1])
        ax.set_xticks(sub_ticks)
        #ax.set_xticks([curr_ticks[0]] + [curr_ticks[i] for i in range(1, len(curr_ticks)-1, 2)] + [curr_ticks[-1]])     
    axn[0].tick_params(axis='both', which='major', pad=0)
    axn[1].tick_params(axis='both', which='major', pad=0) 
    pl.subplots_adjust(wspace=0.7, left=0.15, right=0.8)
    sns.despine(offset=2, trim=False)
    
    putil.label_figure(fig, figid)
    figname = 'facingangle_vs_stimhz_{}_{}_{}'.format(stat_type, behav, data_type)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

    print(figdir)
    print(figname)
    
 
    #%%
    import statsmodels.formula.api as smf

    model = smf.mixedlm("facing_angle_abs_deg ~ species * stim_hz", 
                        data=stimhz_means, groups="acquisition")
    result = model.fit()
    print(result.summary())


    #%% counts
    import re
    # Find the pattern of #x#, where # is a number in the str S
    # and x is a non-digit character
    pattern = r'(\d+)[x](\d+)'
    # Use str.extract to find the pattern in the 'stim_hz' column
    #stimhz_means[['stimsize_x', 'stimsize_y']] = stimhz_means['acquisition'].astype(str).str.extract(pattern)
    fly_cnts = stimhz_means[['species', 'acquisition','acq_fly']].drop_duplicates().groupby(['species', 'acq_fly']).count()
    print("N flies per species:", stimhz_means.groupby('species')['acq_fly'].nunique() )
    print("N acqs. per species:", stimhz_means.groupby('species')['acquisition'].nunique())
    
   
    #%% 
    # Size/Speed?
    # -------------------------------------------------
    ftjaaba['stimsize'] = [re.search(r'\dx\d', s).group() for s in ftjaaba['acquisition']]
    ftjaaba['acq_fly'] = [f"{m.group(1)}_{m.group(2)}" for s in \
                                                ftjaaba['acquisition'] if \
                                                (m := re.match(r"(\d{8})-\d{4}_(fly\d)", s))] 
   
    ftjaaba['stimsize'].unique() 
  
    means_by_stim = ftjaaba.groupby(['species', 'acq_fly', 'stimsize', 'stim_hz'])['chasing_binary'].mean().reset_index()

    #%%
    
    speed_palette = sns.color_palette('flare', n_colors=3)
    stimsize_palette = dict((k, c) for k, c in zip(sorted(means_by_stim['stimsize'].unique()),
                                                   speed_palette))
    fig, axn = pl.subplots(1, 2, sharex=True, sharey=False, figsize=(6, 3))
    for ai, (sp, spdf) in enumerate(means_by_stim.groupby('species')):
        ax=axn[ai]
        sns.lineplot(data=spdf, x='stim_hz', y='chasing_binary', ax=ax,
                    hue='stimsize', errorbar='se', palette=stimsize_palette,
                    legend=ai==1)
        if ai==1:
            sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1,1), frameon=False,
                            title='dot size')
        ax.set_title(sp)
        ax.set_ylim([0, 0.6])
        xlabels =['{:.1f}'.format(i) for i in np.linspace(0, 1, 6)]
        ax.set_xticks(np.linspace(0, 1, 6), xlabels, fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_box_aspect(1)
        ax.set_xlabel('Stim. freq. (Hz)')
        ax.set_ylabel('p(chasing)') 
    sns.despine(offset=2, trim=False) 
    pl.subplots_adjust(wspace=0.5, left=0.1, right=0.9)
    
    #pl.xticks(fontsize=10) 
   
    putil.label_figure(fig, figid)
    figname = 'pChasing_vs_stimsize_{}_{}'.format(behav, data_type)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    pl.savefig(os.path.join(figdir, '{}.svg'.format(figname))) 
     
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