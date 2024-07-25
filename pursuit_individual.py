#!/usr/bin/env python3
# -*- coding: utf-8 -*-"""
"""
Created on Thu May 30, 10:08:00 2024

@filename: pursuit_individual.py
@author: julianarhee
"""
#%%
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl

import utils as util
import plotting as putil
import theta_error as the

import statsmodels.api as sm
import regplot as rpl
import scipy.stats as spstats


#%%
# Set plotting
plot_style='white'
putil.set_sns_style(plot_style, min_fontsize=24)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'


#%%
assay = '2d-projector'
experiment = 'circle_diffspeeds'

# local savedir for giant pkl
localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/{}/{}/FlyTracker'.format(assay, experiment)

minerva_base = '/Volumes/Juliana'

# set figdir
if plot_style == 'white':
    figdir = os.path.join(minerva_base, '2d-projector-analysis', \
                experiment, 'FlyTracker', 'theta_error', 'white')
else:
    figdir = os.path.join(minerva_base, '2d-projector-analysis', \
                experiment, 'FlyTracker', 'theta_error')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print("Saving figs to: ", figdir)

#%%
if __name__ == '__main__':
    #%%
    # =====================================================
    # Single fly dataframes
    # =====================================================
    new_ftjaaba = False # Run theta_error.py section of need to create new
    jaaba_fname = 'mel_yak_20240330' # should be basename contained in ./JAABA/projector_data_{}.pkl 
 
    processed_ftjaaba_fpath = os.path.join(localdir, 'ftjaaba_{}.pkl'.format(jaaba_fname))
    try:
        ftjaaba = pd.read_pickle(processed_ftjaaba_fpath)
        print(ftjaaba['species'].unique())
        ftjaaba.head()
    except Exception as e:
        print(e)
        new_ftjaaba = True
    print(new_ftjaaba)

    if 'strain' in ftjaaba.columns:
        ftjaaba = ftjaaba.drop(columns={'strain': 'species'})

    #%%
    # colormaps
    cmap='viridis'
    stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)

    figid = processed_ftjaaba_fpath
    print(figid)
    print(figdir)


    # ======================================================= 
    # Look at 1 fly
    # =======================================================
    #%
    # acq = '20240216-1434_fly3_Dmel_sP1-ChR_2do_sh_4x4'
    acq = '20240222-1611_fly7_Dmel_sP1-ChR_2do_sh_8x8'
    # acq = '20240216-1434_fly3_Dmel_sP1-ChR_2do_sh_4x4'
    # acq = '20240212-1215_fly3_Dmel_sP1-ChR_3do_sh_8x8'

    # acq = '20240307-1537_fly2_Dyak_sP1-ChR_3do_gh_6x6'

    #acq = acqs[3] #'20240212-1215_fly3_Dmel_sP1-ChR_3do_sh_8x8'a
    #acq = '20240212-1215_fly3_Dmel_sP1-ChR_3do_sh_8x8'
    # acq = '20240216-1434_fly3_Dmel_sP1-ChR_2do_sh_4x4'
    # acq = '20240216-1434_fly3_Dmel_sP1-ChR_2do_sh_4x4'
    #acq = '20240212-1215_fly3_Dmel_sP1-ChR_3do_sh_8x8'
    # acq = '20240222-1611_fly7_Dmel_sP1-ChR_2do_sh_8x8'

    # acq = '20240307-1537_fly2_Dyak_sP1-ChR_3do_gh_6x6'
    # acq = '20240222-1055_fly1_Dyak_sP1-ChR_2do_sh_8x8'
    #acq = '20240216-1254_fly1_Dyak_sP1-ChR_2do_sh_8x8'

    #['20240222-1126_fly1_Dyak_sP1-ChR_2do_sh_6x6',
    #'20240227-1811_fly2_Dyak_sP1-ChR_3do_gh_6x6',
    #'20240301-1504_fly1_Dyak_sP1-ChR_3do_gh_6x6']

    # acq = '20240301-1504_fly1_Dyak_sP1-ChR_3do_gh_6x6'

    df_ = ftjaaba[ftjaaba['acquisition']==acq].copy()

    curr_figdir = os.path.join(figdir, acq)
    if not os.path.exists(curr_figdir):
        os.makedirs(curr_figdir)
    print(curr_figdir)


    #%% PAIRPLOT

    vars_to_compare = ['theta_error', 'theta_error_heading', 'facing_angle', 'facing_angle_vel', 'theta_error_dt', 
                       'turn_size', 'ang_vel', 'ang_vel_fly']

    plotdf = df_[(df_['chasing_binary']>0) & (df_['stim_hz']>0)]
    figname = 'pairplot_{}'.format(acq)

    if not os.path.exists(os.path.join(curr_figdir, '{}.png'.format(figname))):
        sns.set_context("paper", rc={"axes.labelsize":12})
        sns.pairplot(data=plotdf, vars=vars_to_compare, hue='stim_hz', palette=stimhz_palette)

        pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))

    #%% THETA_ERROR vs. ANG_VEL

    # subdivide into smaller boutsa
    bout_dur = 0.20
    ftjaaba = util.subdivide_into_subbouts(ftjaaba, bout_dur=bout_dur)

    min_boutdur = 0.05
    min_dist_to_other = 2

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


    #%%
    # compare SUBBOUTS vs. FRAMES
    xvar = 'theta_error'
    yvar = 'ang_vel_fly_shifted'

    behav = 'chasing'
    # Compare FRAMES vs. MINI-BOUTS
    frames_ = filtdf[ (filtdf['acquisition']==acq) & (filtdf['{}_binary'.format(behav)]>0)].copy()
    meanframes_ = frames_.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()
    mean_ = meanbouts[ (meanbouts['acquisition']==acq) & (meanbouts['{}_binary'.format(behav)]>0)].copy()
    mean_['stim_hz'] = mean_['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))

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

    n_stim = frames_['stim_hz'].nunique() # no zero

    min_nframes = 10


    do_bouts = True 
    xvar = 'theta_error'
    yvar = 'ang_vel_fly_shifted'

    plotd_ = mean_.copy() if do_bouts else frames_.copy()
    data_type = 'BOUTS' if do_bouts else 'FRAMES'

    if 'vel' in xvar:
        xlabel = r'$\omega_{\theta}$'
    else:
        xlabel = r'$\theta_{E}$'
    ylabel = '$\omega_{f}$'

    all_stims = sorted(list(stimhz_palette.keys()))
    fig, axn = pl.subplots(2, n_stim, sharex=True, sharey=True, figsize=(n_stim*3, 7))
    for ci, data_type in enumerate(['BOUTS', 'FRAMES']):
        plotd_ = mean_.copy() if data_type=='BOUTS' else frames_.copy()

        for ri, (stim, sd_) in enumerate(plotd_.groupby('stim_hz')):  
            if len(sd_) < min_nframes:
                continue
            ax=axn[ci, ri]
            ax.set_title(stim, loc='left', fontsize=12)
            sns.regplot(data=sd_, x=xvar, y=yvar, ax=ax, color=stimhz_palette[stim], scatter_kws={'alpha':0.5})

            # OLS 
            model_ = sm.OLS(sd_[yvar], sm.add_constant(sd_[xvar]))
            res_ = model_.fit()
            fit_str = 'OLS: y = {:.2f}x + {:.2f}\nR2={:.2f}'\
                        .format(res_.params[1], res_.params[0], res_.rsquared)
            ax.text(0.1, 0.85, fit_str, transform=ax.transAxes)
            #sns.scatterplot(data=sd_, x=xvar, y=yvar, ax=ax, hue='stim_hz', palette=stimhz_palette,
            #            legend=0)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

    fig.text(0.015, 0.7, 'bouts', rotation=90, fontsize=18)
    fig.text(0.015, 0.3, 'frames', rotation=90, fontsize=18)
   
    pl.subplots_adjust(hspace=0.5, bottom=0.2, top=0.9, left=0.05)

    putil.label_figure(fig, acq)
    fig.suptitle('{:.2f}s {}'.format(bout_dur, data_type), fontsize=24)

    figname = 'bouts_vs_frames_{}_v_{}_{}'.format(yvar, xvar, acq)
    pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))


    #%%
    import parallel_pursuit as pp
    import matplotlib as mpl

    do_bouts=False
    fig, axn = pl.subplots(1, 2, figsize=(10,5), sharex=True, sharey=False,
                                subplot_kw={'projection': 'polar'})
    huevar = 'stim_hz'
    #
    vmin = min(list(stimhz_palette.keys()))
    vmax = max(list(stimhz_palette.keys()))
    hue_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # plotdf = mean_.copy()
    plotd_ = mean_.copy() if do_bouts else frames_.copy()
    data_type = 'BOUTS' if do_bouts else 'FRAMES'

    pp.plot_allo_vs_egocentric_pos(plotd_[plotd_['stim_hz']>0], axn=axn, 
                                   xvar='targ_pos_theta', yvar='targ_pos_radius', 
                                   huevar=huevar, palette_dict=stimhz_palette, hue_norm=hue_norm, 
                                   markersize=5, com_markersize=50, com_lw=1)
    for ax in axn:
        yl = ax.get_yticklabels()
        ax.set_yticklabels([v if i%2==0 else '' for i, v in enumerate(yl)])
    pl.subplots_adjust(wspace=0.6, top=0.8, right=0.8)

    putil.label_figure(fig, acq)
    fig.suptitle('{:.2f}s {}'.format(bout_dur, data_type), fontsize=24)

    figname = 'allo_v_egocentric_targetpos_{}_v_{}_{}'.format(yvar, xvar, acq)
    pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))

    #%% 
    # timecourse 
    # =======================================================
    print(acq)
    #%
    # Only include good frames
    flydf = ftjaaba[(ftjaaba['acquisition']==acq) & (ftjaaba['id']==0)
                & (ftjaaba['good_frames']==1)].copy()
    print(flydf.shape)

    if 'strain' in flydf.columns:
        flydf = flydf.drop(columns=['strain'])

    # get pursuit frames
    behav = 'chasing'
    frames_ = flydf[ (flydf['acquisition']==acq) & (flydf['{}_binary'.format(behav)]>0)].copy()

    # get means
    mean_ = frames_.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()
    mean_['stim_hz'] = mean_['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))

    #%% ----------------------------
    # Does angular vel. INCREASE with stim_hz? with theta-error?
    # ----------------------------
    # Look at moments of high ang vel.

    import theta_error as the

    fps = 60.
    min_vel = 15
    min_dist_to_other = 25
    min_facing_angle = np.deg2rad(90)
    min_ang_acc = 120

    turn_bout_starts, high_ang_start_frames = the.select_turn_bouts_for_plotting(flydf,
                                            min_ang_acc=min_ang_acc, min_dist_to_other=min_dist_to_other,
                                            min_facing_angle=min_facing_angle, min_vel=min_vel)
    #%
    turn_counts = the.count_n_turns_in_window(flydf, turn_bout_starts, high_ang_start_frames, fps=fps)
    print(turn_counts.sort_values(by='n_turns', ascending=False))
    #%
    # PLOT TIME COURSES
    turn_ix = 17
    # Look at a large window around detected turn bouts
    start_ix, stop_ix = turn_counts[turn_counts['turn_ix']==turn_ix][['start_ix', 'stop_ix']].values[0] #= turn_start_frames[turn_ix] - nframes_win #12049 #18320 #turn_start_frames[4] #- nframes_win
    plotdf = flydf.loc[start_ix:stop_ix]
    
    #% PLOT.
    col1 = 'r'
    col2 = 'cornflowerblue'
    accel_color = [0.6]*3

    xvar = 'sec'
    varset = 'varset1' #_smoothed'
    yvar1 = 'theta_error' if 'varset2' in varset else 'facing_angle'
    yvar2 = 'ang_vel_fly' if 'varset2' in varset else 'ang_vel'
    # varset1 = facing_angle, ang_vel
    # varset2 = theta_error, ang_vel_fly
    # varset2_smoothed = theta_error_smoothed, ang_vel_fly_smoothed
    fig = the.plot_timecourses_for_turn_bouts(plotdf, high_ang_start_frames, xvar=xvar, varset=varset, 
                                          targ_color=col1, fly_color=col2)
    fig.axes[0].set_title('{}, frames: {}-{}'.format(acq, start_ix, stop_ix), loc='left', 
                    fontsize=8)

    putil.label_figure(fig, figid)
    figname = 'time-courses_{}_{}_frames-{}-{}'.format(yvar1, yvar2, start_ix, stop_ix)
    pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))


#%% Only plot theta error and ang vel

#   #kPlot top plot only

    plotdf = flydf.loc[start_ix:stop_ix]

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

    targ_color='r'
    fly_color='cornflowerblue'
    accel_color=bg_color

    fig, ax = pl.subplots(1,1,figsize=(6, 1.5), sharex=True)

    # Plot theta_error and fly's angular velocity
    ax.plot(plotdf[xvar], plotdf[var1], targ_color)
    # color y-axis spine and ticks red
    ax.set_ylabel(r'$\theta_{E}$' + '\n{}'.format(var1)) #, color=targ_color) #r'$\theta_{E}$'
    putil.change_spine_color(ax, targ_color, 'left')
    ax.axhline(y=0, color=targ_color, linestyle='--', lw=0.5)

    ax.set_xlabel('time (sec)')

    ax2 = ax.twinx()
    ax2.plot(plotdf[xvar], plotdf[var2], fly_color)
    # Color y-axis spine and ticks blue
    ax2.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(var2)) #, color=fly_color)
    putil.change_spine_color(ax2, fly_color, 'right')
    # Center around 0
    if center_yaxis:
        curr_ylim = np.round(plotdf[var2].abs().max(), 0)
        ax2.set_ylim(-curr_ylim, curr_ylim)

    pl.subplots_adjust(bottom=0.2, top=0.8)

    putil.label_figure(fig, '{}\n{}'.format(figid, acq))

    ax.set_title('{}, frames: {}-{}'.format(acq, start_ix, stop_ix), loc='left', 
                    fontsize=8)

    figname = 'time-courses_single_{}_{}_frames-{}-{}'.format(yvar1, yvar2, start_ix, stop_ix)
    pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))
    pl.savefig(os.path.join(curr_figdir, '{}.svg'.format(figname)))





    #%% PSTH of turn bouts:  FACING_ANGLE and ANG_VEL
    #del mean_v1, mean_v2
    fps = 60.
    nframes_win = 0.1*fps

    yvar1 = 'facing_angle' #'theta_error' #'facing_angle' #'facing_angle'
    yvar2 = 'ang_vel' #'ang_vel_fly' #'ang_vel' #'ang_vel'

    #plotdf = flydf.copy()
    curr_high_ang_start_frames = [f for f in high_ang_start_frames if f in plotdf['frame']]
    print(len(curr_high_ang_start_frames))

    turns_, t_lags = the.get_turn_psth_values(plotdf, curr_high_ang_start_frames, interval=1,
                                                  yvar1=yvar1, yvar2=yvar2, nframes_win=nframes_win, fps=fps)
    # Get mean 
    mean_turns_ = turns_.groupby(['rel_sec']).mean().reset_index() #rop=True)

    # Plot PSTH all turns
    fig = the.plot_psth_all_turns(turns_, yvar1=yvar1, yvar2=yvar2, col1=col1, col2=col2, lw_all=1, lw_mean=2)

    fig.axes[0].set_title('{}, frames: {}-{}'.format(acq, start_ix, stop_ix), loc='left', 
                    fontsize=8)
    fig.axes[0].set_box_aspect(1)
    # save
    putil.label_figure(fig, figid)
    figname = 'psth-turn-bouts_{}_{}_frames-{}-{}'.format(yvar1, yvar2, start_ix, stop_ix)  
    pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))

    #%%  Cross Correlate vars
    correlation, lags, lag_frames, t_lag = the.cross_correlation_lag(mean_turns_[yvar2], mean_turns_[yvar1], fps=60)

    fig = the.plot_mean_cross_corr_results(mean_turns_, correlation, lags, t_lags, t_lag=t_lag,
                                  yvar1=yvar1, yvar2=yvar2, col1=col1, col2=col2, bg_color=bg_color)
    #pl.subplots_adjust(wspace=1)

    # save
    putil.label_figure(fig, figid)
    figname = 'mean-turn-bouts-xcorr-tlags_{}_{}_frames-{}-{}'.format(yvar1, yvar2, start_ix, stop_ix)
    pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))

    #%%
    # =======================================================
    # Do ALL turns
    # =======================================================
    # Just use ANG_ACC
    flydf = ftjaaba[(ftjaaba['acquisition']==acq) & (ftjaaba['id']==0)].copy()

    #% Look at moments of high ang acc.
    #del turnbouts
    #%
    min_ang_acc = 100 # 150
    min_ang_vel = 5
    min_vel = 15
    min_dist_to_other = 15
    min_facing_angle = np.deg2rad(90)
    nframes_win = 0.1*fps

    turnbouts = the.get_turn_bouts(flydf, min_ang_acc=min_ang_acc, #min_ang_vel=min_ang_vel, 
                            min_vel=min_vel, min_dist_to_other=min_dist_to_other,
                            min_facing_angle=min_facing_angle, 
                            nframes_win=nframes_win)
    turnbouts = turnbouts.reset_index(drop=True)

    #%% 
    v1 = 'facing_angle' #'theta_error' #'theta_error'
    v2 = 'ang_vel' #'ang_vel_fly' #'ang_vel_fly'
    xcorr, lags, t_lags = the.cross_corr_each_bout(turnbouts, v1=v1, v2=v2)

    #%% Add delta t to turnbouts DF
    for ((turn_ix, t), l) in zip(turnbouts.groupby('turn_bout_num'), t_lags):
        turnbouts.loc[t.index, 'delta_t_lag'] = l
    #%%
    fig, ax =pl.subplots()
    plotdf = turnbouts[['stim_hz', 'turn_bout_num', 'delta_t_lag']].drop_duplicates()
    sns.barplot(data=plotdf, x='stim_hz', y='delta_t_lag', ax=ax, palette=stimhz_palette)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)

    #%% PLOT
    # Trio: Average aligned turn bouts, cross-correlation, distribution of time lags
    v1_label = r'$\theta_{E}$' + '\n{}'.format(v1)
    v2_label = r'$\omega_{f}$' + '\n{}'.format(v2)
    fig = the.plot_cross_corr_results(turnbouts, xcorr, lags, t_lags, v1=v1, v2=v2, 
                                  col1=col1, col2=col2, v1_label=v1_label, v2_label=v2_label)

    fig.text(0.1, 0.95,  '{}, all turns ang_acc > {:.2f}'.format(acq, min_ang_acc), 
                        fontsize=8)
    # save
    figname = 'mean-turn-bouts-xcorr-tlags_acc-thr-{}_{}'.format(min_ang_vel, acq)
    putil.label_figure(fig, acq)
    pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))
    pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))

    print(curr_figdir, figname)

    #% Aligned INDIVIUAL TURNS, 2 subplots of theta_error and ang_vel
    fig = the.plot_individual_turns(turnbouts, v1=v1, v2=v2)
    fig.text(0.1, 0.95,  '{}, all turns ang_acc > {:.2f}'.format(acq, min_ang_acc), 
                    fontsize=8)

    # save
    figname = 'turn-bouts_acc-thr-{}_{}'.format(min_ang_acc, acq)
    putil.label_figure(fig, acq)
    pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))
    print(curr_figdir, figname)

    #%% SHIFT frames
    # shift target variables by lag_frames, compare with flydf on current frames
    # of high_ang_start_frames
    med_lag = np.median(np.array(t_lags))
    shifted, unshifted = shift_vars_by_lag(flydf, high_ang_start_frames, med_lag, fps=fps)

    #%% Plot correlations between theta_error and ang_vel for same frame and lagged frame
    col = bg_color
    markersize=5

    varset = 'varset2'
    if varset == 'varset2':
        x='theta_error'
        y='ang_vel_fly'
        x1 = 'theta_error_dt' #_angle_vel'
    else:
        x = 'facing_angle'
        y = 'ang_vel'
        x1 = 'facing_angle_vel'
    fig = compare_regr_pre_post_shift(flydf, shifted, x=x, y=y, x1=x1, col=col, markersize=markersize)
    pl.subplots_adjust(bottom=0.2, wspace=0.6, hspace=0.6, left=0.1, right=0.9)

    # save
    putil.label_figure(fig, acq)
    figname = 'lag-shifted-regr_acc-thr-{}_{}'.format(min_ang_acc, acq)
    pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))
    print(curr_figdir, figname)

    #%% DO PAIRPLOT
#    cols = [c for c in shifted.columns if c != 'turn_start_frame']
#    with sns.plotting_context(rc={"axes.labelsize":20}):
#        g = sns.pairplot(data=shifted[cols], diag_kws={'color': bg_color}, plot_kws={'color': bg_color, 's': 5})
#
#    putil.label_figure(g.fig, acq)
#    figname = 'pairplot_lag-shifted_acc-thr-{}_{}'.format(min_acc, acq)
#    pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))
#
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
    figname = 'regr_vel-v-acc_lag-shifted_acc-thr-{}_{}'.format(min_ang_acc, acq)
    pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))

    #%%
    # Theta error on PREVIOUS vs. Turn size after lag
    plotdf_shifted = shifted[(shifted['turn_size']<=np.pi) 
                    & (shifted['turn_size']>=-np.pi)]
    plotdf = unshifted[(unshifted['turn_size']<=np.pi) 
                    & (unshifted['turn_size']>=-np.pi)]

    x = 'theta_error'
    y = 'turn_size'
    fig, axn =pl.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)

    ax=axn[0]
    ax.set_title('original')
    sns.regplot(data=plotdf, x=x, y=y, ax=ax, 
                color=col, scatter_kws=scatter_kws) #markersize=markersize)
    model_unshifted = sm.OLS(plotdf.dropna()[y], sm.add_constant(plotdf.dropna()[x]))
    res_unshifted = model_unshifted.fit()
    print("REGR. UN-shifted: x={}, y={} ----------------------- ".format(x, y))
    print(res_unshifted.summary())
    fit_str = 'OLS: y = {:.2f}x + {:.2f}, R2={:.2f}'\
                .format(res_unshifted.params[1], res_unshifted.params[0],
                        res_unshifted.rsquared)
    ax.text(0.1, 0.9, fit_str, transform=ax.transAxes)

    ax=axn[1]
    ax.set_title('shifted (lag={} frames)'.format(lag_frames))
    sns.regplot(data=plotdf_shifted.dropna(), x=x, y=y, ax=ax, 
                color=col, scatter_kws=scatter_kws) #markersize=markersize)
    model_shifted = sm.OLS(plotdf_shifted.dropna()[y], sm.add_constant(plotdf_shifted.dropna()[x]))
    res_shifted = model_shifted.fit()
    print("REGR. shifted: x={}, y={} ----------------------- ".format(x, y))
    print(res_shifted.summary())
    fit_str = 'OLS: y = {:.2f}x + {:.2f}, R2={:.2f}'\
                .format(res_shifted.params[1], res_shifted.params[0],
                        res_shifted.rsquared)
    ax.text(0.1, 0.9, fit_str, transform=ax.transAxes)

    for ax in axn:
        ax.set_box_aspect(1)

    putil.label_figure(fig, '{}\n{}'.format(figid, acq))
    figname = '{}_v_{}_lag-shifted-{}_acc-thr-{}'.format(x, y, lag_frames, min_ang_acc)
    pl.savefig(os.path.join(curr_figdir, '{}.png'.format(figname)))
    print(curr_figdir, figname)






