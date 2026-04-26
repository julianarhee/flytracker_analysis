#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

"""
This is tester code to see that theta_error.py runs with both
free behavior and projector, with updated paths

"""

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

import transform_data.relative_metrics as rel
import libs.utils as util
import libs.plotting as putil
import libs.stats as lstats

import statsmodels.api as sm
import libs.regplot as rpl
import scipy.stats as spstats

import analyses.pursuit.src.pursuit_funcs as pf
import analyses.theta_error.src.theta_error_funcs as tef


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

if __name__ == '__main__':
    #%% 
    # Set plotting
    plot_style='dark'
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

    create_new = False

    assay = '2d-projector' # '38mm-dyad'
    experiment = 'circle_diffspeeds'

    #assay = '38mm-dyad' 
    #experiment = 'MF'

    #%%
    minerva_base = '/Volumes/Juliana'

    # server loc for aggregated pkl
    #out_fpath = os.path.join(srcdir, 'relative_metrics.pkl')

    # Specify local dirs
    local_basedir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data'
    localdir = os.path.join(local_basedir, assay, experiment, 'FlyTracker')

    if assay == '2d-projector':
        # Set sourcedirs
        srcpath = '2d-projector-analysis/circle_diffspeeds/FlyTracker'       
        # local jaaba_file
        jaaba_fname = 'ftjaaba_mel_yak_20240330'

    elif assay == '38mm-dyad':
        # Set src
        srcpath= 'free-behavior-analysis/38mm-dyad/MF/FlyTracker'
        # local jaaba_file
        jaaba_fname = 'jaaba_20240303.pkl' #'jaaba_free_behavior_data_mel_yak_20240403'

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

    # Only select subset of data (GG):
    #df0 = df0.dropna()
    df0 = util.split_condition_from_acquisition_name(df0)
    ft_acq = df0['acquisition'].unique()

    #%
    # summary of what we've got
    print(df0[['species', 'acquisition']].drop_duplicates().groupby('species').count())

    #%%
    # =====================================================
    # Single fly dataframes
    # =====================================================
    new_ftjaaba = True

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
        f1 = df0[df0['id']==0].copy()
        f1 = rel.calculate_angle_metrics_focal_fly(f1, winsize=5)

    #%% Add JAABA to flydf
    if new_ftjaaba:
        #% Process: split into courting bouts, calculate boutdurs
        ftjaaba = tef.add_jaaba_to_flydf(f1, jaaba) #jaaba_fpath)    
        #% SAVE
        ftjaaba.to_pickle(processed_ftjaaba_fpath)

        print(processed_ftjaaba_fpath)

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
                    #& (ftjaaba['led_level']>0)
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

    if assay == '2d-projector':
        stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)

        # find the closest matching value to one of the keys in stimhz_palette:
        meanbouts['stim_hz'] = meanbouts['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))   

    #%% ------------------------------------------------
    # ANG_VEL vs. THETA_ERROR
    # -------------------------------------------------
    #%
    xvar ='theta_error'
    yvar = 'ang_vel_fly' #'ang_vel_fly_shifted' #'ang_vel' #'ang_vel_fly'
    plot_hue= False #True
    plot_grid = True
    nframes_lag = 2

    shift_str = 'SHIFT-{}frames_'.format(nframes_lag) if 'shifted' in yvar else ''
    hue_str = 'stimhz' if plot_hue else 'no-hue'
    hue_var = 'stim_hz' if plot_hue else None
    
    # Set palettes
    cmap='viridis'
    #stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)

    # Get CHASING bouts 
    behav = 'chasing'
    min_frac_bout = 0.5
    chase_ = meanbouts[ (meanbouts['{}_binary'.format(behav)]>min_frac_bout) ].copy()
    if assay == '2d-projector':
        chase_ = chase_[chase_['stim_hz']>0]

    if 'shifted' in yvar:
        figtitle = '{} bouts, where min fract of bout >= {:.2f}\nshifted {} frames'.format(behav, min_frac_bout, nframes_lag)
    else:
        figtitle = '{} bouts, where min fract of bout >= {:.2f}'.format(behav, min_frac_bout)

    species_str = '-'.join(chase_['species'].unique())

    xlabel = r'$\theta_{E}$ at $\Delta t$ (rad)'
    ylabel = '$\omega_{f}$ (rad/s)'

    # SCATTERPLOT:  ANG_VEL vs. THETA_ERROR -- color coded by STIM_HZ
    fig = tef.plot_regr_by_species(chase_, xvar, yvar, hue_var=hue_var, 
                               plot_hue=plot_hue, plot_grid=plot_grid,
                            xlabel=xlabel, ylabel=ylabel, bg_color=bg_color)
    fig.suptitle(figtitle, fontsize=12)
    pl.subplots_adjust(wspace=0.25)

    for ax in fig.axes:
        #ax.invert_yaxis()
        ax.invert_xaxis()

    putil.label_figure(fig, figid)
    #figname = 'sct_{}_v_{}_stimhz_{}_min-frac-bout-{}'.format(yvar, xvar, species_str, min_frac_bout)
    figname = 'sct_{}_v_{}_{}{}_{}_min-frac-bout-{}'.format(
                yvar, xvar, shift_str, hue_str, species_str, min_frac_bout)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    #pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

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
    data_type = 'BOUTS' if use_bouts else 'FRAMES'

    theta_error_small_deg = 10
    theta_error_large_deg = 25
    theta_error_small = np.deg2rad(theta_error_small_deg)
    theta_error_large = np.deg2rad(theta_error_large_deg)

    for curr_species in ftjaaba['species'].unique():

        nframes_lag_plot = 2 if 'shifted' in var1 or 'shifted' in var2 else 0

        if use_bouts:
            chase_ = meanbouts[(meanbouts['{}_binary'.format(behav)]>min_frac_bout)
                        & (meanbouts['species']==curr_species)].copy().reset_index(drop=True)
        else:
            chase_ = filtdf[(filtdf['{}_binary'.format(behav)]>min_frac_bout)
                        & (filtdf['species']==curr_species)].copy().reset_index(drop=True)
        chase_ = util.split_theta_error(chase_, theta_error_small=theta_error_small, theta_error_large=theta_error_large)

        # plot ------------------------------------------------
        fig = putil.plot_ang_v_fwd_vel_by_theta_error_size(chase_, 
                            var1=var1, var2=var2, err_palette=err_palette, lw=2)
        fig.text(0.1, 0.92, 
                 '{} bouts, frac. of {} > {:.1f}, lag {} frames (small=+/-{}, large=+/-{})'.format(
                behav, data_type, min_frac_bout, nframes_lag_plot, theta_error_small_deg, theta_error_large_deg), fontsize=12)

        fig.text(0.1, 0.85, curr_species, fontsize=24)
        pl.subplots_adjust(wspace=0.6, left=0.1, right=0.9, top=0.9)

        putil.label_figure(fig, figid)
        figname = 'big-v-small-theta-error_ang-v-fwd-vel_{}_thetaS{}_thetaL{}_min-frac-bout-{}_{}'.format(curr_species, theta_error_small_deg, theta_error_large_deg, min_frac_bout, data_type)
        pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
        #pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

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

        g = tef.plot_regr_by_hue(chase_, xvar, yvar, hue_var='stim_hz', stimhz_palette=stimhz_palette, show_scatter=show_scatter)
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
    do_bouts = False

    markersize=5
    huevar='stim_hz'
    cmap='viridis'
    plot_com=True

    xvar= 'facing_angle'
    yvar = 'targ_pos_radius'

    is_flytracker=True
    data_type = 'BOUTS' if do_bouts else 'FRAMES' 

    plotdf = tef.select_data_subset(filtdf, meanbouts, behav=behav, min_frac_bout=min_frac_bout, 
                                do_bouts=do_bouts, is_flytracker=is_flytracker)
    
    for sp, p_ in plotdf.groupby('species'):

        fig = tef.plot_allo_ego_frames_by_species(p_.iloc[0::4], xvar=xvar, yvar=yvar,
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
    plotdf_ = tef.select_data_subset(filtdf, meanbouts, behav=behav, min_frac_bout=min_frac_bout, 
                                do_bouts=do_bouts, is_flytracker=is_flytracker)
    

    currplotdf = plotdf_[plotdf_['acquisition'] == curr_acq].copy()
    print(currplotdf.shape)


    fig = tef.plot_allo_ego_frames_by_species(currplotdf, xvar=xvar, yvar=yvar,
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
    # FIGURE:
    # 1) allocentric frames for 1 example fly
    # 2) Dmel: all bouts, male-centered
    # 3) Dyak: all bouts, male-centered

    min_frac_bout = 0.1

    frames_ = tef.select_data_subset(filtdf, meanbouts, behav=behav, min_frac_bout=min_frac_bout, 
                                do_bouts=False, is_flytracker=is_flytracker)
    bouts_ = tef.select_data_subset(filtdf, meanbouts, behav=behav, min_frac_bout=min_frac_bout, 
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

    aggr_turns = pf.aggregate_turns_across_flies(ftjaaba, v1=v1, v2=v2, min_n_turns=min_n_turns,
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
    turn_starts = pf.get_theta_errors_before_turns(aggr_turns, fps=fps)

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