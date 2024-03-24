#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 2021/5/10 10:47
    Author  : julianarhee
    Software: PyCharm
"""
#%%
import os
import glob
import copy
import cv2

import matplotlib as mpl
import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns

import utils as util
import plotting as putil
import relative_metrics as rem
import importlib

from relative_metrics import load_processed_data, get_video_cap
#%% FUNCTIONS 
def get_indices_of_consecutive_rows(passdf):
    '''
    Find start and stop indices of consecutive rows in a dataframe.

    Arguments:
        passdf -- frames which pass boolean condition(s)

    Returns:
        series of tuples, each containing start and stop indices of consecutive rows
    '''
    passdf['diff'] = passdf.index.to_series().diff().fillna(1)
    passdf['diff'] = passdf['diff'].apply(lambda x: 1 if x>1 else 0)
    passdf['group'] = passdf['diff'].cumsum()

    return passdf.groupby('group').apply(lambda x: (x.index[0], x.index[-1]))

def get_vector_between_flies(f1, f2, curr_frames):
    f1_pos = f1[f1['frame'].isin(curr_frames)][['pos_x', 'pos_y']].values
    f2_pos = f2[f2['frame'].isin(curr_frames)][['pos_x', 'pos_y']].values
    vecs = [(i[0]-j[0], i[1]-j[1]) for i, j in zip(f2_pos, f1_pos)]
    return vecs

def filter_bouts_by_frame_duration(consec_bouts, min_bout_len, fps=60):
    min_bout_len_frames = min_bout_len*fps # corresponds to 0.25s at 60Hz
    incl_bouts = [c for c in consec_bouts if c[1]-c[0]>=min_bout_len_frames]
    #print("{} of {} bouts pass min dur {}sec".format(len(incl_bouts), len(consec_bouts), min_bout_len))
    return incl_bouts

# plotting 
def plot_bout_of_two_flies(df_, curr_frames, ax=None):
    if ax is None:
        fig, ax = pl.subplots()
    sns.scatterplot(x='pos_x', y='pos_y', 
                    data=df_[( df_['id']==0) & (df_['frame'].isin(curr_frames))], ax=ax,
                    hue='sec', palette='viridis', legend=0, edgecolor='none')
    sns.scatterplot(x='pos_x', y='pos_y', 
                    data=df_[(df_['id']==1) & (df_['frame'].isin(curr_frames))], ax=ax, 
                    hue='sec', palette='magma', legend=0, edgecolor='none')
    ax.set_aspect(1)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_xlabel('')

    return

def plot_example_bouts(nr, nc, incl_bouts, df_, title=''):    
    fig, axn = pl.subplots(nr, nc, figsize=(20, 15))
    for i, b_ in enumerate(incl_bouts[:nr*nc]):
        curr_frames = df_.loc[b_[0]:b_[1]]['frame']
        ax=axn.flat[i]
        plot_bout_of_two_flies(df_, curr_frames, ax)
        ax.set_title('{}: {}-{}'.format(i, b_[0], b_[1]), fontsize=8, loc='left')

    for ai in range(i+1, nr*nc):
        axn.flat[ai].axis('off')

    pl.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.text(0.1, 0.92, title, fontsize=8)
    pl.subplots_adjust(left=0.05, right=0.95)
    return fig
#%%
def main():
    #%% set plotting
    plot_style='dark'
    putil.set_sns_style(style=plot_style, min_fontsize=12)
    bg_color = [0.7]*3 if plot_style=='dark' else 'k'

    #%% set source dirs
    rootdir = '/Volumes/Julie'
    assay = '2d-projector' #'38mm_dyad'

    acq = '20240214-0945_f1_Dele-wt_5do_sh_prj10_sz6x6'
    subfolder = 'fly-tracker/*'

    if assay == '2d-projector':
        session = acq.split('-')[0]
        viddir = os.path.join(rootdir, '2d-projector', session)
        found_mats = glob.glob(os.path.join(viddir, 'fly-tracker', '20*', '*feat.mat'))
        
        procdir = os.path.join(rootdir, '2d-projector-analysis/FlyTracker/processed')
    else:
        viddir = os.path.join(rootdir, 'courtship-videos', assay)
        found_mats = glob.glob(os.path.join(viddir,  '20*ele*', '*', '*feat.mat'))

        procdir = os.path.join(rootdir, 'free-behavior-analysis/FlyTracker/38mm_dyad/processed')

    print("Found {} processed -feat.mat files for ele.".format(len(found_mats)))

    #%% set input data
    #acq = '20231226-1137_fly2_eleWT_4do_sh_eleWT_4do_gh' #util.get_acq_from_ftpath(fp, viddir)
    #acq = '20240214-0945_f1_Dele-wt_5do_sh_prj10_sz6x6'
    importlib.reload(util)
    fp = [f for f in found_mats if acq in f][0] #found_mats[0]
    print(fp)
    acqdir = os.path.join(viddir, acq)
    try:
        df_ = load_processed_data(acqdir, load=True, savedir=procdir)
        df_.head()
    except FileNotFoundError:
        subfolder = 'fly-tracker/*'
        calib_, trk_, feat_ = util.load_flytracker_data(viddir, 
                                        subfolder=subfolder,
                                        fps=fps)
        trk_cols = [c for c in trk_.columns if c not in feat_.columns]
        df_ = pd.concat([trk_[trk_cols], feat_], axis=1)
        df_.to_pickle(os.path.join(procdir, '{}_df.pkl'.format(acq))) 

#%%         
    if subfolder=='fly-tracker/*':
        vids = util.get_videos(viddir, vid_type='avi')
        vid_fpath = [v for v in vids if acq in v][0]
        cap = cv2.VideoCapture(vid_fpath)
    else:
        cap = get_video_cap(viddir) #acqdir)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    print(frame_width, frame_height) # array columns x array rows

    #%% get relative metrics
    importlib.reload(rem)
    #df_ = rem.get_relative_metrics(df_, acqdir, savedir=procdir)
    # FLIP ORI if fly-tracker
    df_['ori'] = -1 * df_['ori']
    df_ = rem.do_transformations_on_df(df_, frame_width, frame_height) #, fps=fps)

    #%% output dir
    destdir = os.path.join(os.path.split(procdir)[0], 'predictive_coding')
    figdir = os.path.join(destdir, acq)
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    print(figdir)

    #%% find candidate approach bouts
    min_vel = 12 # at lower vel, male may be doing bilateral display
    min_angle_between = 1.0
    max_facing_angle = 0.5
    passdf = df_[(df_['vel']>min_vel) 
                & (df_['angle_between']>=min_angle_between)
                & (df_['facing_angle']<=max_facing_angle)]

    # get start/stop indices of consecutive rows
    consec_bouts = get_indices_of_consecutive_rows(passdf)

    #%% Filter bouts based on duration
    min_bout_len = 0.25
    fps = 60.
    incl_bouts = filter_bouts_by_frame_duration(consec_bouts, min_bout_len, fps)
    print("{} of {} bouts pass min dur {}sec".format(len(incl_bouts), len(consec_bouts), min_bout_len))

    #%% Plot several bouts
    plot_bouts = False
    filter_str = 'min_bout_len-{}_min_vel-{}_min_angle-{}_max_facing-{}'.format(min_bout_len, min_vel, min_angle_between, max_facing_angle) 
    nr=7; nc=10;
    if plot_bouts:
        fig = plot_example_bouts(nr, nc, incl_bouts, df_, filter_str)
        # save
        putil.label_figure(fig, acq)
        figname = 'filtered-bouts_{}'.format(acq)
        pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
        print(figdir, figname)

    #%% ----------------------------------------------
    # look at 1 specific bout
    # ------------------------------------------------
    b_ = list(copy.copy(incl_bouts[7]))
    #b_ = [7955, 8000] # plotted, big
    # b_ = [9110, 9340] # plotted, smaller
    # b_ = [10336, 10492] 
    # b_ = [10590, 10760]
    # b_ = [10946, 11023]
    # b_ = [11262, 11357] # big one (plotted)
    # b_ = [11691, 11843] # big one (plotted)
    # b_ = [12176, 13402] # lots of inner-circle chasing
    b_ = [14001, 14160] # 1 inner-circle bout
    #b_ = [14164, 14313] # same
    b_ = [17858, 18663] # very long inner-circle bout, continuous
    bout_start_frame = df_.loc[b_[0]]['frame']
    bout_end_frame = df_.loc[b_[1]]['frame']
    nsec_pre = 0
    b_[0] = b_[0] - nsec_pre*fps
    b_[1] = b_[1] + nsec_pre*fps # look at a few sec before/after

    curr_frames = df_.loc[b_[0]:b_[1]]['frame']
    # get current df
    f1 = df_[(df_['frame'].isin(curr_frames)) & (df_['id']==0) ]
    f2 = df_[(df_['frame'].isin(curr_frames)) & (df_['id']==1) ]

    # plot traj
    fig, ax = pl.subplots(figsize=(4,8))
    ax.scatter(f1['pos_x'], f1['pos_y'], c=f1['sec'], cmap='viridis')
    ax.scatter(f2['pos_x'], f2['pos_y'], c=f2['sec'], cmap='viridis')
    ax.set_aspect(1)

    # plot vectors
    for i, fr in enumerate(curr_frames): 
        ax.plot([f1[f1['frame']==fr]['pos_x'], f2[f2['frame']==fr]['pos_x']],
                [f1[f1['frame']==fr]['pos_y'], f2[f2['frame']==fr]['pos_y']], 
                'w-', lw=0.5)
        
    vecs  = get_vector_between_flies(f1, f2, curr_frames)

    #%% calculate differences between pursuer-target vectors

    # get angle between consecutive vectors
    ang_diffs=[]
    for vi, v in enumerate(vecs):
        if vi == len(vecs)-1:
            break
        next_v = vecs[vi+1]
        ang_diffs.append(np.rad2deg(util.angle_between(next_v, v)))

    # calculate change in length between consecutive vectors
    vec_lens = [np.linalg.norm(v) for v in vecs]
    vec_len_diffs = np.diff(vec_lens)

    vecdf = pd.DataFrame({'ang_diff': ang_diffs, 'vec_len': vec_lens[1:],
                        'vec_len_diff': vec_len_diffs,
                'frame': curr_frames[1:]})

    #%% plot trajectories, color by angular diff
    min_val, max_val = min(ang_diffs), max(ang_diffs)
    cmap = mpl.cm.Greys # 
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    color_list = cmap(ang_diffs)

    fig, ax = pl.subplots(figsize=(5,4)) #figsize=(6,6))
    ax.scatter(f1['pos_x'], f1['pos_y'], c=f1['sec'], cmap='viridis', s=5)
    ax.scatter(f2['pos_x'], f2['pos_y'], c=f2['sec'], cmap='viridis', s=5)
    ax.set_aspect(1)
    for i, v in enumerate(vecs[0:-1]): 
        c='k' if i==0 else color_list[i]
        pos1 = f1.iloc[i][['pos_x', 'pos_y']].values
        pos2 = f2.iloc[i][['pos_x', 'pos_y']].values
        ax.plot([ pos1[0], pos1[0]+v[0]], 
                [ pos1[1], pos1[1]+v[1]], c=c)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                shrink=0.5, label='ang. diff between vectors')

    putil.label_figure(fig, acq)
    figname = 'vectors_example-bout_frames-{}-{}_nsec-pre-{}_{}'.format(b_[0], b_[1], nsec_pre, acq)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

    #%% Scatter: vector length vs angle difference
    # color by frame/time
    min_val, max_val = min(vecdf['frame']), max(vecdf['frame'])
    cmap = mpl.cm.viridis # 
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    color_list = cmap(ang_diffs)

    fig, ax = pl.subplots()
    sns.scatterplot(data=vecdf, x='vec_len', y='ang_diff', ax=ax,
                    hue='frame', palette='viridis', legend=0)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                shrink=0.5, label='frame')
    ax.set_title('Vector length vs angle diff')

    putil.label_figure(fig, acq)
    figname = 'vec_len_vs_ang_diff_frames-{}-{}_nsec-pre-{}__{}'.format(b_[0], b_[1], nsec_pre, acq)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

    #%% Scatter2: vector length diff vs angle difference
    # color by frame/time
    min_val, max_val = min(vecdf['frame']), max(vecdf['frame'])
    cmap = mpl.cm.viridis # 
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    color_list = cmap(ang_diffs)

    fig, ax = pl.subplots()
    sns.scatterplot(data=vecdf, x='vec_len_diff', y='ang_diff', ax=ax,
                    hue='frame', palette='viridis', legend=0)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                shrink=0.5, label='frame')
    ax.set_title('Vector length diff vs angle diff')

    putil.label_figure(fig, acq)
    figname = 'vec_len_diff_vs_ang_diff_frames-{}-{}_nsec-pre-{}__{}'.format(b_[0], b_[1], nsec_pre, acq)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

        
    #%%a
    # find vectors with shorter length than previous 
    shorter = np.where(vec_len_diffs<0)[0]

    # find vectors with small angle differences from previous
    ang_diff_thr=2
    smaller_angs = np.where(np.array(ang_diffs)<ang_diff_thr)[0]

    # plot
    fig, ax = pl.subplots(figsize=(6, 12))
    ax.scatter(f1['pos_x'], f1['pos_y'], c=f1['sec'], cmap='viridis', s=5)
    ax.scatter(f2['pos_x'], f2['pos_y'], c=f2['sec'], cmap='viridis', s=5)
    ax.set_aspect(1)

    ang_diff_thr=1
    smaller_angs = np.where(np.array(ang_diffs)<ang_diff_thr)[0]

    for i, v in enumerate(vecs[1:]):
        pos1 = f1.iloc[i][['pos_x', 'pos_y']].values
        pos2 = f2.iloc[i][['pos_x', 'pos_y']].values
        if i+1 in shorter and i+1 in smaller_angs:
            ax.plot([ pos1[0], pos1[0]+v[0]], 
                    [ pos1[1], pos1[1]+v[1]], c='r', lw=0.5)
    ax.set_title('Vectors with vec len shorter than prev. and small ang diff')

    start_marker = b_[0] + nsec_pre*fps
    stop_marker = b_[1] - nsec_pre*fps
    for m_ in [start_marker, stop_marker]:
        ax.plot(f1[f1['frame']==m_]['pos_x'], 
                f1[f1['frame']==m_]['pos_y'], 'ro')   

    putil.label_figure(fig, acq)
    figname = 'traj_with_shorter_vecs_and_small_ang_diff_{}_frames-{}-{}_nsec-pre-{}__{}'.format(ang_diff_thr, b_[0], b_[1], nsec_pre, acq)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

    #%% ANG VEL range?
    pl.figure()
    pl.plot(f1['ang_vel'])

    #%% -----------------------------------------
    # plot TARG_POS_THETA vs. TIME
    # ------------------------------------------
    fig, ax =pl.subplots()
    huevar = 'ang_vel'
    yvar = 'targ_pos_theta_deg'
    f1['targ_pos_theta_deg'] = np.rad2deg(f1['targ_pos_theta'])
    vmin, vmax = f1[huevar].min(), 10 #f1[huevar].max()
    sns.scatterplot(data=f1, x='frame', y=yvar, ax=ax,
                    hue=huevar, s=30, edgecolor='none', 
                    palette='viridis', legend=0,
                    hue_norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    putil.add_colorbar(fig, ax, vmin, vmax, 
                       label=huevar, cmap='viridis')
    for i in [11801]:
        ax.plot(f1[f1['frame']==i]['frame'], 
                f1[f1['frame']==i][yvar], 'r*')

    ax.axvline(x=bout_start_frame)
    ax.axvline(x=bout_end_frame)
    ax.axhline(y=0, c=bg_color, ls='--')
    ax.set_ylim([-50, 50])

    putil.label_figure(fig, acq)
    figname = 'targ_pos_theta_vs_time_frames-{}-{}_nsec-pre-{}'.format(b_[0], b_[1], nsec_pre)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

    #% plot on POLAR
    fig, ax =pl.subplots(subplot_kw={'projection': 'polar'})
    sns.scatterplot(data=f1, x='targ_pos_theta', y='targ_pos_radius', 
                    ax=ax, edgecolor='none',
                    hue='frame',palette='viridis' , legend=0,
                    )
    if nsec_pre > 0:
        for i in [bout_start_frame, bout_end_frame]:
            ax.plot(f1[f1['frame']==i]['targ_pos_theta'], 
                    f1[f1['frame']==i]['targ_pos_radius'], 'ro')
    for i in [11801]:
        ax.plot(f1[f1['frame']==i]['targ_pos_theta'], 
                f1[f1['frame']==i]['targ_pos_radius'], 'r*')

    putil.label_figure(fig, acq)
    figname = 'targ_pos_polar_frames-{}-{}_nsec-pre-{}'.format(b_[0], b_[1], nsec_pre)        
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))


    #%% ------------------------------
    # MAKE VIDEO OF BOUT
    # --------------------------------
    #%% make video of bout?
    tmp_outdir = '/Users/julianarhee/Documents/rutalab/projects/predictive_coding'
    from matplotlib.animation import FuncAnimation
    from matplotlib import animation

    #frame_num = curr_frames.values[0]
    ix = 0 
    # get list of colors for scatter plot based on angular size using viridis cmap and normalizing to range of angular size values
    huevar = 'ang_vel' #'targ_ang_size'
    #vmin, vmax = f1[huevar].min(), f1[huevar].max()
    #hue_norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    print(vmin, vmax)
    ylim = np.ceil(f1['targ_pos_radius'].max())
    # Create a Normalize object to map the data values to the range [0, 1]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax) #f1[hue_var].min(), 
                                #vmax=0.3) #f1[hue_var].max())
    # Create a ScalarMappable object with the Viridis colormap and the defined normalization
    sm = mpl.cm.ScalarMappable(cmap='viridis', norm=norm)

    # Get a list of colors corresponding to the data values
    colors = [sm.to_rgba(value) for value in f1[huevar].values]

    fig = pl.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='polar')
    ax2.set_ylim([0, ylim])

    #while im is None:
    #for ix in range(len(curr_frames)):
    cap.set(1, curr_frames.iloc[ix])
    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    print(ix)

    p1 = ax1.imshow(im, aspect='equal', cmap='gray')
    ax1.invert_yaxis()
    p2 = ax2.scatter(f1.iloc[:ix]['targ_pos_theta'],
                      f1.iloc[:ix]['targ_pos_radius'],
                      c=f1.iloc[:ix][huevar],                       
                      vmin=vmin, vmax=vmax,
                      s=20, #f1.iloc[:ix]['targ_ang_size'],
                      cmap='viridis' #f1.loc[:frame_num]['targ_ang_size'],
                      )

    def init():
        p1.set_data(np.zeros((frame_height, frame_width)))
        # set x, y data for scatter plot
        p2.set_data([], [])
        # set sizes
        #p2.set_sizes([], [])
        p2.set_array([])
        return (p1, p2,)

    def update_figure(ix): #, p1, p2, cap, f1):        
        # udpate imshow
        cap.set(1, curr_frames.iloc[ix])
        ret, im = cap.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)
        p1.set_data(im)
        #p1.set_title('Frame {}'.format(frame_num))
        
        # Update subplot 2 (plot)
        xys_ = f1.iloc[:ix][['targ_pos_theta', 'targ_pos_radius']].values
        # update x, y
        p2.set_offsets(xys_)
        # uppdate size
        #p2.set_sizes(f1.iloc[:ix]['targ_ang_size'] * 100) 
        # update color
        p2.set_array(f1.iloc[:ix][huevar])

        # Adjust layou+ 1t
        #pl.tight_layout()

    frame_numbers = np.arange(0, len(curr_frames)) #curr_frames.values #range(1, 11)  # List of frame numbers

    # Create the animation
    anim = FuncAnimation(fig, update_figure, frames=frame_numbers, 
                        interval=1000//fps)
    
    # Save the animation as a movie file
    video_outrate = 30
    save_movpath = os.path.join(figdir, 
                        'anim_frames-{}-{}_nsec-pre-{}_hue-{}_{}hz.mp4'.format(b_[0], b_[1], nsec_pre,
                                                                                   huevar, video_outrate))

    anim.save(save_movpath, fps=video_outrate, extra_args=['-vcodec', 'libx264'])
    #writervideo = animation.FFMpegWriter(fps=video_outrate) 
    #anim.save(save_movpath, writer=writervideo)

    #%% check transformationsa ---------------------------------
    importlib.reload(rem)
    f2.index = f1.index.copy() 
    ix = 250
    frame_ = curr_frames.iloc[ix]
    fig = rem.plot_frame_check_affines(frame_, f1, f2, cap, 
                                    frame_width, frame_height)

    currdf = df_[df_['frame'].isin(curr_frames)].copy()
    fig = rem.check_rotation_transform(frame_, currdf, cap) # frame_width, frame_height)








#%%

    # %% Parallel pursuit in terms of ANGLES.

    #%% smooth x, y, 
    winsize=5
    df_['pos_x_smoothed'] = df_.groupby('id')['pos_x'].transform(lambda x: x.rolling(winsize, 1).mean())
    df_['pos_y_smoothed'] = -1*df_.groupby('id')['pos_y'].transform(lambda x: x.rolling(winsize, 1).mean())  

    # calculate heading
    df_['heading'] = np.arctan2(df_['pos_y_smoothed'].diff(), df_['pos_x_smoothed'].diff())
    df_['heading_deg'] = np.rad2deg(df_['heading']) #np.rad2deg(np.arctan2(df_['pos_y_smoothed'].diff(), df_['pos_x_smoothed'].diff())) 

    #%%
    def set_angle_range_to_neg_pos_pi(ang):
        if ang>np.pi:
            ang = ang - 2*np.pi
        elif ang<-np.pi:
            ang = ang + 2*np.pi
        return ang
    #%%
    bearing_ang = df_['facing_angle'] + df_['ori']
    df_['bearing_ang']  = [set_angle_range_to_neg_pos_pi(a) for a in bearing_ang]

    error_ang = df_['bearing_ang'] - df_['heading']  
    df_['error_ang'] = [set_angle_range_to_neg_pos_pi(a) for a in error_ang]    
    # %%
    fig, ax =pl.subplots()
    plotdf = df_[(df_['frame'].isin(curr_frames))
                & (df_['id']==0)].copy()

    colors = ['b', 'orange', 'green', 'r', 'm']
    labels = ['ori', 'heading', 'bearing_ang', 'error_ang', 'facing_angle']
    for c, l in zip(colors, labels):
        ax.plot(plotdf[l], c, label=l)
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')

    # %%
    fig, ax =pl.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(plotdf['heading'], plotdf['sec'])
    ax.scatter(plotdf['bearing_ang'], plotdf['sec'])    
    ax.scatter(plotdf['error_ang'], plotdf['sec'])    
    ax.scatter(plotdf['facing_angle'], plotdf['sec'])    

    # %%
    fig, axn =pl.subplots(1, 2, sharex=True, sharey=True)
    ax=axn[0]
    ax.scatter(plotdf['bearing_ang'], plotdf['heading'], c=plotdf['sec'], cmap='viridis')
    ax.set_aspect(1)
    ax=axn[1]
    ax.scatter(plotdf['facing_angle'], plotdf['heading'], c=plotdf['sec'], cmap='viridis')
    ax.set_aspect(1)


    # %%
    pl.figure()
    pl.scatter(plotdf['facing_angle'], plotdf['error_ang'], c=plotdf['sec'], cmap='viridis')
    pl.gca().set_aspect(1)

    # %% traveling vector vs. heading?

    min_vel = 18 # at lower vel, male may be doing bilateral display
    max_angle_between = np.deg2rad(30)
    max_facing_angle = np.deg2rad(30)
    passdf = df_[(df_['vel']>min_vel) 
                & (df_['angle_between']<=max_angle_between)
                & (df_['facing_angle']<=max_facing_angle)]

    # get start/stop indices of consecutive rows
    consec_bouts = get_indices_of_consecutive_rows(passdf)
    consec_bouts[0]

    min_bout_len = 0.25
    fps = 60.
    incl_bouts = filter_bouts_by_frame_duration(consec_bouts, min_bout_len, fps)
    print("{} of {} bouts pass min dur {}sec".format(len(incl_bouts), len(consec_bouts), min_bout_len))

    # %%

    #cap = get_video_cap(acqdir)
    #n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    #frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    #print(frame_width, frame_height) # array columns x array rows

    #
    #%%
    b_ = list(copy.copy(incl_bouts[0]))
    bout_dur = (b_[1]-b_[0]) / fps 
    bout_nframes = b_[1] - b_[0]
    print("curr bout: {} frames, {:.2f} sec".format(bout_nframes, bout_dur))
    curr_frames = df_.loc[b_[0]:b_[1]]['frame']
    #% plot M/F traj
    fig, ax = pl.subplots()
    sns.scatterplot(data=df_[(df_['frame'].isin(curr_frames))], x='pos_x', y='pos_y', 
                    ax=ax, hue='sec', palette='viridis', 
                    style='id', legend=0)   
    ax.set_aspect(1)
    fig, ax =pl.subplots()
    plotdf = df_[(df_['frame'].isin(curr_frames))
                & (df_['id']==0)].copy()

    ax.plot(plotdf['ori'])
    ax.plot(plotdf['heading'])

    # %%