#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:07:00 2024

Author:         julianarhee
Name:           pursuit_vs_interceptions.py
Description:    compare pursuit vs. interceptions

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

import parallel_pursuit as pp

import plot_dlc_frames as pdlc
import dlc as dlc


import importlib

#%%
import scipy

#def ft_actions_to_bout_df(action_fpath):
#    '''
#    Take manually annoted bouts from FlyTracker -actions.mat file and convert to a pandas df
#
#    Arguments:
#        action_fpath -- _description_
#
#    Returns:
#        _description_
#    '''
#    # mat['bouts'] is (n_flies, action_types)
#    # mat['bouts'][0, 10] gets bout start/end/likelihood for fly1, action #10
#    # mat['behs'] are the behavior names
#
#    # load mat
#    mat = scipy.io.loadmat(action_fpath)
#
#    # behavior names
#    behs = [i[0][0] for i in mat['behs']]
#
#    # aggregate into list
#    b_list = []
#    for i, beh in enumerate(behs):
#        # get male action's 
#        if mat['bouts'][0, i].shape[1]==3:
#            b = mat['bouts'][0, i]
#            b_df = pd.DataFrame(data=b, columns=['start', 'end', 'likelihood'])
#            b_df['action'] = beh
#            b_df['id'] = 0
#            b_list.append(b_df)
#
#    boutdf = pd.concat(b_list)
#
#    return boutdf

def plot_heading_vs_travelingdir_frames(curr_frames, f1, f2, cap=None, ax=None,
                                     var1='ori', var2='traveling_dir', plot_interval=2,
                                     col1='magenta', col2='dodgerblue', target_col='g'):
    if ax is None:
        fig = pl.figure(figsize=(10,8))
        ax = fig.add_subplot(121) # axn = pl.subplots(1, 2)

    if cap is not None:
        cap.set(1, curr_frames[0])
        ret, im = cap.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

        ax.imshow(im, cmap='gray')
        ax.set_title("Frame {}".format(curr_frame), fontsize=8, loc='left')
        ax.invert_yaxis()

    for ix in curr_frames[0::plot_interval]:
        # set fly oris as arrows
        fly_marker = '$\u2192$' # https://en.wikipedia.org/wiki/Template:Unicode_chart_Arrows
        m_ori = np.rad2deg(f1.loc[ix][var1])
        m_trav = np.rad2deg(f1.loc[ix][var2])

        f_ori = np.rad2deg(f2.loc[ix][var2])

        marker_m1 = mpl.markers.MarkerStyle(marker=fly_marker)
        marker_m1._transform = marker_m1.get_transform().rotate_deg(m_ori)

        marker_m2 = mpl.markers.MarkerStyle(marker=fly_marker)
        marker_m2._transform = marker_m2.get_transform().rotate_deg(m_trav)

        marker_f = mpl.markers.MarkerStyle(marker=fly_marker)
        marker_f._transform = marker_f.get_transform().rotate_deg(f_ori)
        #print(np.rad2deg(fly1.loc[ix]['ori'])) #m_ori)
        #print(f_ori)


        # make a markerstyle class instance and modify its transform prop
        ax.plot([f1['pos_x'].loc[ix]], [f1['pos_y'].loc[ix]], 'w', 
                marker=marker_m1, markerfacecolor=col1, markersize=20, markeredgewidth=0.5, label=var1) 
        
        ax.plot([f1['pos_x'].loc[ix]], [f1['pos_y'].loc[ix]], 'w', 
                marker=marker_m2, markerfacecolor=col2, markersize=20, markeredgewidth=0.5, label=var2) 

        ax.plot([f2['pos_x'].loc[ix]], [f2['pos_y'].loc[ix]], 'w',
                marker=marker_f, markerfacecolor=target_col, markersize=20, markeredgewidth=0.5)
        ax.set_aspect(1)

    leg =putil.custom_legend(labels=[var1, var2, 'target'], colors=[col1, col2, target_col])

    ax.set_title('Frames: {}-{}'.format(int(curr_frames[0]), int(curr_frames[-1]) ))
    ax.legend(handles=leg, loc='upper left', bbox_to_anchor=(1,1), frameon=False)

    return


#%%
plot_style='dark'
putil.set_sns_style(style=plot_style, min_fontsize=18)
bg_color='w' if plot_style=='dark' else 'k'


#%% set source dirs
minerva_base = '/Volumes/Julie'
assay = '2d-projector' #'38mm_dyad'
fps = 60.

# acq = '20240214-0945_f1_Dele-wt_5do_sh_prj10_sz6x6'
# acq = '20240214-1010_f1_Dele-wt_5do_sh_prj10_sz4x4'
acq = '20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10'
#acq = '20240222-1611_fly7_Dmel_sP1-ChR_2do_sh8x8'

input_is_flytracker = False

#%% 
if input_is_flytracker:
    #% FlyTracker output
    subfolder = 'fly-tracker/*'
    # Get processed dir info
    if assay == '2d-projector':
        session = acq.split('-')[0]
        viddir = os.path.join(minerva_base, '2d-projector', session)
        found_mats = glob.glob(os.path.join(viddir, 'fly-tracker', '20*', '*feat.mat'))
        
        procdir = os.path.join(minerva_base, '2d-projector-analysis/FlyTracker/processed')
    else:
        viddir = os.path.join(minerva_base, 'courtship-videos', assay)
        found_mats = glob.glob(os.path.join(viddir,  '20*ele*', '*', '*feat.mat'))

        procdir = os.path.join(minerva_base, 'free-behavior-analysis/FlyTracker/38mm_dyad/processed')

    print("Found {} processed -feat.mat files for ele.".format(len(found_mats)))

    # Load processed data
    fp = [f for f in found_mats if acq in f][0] #found_mats[0]
    print(fp)
    acqdir = os.path.join(viddir, acq)
    try:
        df_ = pp.load_processed_data(acqdir, load=True, savedir=procdir)
        df_.head()
    except FileNotFoundError:
        print("creating feat/trk df.")
        subfolder = 'fly-tracker/*'
        df_ = pp.load_flytracker_data(acq, viddir, subfolder=subfolder, fps=fps)
else:
    importlib.reload(dlc)
    # DLC
    projectname='projector-1dot-jyr-2024-02-18' 
    procdir = os.path.join(minerva_base, '2d-projector-analysis/DeepLabCut', projectname)
    print(len(os.listdir(procdir)))

    #% get src paths
    localroot = '/Users/julianarhee/DeepLabCut' # all these functions assume this is the local rootdir
    #% Look at 1 data file
    analyzed_dir = dlc.get_dlc_analysis_dir(projectname=projectname)
    #acq_prefix = '20240214-1025_f1_*sz10x10'
    #acq_prefix = '20240216-*fly3*6x6'
    #acq_prefix = '20240222-*fly7_Dmel*8x8'
    # acq_prefix = '20240216-*fly3_Dmel*4x4'
    # acq_prefix = '20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10'

    acq_prefix = '20240214-1002_f1_Dele-wt_5do_sh_prj10_sz8x8'
    fpath = dlc.get_fpath_from_acq_prefix(analyzed_dir, acq_prefix)

    flyid = 'fly' # double check in the plots for abdomen lengths
    dotid = 'single'
    fps = 60  # Hz
    max_jump = 6
    pcutoff=0.8 #0.99

    # includes calc theta error
    df_ = dlc.load_and_transform_dlc(fpath, winsize=10,
                                     localroot=localroot, projectname=projectname,
                                     assay=assay, flyid=flyid, dotid=dotid, fps=fps, max_jump=max_jump, pcutoff=pcutoff,
                                     heading_var='ori')

# Create colormap for velocities
stimhz_vals = np.array([0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.625, 0.8, 1])
stimhz_dict = dict((k, v) for k, v in zip(np.arange(0, len(stimhz_vals)), stimhz_vals))
df_['stim_hz'] = [stimhz_dict[v] if v in stimhz_dict.keys() else None for v in df_['epoch']]

#%%
importlib.reload(util)
cap, viddir = util.get_video_cap_check_multidir(acq, assay=assay, return_viddir=True)

# get frame info
n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
print(frame_width, frame_height) # array columns x array rows

#%% get relative metrics

if input_is_flytracker:
    # FLIP ORI if fly-tracker
    df_['ori'] = -1 * df_['ori']
    df_ = rem.do_transformations_on_df(df_, frame_width, frame_height) #, fps=fps)
    df_.to_pickle(os.path.join(procdir, '{}_df.pkl'.format(acq))) 
#else:
#    # input is DLC
#    df_ = rem.do_transformations_on_df(df_, frame_width, frame_height) #, fps=fps)
#    df_['ori_deg'] = np.rad2deg(df_['ori'])
#    #df['targ_pos_theta'] = -1*df['targ_pos_theta']

#%%    
#importlib.reload(util)
#if not input_is_flytracker:
#    # convert centered cartesian to polar
#    rad, th = util.cart2pol(df_['ctr_x'].values, df_['ctr_y'].values)
#    df_['pos_radius'] = rad
#    df_['pos_theta'] = th
#
#    # angular velocity
#    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='ori', 
#                                    vel_var='ang_vel', time_var='sec', winsize=10)
#    #df_.loc[(df_['ang_vel']>200) | (df_['ang_vel']<-200), 'ang_vel'] = np.nan
#    #df_.loc[(df_['ang_vel_smoothed']>200) | (df_['ang_vel_smoothed']<-200), 'ang_vel_smoothed'] = np.nan
#    df_.loc[ (df_['ang_vel']>100) | (df_['ang_vel']<-100), 'ang_vel' ] = np.nan
#    df_['ang_vel_deg'] = np.rad2deg(df_['ang_vel'])
#    df_['ang_vel_abs'] = np.abs(df_['ang_vel'])

if input_is_flytracker:
    df_['targ_pos_theta_abs'] = np.abs(df_['targ_pos_theta'])

    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='targ_pos_theta', vel_var='targ_ang_vel',
                                    time_var='sec', winsize=10)

    #% smooth x, y, 
    winsize=10
    df_['pos_x_smoothed'] = df_.groupby('id')['pos_x'].transform(lambda x: x.rolling(winsize, 1).mean())
    sign = -1 if input_is_flytracker else 1
    df_['pos_y_smoothed'] = sign * df_.groupby('id')['pos_y'].transform(lambda x: x.rolling(winsize, 1).mean())  

    # calculate heading
    for i, d_ in df_.groupby('id'):
        df_.loc[df_['id']==i, 'traveling_dir'] = np.arctan2(d_['pos_y_smoothed'].diff(), d_['pos_x_smoothed'].diff())
    df_['traveling_dir_deg'] = np.rad2deg(df_['traveling_dir']) #np.rad2deg(np.arctan2(df_['pos_y_smoothed'].diff(), df_['pos_x_smoothed'].diff())) 
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='traveling_dir', vel_var='traveling_dir_dt',
                                    time_var='sec', winsize=3)

    df_['heading_travel_diff'] = np.abs( np.rad2deg(df_['ori']) - np.rad2deg(df_['traveling_dir']) ) % 180  #% 180 #np.pi 

    winsize=5
    df_['vel_smoothed'] = df_.groupby('id')['vel'].transform(lambda x: x.rolling(winsize, 1).mean())

#%% 
stimhz_vals = np.array([0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.625, 0.8, 1])
stimhz_dict = dict((k, v) for k, v in zip(np.arange(0, len(stimhz_vals)), stimhz_vals))
df_['stim_hz'] = [stimhz_dict[v] if v in stimhz_dict.keys() else None for v in df_['epoch']]


stimhz_palette = putil.get_palette_dict(df_, 'stim_hz', cmap='viridis')

#%% output dir
destdir = os.path.join(os.path.split(procdir)[0], 'pursuit_v_interception')
figdir = os.path.join(destdir, acq_prefix)
if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

acq = acq_prefix
print(acq_prefix)


#%% ------------------------------------------
# PURSUIT v. INTERCEPTIONS
# --------------------------------------------
# %% Get manually annotated actions -- annoted with FlyTracker

all_action_paths = glob.glob(os.path.join(viddir, 'fly-tracker', '20*', '*actions.mat'))

a_list = []
for fp in all_action_paths:
    # load actions to df
    actions_ = util.ft_actions_to_bout_df(fp)
    acqname = '_'.join(os.path.split(fp)[1].split('_')[0:-1])
    actions_['acquisition'] = acqname
    a_list.append(actions_)
actions = pd.concat(a_list)

#%%
# get path to actions file for current acquisition
#viddir = acqdir
action_fpaths = glob.glob(os.path.join(viddir, 'fly-tracker', '{}*'.format(acq), '*actions.mat'))
action_fpath = action_fpaths[0]
print(action_fpath)

# load actions to df
boutdf = util.ft_actions_to_bout_df(action_fpath)

#%% assign
#%
df_['action'] = None
for i, b_ in boutdf.iterrows():
    start = b_['start']-1
    end = b_['end']-1
    action = b_['action']
    curr_frames = np.arange(start, end+1)
    df_.loc[df_['frame'].isin(curr_frames), 'action'] = action
    df_.loc[df_['frame'].isin(curr_frames), 'group'] = i
action_palette={None: bg_color,
                'interception': 'red', 
                'pursuit': 'cornflowerblue'}


#%% Add actions 
f1 = df_[df_['id']==0].copy().reset_index(drop=True)
f2 = df_[df_['id']==1].copy().reset_index(drop=True)

#%% 
importlib.reload(pp)
import theta_error as the
# Is theta error same as targ_pos_theta??
#th_err = f2['pos_theta'].values - f1['traveling_dir'].values
f1 = the.calculate_theta_error(f1, f2) #, heading_var='ori')

fig, axn =pl.subplots(1, 2, sharex=True, sharey=True)
ax=axn[0]
sns.scatterplot(data=f1, x='targ_pos_theta', y='vel', ax=ax)
ax=axn[1]
sns.scatterplot(data=f1, x='theta_error', y='vel', ax=ax)
#ax.scatter(th_err, f1['targ_pos_theta'])
for ax in axn:
    ax.set_box_aspect(1)

#%%
fig, axn =pl.subplots(1, 2, sharex=False, sharey=False)
ax=axn[0]
sns.scatterplot(data=f1, x='theta_error', y='traveling_dir_dt', ax=ax)
ax=axn[1]
#sns.scatterplot(data=f1, x='theta_error_dt', y='traveling_dir_dt', ax=ax)
nsec_pre = int(np.ceil(0.03 * fps))
print(nsec_pre)
ax.plot(f1['theta_error_dt'], f1['traveling_dir_dt'], 'o')
# for ax in axn:
#     ax.set_aspect(1)

#%% ----
# Sanity check FACING ANGLE
# ----
# curr_frame = 7450 #2430 # to the left is positive?
#curr_frame = 9752
curr_frame = 7470 #7420 # INTERCEPTION EX.
#curr_frame = 10150 # TRACKING EX.
curr_frames = [curr_frame]
nsec_win = 0.4
plot_frame = True

error_var = 'theta_error_deg'

several_frames = np.arange(curr_frame - nsec_win*fps,  curr_frame + nsec_win*fps*3)[0::6]

df_['facing_angle_deg'] = np.rad2deg(df_['facing_angle'])
df_['theta_error_deg'] = np.rad2deg(df_['theta_error'])

f1['ori_deg'] = np.rad2deg(f1['ori'])
f1['traveling_deg'] = np.rad2deg(f1['traveling_dir'])
f1['facing_angle_deg'] = np.rad2deg(f1['facing_angle'])

#df_['theta_error_deg'] = np.rad2deg(df_['theta_error'])

f_ = df_.loc[df_['frame']==curr_frame] #[['pos_x', 'pos_y', 'ori', 'id', 'facing_angle_deg']]
print(f_[['pos_x', 'pos_y', 'ori', 'id', 'facing_angle_deg']])

# this is what DLC.GET_RELATIVE_orientations does:
normPos = f2.loc[f2['frame']==curr_frame][['pos_x', 'pos_y']] - f1.loc[f1['frame']==curr_frame][['pos_x', 'pos_y']]
#absoluteAngle = np.arctan2(normPos['pos_y'], normPos['pos_x'])
#fA = dlc.circular_distance(absoluteAngle, f1.loc[f2['frame']==curr_frame]['ori'])

# plot ------------------------------------------------
fig = pl.figure(figsize=(12, 8))
gs = mpl.gridspec.GridSpec(3, 3)

ax = pl.subplot(gs[0:2, 0:2]) 
curr_cap = cap if plot_frame else None
plot_heading_vs_travelingdir_frames(several_frames, f1, f2, cap=curr_cap, ax=ax, plot_interval=1,
                                          var1='ori', var2='traveling_dir')
#ax.plot(df_[df_['frame'].isin(curr_frames)]['pos_x'].values,
#        df_[df_['frame'].isin(curr_frames)]['pos_y'].values, 'r')
ax.set_xticks([0, 200, 400, 600, 800])
# plot FACING ANGLE
ax = pl.subplot(gs[2, 0])
ax.plot([0, float(normPos['pos_x'])], [0, float(normPos['pos_y'])], 'g')
fA = f_.iloc[0][error_var]
ax.set_title('{}:\nmale ori vs. position = {:.1f} deg'.format(error_var, float(fA)), loc='left')

ax.plot(f_[f_['id']==0]['pos_x'], f_[f_['id']==0]['pos_y'], 'ro')
ax.plot(f_[f_['id']==1]['pos_x'], f_[f_['id']==1]['pos_y'], 'bo')
ax.plot(df_[df_['frame'].isin(curr_frames)]['pos_x'].values,
        df_[df_['frame'].isin(curr_frames)]['pos_y'].values, 'r')
# timesteps
for fi in several_frames:
    f_ = df_.loc[df_['frame']==fi]
    prev_f = df_.loc[df_['frame']==fi-1]
    start_pos = [f_[f_['id']==0]['pos_x'], f_[f_['id']==0]['pos_y'] ]
    dx = np.cos(f1[f1['frame']==fi]['facing_angle'])
    dy = np.sin(f1[f1['frame']==fi]['facing_angle'])
    vlen = 100
    ax.plot( [start_pos[0], start_pos[0] + dx*vlen], [start_pos[1], start_pos[1] + dy*vlen], 'magenta')

ax.set_aspect(1)

# plot THETA ERROR vectors
ax = pl.subplot(gs[2, 1])
ax.plot(f_[f_['id']==0]['pos_x'], f_[f_['id']==0]['pos_y'], 'ro')
ax.plot(f_[f_['id']==1]['pos_x'], f_[f_['id']==1]['pos_y'], 'bo')
ax.plot(df_[df_['frame'].isin(curr_frames)]['pos_x'].values,
        df_[df_['frame'].isin(curr_frames)]['pos_y'].values, 'r')
ax.set_aspect(1)
# traveling vec
for fi in several_frames:
    f_ = df_.loc[df_['frame']==fi]
    prev_f = df_.loc[df_['frame']==fi-1]
    start_pos = [f_[f_['id']==0]['pos_x'], f_[f_['id']==0]['pos_y'] ]
    dx = np.cos(f1[f1['frame']==fi]['traveling_dir'])
    dy = np.sin(f1[f1['frame']==fi]['traveling_dir'])
    vlen = 100
    ax.plot( [start_pos[0], start_pos[0] + dx*vlen], [start_pos[1], start_pos[1] + dy*vlen], 'dodgerblue')

ax.set_aspect(1)

# plot THETA ERROR
ax =pl.subplot(gs[2, 2]) #, projection='polar')
sns.scatterplot(data=f1[f1['frame'].isin(several_frames)], x='sec', y=error_var, ax=ax,
                hue='ang_vel', palette='magma', legend=0)

pl.subplots_adjust(wspace=0.6, hspace=0.6)


figname = 'vecs_heading_traveldir_theta_error_frame{}-{}_{}'.format(several_frames[0], several_frames[-1], acq)
print(figname)

putil.label_figure(fig, acq)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))
print(figdir, figname)

#%% Compare TRAVEL vs. HEADING
import scipy.stats as spstats
#curr_frame = 7446 # interception example
curr_frame = 9720 # 9752  # pursuit example

nframes_add = 1.5 * fps
curr_frames = np.arange(curr_frame, curr_frame+nframes_add)[0::2]
#curr_frames = [curr_frame]

mean_heading = spstats.circmean(f1[f1['frame'].isin(curr_frames)]['ori'], low=-np.pi, high=np.pi)
mean_travel = spstats.circmean(f1[f1['frame'].isin(curr_frames)]['traveling_dir'], low=-np.pi, high=np.pi)

diff_deg = np.rad2deg(mean_heading - mean_travel) % 180.
print(np.rad2deg(f1[f1['frame'].isin(curr_frames)]['facing_angle'].iloc[0:10]))
print(np.rad2deg(f2[f2['frame'].isin(curr_frames)]['facing_angle'].iloc[0:10]))

#%
fig, axn =pl.subplots(1, 2, figsize=(12,5))
ax=axn[0]
plot_heading_vs_travelingdir_frames(curr_frames, f1, f2, ax=ax, cap=cap,
                                          var1='ori', var2='traveling_dir'
                                          )
ax.set_xticks([0, 200, 400, 600, 800])
#ax.plot(df_[df_['frame'].isin(curr_frames)]['pos_x'].values,
#        df_[df_['frame'].isin(curr_frames)]['pos_y'].values, 'r')

ax=axn[1]
ax.plot(f1[f1['frame'].isin(curr_frames)]['sec'], 
        np.unwrap(f1[f1['frame'].isin(curr_frames)]['ori']), color='magenta')
ax.plot(f1[f1['frame'].isin(curr_frames)]['sec'], 
        np.unwrap(f1[f1['frame'].isin(curr_frames)]['traveling_dir']), color='dodgerblue')
ax.set_box_aspect(1)
ax.set_title('heading v travel, diff {:.2f} deg'.format(diff_deg))

pl.subplots_adjust(wspace=0.5)
#%
# save
putil.label_figure(fig, acq)
figname = 'heading_vs_travelingdir_frames_{}'.format(curr_frames[0], curr_frames[-1])
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(os.path.join(figdir, figname))

#%%



# plot ------------------------------------------------
fig, axn = pl.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))

# interception ex
curr_frame = 7470 #7420 # INTERCEPTION EX.
nsec_win = 0.4
interception_frames = np.arange(curr_frame - nsec_win*fps,  curr_frame + nsec_win*fps*3)[0::6]
# tracking ex
curr_frame = 10150 #7420 # INTERCEPTION EX.
nsec_win = 0.4
tracking_frames = np.arange(curr_frame - nsec_win*fps,  curr_frame + nsec_win*fps*3)[0::6]

plot_frame = True

ax=axn[0]
curr_cap = cap if plot_frame else None
plot_heading_vs_travelingdir_frames(tracking_frames, f1, f2, cap=curr_cap, ax=ax, plot_interval=1,
                                          var1='ori', var2='traveling_dir')
ax.legend_.remove()
ax.set_xticks([0, 200, 400, 600, 800])

ax=axn[1]
plot_heading_vs_travelingdir_frames(interception_frames, f1, f2, cap=curr_cap, ax=ax, plot_interval=1,
                                          var1='ori', var2='traveling_dir')

putil.label_figure(fig, acq)
figname = 'interception_v_tracking_ex_{}'.format(acq)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

print(os.path.join(figdir, figname))


#%%

#curr_frames = np.arange(9750, 9780)[0::5] # #9791)
curr_frames = np.arange(10118, 10178)[0::5]
f_ = df_[df_['frame'].isin(curr_frames)].copy()

# plot vectors
f_1 = f_[f_['id']==0]
f_2 = f_[f_['id']==1]

#for i, fr in enumerate(curr_frames): 
#    ax.plot([f_1[f_1['frame']==fr]['pos_x'], f_2[f_2['frame']==fr]['pos_x']],
#            [f_1[f_1['frame']==fr]['pos_y'], f_2[f_2['frame']==fr]['pos_y']], 
#            'w-', lw=0.5)
#ax.set_aspect(1) 
#

var1='ori'
var2 = 'traveling_dir'
ix = curr_frames[0]

col1 = 'magenta'
col2 = 'dodgerblue'
target_col = 'green'

# set fly oris as arrows
fly_marker = '$\u2192$' # https://en.wikipedia.org/wiki/Template:Unicode_chart_Arrows

n_frames_plot = 7

# plot ------------------------------------------------
fig = pl.figure(figsize=(12, 8))
gs = mpl.gridspec.GridSpec(3, n_frames_plot)

ax = pl.subplot(gs[0:2, 0:2]) 
plot_heading_vs_travelingdir_frames(curr_frames[0:n_frames_plot], f1, f2, cap=cap, ax=ax, plot_interval=1,
                                          var1='ori', var2='traveling_dir')
for i, ix in enumerate(curr_frames[0:n_frames_plot]):
    print(ix)
    ax = pl.subplot(gs[2, i]) #=axn[2, i]

    # make a markerstyle class instance and modify its transform prop
    m_ori = np.rad2deg(f1.loc[ix][var1])
    m_trav = np.rad2deg(f1.loc[ix][var2])
    f_ori = np.rad2deg(f2.loc[ix][var2])

    marker_m1 = mpl.markers.MarkerStyle(marker=fly_marker)
    marker_m1._transform = marker_m1.get_transform().rotate_deg(m_ori)

    marker_m2 = mpl.markers.MarkerStyle(marker=fly_marker)
    marker_m2._transform = marker_m2.get_transform().rotate_deg(m_trav)

    marker_f = mpl.markers.MarkerStyle(marker=fly_marker)
    marker_f._transform = marker_f.get_transform().rotate_deg(f_ori)


    ax.plot([f1['pos_x'].loc[ix]], [f1['pos_y'].loc[ix]], 'w', 
            marker=marker_m1, markerfacecolor=col1, markersize=30, markeredgewidth=0.5, label=var1) 

    ax.plot([f1['pos_x'].loc[ix]], [f1['pos_y'].loc[ix]], 'w', 
            marker=marker_m2, markerfacecolor=col2, markersize=30, markeredgewidth=0.5, label=var2) 

    ax.plot([f2['pos_x'].loc[ix]], [f2['pos_y'].loc[ix]], 'w',
            marker=marker_f, markerfacecolor=target_col, markersize=20, markeredgewidth=0.5)

    ax.plot([f_1[f_1['frame']==ix]['pos_x'], f_2[f_2['frame']==ix]['pos_x']],
                [f_1[f_1['frame']==ix]['pos_y'], f_2[f_2['frame']==ix]['pos_y']], 
                'w-', lw=0.5)
    ax.set_title("th-err: {:2f}".format(np.rad2deg(f1.loc[ix]['theta_error'])), loc='left')
    ax.set_aspect(1)

#%% -------
#curr_frames = np.arange(13322, 13412)
curr_frames = np.arange(14757, 15065)

#fig, ax = pl.subplots()
#sns.regplot(data=f1[f1['frame'].isin(curr_frames)], 
#            x='theta_error', y='ang_vel', ax=ax)

offsets = [1, 2, 3, 4, 5, 6]
n_offsets = len(offsets)
fig, axn = pl.subplots(1, n_offsets, sharex=True, sharey=True)
for ai, offset in enumerate(offsets):
    ax=axn[ai]
    sns.regplot( x = f1[f1['frame'].isin(curr_frames)]['theta_error'].iloc[:-offset],
                 y = f1[f1['frame'].isin(curr_frames)]['traveling_dir_dt'].iloc[offset:], ax=ax)
    ax.set_box_aspect(1)

#%%
fig, ax = pl.subplots(figsize=(12, 4))
ax.plot(f1[f1['frame'].isin(curr_frames)]['sec'], 
        f1[f1['frame'].isin(curr_frames)]['theta_error'], 'r')

ax2 = ax.twinx()
ax2.plot(f1[f1['frame'].isin(curr_frames)]['sec'],
        f1[f1['frame'].isin(curr_frames)]['traveling_dir_dt'])


#%% ASSIGN ACTIONS TO DF
# ----------------------------------------------

srcdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/2d-projector/JAABA'
#% Load jaaba-traansformed data
jaaba_fpath = os.path.join(srcdir, 'jaaba_transformed_data_elegans.pkl')
assert os.path.exists(jaaba_fpath), "File not found: {}".format(jaaba_fpath)
jaaba = pd.read_pickle(jaaba_fpath)   

jaa_ = jaaba[jaaba['name']==acq].copy()
print(jaa_.shape)
has_jaaba = jaa_.shape[0] == f1.shape[0]

print(has_jaaba) #
#%%


#%%% SANITY CHECK: Can we pull out Pursuit frames?

min_vel = 1
max_facing_angle = np.deg2rad(45)
min_facing_angle = np.deg2rad(-45)
max_dist_to_other=30

f1['vel_smoothed'] = f1.groupby('id')['vel'].transform(lambda x: x.rolling(3, 1).mean())
print(f1['vel_smoothed'].max(), f1['vel_smoothed'].min())

possible_chasing = f1[ (f1['vel_smoothed']>=min_vel) & (f1['heading_travel_diff']<45)
                     & (f1['facing_angle']<max_facing_angle)
                     & (f1['facing_angle']>-max_facing_angle) 
                     & (f1['dist_to_other']<=max_dist_to_other)
                     & (f2['vel_smoothed']>0.5)].copy()

possible_chasing.groupby('action').count()

#

#%%
## Look at a bunch of vars across time, color code events

plot_vars = ['pos_x', 'vel', 'ang_vel', 'dist_to_other',
             # 'facing_angle', 'targ_pos_theta', 'theta_error',
             'theta_error',
             'theta_error_dt', 'targ_ang_vel']

t1 = 120 #150 # 120
t2 = 160 #180 #160 
plotdf = f1[(f1['sec']>t1) & (f1['sec']<t2)]

plotdf2 = f2[(f2['sec']>t1) & (f2['sec']<t2)]
fig, axn = pl.subplots(len(plot_vars)+1, 1, figsize=(10,2*len(plot_vars)), sharex=True)
ax=axn[0]
ax.plot(plotdf2['sec'], plotdf2['pos_x'], color=bg_color, alpha=0.5)
ax.set_ylabel('target pos x')

for i, yvar in enumerate(plot_vars):
    ax = axn[i+1]
    ax.plot(plotdf['sec'], plotdf[yvar], color=bg_color, alpha=0.5)
    sns.scatterplot(data=plotdf, x='sec', y=yvar, ax=ax,
                    hue='action', palette=action_palette,
                    edgecolor='none', s=10, alpha=0.5, legend=i==0)
    if i==0:
        sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1,1), frameon=False)
    if yvar=='targ_pos_theta':
        ax.axhline(y=0, color='w', linestyle=':')
    if yvar in ['ang_vel', 'ang_vel_abs', 'theta_error_dt', 'targ_ang_vel']:
        ax.set_ylim([-30, 30])
    if yvar == 'theta_error':
        ax.axhline(y=0, color='w', linestyle=':', lw=0.5)

for ax in axn:
    putil.remove_spines(ax)
    sns.despine(bottom=True)
pl.subplots_adjust(wspace=0.5)

# save
putil.label_figure(fig, acq)
figname = 'pursuit_interception_bouts_by_time_t{}-{}'.format(t1, t2) #plotdf['sec'].min(), plotdf['sec'].max())
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(os.path.join(figdir, figname))
#%%
fig, ax = pl.subplots(figsize=(8,4))
yvar = 'theta_error'
ax.plot(plotdf['sec'], plotdf[yvar], color=bg_color, alpha=0.5)
sns.scatterplot(data=plotdf, x='sec', y=yvar, ax=ax,
                hue='action', palette=action_palette,
                edgecolor='none', s=10, alpha=0.5, legend=i==0)

ax.axhline(y=0, color='w', linestyle=':', lw=0.5)
ax.set_ylim([-1, 1])

# %% ------------------------------------------
# Separate bouts
#% --------------------------------------------
has_jaaba=False
if has_jaaba:
    f1['action'] = None
    f1.loc[f1['frame'].isin(jaa_[jaa_['chasing']>20]['frame']), 'action'] = 'pursuit'
    pursuits = f1[f1['action']=='pursuit'].copy()
    print(pursuits.shape)
else:
    pursuits = f1[(f1['epoch']>0) & (f1['action']=='pursuit')]


interceptions = f1[ (f1['action']=='interception')].copy()
#consec_bouts_int = pp.get_indices_of_consecutive_rows(interceptions)

#%% Remove too-short
filter_duration = False
if filter_duration:
    importlib.reload(pp)
    consec_bouts = pp.get_indices_of_consecutive_rows(pursuits)

    # filter duration?
    min_bout_len = 0.25
    fps = 60.
    incl_ixs, incl_bouts = pp.filter_bouts_by_frame_duration(consec_bouts, min_bout_len, fps, return_indices=True)

    print("{} of {} bouts pass min dur {}sec".format(len(incl_bouts), len(consec_bouts), min_bout_len))
                                                    
    pursuits = pursuits[pursuits['group'].isin(incl_ixs)] #pursuits[~pursuits['group'].isin(incl_ixs), 'group'] = None #s


#%%# Do pursuit bouts change with stim speed
importlib.reload(pp)
fig, axn = pl.subplots(1, 2,#figsize=(10,5), sharex=True, sharey=True,
                       subplot_kw={'projection': 'polar'})
pp.plot_allo_vs_egocentric_pos(pursuits, axn, huevar='stim_hz', cmap='viridis',
                        palette_dict=stimhz_palette, hue_norm=mpl.colors.Normalize(vmin=0, vmax=1),
                        plot_com=True, bg_color=bg_color) 
pl.subplots_adjust(left=0.1, right=0.85, wspace=0.5)

fig.text(0.05, 0.9, 'Pursuit bouts')
putil.label_figure(fig, acq)

# save
putil.label_figure(fig, acq)
figname = 'allo_v_ego_by_stimhz_pursuit'
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(os.path.join(figdir, figname))


#%% PURSUIT:  plot all vs. ego, color by ANG_VEL
nbins=5
f1['ang_vel_abs_binned'] = pd.cut(f1['ang_vel_abs'], bins=nbins)
f1['ang_vel_abs_leftbin'] = [v.left if isinstance(v, pd.Interval) else v for v in f1['ang_vel_abs_binned']]

#% plot.
#plotdf = f1[(f1['epoch']>0) & (f1['action']=='pursuit')]
fig, ax =pl.subplots()
sns.histplot(data=f1, x='ang_vel_abs', hue='ang_vel_abs_leftbin',  
            palette='viridis', ax=ax,
            stat='probability', common_norm=False, common_bins=True,
            element='bars', fill=False, alpha=1)
#%
importlib.reload(pp)
fig, axn = pl.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True,
                       subplot_kw={'projection': 'polar'})
pp.plot_allo_vs_egocentric_pos(pursuits, axn, huevar='ang_vel_abs', cmap='viridis',
                        hue_norm=mpl.colors.Normalize(vmin=0, vmax=20), 
                        markersize=10,
                        #palette_dict=stimhz_palette, 
                        plot_com=False, bg_color=bg_color) 
pl.subplots_adjust(left=0.1, right=0.85, wspace=0.5)

# save
putil.label_figure(fig, acq)
figname = 'allo_v_ego_by_stimhz_pursuit_color-by-ang_vel_abs'
#pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(os.path.join(figdir, figname))


# #%% take a closer look at ang_vel_abs
# fig, ax =pl.subplots(subplot_kw={'projection': 'polar'})
# sns.scatterplot(data=plotdf, ax=ax,
#                 x='targ_pos_theta', y='targ_pos_radius', s=10,
#                 hue='ang_vel_abs', palette='viridis',
#                 hue_norm=(0, 30), 
#                 edgecolor='none', legend=0, alpha=0.7)

#%% INTERCEPTIONS:  Plot EGO, but only for INTERCEPTION events

fig, axn = pl.subplots(1, 2,#figsize=(10,5), sharex=True, sharey=True,
                       subplot_kw={'projection': 'polar'})
pp.plot_allo_vs_egocentric_pos(interceptions, axn, huevar='stim_hz', cmap='viridis',
                        palette_dict=stimhz_palette, hue_norm=mpl.colors.Normalize(vmin=0, vmax=1),
                        plot_com=True, bg_color=bg_color) 
pl.subplots_adjust(left=0.1, right=0.85, wspace=0.5)

fig.text(0.05, 0.9, 'Interception bouts')

# save
putil.label_figure(fig, acq)
figname = 'allo_v_ego_by_stimhz_interception'
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(os.path.join(figdir, figname))

#%% ------------------------------------------
# SELECTION ACTION TYPE
# --------------------------------------------

action_type = 'interceptions'

if action_type=='pursuit':
    actions = pursuits.copy()
elif action_type=='interception':
    actions = interceptions.copy()
else:
    actions = f1.copy()


#%% POLAR:  plot THETA_ERROR 
fig, axn = pl.subplots(1, 2,#figsize=(10,5), sharex=True, sharey=True,
                       subplot_kw={'projection': 'polar'})
pp.plot_allo_vs_egocentric_pos(actions, axn, xvar='theta_error', yvar='dist_to_other',
                               huevar='stim_hz', cmap='viridis',
                        palette_dict=stimhz_palette, hue_norm=mpl.colors.Normalize(vmin=0, vmax=1),
                        plot_com=True, bg_color=bg_color) 
pl.subplots_adjust(left=0.1, right=0.85, wspace=0.5)


#%%

# PLOT ALL BOUTS

nrows = interceptions['stim_hz'].nunique()
bouts_per = interceptions.groupby('stim_hz')['group'].unique()
ncols = max([len(v) for k, v in bouts_per.items()])
ncols

fig, axn = pl.subplots(nrows, ncols, figsize=(ncols*5, nrows*5), sharex=True, sharey=True)

for r, (stim, b_) in enumerate(interceptions.groupby('stim_hz')):
    for c, (boutnum, g_) in enumerate(b_.groupby('group')):
        ax = axn[r, c]
        bout_frames = g_['frame'].unique()
        if stim==1:
            nsec_pre = 0.
        else:
            nsec_pre = 0.75
        start_ix = bout_frames[0] - nsec_pre*fps
        stop_ix = bout_frames[-1] + 0.5*fps
        curr_frames = np.arange(start_ix, stop_ix)
#        sns.scatterplot(data=g_[g_['frame'].isin(curr_frames)], x='pos_x', y='pos_y', ax=ax,
#                        hue='sec', palette='viridis')
#        sns.scatterplot(data=f2[f2['frame'].isin(curr_frames)], ax=ax,
#                        x='pos_x', y='pos_y', color='red', s=10)
#                        #hue='sec', palette='viridis')         
        plot_heading_vs_travelingdir_frames(curr_frames, f1, f2, ax=ax, plot_interval=5,
                                          var1='ori', var2='traveling_dir')

        ax.set_aspect(1)
        ax.set_title('Frame: {}'.format(curr_frames[0]), loc='left')

pl.savefig(os.path.join(figdir, 'interception_bouts_traveling-v-ori.png'))

#%%

pre_frames = np.arange(9805, 9826)
pre_ = f1[f1['frame'].isin(pre_frames)]
pre_['vel']


nsec_pre = 2
n_bouts = interceptions['group'].nunique()
fig, axn = pl.subplots(n_bouts, 1, figsize=(8, 3*n_bouts), sharex=True, sharey=True)
for ai, (i, b_) in enumerate(interceptions.groupby('group')):
    ax=axn[ai]
    bout_frames = np.arange( b_.iloc[0]['frame'] - nsec_pre*fps, b_.iloc[-1]['frame']) 
    sec = f1[f1['frame'].isin(bout_frames)]['sec'] - b_.iloc[0]['sec']
    stim = b_.iloc[0]['stim_hz']
    ax.plot(sec, f1[f1['frame'].isin(bout_frames)]['accel'], color=stimhz_palette[stim])
    ax.axvline(x=0, color='w', linestyle='--')

#%% 
 
fig, ax = pl.subplots()

interceptions['theta_error_abs'] = np.abs(interceptions['theta_error'])
#interceptions['theta_error_abs_dt'] = interceptions['theta_error_abs'].diff() / interceptions['sec_diff'].mean()

for i, b_ in interceptions.groupby('group'):
    sns.scatterplot(data=b_, x='targ_pos_theta', y='ang_vel', ax=ax)

ax.set_ylim([-30, 30])


fig, ax = pl.subplots()
for i, b_ in interceptions.groupby('group'):
    bb = b_[b_['vel']>=3]
    ax.hist( bb['theta_error'], color='w', alpha=0.5)

fig, ax = pl.subplots()
plotdf = interceptions[interceptions['theta_error_abs'] < np.deg2rad(35)]
#plotdf = pursuits.copy() #< np.deg2rad(35)]
sns.regplot(data=plotdf, x='theta_error', y='sec', ax=ax)

ax.set_ylim([-30, 30])

fig, ax = pl.subplots()
sns.regplot(data=plotdf, x='theta_error_dt', y='traveling_dir_dt', ax=ax)
ax.set_ylim([-30, 30])


#%%

fig, ax = pl.subplots()
for i, b_ in pursuits.groupby('group'):
    bb = b_[b_['vel']>=3]
    ax.hist( bb['theta_error'], color='w', alpha=0.5)

fig, ax = pl.subplots()
sns.regplot(data=pursuits[pursuits['vel']>3], x='theta_error', y='ang_vel', ax=ax)

ax.set_ylim([-30, 30])




#%% Calcualte some more variables....

#f1['targ_ang_vel'] = f1['targ_pos_theta'].diff() / f1['sec_diff'].mean()

def calculate_speeds(positions_over_time, xvar='pos_x', yvar='pos_y', time='sec'):

    movements_over_timesteps = (
        np.roll(positions_over_time, -1, axis=0)
        - positions_over_time)[:-1]

    speeds = np.sqrt(
            movements_over_timesteps[xvar] ** 2 +
            movements_over_timesteps[yvar] ** 2) / movements_over_timesteps[time].mean()

    speeds = np.concatenate([[0], speeds.values])

    return pd.DataFrame({
        time: positions_over_time[time], #positions_over_time[time][:-1],
        'calc_speed': speeds,
    })


xvar = 'targ_rel_pos_x_mm'
yvar = 'targ_rel_pos_y_mm'
#positions_over_time = f1[[xvar, yvar, 'sec']].copy()

f1['pos_x_mm'] = f1['pos_x'] * 0.04
f1['pos_y_mm'] = f1['pos_y'] * 0.04
f1['targ_rel_pos_x_mm'] = f1['targ_rel_pos_x'] * 0.04
f1['targ_rel_pos_y_mm'] = f1['targ_rel_pos_y'] * 0.04
f1['targ_vel'] = f2['vel'].values

f'] = f1['vel'].diff() / f1['sec_diff'].mean()

speeds = calculate_speeds(f1[['targ_rel_pos_x_mm', 'targ_rel_pos_y_mm', 'sec']].interpolate(), 
                          xvar='targ_rel_pos_x_mm', yvar='targ_rel_pos_y_mm', time='sec')    
f1['targ_rel_vel'] = speeds['calc_speed'].values


f1['targ_rad_vel'] = f1['targ_pos_radius'].diff() / f1['sec_diff'].mean()
f1['targ_size_change_speed'] = f1['targ_ang_size'].diff() / f1['sec_diff'].mean()
#fig, ax =pl.subplots()
#ax.scatter(speeds['calc_speed'], f1['vel'], color='w', alpha=0.5)
#ax.scatter(f1['pos_x'], f1['pos_x_mm'], color='w', alpha=0.5)


#%% Look at INTERCEPTION bouts and a bunch of variables across time: (line plots)

#interceptions = f1[ (f1['action']=='interception')].copy()
#consec_bouts = pp.get_indices_of_consecutive_rows(interceptions)
#plotdf = plotdf[plotdf['stim_hz']>0].copy()
#interceptions[interceptions['stim_hz']==0] = None
#plotdf = plotdf.interpolate()

plotvars = ['targ_pos_theta', 'targ_ang_vel', 'theta_error', 'theta_error_dt',
            'ang_vel', 'vel', 'accel', #'targ_rel_vel', 'targ_ang_vel',
            'targ_ang_size_deg', 'dist_to_other', 'targ_rel_vel']

curr_stim_hz_vals = [0, 0.1, 0.2, 0.4, 0.8, 1]
#plotdf = interceptions[interceptions['stim_hz'].isin(curr_stim_hz_vals)]

palette_dict = dict((k, v) for k, v in zip(actions['stim_hz'].unique(), 
                                           sns.color_palette('viridis', n_colors=len(actions['stim_hz'].unique())))) 

nsec_pre = 0 
fig, axn =pl.subplots(len(plotvars), 1, figsize=(10, 2*len(plot_vars)))
d_list=[]
for i, b_ in actions.groupby('group'):
    start_ix = b_.iloc[0]['frame'] - nsec_pre*fps*2
    stop_ix = b_.iloc[-1]['frame'] #b_.iloc[0]['frame'] + nsec_pre*fps*2#*4
    stim_hz_vs = b_['stim_hz'].dropna().unique()
    if len(stim_hz_vs)>1:
        cnts = b_.groupby('stim_hz').count()['frame'].argmax()
        vals = list(b_.groupby('stim_hz').groups.keys())
        stim_hz = vals[cnts]
    else:
        stim_hz = stim_hz_vs[0]
    #print(stim_hz)
    #f1[f1['frame']==b_.iloc[0]['frame']]
    currdf = f1.loc[start_ix:stop_ix].copy()
    currdf['sec'] = currdf['sec'] - b_.iloc[0]['sec'] 
    currdf['group'] = i
    d_list.append(currdf)
    for ai, yvar in enumerate(plotvars):
        ax=axn[ai]
        ax.plot(currdf['sec'],
                currdf[yvar], 
                color=palette_dict[stim_hz]) #alpha=0.5  )
        ax.axvline(x=0, color=bg_color, linestyle=':')
        ax.set_ylabel(yvar)
        if yvar in ['targ_pos_theta', 'theta_error']:
            ax.axhline(y=0, color='w', linestyle=':')
        if yvar=='targ_ang_vel':
            ax.set_ylim([-50, 50])
        if yvar=='ang_vel':
            ax.set_ylim([-50, 50])
centered_bouts = pd.concat(d_list).reset_index(drop=True)

# save
putil.label_figure(fig, acq)
figname = 'compare_yvars_{}_bouts_by_time_and_stimhz'.format(action_type)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(os.path.join(figdir, figname))
#%%

yvar = 'theta_error' #'targ_pos_theta'

ycol1 = 'dodgerblue'
ycol2 = 'red'

bnum = 18 #7
b_ = centered_bouts[centered_bouts['group']==bnum].copy()
#start_ix = boutdf[boutdf['action']=='pursuit'].loc[bnum]['start']
#b_[b_['frame']==start_ix]

b_.loc[(b_['ang_vel']>100) | (b_['ang_vel']<-100), 'ang_vel'] = np.nan

#fig = plot_heading_vs_travelingdir_frames(cap, b_['frame'].values[20:60], f1, f2)
#fig = plot_heading_vs_travelingdir_frames(cap, b_['frame'].values[60:80], f1, f2)

fig, ax =pl.subplots(figsize=(10,4))
ax.plot(b_['sec'], b_[yvar], color=ycol1)
# color spines blue:
ax.spines['left'].set_color(ycol1)
# set tick labels blue:
ax.tick_params(axis='y', colors=ycol1) 
ax.set_ylabel(yvar, color=ycol1)

ax2 = ax.twinx()
ax2.plot(b_['sec'], b_['ang_vel'], color=ycol2)
ax2.spines['right'].set_color(ycol2)
ax2.tick_params(axis='y', colors=ycol2) 
ax2.set_ylabel('ang_vel', color=ycol2)


ax.set_xlim([0, 5])
#%%

fig, ax =pl.subplots()
ax.scatter(centered_bouts['theta_error'], centered_bouts['ang_vel'])
ax.set_ylim([-30, 30])

sns.regplot(data=centered_bouts, x='theta_error', y='ang_vel', ax=ax, scatter=False)

#%%

plotvars = ['vel', 'ang_vel', 'targ_pos_theta', 'targ_ang_vel',
            'theta_error', 'theta_error_dt',
             'dist_to_other', 'targ_rad_vel', 'targ_size_change_speed']
for yvar in plotvars:
    fig, ax = pl.subplots()
    sns.lineplot(data=centered_bouts, x='sec', y=yvar,
                 hue='stim_hz', palette='viridis', lw=0.5, ax=ax, legend=0)
                #hue='group', palette=dict((k, bg_color) for k in curr_plotdf['group'].unique()), lw=0.5,
                #legend=0)
    #palette=palette_dict, ax=ax)
    ax.axhline(y=0, color='w', linestyle=':')
    ax.axvline(x=0, color='w', linestyle=':')
    if yvar in ['ang_vel', 'targ_ang_vel']:
        ax.set_ylim([-50, 50])
    elif yvar=='targ_size_change_speed':
        ax.set_ylim([-20, 20])
    #ax.set_xlim([-0.5, 0.5])
    # save
    putil.label_figure(fig, acq)
    figname = '{}-bouts_{}_stimhz'.format(action_type, yvar)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    print(os.path.join(figdir, figname))

    #curr_plotdf.groupby('stim_hz')['group'].count()
#%% polar: Plot TARG_POS_THETA for each bout, separate by stim_hz level
cmap='viridis'

#nsec_pre = 0.5
curr_bouts =[]
for i, currbouts in actions[actions['stim_hz']>=0.1].groupby('stim_hz'):
    d_list = []
    fig, ax = pl.subplots(subplot_kw={'projection': 'polar'})
    for ii, b_ in currbouts.groupby('group'):
        start_ix = b_.iloc[0]['frame'] - nsec_pre*fps*2
        stop_ix = b_.iloc[0]['frame'] #+ nsec_pre*fps #*4
        stim_hz_vs = b_['stim_hz'].dropna().unique()
        assert len(stim_hz_vs)==1, "Cross >1 stimHz val: {}".format(stim_hz_vs)
        stim_hz = stim_hz_vs[0]
        #f1[f1['frame']==b_.iloc[0]['frame']]
        currdf = f1.loc[start_ix:stop_ix].copy()
        ax.plot(b_.iloc[0]['targ_pos_theta'], b_.iloc[0]['targ_pos_radius'], 'r*')

        currdf['sec'] = currdf['sec'] - b_.iloc[0]['sec'] 
        ax.plot(currdf['targ_pos_theta'], currdf['targ_pos_radius'], 'w', lw=0.5)

        d_list.append(currdf)
    plotdf_ = pd.concat(d_list).reset_index(drop=True)

    sns.scatterplot(data=plotdf_, ax=ax,  
                x='targ_pos_theta', y='targ_pos_radius', 
                hue='sec', palette='viridis', alpha=0.5, legend=0)

    putil.colorbar_from_mappable(ax, norm=mpl.colors.Normalize(vmin=plotdf_['sec'].min(), vmax=plotdf_['sec'].max()),
                                 cmap=cmap, hue_title='sec')
    ax.set_title('Stim Hz: {}'.format(stim_hz))
    curr_bouts.append(plotdf_)

pre_intereceptions = pd.concat(curr_bouts).reset_index(drop=True)

#sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', ax=ax)

#%% Look at 1 more carefully

xvar = 'theta_error'
currbouts = actions[actions['stim_hz']==0.4].copy()
n_bouts = currbouts['group'].nunique()
print(n_bouts)

#fig, axn = pl.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': 'polar'})
fig, ax =pl.subplots(subplot_kw={'projection': 'polar'})

for i, (bnum, b_) in enumerate(currbouts.groupby('group')):
    #ax=axn.flat[i]
    ax.set_title('bout {} (n={})'.format(i, bnum))
    start_ix = b_.iloc[0]['frame']# - nsec_pre*fps*2
    stop_ix = b_.iloc[0]['frame'] + nsec_pre*fps #*4
    stim_hz_vs = b_['stim_hz'].dropna().unique()
    assert len(stim_hz_vs)==1, "Cross >1 stimHz val: {}".format(stim_hz_vs)
    stim_hz = stim_hz_vs[0]
    #f1[f1['frame']==b_.iloc[0]['frame']]
    currdf = f1.loc[start_ix:stop_ix].copy()
    ax.plot(b_.iloc[0][xvar], b_.iloc[0]['targ_pos_radius'], 'r*')

    sns.scatterplot(data=currdf, ax=ax,  
                x=xvar, y='targ_pos_radius', s=100, 
                hue='sec', palette='viridis', alpha=0.7, legend=0)
    ax.set_ylim([0, 600])

fig.suptitle("Stim hz = {:.2f} bouts".format(stim_hz))

#%% linearly?

fig, axn = pl.subplots(1, 2)
ax=axn[0]
sns.lineplot(data=centered_bouts, x='sec', y='theta_error', 
                hue='stim_hz', palette='viridis', ax=ax, legend=0)
ax=axn[1]
sns.lineplot(data=centered_bouts, x='sec', y='ang_vel', 
                hue='stim_hz', palette='viridis', ax=ax, legend=0)
ax.set_ylim([-50, 50])

for ax in axn.flat:
    ax.set_box_aspect(1)
    ax.axvline(x=0, color='w', linestyle=':')


#%% TWINx:  Plot TARG_POS_THETA *and* Male metric across TIME
yvar1 = 'theta_error'
yvar2 = 'ang_vel'

nsec_pre = 5

#thisdf = interceptions.copy()
if action_type=='pursuit':
    thisdf = pursuits.copy()
else:
    thisdf = interceptions.copy()

_bouts = pp.get_indices_of_consecutive_rows(thisdf)

#%%
incl_hz = thisdf[thisdf['stim_hz']>=0.1]['stim_hz'].unique()

n_cols = len(incl_hz)

fig, axn = pl.subplots(n_cols, 1, figsize=( 5, 5*n_cols)) #subplot_kw={'projection': 'polar'})
d_list = []
for ai, (i, currbouts) in enumerate(thisdf[thisdf['stim_hz']>0].groupby('stim_hz')):
    ax=axn[ai]
    #d_list = []
    for ii, b_ in currbouts.groupby('group'):
        start_ix = b_.iloc[0]['frame'] #- nsec_pre*fps
        stop_ix = b_.iloc[-1]['frame'] #b_.iloc[0]['frame'] + nsec_pre*fps #+ nsec_pre*fps  #nsec_pre*fps #*4
        stim_hz_vs = b_['stim_hz'].dropna().unique()
        assert len(stim_hz_vs)==1, "Cross >1 stimHz val: {}".format(stim_hz_vs)
        stim_hz = stim_hz_vs[0]
        #f1[f1['frame']==b_.iloc[0]['frame']]
        currdf = f1.loc[start_ix:stop_ix].copy()
        # plot dot pos
        currdf['sec'] = currdf['sec'] - b_.iloc[0]['sec'] 
        ax.plot(currdf['sec'], currdf[yvar1], 'w', lw=0.5)
        ax.set_ylabel(yvar1)
        ax.set_ylim([-np.pi, np.pi])

        # plot male velocity
        ax2 = ax.twinx()
        ax2.plot(currdf['sec'], currdf[yvar2], 'r', lw=0.5)
        ax2.set_ylim([-20, 20])
        ax2.set_ylabel(yvar2)

        ax.axhline(y=0, color='w', linestyle=':', lw=0.25)
        ax.axvline(x=0, color='w', linestyle=':', lw=0.25)

        d_list.append(currdf)
    ax.set_title('Stim Hz: {}'.format(stim_hz))

prepost_bouts = pd.concat(d_list).reset_index(drop=True)
#prepost_bouts = prepost_bouts[prepost_bouts['frame'].isin(nonframes)]



#for ax in axn:
##    ax.set_box_aspect(1)
#   ax.set_xlabel('time')
pl.subplots_adjust(wspace=0.7)

#%%
th_err_bound = np.rad2deg(10) #20)

plotdf = prepost_bouts[(prepost_bouts[xvar]<th_err_bound) 
                     | (prepost_bouts[xvar]>=-th_err_bound)].copy()
#fig, ax =pl.subplots()
fig, axn = pl.subplots(1, n_cols, figsize=(n_cols*5, 5)) #subplot_kw={'projection': 'polar'})
for ai, (s_, d_) in enumerate(plotdf.groupby('stim_hz')):
    ax=axn[ai]
    sns.scatterplot(data=d_, ax=ax,
                x='theta_error', y='ang_vel', s=50,
                hue='sec', palette='viridis',
                edgecolor='none', legend=0, alpha=0.7)
    ax.set_box_aspect(1)

#%%
s_ = 0.4
cmap='RdBu'

d_ = plotdf[plotdf['stim_hz']==s_]
fig, axn =pl.subplots(1, 2)
ax=axn[0]
sns.scatterplot(data=d_, x='theta_error', y='ang_vel', hue='sec', ax=ax, legend=0, palette=cmap)
ax=axn[1]
sns.scatterplot(data=d_, x='theta_error_dt', y='ang_vel', hue='sec', ax=ax, palette=cmap)
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1,1))
for ax in axn:
    ax.set_box_aspect(1)










#%%
import scipy.stats as spstats

for s_, d_ in curr_plotdf.groupby('group'):
    mean_ang_vel_post = d_[d_['sec']>=0]['ang_vel'].mean() # -neg to pos dir
    mean_ang_vel_pre = d_[d_['sec']<0]['ang_vel'].mean() # -neg to pos dir
    
    mean_theta_err_pre = spstats.circmean(d_[d_['sec']<0]['targ_pos_theta'], low=-np.pi, high=np.pi)
    mean_theat_err_post = spstats.circmean(d_[d_['sec']>=0]['targ_pos_theta'], low=-np.pi, high=np.pi)


#%%  TRAVELING vs. HEADING DIR

plotdf = f1[ (f1['action']=='interception')].copy()
consec_bouts = pp.get_indices_of_consecutive_rows(plotdf)
#plotdf = plotdf[plotdf['stim_hz']>0].copy()
plotdf[plotdf['stim_hz']==0] = None
#plotdf = plotdf.interpolate()


fig, axn  =pl.subplots(7, 2)
for bnum, b_ in plotdf.groupby('group'):
    ax = axn.flat[bnum]
    ax.plot(b_['sec'], np.unwrap(b_['traveling_dir']))
    ax.plot(b_['sec'], b_['ori'])

#%% # make movie of dlc
    
#cap.set(1, 12928); ret, im = cap.read(); im.shape
bnum = 3
b_ =   plotdf[plotdf['group']==bnum].copy()
fig, ax =pl.subplots()
ax.plot(b_['sec'], b_['traveling_dir'])
ax.plot(b_['sec'], b_['ori'])

#%%
tmpdir = os.path.join(figdir, 'tmp')
if not os.path.exists(tmpdir):
    os.makedirs(tmpdir)
if len(os.listdir(tmpdir))>0:
    for i in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, i))

#%%
# make movie
curr_frames = b_['frame'].values

#ix = curr_frames[0]
importlib.reload(pdlc)
for ix in curr_frames:
    print(ix)
    fig, ax = pl.subplots(figsize=(4,4))
    pdlc.plot_skeleton_on_ax(int(ix), df0, cap, cfg, ax=ax, markersize=10,
                                pcutoff=0.01, animal_colors={'fly': 'm', 'single': 'c'})
    pl.savefig(os.path.join(tmpdir, 'frame_{}.png'.format(ix)))
    pl.close()

#%%


import cv2
import os

image_folder = tmpdir #'images'
video_name = os.path.join(os.path.split(tmpdir)[0], 'video.avi')

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")],
                key=util.natsort)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#fourcc = cv2.VideoWriter_fourcc(*'H264')
video_outrate=30
video = cv2.VideoWriter(video_name, fourcc, video_outrate, (width,height))

for image in images:
    print(image)
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()

cv2.destroyAllWindows()



#%% DEBUG
start_frame = 7514
end_frame = 7634
fig = pl.figure()
ax = fig.add_subplot(121) #, ax =pl.subplots()
sns.scatterplot(data=f1.loc[start_frame:5:end_frame], ax=ax,
                x='targ_rel_pos_x',
                y='targ_rel_pos_y', hue='frame', palette='viridis', legend=0)
ax.plot(0, 0, 'k*')
ax.set_aspect(1)

ax = fig.add_subplot(122, projection='polar')
sns.scatterplot(data=f1.loc[start_frame:end_frame], ax=ax,
                x='targ_pos_theta',
                y='targ_pos_radius', hue='targ_rel_vel', palette='viridis')
ax.plot(0, 0, 'k*')


curr_frames = np.arange(start_frame, end_frame+1)
fig, ax = pl.subplots(1, 1) #len(curr_frames), figsize=(12,5))
#ax=axn[0]
for i, ix in enumerate(curr_frames):
    # plot DLC skeleton
    #ax=axn[i]
    pdlc.plot_skeleton_on_ax(ix, df0, cap, cfg, ax=ax,
                            pcutoff=0.01, animal_colors={'fly': 'm', 'single': 'c'})
   


#%% CHECK TRANSFORMATIONS
ix = 1002 #4000

# plot DLC skeleton
fig = pdlc.plot_skeleton_on_image([ix, ix+1], df0, cap, cfg, 
                            pcutoff=0.01, animal_colors={'fly': 'm', 'single': 'c'})

#% plot centered and rotated with orientation vectors
fly1 = df_[df_['id']==0].copy().reset_index(drop=True)
fly2 = df_[df_['id']==1].copy().reset_index(drop=True)


fig = rem.plot_frame_check_affines(ix, fly1, fly2, cap, frame_width, frame_height)

#% double-check rotation matrix
importlib.reload(rem)
fig = rem.check_rotation_transform(ix, df_, cap) # frame_width, frame_height)
         
# %%
