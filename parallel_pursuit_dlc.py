#!/usr/bin/env 
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:10:00 2020

@author: julianarhee
@description: import DLC tracks and do calculations
"""
#%%
import os
import glob
import yaml
import re

import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl

import matplotlib as mpl

import plotting as putil
import dlc as dlc
import utils as util
import cv2
import importlib

#from relative_metrics import get_video_cap, do_transformations_on_df
import relative_metrics as rem


plot_style='white'
putil.set_sns_style(style=plot_style)
bg_color='w' if plot_style=='dark' else 'k'

#%% functions

#def get_acq_from_dlc_fpath(fpath):
#    return '_'.join(os.path.split(fpath.split('DLC')[0])[-1].split('_')[0:-1])

#%%
rootdir = '/Users/julianarhee/DeepLabCut'
projectname = 'projector-1dot-jyr-2024-02-18'
project_dir = os.path.join(rootdir, projectname) 
# load config file
cfg_fpath = os.path.join(project_dir, 'config.yaml')
with open(cfg_fpath, "r") as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
# analyzed files directory
minerva_base = '/Volumes/Julie/2d-projector-analysis'
analyzed_dir = os.path.join(minerva_base, 'DeepLabCut', projectname) #'analyzed')

analyzed_files = glob.glob(os.path.join(analyzed_dir, '*_el.h5'))
print("Found {} analyzed files".format(len(analyzed_files)))

# %% Look at 1 data file
acq_prefix = '20240214-1025_f1_*sz10x10'
match_acq = glob.glob(os.path.join(analyzed_dir, '{}*_el.h5'.format(acq_prefix)))
fpath = match_acq[0]

# get fig id 
fig_id = os.path.split(fpath.split('DLC')[0])[-1]
# get acq name
acq = dlc.get_acq_from_dlc_fpath(fpath) #'_'.join(os.path.split(fpath.split('DLC')[0])[-1].split('_')[0:-1])
print(acq)

#%%

importlib.reload(dlc)
# %%
flyid = 'fly' # double check in the plots for abdomen lengths
dotid = 'single'
fps = 60  # Hz
max_jump = 6
pcutoff=0.8

# load _eh.h5
#trk = pd.read_hdf(fpath) #os.path.join(pathname, filename))
#scorer = trk.columns.get_level_values(0)[0]
#tstamp = np.linspace(0, len(trk) * 1 / fps, len(trk))
#nframes = len(trk)

# get dataframes
flydf = dlc.load_trk_df(fpath, flyid='fly', fps=fps, 
                        max_jump=max_jump, cop_ix=None,
                        filter_bad_frames=True, pcutoff=pcutoff)
dotdf = dlc.load_trk_df(fpath, flyid='single', fps=fps, 
                        max_jump=max_jump, cop_ix=None, 
                        filter_bad_frames=False, pcutoff=pcutoff)
# set nans
nan_rows = flydf.isna().any(axis=1)
dotdf.loc[nan_rows] = np.nan

flydf, dotdf = dlc.get_interfly_params(flydf, dotdf, cop_ix=None)

#%% add speed info
dotdf, flydf = dlc.add_speed_epochs(dotdf, flydf, acq)

#%% process for relative metrics
# assign IDs like FlyTracker DFs
flydf['id'] = 0
dotdf['id'] = 1
trk_ = pd.concat([flydf, dotdf], axis=0)

mm_per_pix = 3 / trk_['body_length'].mean()
trk_['vel'] = trk_['lin_speed'] * mm_per_pix

# %% rename columns to get RELATIVE pos info
trk_ = trk_.rename(columns={'centroid_x': 'pos_x',
                           'centroid_y': 'pos_y',
                           'heading': 'ori', # should rename this
                           'body_length': 'major_axis_len',
                           'inter_wing_dist': 'minor_axis_len',
                           'time': 'sec'
                           }) 
# get video info
#minerva_video_base = '/Volumes/Julie/2d-projector'
#match_vids = glob.glob(os.path.join(minerva_video_base, '20*', '{}*.avi'.format(acq)))
#video_fpath = match_vids[0]
video_fpath = glob.glob(os.path.join(project_dir, 'videos', '{}*.avi'.format(acq)))

#acqdir = os.path.join(minerva_base, 'videos', acq)
#vids = util.get_videos(acqdir, vid_type='avi')

#cap = get_video_cap(fpath)
cap = cv2.VideoCapture(video_fpath[0])
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width, frame_height)
#%% do transformations
importlib.reload(rem)
#trk_ = util.center_coordinates(trk_, frame_width, frame_height) 
df = rem.do_transformations_on_df(trk_, frame_width, frame_height) #, fps=fps)
df['ori_deg'] = np.rad2deg(df['ori'])

df['targ_pos_theta'] = -1*df['targ_pos_theta']

#%%
# add polar conversion
#polarcoords = util.cart2pol(df['pos_x'], df['pos_y']) 
#df['pos_radius'] = polarcoords[0]
#df['pos_theta'] = -1*polarcoords[1]
 
def smooth_orientations(y, winsize=5):
    yy = np.concatenate((y, y))
    smoothed = np.convolve(np.array([1] * winsize), yy)[winsize: len(y) + winsize]
    return smoothed #% (2 * np.pi)

def smooth_orientations_pandas(x, winsize=3): 
    # 'unwrap' the angles so there is no wrap around
    x1 = pd.Series(np.rad2deg(np.unwrap(x)))
    # smooth the data with a moving average
    # note: this is pandas 17.1, the api changed for version 18
    x2 = x1.rolling(winsize, min_periods=1).mean() #pd.rolling_mean(x1, window=3)
    # convert back to wrapped data if desired
    x3 = x2 % 360
    return np.deg2rad(x3)

def smooth_and_calculate_velocity_circvar(df, smooth_var='ori', vel_var='ang_vel',
                                  time_var='sec', winsize=3):
    '''
    Smooth circular var and then calculate velocity. Takes care of NaNs.

    Arguments:
        df -- _description_

    Keyword Arguments:
        smooth_var -- _description_ (default: {'ori'})
        vel_var -- _description_ (default: {'ang_vel'})
        time_var -- _description_ (default: {'sec'})
        winsize -- _description_ (default: {3})

    Returns:
        _description_
    '''
    df[vel_var] = np.nan
    df['{}_smoothed'.format(smooth_var)] = np.nan
    for i, df_ in df.groupby('id'): 
        # unwrap for continuous angles, then interpolate NaNs
        nans = df_[df_[smooth_var].isna()].index
        unwrapped = pd.Series(np.unwrap(df_[smooth_var].fillna(method='ffill')),
                            index=df_.index) #.interpolate().values))
        # replace nans 
        unwrapped.loc[nans] = np.nan 
        # interpolate over nans now that the values are unwrapped
        oris = unwrapped.interpolate() 
        # revert back to -pi, pi
        oris = [util.set_angle_range_to_neg_pos_pi(i) for i in oris]
        # smooth with rolling()
        smoothed = smooth_orientations_pandas(oris, winsize=2) #smoothed = smooth_orientations(df_['ori'], winsize=3)
        # unwrap again to take difference between oris
        smoothed_wrap = pd.Series(np.unwrap([util.set_angle_range_to_neg_pos_pi(i) \
                                            for i in smoothed]), index=df_.index)
        # take difference
        smoothed_diff = smoothed_wrap.diff()
        #smoothed_diff = np.concatenate([[0], smoothed_diff])

        # get angular velocity
        ang_vel = smoothed_diff / df_[time_var].diff().mean()
        df.loc[df_.index, vel_var] = ang_vel
        df.loc[df_.index, '{}_smoothed'.format(smooth_var)] = smoothed_wrap
        df.loc[df_.index, '{}_smoothed_range'.format(smooth_var)] = [util.set_angle_range_to_neg_pos_pi(i) for i in smoothed_wrap]

    #df.loc[df[df[smooth_var].isna()].index, :] = np.nan
    bad_ixs = df[df[smooth_var].isna()]['frame'].dropna().index.tolist()
    cols = [c for c in df.columns if c not in ['frame', 'id']]
    df.loc[bad_ixs, cols] = np.nan

    return df

#%%
# angular vel
df = smooth_and_calculate_velocity_circvar(df, smooth_var='ori', 
                                vel_var='ang_vel', time_var='sec')

#%%

flydf = df[df['id']==0].copy()
dotdf = df[df['id']==1].copy()

flydf.index = flydf['frame']
fig, ax = pl.subplots()
ax.plot(flydf['frame'].loc[0:50], flydf['ori'].loc[0:50]) ##, 'ro')
ax.plot(flydf['frame'].loc[0:50], flydf['ori_smoothed'].loc[0:50],'r') # 'ro')

#%%
flydf['ang_vel_deg'] = np.rad2deg(flydf['ang_vel'])
flydf['ang_vel_abs'] = np.abs(flydf['ang_vel'])

# filter
flydf[flydf['ang_vel_abs']>50] = np.nan
flydf['ang_vel_abs'].plot()
#df_[df_['ang_vel_deg']>=100]


#%%
plotdf = flydf[(flydf['frame']>=15277) & (flydf['frame']<=15284)]
#plotdf = df_[(df_['frame']>=15299) & (df_['frame']<=15304)]
fig, axn = pl.subplots(2, 1)
ax=axn[0]
ax.plot(plotdf['ori'])
ax=axn[1]
ax.plot(plotdf['frame'], plotdf['ang_vel_abs'], 'ro')

#%% Calculate relative angular velocity of target

flydf = smooth_and_calculate_velocity_circvar(flydf, smooth_var='targ_pos_theta',
                                              vel_var='targ_ang_vel')
flydf.columns
#%%

flydf['targ_ang_vel_abs'] = np.abs(flydf['targ_ang_vel'])
fig, ax =pl.subplots()
ax.plot(flydf['targ_ang_vel_abs'])

#%%k
#plotdf = flydf[flydf['epoch']>0]
min_pos_theta = np.deg2rad(-160)
max_pos_theta = np.deg2rad(160)
min_dist_to_other = 1

min_ang_vel = 18
passdf = flydf[ (flydf['epoch']>0) 
               & (flydf['epoch']<7) 
               & (flydf['facing_angle']<=0.25)
               & (flydf['targ_pos_theta']>=min_pos_theta) 
               & (flydf['targ_pos_theta']<=max_pos_theta)
               & (flydf['id']==0) & (flydf['ang_vel_abs']>min_ang_vel)]

# get start/stop indices of consecutive rows
consec_bouts = pp.get_indices_of_consecutive_rows(passdf)

#min_bout_len = 0.25
#fps = 60.
#incl_bouts = filter_bouts_by_frame_duration(consec_bouts, min_bout_len, fps)
#print("{} of {} bouts pass min dur {}sec".format(len(incl_bouts), len(consec_bouts), min_bout_len))

turn_start_frames = [c[0] for c in consec_bouts]
print(len(turn_start_frames))

nsec_pre = 0.5

fig, ax =pl.subplots(subplot_kw={'projection': 'polar'})
for ix in turn_start_frames[0::20]:
    start_ = ix - nsec_pre*fps
    stop_ = ix + nsec_pre*fps

    sns.scatterplot(data=plotdf.loc[start_:stop_], ax=ax,
                x='targ_pos_theta', y='targ_pos_radius', 
                hue='epoch', palette='viridis',
                hue_norm=mpl.colors.Normalize(vmin=1, vmax=6))



#%%
fig, ax =pl.subplots(subplot_kw={'projection': 'polar'})

sns.scatterplot(data=plotdf, ax=ax,
                x='targ_pos_theta', y='targ_pos_radius', 
                hue='epoch', palette='viridis')

#%%


import math

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def rotate2(p, angle, origin=(0, 0)): #degrees=0):
    #angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


#%% center x- and y-coordinates
trk_ = util.center_coordinates(trk_, frame_width, frame_height) 

# separate fly1 and fly2
fly1 = trk_[trk_['id']==flyid1].copy().reset_index(drop=True)
fly2 = trk_[trk_['id']==flyid2].copy().reset_index(drop=True)

# translate coordinates so that focal fly is at origin
fly1, fly2 = util.translate_coordinates_to_focal_fly(fly1, fly2)

# rotate coordinates so that fly1 is facing 0 degrees (East)
fly1, fly2 = util.rotate_coordinates_to_focal_fly(fly1, fly2)

polarcoords = util.cart2pol(fly2['rot_x'], fly2['rot_y']) 
fly1['targ_pos_radius'] = polarcoords[0]
fly1['targ_pos_theta'] = polarcoords[1]
fly2['targ_pos_radius'] = polarcoords[0]
fly2['targ_pos_theta'] = polarcoords[1]

fly1['targ_rel_pos_x'] = fly2['rot_x']
fly1['targ_rel_pos_y'] = fly2['rot_y']
fly2['targ_rel_pos_x'] = fly2['rot_x']
fly2['targ_rel_pos_y'] = fly2['rot_y']


#%%a
id_colors = ['r', 'b']
ix = 2262 #5585 # 7764 #2262 #7062 #2262

old = False#False #True

cap.set(1, ix)
ret, im = cap.read()
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

fig = pl.figure(figsize=(12,5)) #pl.subplots(1, 2)
ax = fig.add_subplot(151)
ax.imshow(im, cmap='gray')
ax.invert_yaxis()

for i, d_ in trk_.groupby('id'):
    print('pos:', i, d_[d_['frame']==ix]['pos_x'], d_[d_['frame']==ix]['pos_y'])
    ax.plot(d_.iloc[ix]['pos_x'], d_.iloc[ix]['pos_y'], 
            marker='o', color=id_colors[i], markersize=3)

ax=fig.add_subplot(152) #projection='polar')
for i, d_ in trk_.groupby('id'):
    print('ctr:', i, d_[d_['frame']==ix]['ctr_x'], d_[d_['frame']==ix]['ctr_y'])
    ax.plot(d_.iloc[ix]['ctr_x'], d_.iloc[ix]['ctr_y'], 
            marker='o', color=id_colors[i], markersize=3)
#ax.invert_yaxis()
ax.set_aspect(1)
ax.set_title('center frame')

ax = fig.add_subplot(153)
for i, d_ in enumerate([fly1, fly2]):
    print('trans:', i, d_.loc[ix]['trans_x'], d_.loc[ix]['trans_y'])
    ax.plot(d_.loc[ix]['trans_x'], d_.loc[ix]['trans_y'], 
            marker='o', color=id_colors[i], markersize=3)
#ax.invert_yaxis()
ax.set_aspect(1)
ax.set_title('trans')

ax = fig.add_subplot(154) #, projection='polar')
for i, d_ in enumerate([fly1, fly2]):
    #print('rot:', i, d_.iloc[ix]['rot_x'], d_.iloc[ix]['rot_y'])
    if old:
        ax.plot(d_.loc[ix]['rot_x'], d_.loc[ix]['rot_y'], 
           marker='o', color=id_colors[i], markersize=3)
    else:
        pt = np.array([d_.iloc[ix]['trans_x'], d_.iloc[ix]['trans_y']])
        #ang = rotation_angs[ix]        
        #rx, ry = rotate([0, 0], pt, ang)
        ang = -1*fly1.loc[ix]['ori'] 
        rotmat = np.array([[np.cos(ang), -np.sin(ang)],
                          [np.sin(ang), np.cos(ang)]])
        #rx, ry = (rotmat @ pt.T).T
        rx, ry = rotate2(pt, ang) #[0, 0], pt, ang)
        print('rot:', i, rx, ry)
        ax.plot(rx, ry,marker='o', color=id_colors[i], markersize=3)

ax.set_aspect(1)
ax.set_title('rot')

ax = fig.add_subplot(155, projection='polar')
for i, d_ in enumerate([fly1, fly2]):
    print('polar:', i, d_.iloc[ix]['targ_pos_theta'], d_.iloc[ix]['targ_pos_radius'])
    if i==0:
        ax.plot(0, 0, 'r*')
    if old:
        ax.plot(d_.iloc[ix]['targ_pos_theta'], d_.iloc[ix]['targ_pos_radius'], 
            marker='o', color=id_colors[i], markersize=3)
    else:
        #ang = fly1.iloc[ix]['ori'] #* -1
        pt = [d_.loc[ix]['trans_x'], d_.loc[ix]['trans_y']]
        #ang = rotation_angs[ix]  
        #rx, ry = rotate((0,0), pt, ang)      
        #rx, ry = rotate2(pt, ang) #[0, 0], pt, ang)
        rad, th = util.cart2pol(rx, ry)
        ax.plot(th, rad, marker='o', color=id_colors[i], markersize=3)
ax.set_aspect(1)
ax.set_title('polar')
#ax.set_theta_direction(-1)

fig.suptitle('{}, ang={:.2f}'.format(ix, np.rad2deg(ang)))


#%% OLD WAY
ix =  2262#7062
fig = rem.plot_frame_check_affines(ix, fly1, fly2, cap, frame_width, frame_height)

#%%





#%% check polar coords
ix =  9000 #2500 #1254 #00 #3179
cap.set(1, ix)
ret, im = cap.read()
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)


fig = pl.figure() #pl.subplots(1, 2)
ax = fig.add_subplot(131)
ax.imshow(im, cmap='gray')
id_colors = ['r', 'b']
for i, df_ in df.groupby('id'):
    ax.plot(df_[df_['frame']==ix]['pos_x'], df_[df_['frame']==ix]['pos_y'], 
            marker='*', color=id_colors[i])

ax = fig.add_subplot(132)
for i, df_ in df.groupby('id'):
    ax.plot(df_[df_['frame']==ix]['rot_x'], df_[df_['frame']==ix]['rot_y'], 
            marker='*', color=id_colors[i])
ax.invert_yaxis() 

ax=fig.add_subplot(133, projection='polar')
for i, df_ in df.groupby('id'):
    ax.plot(0, 0, 'r*')
    ax.plot(df_[df_['frame']==ix]['targ_pos_theta'], 
            df_[df_['frame']==ix]['targ_pos_radius'],
            marker='*', color=id_colors[i])
    

#%%
    
fly1 = df[df['id']==0].copy().reset_index(drop=True)
fly2 = df[df['id']==1].copy().reset_index(drop=True)    
# check affine transformations for centering and rotating male
ix = 100 #5000 #2500 #590
fig = rem.plot_frame_check_affines(ix, fly1, fly2, cap, frame_width, frame_height)
#fig.text(0.1, 0.95, os.path.split(acqdir)[-1], fontsize=4)


# %%
import parallel_pursuit as pp

#%%
# manually select frames
start_ix = 7447
stop_ix = 7511
curr_frames = np.arange(start_ix, stop_ix)
curr_frames
currdf = df[(df['frame'].isin(curr_frames))].copy()

#%%
importlib.reload(putil)

fig, axn = pl.subplots(1, 2)
ax=axn[0]
sns.scatterplot(data=currdf, x='pos_x', y='pos_y', hue='frame', ax=ax,
                style='id', palette='viridis', legend=False)
putil.add_colorbar(fig, ax, vmin=currdf['frame'].min(), 
                   vmax=currdf['frame'].max(), cmap='viridis')
ax=axn[1]
hue_min, hue_max = -np.pi, np.pi
sns.scatterplot(data=currdf, x='pos_x', y='pos_y', hue='facing_angle', ax=ax,
                style='id', palette='hsv', hue_norm=mpl.colors.Normalize(vmin=hue_min, vmax=hue_max),
                legend=False)
putil.add_colorbar(fig, ax, vmin=hue_min, vmax=hue_max, cmap='hsv')
for ax in axn:
    ax.set_aspect(1)
    ax.invert_yaxis()

#%%
min_vel = 12 # at lower vel, male may be doing bilateral display
#min_angle_between = 1.0
max_facing_angle = 0.5
passdf = df[(df['vel']>min_vel) 
             #& (df['angle_between']>=min_angle_between)
             & (df['facing_angle']<=max_facing_angle)]

# get start/stop indices of consecutive rows
consec_bouts = pp.get_indices_of_consecutive_rows(passdf)

# %%
