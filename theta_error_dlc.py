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
import plot_dlc_frames as pdlc

plot_style='white'
putil.set_sns_style(style=plot_style)
bg_color='w' if plot_style=='dark' else 'k'

#%% functions

#def get_acq_from_dlc_fpath(fpath):
#    return '_'.join(os.path.split(fpath.split('DLC')[0])[-1].split('_')[0:-1])

#%%
rootdir = '/Users/julianarhee/DeepLabCut'
projectname = 'projector-1dot-jyr-2024-02-18'
#%% Look at 1 data file
projectname='projector-1dot-jyr-2024-02-18' 
analyzed_dir = pdlc.get_dlc_analysis_dir(projectname=projectname)
#acq_prefix = '20240214-1025_f1_*sz10x10'
#acq_prefix = '20240216-*fly3*6x6'
acq_prefix = '20240222-*fly7_Dmel*8x8'
# acq_prefix = '20240216-*fly3_Dmel*4x4'

fpath = pdlc.get_fpath_from_acq_prefix(analyzed_dir, acq_prefix)
acq = dlc.get_acq_from_dlc_fpath(fpath) #'_'.join(os.path.split(fpath.split('DLC')[0])[-1].split('_')[0:-1])
project_dir = os.path.join('/Users/julianarhee/DeepLabCut', projectname)

# load dataframe
df0 = pd.read_hdf(fpath)
scorer = df0.columns.get_level_values(0)[0]

# load config file
cfg = pdlc.load_dlc_config(projectname=projectname)

# get fig id 
fig_id = os.path.split(fpath.split('DLC')[0])[-1]

#%% get video info
minerva_base2 = '/Volumes/Giacomo/JAABA_classifiers/projector/changing_dot_size_speed'
found_vids = glob.glob(os.path.join(minerva_base2, '{}*'.format(acq), 'movie.avi'))
assert len(found_vids)>0, "No video found for acq {}".format(acq)
video_fpath = found_vids[0]
#cap = pdlc.get_video_cap_tmp(acq)
cap = cv2.VideoCapture(video_fpath) 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width, frame_height)

#%%
# %% Load dlc into dataframes
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

df = dlc.load_dlc_df(fpath, fly1=flyid, fly2=dotid, fps=fps, 
                           max_jump=max_jump, pcutoff=pcutoff, diff_speeds=True)
flydf = df[df['id']==0].copy()
dotdf = df[df['id']==1].copy()


#%% add speed info
importlib.reload(dlc)

#%% Change var names for relative metrics
trk_ = dlc.convert_dlc_to_flytracker(df) #flydf, dotdf)

# do transformations
df = rem.do_transformations_on_df(trk_, frame_width, frame_height) #, fps=fps)
df['ori_deg'] = np.rad2deg(df['ori'])
#df['targ_pos_theta'] = -1*df['targ_pos_theta']

#%%

import parallel_pursuit as pp

#%%
flydf = df[df['id']==0].copy()

#%%
#srcdir = '/Volumes/Julie/2d-projector-analysis/processed'
srcdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/2d-projector/JAABA'
# Load jaaba-traansformed data
jaaba_fpath = os.path.join(srcdir, 'jaaba_transformed_data.pkl')
assert os.path.exists(jaaba_fpath), "File not found: {}".format(jaaba_fpath)
jaaba = pd.read_pickle(jaaba_fpath)   

#%%
jaa_ = jaaba[jaaba['filename']==acq].copy()
print(jaa_.shape)

chasing_ = jaa_[jaa_['chasing']>10].index.tolist()

flydf['chasing'] = 0
flydf.loc[chasing_, 'chasing'] = 1   
print(flydf.loc[chasing_].shape)

#%% theta error

min_pos_theta = np.deg2rad(-160)
max_pos_theta = np.deg2rad(160)
max_facing_angle = np.deg2rad(45)
min_vel = 5
max_dist_to_other = 20
min_dist_to_other = 3

min_ang_vel = 0.04
max_ang_vel = 24.8

passdf = flydf[ (flydf['epoch']>0) 
               & (flydf['facing_angle']<=max_facing_angle)
               & (flydf['targ_pos_theta']>=min_pos_theta) 
               & (flydf['targ_pos_theta']<=max_pos_theta)
               & (flydf['vel']>=min_vel)
               & (flydf['dist_to_other'] <= max_dist_to_other)
               & (flydf['dist_to_other'] >= min_dist_to_other)
               #& (flydf['ang_vel']<=max_ang_vel)
               #& (flydf['ang_vel']>=min_ang_vel)
               #& (flydf['chasing']==1)
               ]

# get start/stop indices of consecutive rows
consec_bouts = pp.get_indices_of_consecutive_rows(passdf)

# filter duration?
min_bout_len = 0.25
fps = 60.
incl_bouts = pp.filter_bouts_by_frame_duration(consec_bouts, min_bout_len, fps)
print("{} of {} bouts pass min dur {}sec".format(len(incl_bouts), len(consec_bouts), min_bout_len))

# find turn starts
turn_start_frames = [c[0] for c in incl_bouts]
print(len(turn_start_frames))

nsec_pre = 10

#stim_hz_vals = passdf['epoch'].unique()
stimhz_palette = putil.get_palette_dict(passdf, 'epoch', 'viridis')
#stimhz_palette = dict((k, v) for k, v in zip(stim_hz_vals, 
#                        sns.color_palette('viridis', n_colors=len(stim_hz_vals))))
d_list = []
fig, axn = pl.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True,
                       subplot_kw={'projection': 'polar'})
ax=axn[1]
ax.set_title('egocentric (targ. pos.)')
for ix in turn_start_frames: #[0::2]: #0::20]:
    start_ = ix #nsec_pre*fps
    stop_ = ix + nsec_pre*fps
    sns.scatterplot(data=passdf.loc[start_:stop_], ax=ax,
                x='targ_pos_theta', y='targ_pos_radius', s=3,
                hue='epoch', palette=stimhz_palette,
                edgecolor='none', legend=0, alpha=0.7)
    d_list.append(passdf.loc[start_:stop_])
plotted_passdf = pd.concat(d_list)

# plot Center of Mass
importlib.reload(putil)
for stimhz, df_ in plotted_passdf.groupby('epoch'):
    cm_theta = df_['targ_pos_theta'].mean()
    cm_radius = df_['targ_pos_radius'].mean()
    ax.scatter(cm_theta, cm_radius, s=30, c=stimhz_palette[stimhz],
               marker='o', edgecolor='k', lw=0.5,
               label='COM: {:.2f}Hz'.format(stimhz))
putil.add_colorbar(fig, ax, label='Stim Hz', cmap='viridis',
                   pad=0.2, shrink=0.3,
                   vmin=stim_hz_vals.min(), vmax=stim_hz_vals.max())

# plot dot in allocentric
rad, th = util.cart2pol(flydf['ctr_x'].values, flydf['ctr_y'].values)
flydf['pos_radius'] = rad
flydf['pos_theta'] = th
ax = axn[0] #fig.add_subplot(121,projection='polar')
ax.set_title('allocentric (male pos.)')
plotted_allocentric = flydf.loc[plotted_passdf.index].copy()
sns.scatterplot(data=plotted_allocentric, ax=ax,
                x='pos_theta', y='pos_radius', s=3,
                hue='epoch', palette=stimhz_palette,
                edgecolor='none', legend=0, alpha=0.7)
putil.add_colorbar(fig, ax, label='Stim Hz', cmap='viridis',
                   pad=0.2, shrink=0.3,
                   vmin=stim_hz_vals.min(), vmax=stim_hz_vals.max())
for ax in axn:
    ax.tick_params(pad=10)
    ax.set_xlabel('')
    ax.set_ylabel('')

pl.subplots_adjust(left=0.1, right=0.85, wspace=0.5)

putil.label_figure(fig, acq)


#%%
importlib.reload(util)

#%% Calculate relative angular velocity of target
# angular vel
check_smoothing = False 
df = util.smooth_and_calculate_velocity_circvar(df, smooth_var='ori', 
                                vel_var='ang_vel', time_var='sec')
df.loc[(df['ang_vel']>200) | (df['ang_vel']<-200), 'ang_vel'] = np.nan
df.loc[(df['ang_vel_smoothed']>200) | (df['ang_vel_smoothed']<-200), 'ang_vel_smoothed'] = np.nan


df['ang_vel_deg'] = np.rad2deg(df['ang_vel'])
df['ang_vel_abs'] = np.abs(df['ang_vel'])

# separate into fly and dot
flydf = df[df['id']==0].copy()
dotdf = df[df['id']==1].copy()

# check smoothing
if check_smoothing:
    flydf.index = flydf['frame']
    fig, ax = pl.subplots()
    ax.plot(flydf['frame'].loc[0:50], flydf['ori'].loc[0:50]) ##, 'ro')
    ax.plot(flydf['frame'].loc[0:50], flydf['ori_smoothed'].loc[0:50],'r') # 'ro')

#%% filter
#flydf[flydf['ang_vel_abs']>50] = np.nan
fig, ax =pl.subplots()
ax.plot(flydf['ang_vel_abs'])
ax.set_ylabel('ang_vel_abs')
#df_[df_['ang_vel_deg']>=100]

#%%
#bins = np.arange(0, nbins)
nbins=5
flydf['targ_pos_theta_binned'] = pd.cut(flydf['targ_pos_theta'], bins=nbins)
flydf['targ_pos_theta_leftbin'] = [v.left if isinstance(v, pd.Interval) else v for v in flydf['targ_pos_theta_binned']]

flydf['chasing'] = 0
#flydf.loc[jaa_['chasing']>20, 'chasing'] = 1# jaa_['chasing'].values

flydf.loc[ (flydf['epoch']>0) 
               & (flydf['facing_angle']<=max_facing_angle)
               & (flydf['targ_pos_theta']>=min_pos_theta) 
               & (flydf['targ_pos_theta']<=max_pos_theta)
               & (flydf['vel']>=min_vel)
               & (flydf['dist_to_other'] <= max_dist_to_other)
               & (flydf['dist_to_other'] >= min_dist_to_other)
               #& (flydf['ang_vel']<=max_ang_vel)
               #& (flydf['ang_vel']>=min_ang_vel)
               #& (flydf['chasing']==1)
               , 'chasing'] = 1

#%%
plotdf = flydf[(flydf['epoch']>0)
                 #& (transft['stim_hz']<1)
                #(transft['species']==sp)
                 & (flydf['chasing']==1)].copy()

fig, ax = pl.subplots( figsize=(10,5)) #, sharex=True, sharey=True)
#for i, (sp, df_) in enumerate(plotdf.groupby('species')):
#    ax=axn[i]
sns.pointplot(data=plotdf, ax=ax,
                x='targ_pos_theta_leftbin', y='ang_vel_smoothed',
                hue='epoch', palette=stimhz_palette) # , legend=0)
                #edgecolor='none', s=3, legend=0, alpha=0.5)
#ax.set_title(sp, loc='left')
if i==0:
    ax.legend_.remove()
else:
    sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', frameon=False)
ax.axhline(0, color='k', linestyle='--')

#%%
fig, ax =pl.subplots()









#%% check polar coords
ix =  9000 #2500 #1254 #00 #3179
cap.set(1, ix)
ret, im = cap.read()
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

fig = pl.figure() #pl.subplots(1, 2)
ax = fig.add_subplot(131)
ax.imshow(im, cmap='gray')
ax.invert_yaxis()
id_colors = ['r', 'b']
for i, df_ in df.groupby('id'):
    ax.plot(df_[df_['frame']==ix]['pos_x'], df_[df_['frame']==ix]['pos_y'], 
            marker='*', color=id_colors[i])

ax = fig.add_subplot(132)
for i, df_ in df.groupby('id'):
    ax.plot(df_[df_['frame']==ix]['rot_x'], df_[df_['frame']==ix]['rot_y'], 
            marker='*', color=id_colors[i])
ax.set_aspect(1)

ax=fig.add_subplot(133, projection='polar')
for i, df_ in df.groupby('id'):
    ax.plot(0, 0, 'r*')
    ax.plot(df_[df_['frame']==ix]['targ_pos_theta'], 
            df_[df_['frame']==ix]['targ_pos_radius'],
            marker='*', color=id_colors[i])

# plot direction vectors    
fly1 = df[df['id']==0].copy().reset_index(drop=True)
fly2 = df[df['id']==1].copy().reset_index(drop=True)    
# check affine transformations for centering and rotating male
#ix = 100 #5000 #2500 #590
fig = rem.plot_frame_check_affines(ix, fly1, fly2, cap, frame_width, frame_height)
#fig.text(0.1, 0.95, os.path.split(acqdir)[-1], fontsize=4)


# %%
