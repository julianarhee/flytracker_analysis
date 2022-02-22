#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   create_gif.py
@Time    :   2022/02/21 12:48:04
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com
'''

import scipy.io
import os
import re
import glob
import importlib

import pandas as pd
import pylab as pl
import numpy as np
import seaborn as sns

import utils as util

import cv2
import matplotlib.gridspec as gridspec
from matplotlib import animation, rc
from IPython.display import HTML, Image


#%%
def get_acq_dir(sessionid, assay_prefix='single_20mm*', rootdir='/mnt/sda/Videos'):

    acquisition_dirs = sorted([f for f in glob.glob(os.path.join(rootdir, '%s*' % assay_prefix, 
                            '%s*' % sessionid)) if os.path.isdir(f)], key=util.natsort)
    #print("Found %i acquisitions from %s" % (len(acquisition_dirs), sessionid))
    assert len(acquisition_dirs)==1, "Unable to find unique acq. from session ID: %s" % sessionid
    acq_dir = acquisition_dirs[0]
    acquisition = os.path.split(acq_dir)[-1]

    return acq_dir

def add_frame_nums(trackdf, fps=None):
    frame_ixs = trackdf[trackdf['id']==0].index.tolist()
    trackdf['frame'] = None
    for i, g in trackdf.groupby('id'):
        trackdf.loc[g.index, 'frame'] = frame_ixs
    
    # add sec
    if fps is not None:
        trackdf['sec'] = trackdf['frame']/float(fps)
    
    return trackdf

def plot_wing_extensions(trk, start_frame=0, end_frame=None, ax=None, figsize=(20,3),
                         c1='lightblue', c2='steelblue', l1='var1', l2='var2', xaxis='sec'):
    if ax is None:
        fig, ax = pl.subplots(figsize=figsize)
    if end_frame is None:
        end_frame = int(trk.index.tolist()[-1])
    bout_dur_sec = (end_frame-start_frame)/fps
    df_ = trk.loc[start_frame:end_frame]
    ax.plot(df_[xaxis], np.rad2deg(df_['wing_r_ang']), color=c1, label=l1)
    ax.plot(df_[xaxis], np.rad2deg(df_['wing_l_ang']), color=c2, label=l2)

    return ax


#%%

rootdir = '/mnt/sda/Videos'
assay_prefix='single_20mm'
#sessionid = '20220128-1555'
sessionid = '20220202-1415'

acq_dir = get_acq_dir(sessionid, assay_prefix=assay_prefix)
acquisition = os.path.split(acq_dir)[-1]
print('Processing acq: %s' % acquisition)

#%% Load data
# Get corresponding calibration file
calib = util.load_calibration(acq_dir)

#% Load feature mat
featdf = util.load_feat(acq_dir)
trackdf = util.load_tracks(acq_dir)

trackdf = add_frame_nums(trackdf, fps=calib['FPS'])
featdf = add_frame_nums(featdf, fps=calib['FPS'])


#%% output dirs
project_dir = '/home/julianarhee/Documents/projects'
dst_dir = os.path.join(project_dir, 'flytracker-analysis', 'examples')
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
print("Saving output to: %s" % dst_dir )

#%%
fly_id = 0
feat = featdf[featdf['id']==fly_id].copy()
trk = trackdf[trackdf['id']==fly_id].copy()

del trackdf
del featdf

#%% Identify bouts
ppm = calib['PPM']
fps = calib['FPS']

male_id=0
female_id=1

# interaction params:
interaction_dist_mm = 5
max_dist_to_other = interaction_dist_mm #* ppm
max_facing_angle=80
fly_id = 0
ibi_min_sec=5

# -----------------------------------------
feat = util.threshold_courtship_bouts(feat, max_dist_to_other=max_dist_to_other, max_facing_angle=max_facing_angle)

feat, bout_dict = util.get_true_bouts(feat, calib, ibi_min_sec=ibi_min_sec)
feat['angle_between_deg'] = np.rad2deg(feat['angle_between'])

print("Found %i bouts, with min. IBI=%.2f sec" % (len(bout_dict), ibi_min_sec))

bouts = [b for i, b in bout_dict.items()]

#%%

import plotting as plotutils
cdict = {'wing_r': 'mediumpurple',
        'wing_l': 'orchid'}
plotutils.set_plot_params(light=True, default_sizes=False, lw_axes=1, 
                          axis_labelsize=10, tick_labelsize=8) 
#lw_axes=1, axis_labelsize=12, tick_labelsize=10, color=[0.7]*3, dpi=300)

param_list=['wing_l_ang', 'wing_r_ang']
for param in param_list:
    trk[param]=trk[param].interpolate()
    
plot_params=['wing_l_ang', 'wing_r_ang', 'vel', 'leg_dist', 'angle_between_deg']
limits={}
for param in plot_params:
    if param in feat.columns:
        limits.update({param: (feat[param].min(), feat[param].max())})
    else:
        limits.update({param: (trk[param].min(), trk[param].max())})
# limits
limits.update({'leg_dist': (0, 10)})
limits.update({'angle_between_deg': (0, 180)})
limits.update({'vel': (0, 15)})

#%%

xaxis='sec'

reg_color = [0.8]*3

example = 'circling'

if example=='rowing':
    start_frame = 12400
    end_frame=14000
elif example=='circling':
    # circling:
    start_frame=18500 #700
    end_frame=20900 #500
    
    
    
start_str = int(round(start_frame/fps))
end_str = int(round(end_frame/fps))

#fig, axn = pl.subplots(4, 1, figsize=(20,12), sharex=True)
#plotutils.label_figure(fig, acquisition)
#
#ax=axn[0]
#ax = plot_wing_extensions(trk, start_frame=start_frame, end_frame=end_frame, ax=ax, xaxis=xaxis, #figsize=(20,3),
#                         c1=cdict['wing_l'], l1='wing_l', c2=cdict['wing_r'], l2='wing_r')
#ax.set_ylim([-100, 100])
#ax.set_ylabel('wing angle')
#
#plot_params=['vel', 'leg_dist', 'angle_between_deg']
#labels = ['vel', 'leg dist', 'angle between']
#
#for ax, param, lab in zip(axn.flat[1:], plot_params, labels):
#
#    ax.plot(feat.loc[start_frame:end_frame][xaxis], feat.loc[start_frame:end_frame][param], color=reg_color)
#    ax.set_ylabel(lab)
#    ax.set_ylim(limits[param])
#    ax.get_yaxis().set_label_coords(-0.07,0.5)
#
#
#for ax in axn.flat:
#    ax.patch.set_alpha(0)
#
#axn[-1].set_xlabel(xaxis)
#    
#sns.despine(trim=True, offset={'left': -5, 'bottom': 10})
#
#figname = '%s-%i-%isec_%s' % (example, start_str, end_str, acquisition)
#pl.savefig(os.path.join(dst_dir, '%s.svg' % figname))
#print(dst_dir, figname)

##%%



fmt='avi'
movie_path = os.path.join(acq_dir, '%s.%s' % (acquisition, fmt))
assert os.path.exists(movie_path), "Mov does not exist: %s" % movie_path

#%% loa frames
print(start_frame, end_frame)
cap = cv2.VideoCapture(movie_path)
#movie=[]
#for fi, fr in enumerate(range(start_frame, end_frame)):
#    if fi % 100 == 0:
#        print("... %i of %i frames" % (int(fi+1), (end_frame-start_frame)))
#    cap.set(1, fr)
#    ret, frame = cap.read()
#    movie.append(frame[:, :, 0])
#    
    
cap.set(1, start_frame)
ret, frame = cap.read()

# Data points to draw
wingL = np.rad2deg(trk.loc[start_frame:end_frame]['wing_l_ang'].values)
wingR = np.rad2deg(trk.loc[start_frame:end_frame]['wing_r_ang'].values)
wingR = util.check_nan(wingR)
wingL = util.check_nan(wingL)

vel = util.check_nan(feat.loc[start_frame:end_frame]['vel'].values)
leg_dist = util.check_nan(feat.loc[start_frame:end_frame]['leg_dist'].values)
angle_between = util.check_nan(feat.loc[start_frame:end_frame]['angle_between_deg'].values)

tsec = trk.loc[start_frame:end_frame]['sec'].values - float(trk.loc[start_frame]['sec'])

#%% plot test

#fig = pl.figure(figsize=(17,10), constrained_layout=True)
#
#spec = gridspec.GridSpec(ncols=8, nrows=5, figure=fig)
#ax0 = fig.add_subplot(spec[0:3, 0:3]) # video
#ax1 = fig.add_subplot(spec[0, 3:])  # wing extensions
#ax2 = fig.add_subplot(spec[1, 3:], sharex=ax1)  # vel
#ax3 = fig.add_subplot(spec[2, 3:], sharex=ax1)  # dist
#ax4 = fig.add_subplot(spec[3, 3:], sharex=ax1)  # angle between
#
#ax0.axis('off')
#im = ax0.imshow(movie[0], aspect='equal', cmap='gray')
#
#ax1.set_ylim([-100, 100])
#ax1.set_ylabel('wing angle')
#p1, = ax1.plot(tsec, wingL, color=cdict['wing_l'], lw=1)
#p2, = ax1.plot(tsec, wingR, color=cdict['wing_r'], lw=1)
#
#ax2.set_ylabel('vel')
#p3, = ax2.plot(tsec, vel, color=reg_color)
#
#ax3.set_ylabel('leg dist')
#p4, = ax3.plot(tsec, leg_dist, color=reg_color)
#
#ax4.set_ylabel('angle between')
#p5, = ax4.plot(tsec, angle_between, color=reg_color)
#
#for ax in fig.axes:
#    ax.patch.set_alpha(0)
#    ax.get_yaxis().set_label_coords(-0.07,0.5)
#
#sns.despine(trim=True, offset={'left': -10, 'bottom': 10})
#
#
#figname = '%s-%i-%isec_video_%s' % (example, start_str, end_str, acquisition)
#pl.savefig(os.path.join(dst_dir, '%s.svg' % figname))
#print(dst_dir, figname)
#
#%% GIF

outf = os.path.join(dst_dir, '%s-bout_%s.gif' % (example, acquisition))
print("Saving GIF to: %s" % outf)

# Set playblack
frame_ixs = list(range(start_frame, end_frame))
nframes = len(frame_ixs) #start_frame
video_rate=20.
interval = (1./video_rate) * 1E3
nx, ny, _ = frame.shape

#%%

print("Setting up GIF")

fig = pl.figure(figsize=(12,6), constrained_layout=True)

spec = gridspec.GridSpec(ncols=8, nrows=4, figure=fig)
ax0 = fig.add_subplot(spec[0:3, 0:3]) # video
ax1 = fig.add_subplot(spec[0, 3:])  # wing extensions
ax2 = fig.add_subplot(spec[1, 3:], sharex=ax1)  # vel
ax3 = fig.add_subplot(spec[2, 3:], sharex=ax1)  # dist
ax4 = fig.add_subplot(spec[3, 3:], sharex=ax1)  # angle between

ax0.axis('off')
im = ax0.imshow(frame, aspect='equal', cmap='gray')

ax1.set_ylim([-100, 100])
ax1.set_ylabel('wing angle')
p1, = ax1.plot(tsec, wingL, color=cdict['wing_l'], lw=1)
p2, = ax1.plot(tsec, wingR, color=cdict['wing_r'], lw=1)

ax2.set_ylabel('vel')
p3, = ax2.plot(tsec, vel, color=reg_color)

ax3.set_ylabel('leg dist')
p4, = ax3.plot(tsec, leg_dist, color=reg_color)

ax4.set_ylabel('angle between')
p5, = ax4.plot(tsec, angle_between, color=reg_color)

for ax in fig.axes:
    ax.patch.set_alpha(0)
    ax.get_yaxis().set_label_coords(-0.07,0.5)

sns.despine(trim=True, offset={'left': -10, 'bottom': 10})

# fig = pl.figure(figsize=(20,6), dpi=300)
# spec = gridspec.GridSpec(ncols=9, nrows=5, figure=fig)
# ax0 = fig.add_subplot(spec[0:3, 0:3])
# ax1 = fig.add_subplot(spec[0:2, 3:])

# im = ax0.imshow(movie[0], aspect='equal', cmap='gray')
# p1, = ax1.plot(tsec[0], wingL[0], color=cdict['wing_l'], lw=2)
# p2, = ax1.plot(tsec[0], wingR[0], color=cdict['wing_r'], lw=2)
# ax1.set_ylim([-100, 100])
# ax1.set_xlim([0, tsec[-1]])

# def animate_with_movie(nframes, movie, x, p1_vals, p2_vals, video_rate, cdict):
# initialization function: plot the background of each frame
def init():
    im.set_data(np.zeros((nx, ny)))
    p1.set_data([], [])
    p2.set_data([], [])
    p3.set_data([], [])
    p4.set_data([], [])
    p5.set_data([], [])

    return (im, p1, p2, p3, p4, p5,)

# animation function. This is called sequentially
def update(i): #, im, movie, tsec, p1, p2, p1_vals, p2_vals):
#def animate(i):
    p1.set_data(tsec[:i], wingL[:i])
    p2.set_data(tsec[:i], wingR[:i])
    p3.set_data(tsec[:i], vel[:i])
    p4.set_data(tsec[:i], leg_dist[:i])
    p5.set_data(tsec[:i], angle_between[:i])
#     p1.set_data(wingL[:i])
#     p2.set_data( wingR[:i])
#     p3.set_data(vel[:i])
#     p4.set_data(leg_dist[:i])
#     p5.set_data(angle_between[:i])
    cap.set(1, frame_ixs[i])
    ret, frame = cap.read()
    im.set_data(frame[:,:,0])
    #im.set_data(movie[i])
    return (im, p1, p2, p3, p4, p5)

# call the animator. blit=True means only re-draw the parts that have changed.
#fargs = [im, movie, tsec, p1, p2, wingL, wingR]
anim = animation.FuncAnimation(fig, update, nframes, init_func=init, ##fargs=fargs, init_func=init, 
                        interval=interval, blit=True)

# anim = animate_with_movie(nframes, movie, tsec, wingL, wingR, video_rate, cdict)
outf = os.path.join(dst_dir, '%s-bout_%s.mp4' % (example, acquisition))
writervideo = animation.FFMpegWriter(fps=video_rate) 
#writervideo = animation.PillowWriter(fps=video_rate) 
anim.save(outf, writer=writervideo)

print("done.")
