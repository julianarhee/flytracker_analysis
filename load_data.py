#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   load_data.py
@Time    :   2022/01/28 13:16:26
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com
'''
# %%
import scipy.io
import os
import re
import glob
import importlib

import pylab as pl
import numpy as np
import seaborn as sns

import utils as util
# %%
importlib.reload(util)

# %%

rootdir = '/mnt/sda/Videos'
assay_prefix='single_20mm'
#sessionid = '20220128-1555'
sessionid = '20220202-1415'
#%% Select 1 acquisition

def get_acq_dir(sessionid, assay_prefix='single_20mm*', rootdir='/mnt/sda/Videos'):

    acquisition_dirs = sorted([f for f in glob.glob(os.path.join(rootdir, '%s*' % assay_prefix, 
                            '%s*' % sessionid)) if os.path.isdir(f)], key=util.natsort)
    #print("Found %i acquisitions from %s" % (len(acquisition_dirs), sessionid))
    assert len(acquisition_dirs)==1, "Unable to find unique acq. from session ID: %s" % sessionid
    acq_dir = acquisition_dirs[0]
    acquisition = os.path.split(acq_dir)[-1]

    return acq_dir

acq_dir = get_acq_dir(sessionid, assay_prefix=assay_prefix)
acquisition = os.path.split(acq_dir)[-1]
print('Processing acq: %s' % acquisition)

#%% Get corresponding calibration file

calib = util.load_calibration(acq_dir)

#importlib.reload(util)
#% Load feature mat
featdf = util.load_feat(acq_dir)
trackdf = util.load_tracks(acq_dir)

importlib.reload(util)
  
# %% Identify courtship bouts

# Select 1 fly
# -----------------------------------------
# male_id = 0
#female_id=1

max_dist_to_other=10
max_facing_angle=30

fly_id = 0
feat = featdf[featdf['id']==fly_id].copy()
feat = util.threshold_courtship_bouts(feat, max_dist_to_other=max_dist_to_other, max_facing_angle=max_facing_angle)
feat.head()
trk = trackdf[trackdf['id']==fly_id].copy()
trk.loc[feat['facing_angle'].isna()] = np.nan


fps = calib['FPS']
trk['frame'] = trk.index.tolist()
trk['sec'] = trk['frame']/fps


#%%
importlib.reload(util)
# Get courtship frame indices
ibi_min_sec=5
feat, bouts = util.get_true_bouts(feat, calib, ibi_min_sec=ibi_min_sec)

print("Found %i bouts, with min. IBI=%.2f sec" % (len(bouts), ibi_min_sec))

#%% PLOT:  
# Histogram of courtship durs and inter-bout durs
# --------------------------------------------------
hist_c = [0.7]*3
plot_cdf=True

bout_durs = [(b[-1]-b[0]+1)/fps for b in bouts]
ibi_durs = np.array([(bouts[i+1][0] - b[-1])/fps for i, b in enumerate(bouts[0:-1])])

fig, axn = pl.subplots(1, 2, sharey=True)
sns.histplot(bout_durs, ax=axn[0], color=hist_c, cumulative=plot_cdf)
axn[0].set_xlabel('bout dur (s)')
sns.histplot(ibi_durs, ax=axn[1], color=hist_c, cumulative=plot_cdf)
axn[1].set_xlabel('interbout dur (s)')
for ax in axn:
    ax.set_box_aspect(1)

fig, axn = pl.subplots(1, 2)
axn[0].eventplot(feat[feat['courtship']==True].index.tolist())


#%% Wing Extensions?

# NOTE:  20220202-1415: 
# frame 6888 -> 6889, Fly1 wings wrong (jump) - back to ok on frame 7075.
#start_frame = 5921
#end_frame = 7705

#curr_bout = bouts[0]

cdict = {'wing_r': 'lightblue',
        'wing_l': 'steelblue'}

fig, axn = pl.subplots(5,5, figsize=(8,8), sharey=True)

curr_bout= bouts[1]

for ai, curr_bout in enumerate(bouts):
    bout_dur_sec = (curr_bout[-1]-curr_bout[0])/fps
    print("Current bout: %.2fmin" % (bout_dur_sec/60.) )

    #curr_trk = trk.loc[curr_bout].copy()

    start_frame, end_frame = curr_bout[0], curr_bout[-1]
    df_ = trk.loc[start_frame:end_frame]

    # plot
    ax = axn.flatten()[ai]
    #fig,ax = pl.subplots()

    ax.plot(np.rad2deg(df_['wing_r_ang']), color=cdict['wing_r'], label='wing_r')
    ax.plot(np.rad2deg(df_['wing_l_ang']), color=cdict['wing_l'], label='wing_l')
    ax.set_ylim([-100, 100])
    #ax.legend(bbox_to_anchor=(1,1), loc='upper right')

#%%
#end_frame=21564
fig, ax = pl.subplots()
ax.plot(feat.loc[start_frame:end_frame]['dist_to_other'], color='r')
ax.plot(feat.loc[start_frame:end_frame]['facing_angle_deg'], color='k')

#fig.sup_title('rowing example (%s)' % acquisition)

#%%
fig,ax = pl.subplots()
ax.plot(np.rad2deg(trk['wing_r_ang']), color=cdict['wing_r'], label='wing_r')
ax.plot(np.rad2deg(trk['wing_l_ang']), color=cdict['wing_l'], label='wing_l')
ax.set_ylim([-100, 100])

#%%

fig, ax = pl.subplots()
ax.plot(feat['facing_angle'], color='r')
ax.plot(feat['dist_to_other'])

#%%
#pl.figure()
#pl.plot(feat.loc[6888:7100]['facing_angle'].diff())

#pl.plot(feat.loc[6700:7100]['facing_angle'])

#%%

#%% Plot wing ext. as EVENTS

min_wing_angle=10.
max_wing_angle=115.
min_thr = np.deg2rad(min_wing_angle)
max_thr= np.deg2rad(max_wing_angle)


wing_r = df_[(df_['wing_r_ang'].abs()<max_thr) & (df_['wing_r_ang'].abs()>min_thr)].copy()

wing_l = df_[(df_['wing_l_ang'].abs()<max_thr) & (df_['wing_l_ang'].abs()>min_thr)].copy()
print(wing_r.shape, wing_l.shape)

#fig, axn = pl.subplots(1, 2)
#axn[0].eventplot(wing_ext.index.tolist())

wing_r.head()

