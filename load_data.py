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
assay = 'singlechoice_15mm_1x_sessions'
session = '20211203'

acquisition_paths = sorted([f for f in glob.glob(os.path.join(rootdir, assay, session, '%s*' % session))
                            if os.path.isdir(f)], key=util.natsort)
print("Found %i acquisitions from %s" % (len(acquisition_paths), session))

#%% Select 1 acquisition

curr_acq = acquisition_paths[0]

#%% Get corresponding calibration file

calib = util.load_calibration(curr_acq)
calib


importlib.reload(util)
# %% Load feature mat
feat = util.load_feat(curr_acq)
trk = util.load_tracks(curr_acq)

   
# %% Identify courtship bouts
feat = util.thresh_courtship_bouts(feat, max_dist=5, max_angle=30)

# Get courtship frame indices
bouts = util.get_true_bouts(feat, calib)

#%% PLOT:  Histogram of courtship durs and inter-bout durs

hist_c = [0.7]*3
plot_cdf=True

fps = calib['FPS']
#bout_edges = [(b[0], b[-1]) for b in bouts]
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

male=0
female=1

curr_bout = bouts[0]
curr_trk = trk.loc[curr_bout].copy()
curr_trk['frame'] = curr_trk.index.tolist()
curr_trk['sec'] = curr_trk['frame']/fps


trk_male = curr_trk[(curr_trk['id']==male)]
trk_female = curr_trk[(curr_trk['id']==female)]

min_thr = 10.
max_thr= 115.

fig, ax = pl.subplots()
sns.lineplot(x='sec', y='wing_r_ang', hue='id', data=curr_trk, ax=ax)



# %%

mat_fpath = acquisition_paths[0]

mat = scipy.io.loadmat(mat_fpath)
mdata = mat.get('trk')
