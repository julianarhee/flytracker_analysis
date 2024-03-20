#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

#%%
import glob
import os
import numpy as np
import pandas as pd
import dlc as dlc
import yaml
import cv2

import matplotlib as mpl
import seaborn as sns
import pylab as pl
import plotting as putil
import utils as util
#%%
plot_style='white'
putil.set_sns_style(style=plot_style)
bg_color='w' if plot_style=='dark' else 'k'

#%%
projectname = 'projector-1dot-jyr-2024-02-18'
# analyzed files directory
minerva_base = '/Volumes/Julie/2d-projector-analysis'
analyzed_dir = os.path.join(minerva_base, 'DeepLabCut', projectname) #'analyzed')

analyzed_files = glob.glob(os.path.join(analyzed_dir, '*_el.h5'))
print("Found {} analyzed files".format(len(analyzed_files)))

# %% Look at 1 data file
acq_prefix = '20240214-1025_f1_*sz10x10'
match_acq = glob.glob(os.path.join(analyzed_dir, '{}*_el.h5'.format(acq_prefix)))
fpath = match_acq[0]
# load dataframe
df0 = pd.read_hdf(fpath)
scorer = df0.columns.get_level_values(0)[0]

# %% # get video info
#minerva_video_base = '/Volumes/Julie/2d-projector'
#match_vids = glob.glob(os.path.join(minerva_video_base, '20*', '{}*.avi'.format(acq)))
#video_fpath = match_vids[0]
acq = dlc.get_acq_from_dlc_fpath(fpath) #'_'.join(os.path.split(fpath.split('DLC')[0])[-1].split('_')[0:-1])

rootdir = '/Users/julianarhee/DeepLabCut'
projectname = 'projector-1dot-jyr-2024-02-18'
project_dir = os.path.join(rootdir, projectname) 
video_fpath = glob.glob(os.path.join(project_dir, 'videos', '{}*.avi'.format(acq)))
#acqdir = os.path.join(minerva_base, 'videos', acq)
#vids = util.get_videos(acqdir, vid_type='avi')

#cap = get_video_cap(fpath)
cap = cv2.VideoCapture(video_fpath[0])

# %% Load DLC config
# load config file
cfg_fpath = os.path.join(project_dir, 'config.yaml')
with open(cfg_fpath, "r") as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)


# %% use these to plot if only 2 flies (no dot)
#bodyparts2connect = cfg["skeleton"]

#bodyparts2plot = set(df0.columns.get_level_values("bodyparts"))
#individuals = set(df0.columns.get_level_values("individuals"))

#n_bodyparts = len(np.unique(df0.columns.get_level_values("bodyparts")[::3]))
#print(n_bodyparts)
#all_bpts = df0.columns.get_level_values("bodyparts")[::3][0:n_bodyparts]

#%%
# colors = visualization.get_cmap(len(bodyparts2plot), name=cfg["colormap"])
pcutoff=0.01
animal_colors={'fly': 'm', 'single': 'c'}
alphavalue=1

skeleton_color=bg_color

#%%

skeleton_color0 = [0.7]*3
markersize=3
skeleton_lw=0.5
lw=1

#ixs2plot = [3042, 3043, 3044, 3045]
#ixs2plot = [7484, 7485, 7486, 7487]
#ixs2plot = [7960, 7961, 7962, 7963]
#ixs2plot = [10670, 10671, 10672, 10673]
ixs2plot= np.arange(2499, 2504) #[2500]
ixs2plot = [2262, 2263]
ixs2plot = [7062, 7063, 7064]
#ixs2plot = [5585, 5586]
#ixs2plot = [7763, 7764, 7765]
#ixs2plot = [10164, 10165]
ixs2plot = np.arange(8215, 8217)

ixs2plot = [7000, 7040, 7060]
fig, axn = pl.subplots(1, len(ixs2plot))

#ix=3043
for ai, ix in enumerate(ixs2plot):
    ax=axn[ai]
    cap.set(1, ix)
    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

    ax.imshow(im, cmap='gray')
    ax.set_title('Frame {}'.format(ix), loc='left')
    #bodyparts2plot = set(df0.columns.get_level_values("bodyparts"))
    #bodyparts1 = df0[scorer]['fly'].columns.get_level_values(0).tolist()
    #bodyparts2 = df0[scorer]['fly'].columns.get_level_values(0).tolist()

    individuals = set(df0.columns.get_level_values("individuals"))
    bodyparts = dict((ind, np.unique(df0[scorer][ind].columns.get_level_values(0)).tolist()) \
                    for ind in individuals)
    for ind2plot in individuals:
        curr_col = animal_colors[ind2plot]
        df = df0.loc(axis=1)[:, ind2plot]
        bodyparts2plot = bodyparts[ind2plot]

        # original indices specified in cfg file:
        bodyparts2connect = [v for v in cfg['skeleton'] if v[0] in bodyparts2plot]
        inds  = dlc.get_segment_indices(bodyparts2connect, all_bpts)
        skeleton_edges=bodyparts2connect

        n_bodyparts = len(np.unique(df.columns.get_level_values("bodyparts")[::3]))
        print(n_bodyparts)
        all_bpts = df.columns.get_level_values("bodyparts")[::3][0:n_bodyparts]
        curr_colors = sns.color_palette("husl", n_bodyparts)

        for bpindex, bp in enumerate(bodyparts2plot):
            curr_col = curr_colors[bpindex]
            prob = df.xs(
                (bp, "likelihood"), level=(-2, -1), axis=1
            ).values.squeeze()
            mask = prob < pcutoff
            temp_x = np.ma.array(
                df.xs((bp, "x"), level=(-2, -1), axis=1).values.squeeze(),
                mask=mask,
            )
            temp_y = np.ma.array(
                df.xs((bp, "y"), level=(-2, -1), axis=1).values.squeeze(),
                mask=mask,
            )
            ax.plot(temp_x[ix], temp_y[ix], ".", color=curr_col, alpha=alphavalue, markersize=markersize)

        nx = int(np.nanmax(df.xs("x", axis=1, level="coords")))
        ny = int(np.nanmax(df.xs("y", axis=1, level="coords")))

        n_frames = df.shape[0]
        xyp = df.values.reshape((n_frames, -1, 3))
        coords = xyp[ix, :, :2]
        coords[xyp[ix, :, 2] < pcutoff] = np.nan
        segs = coords[tuple(zip(*tuple(inds))), :].swapaxes(0, 1) if inds else []
        coll = mpl.collections.LineCollection(segs, colors=skeleton_color, alpha=alphavalue, lw=lw)
        # plot
        ax.add_collection(coll)
        
        segs0 = coords[tuple(zip(*tuple(inds))), :].swapaxes(0, 1) if inds else []
        col0 = mpl.collections.LineCollection(segs0, colors=skeleton_color0, alpha=alphavalue, lw=skeleton_lw)
        ax.add_collection(col0)
        # axes
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_aspect(1)
    # %%
