#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : plotting.py
Created        : 2023/03/27 18:57:30
Project        : /Users/julianarhee/Repositories/plume-tracking
Author         : jyr
Email          : juliana.rhee@gmail.com
Last Modified  : 
'''
import numpy as np
import pandas as pd
import scipy.stats as spstats
import matplotlib as mpl
import pylab as pl
import seaborn as sns

import utils as util


from shapely.geometry import Point, MultiPoint
#from shapely.geometry.polygon import Polygon

## generic
def get_valid_coords(df, pcutoff=0.9):
    n_frames = df.shape[0]
    xyp = df.values.reshape((n_frames, -1, 3))
    xyp[xyp[:, :, 2] < pcutoff] = np.nan
    return xyp[:, :, :2]

## DEEPLCUT functions
def get_segment_indices(bodyparts2connect, all_bpts):
    '''
    https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/utils/make_labeled_video.py
    '''
    bpts2connect = []
    for bpt1, bpt2 in bodyparts2connect:
        if bpt1 in all_bpts and bpt2 in all_bpts:
            bpts2connect.extend(
                zip(
                    *(
                        np.flatnonzero(all_bpts == bpt1),
                        np.flatnonzero(all_bpts == bpt2),
                    )
                )
            )
    return bpts2connect

## shapes
def polygon_from_coords(cc):
    '''
    coords is a Nx2 array of coordinates x, y positions
    inds is a list of paired values (get_segment_indices)
    '''
    xv = cc[:, 0] #coords[np.unique(inds), 0]
    yv = cc[:, 1] #coords[np.unique(inds), 1]
    polygon = list(zip(xv, yv))
    polyshape = MultiPoint(polygon).convex_hull
    
    return polyshape


# DEEPLABCUT plotting
def plot_bodyparts(df, frame_ix, bodyparts2plot, ax=None, pcutoff=0.9, 
                    color='r', alpha=1.0, markersize=6):
    if ax is None:
        fig, ax = pl.subplots()
    for bpindex, bp in enumerate(bodyparts2plot):
        prob = df.xs(
            (bp, "likelihood"), level=(-2, -1), axis=1
        ).values.squeeze()
        mask = prob < pcutoff # confident predictions have likelihood > pcutoff
        temp_x = np.ma.array(
            df.xs((bp, "x"), level=(-2, -1), axis=1).values.squeeze(),
            mask=mask)
        temp_y = np.ma.array(
            df.xs((bp, "y"), level=(-2, -1), axis=1).values.squeeze(),
            mask=mask)
        ax.plot(temp_x[frame_ix], temp_y[frame_ix], ".", color=color, alpha=alpha, markersize=markersize)


def plot_skeleton(coords, inds=None, ax=None, color='k', alpha=1, lw=1):
    if ax is None:
        fig, ax = pl.subplots()
    # abdomen lines
    segs = coords[tuple(zip(*tuple(inds))), :].swapaxes(0, 1) if inds else []
    coll = mpl.collections.LineCollection(segs, colors=color, alpha=alpha, lw=lw)
#         # og skeleton
#         segs0 = coords[tuple(zip(*tuple(inds_og))), :].swapaxes(0, 1) if inds_og else []
#         col0 = mpl.collections.LineCollection(segs0, colors=skeleton_color, alpha=alphavalue, lw=skeleton_lw)
    # plot
    ax.add_collection(coll)
#         ax.add_collection(col0)



