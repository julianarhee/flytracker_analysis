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

# ---------------------------------------------------------------------
# general
# ---------------------------------------------------------------------

def butter_lowpass(data, cutoff, fs, order):
    # implements a butterworth lowpass filter to a signal (data) sampled at fs, with a set cutoff. 

    nyq = fs / 2
    adjustedCutoff = cutoff / nyq
    b, a = signal.butter(order, adjustedCutoff, btype='low', analog=False)
    filteredSignal = signal.filtfilt(b, a, data)

    return filteredSignal


def get_continuous_numbers(nums):
    # returns a list of continous numbers found in an array

    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))




# ---------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------

def get_bodypart_distance(dataframe, partname1, partname2):
    # retrieves the pixel distance between two bodyparts (from Tom)

    bpt1 = dataframe.xs(partname1, level='bodyparts', axis=1).to_numpy()
    bpt2 = dataframe.xs(partname2, level='bodyparts', axis=1).to_numpy()

    bptDistance = np.sqrt(np.sum(np.square(bpt1[:, [0, 1]] - bpt2[:, [0, 1]]), axis=1))
    return bptDistance


def get_animal_centroid(dataframe):
    # From Tom/Rufei
    scorer = dataframe.columns.get_level_values(0).unique()[0]
    bptNames = dataframe.columns.get_level_values(1).unique()
    aniXpos, aniYpos = np.zeros((nFrames, len(bptNames[0:8]))), np.zeros((nFrames, len(bptNames[0:8])))

    for i, nm in enumerate(bptNames[0:8]):
        aniXpos[:, i] = dataframe[scorer][nm]['x']
        aniYpos[:, i] = dataframe[scorer][nm]['y']

    xCenter, yCenter = np.nanmean(aniXpos, axis=1), np.nanmean(aniYpos, axis=1)
    centroid = np.column_stack((xCenter, yCenter))

    return centroid


def get_bodypart_angle(dataframe, partName1, partName2):
    # retrieves the angle between two bodyparts (Tom/Rufei)

    bpt1 = dataframe.xs(partName1, level='bodyparts', axis=1).to_numpy()
    bpt2 = dataframe.xs(partName2, level='bodyparts', axis=1).to_numpy()

    angle = np.arctan2(bpt2[:, 1] - bpt1[:, 1], bpt2[:, 0] - bpt1[:, 0])
    return angle


def circular_distance(ang1, ang2):
    # efficiently computes the circular distance between two angles (Tom/Rufei)

    circdist = np.angle(np.exp(1j * ang1) / np.exp(1j * ang2))

    return circdist


def wrap2pi(ang):
    # wraps a set of values to fit between zero and 2Pi (Tom/Rufei)

    positiveValues = (ang > 0)
    wrappedAng = ang % (2 * pi)
    wrappedAng[ang == 0 & positiveValues] = 2 * pi

    return wrappedAng


def get_relative_orientations(ani1, ani2):

    normPos = ani2[['XCenter', 'YCenter']] - ani1[['XCenter', 'YCenter']]
    absoluteAngle = np.arctan2(normPos['YCenter'], normPos['XCenter'])
    fA = circular_distance(absoluteAngle, ani2['heading'])
    aBetween = circular_distance(ani1['heading'],ani2['heading'])

    return fA, aBetween



# ---------------------------------------------------------------------
# quality-control checks
# ---------------------------------------------------------------------
def get_valid_coords(df, pcutoff=0.9):
    n_frames = df.shape[0]
    xyp = df.values.reshape((n_frames, -1, 3))
    xyp[xyp[:, :, 2] < pcutoff] = np.nan
    return xyp[:, :, :2]


def removeJumps(dataframe, maxJumpLength):
    # removes large jumps in the x/y position of bodyparts, usually resulting from swaps between animals

    # get all column names
    scorer = dataframe.columns.get_level_values(0)[0]
    bps = list(dataframe.columns.levels[1])
    params = list(dataframe.columns.levels[2])
    dataframeMod = dataframe.copy()

    for i, partName in enumerate(bps):

        xDiff = pd.Series(np.diff(dataframe[scorer][partName]['x']))
        yDiff = pd.Series(np.diff(dataframe[scorer][partName]['y']))

        xJumpsPositive = signal.find_peaks(xDiff.interpolate(), threshold=200)
        xJumpsNegative = signal.find_peaks(xDiff.interpolate() * -1, threshold=200)
        yJumpsPositive = signal.find_peaks(yDiff.interpolate(), threshold=200)
        yJumpsNegative = signal.find_peaks(yDiff.interpolate() * -1, threshold=200)

        toKill = np.zeros((len(yDiff),), dtype=bool)

        for j in range(len(xJumpsPositive[0])):
            if np.any((xJumpsNegative[0] > xJumpsPositive[0][j]) & (
                    xJumpsNegative[0] < xJumpsPositive[0][j] + maxJumpLength)):
                endIdx = np.where((xJumpsNegative[0] > xJumpsPositive[0][j]) & (
                        xJumpsNegative[0] < xJumpsPositive[0][j] + maxJumpLength))
                toKill[xJumpsPositive[0][j]:xJumpsNegative[0][endIdx[0][0]]] = True
            else:
                toKill[xJumpsPositive[0][j]] = True

        for j in range(len(xJumpsNegative[0])):

            if np.any((xJumpsPositive[0] > xJumpsNegative[0][j]) & (
                    xJumpsPositive[0] < xJumpsNegative[0][j] + maxJumpLength)):
                endIdx = np.where((xJumpsPositive[0] > xJumpsNegative[0][j]) & (
                        xJumpsPositive[0] < xJumpsNegative[0][j] + maxJumpLength))
                toKill[xJumpsNegative[0][j]:xJumpsPositive[0][endIdx[0][0]]] = True
            else:
                toKill[xJumpsNegative[0][j]] = True

        for j in range(len(yJumpsPositive[0])):
            if np.any((yJumpsNegative[0] > yJumpsPositive[0][j]) & (
                    yJumpsNegative[0] < yJumpsPositive[0][j] + maxJumpLength)):
                endIdx = np.where((yJumpsNegative[0] > yJumpsPositive[0][j]) & (
                        yJumpsNegative[0] < yJumpsPositive[0][j] + maxJumpLength))
                toKill[yJumpsPositive[0][j]:yJumpsNegative[0][endIdx[0][0]]] = True
            else:
                toKill[yJumpsPositive[0][j]] = True

        for j in range(len(yJumpsNegative[0])):
            if np.any((yJumpsPositive[0] > yJumpsNegative[0][j]) & (
                    yJumpsPositive[0] < yJumpsNegative[0][j] + maxJumpLength)):
                endIdx = np.where((yJumpsPositive[0] > yJumpsNegative[0][j]) & (
                        yJumpsPositive[0] < yJumpsNegative[0][j] + maxJumpLength))
                toKill[yJumpsNegative[0][j]:yJumpsPositive[0][endIdx[0][0]]] = True
            else:
                toKill[yJumpsNegative[0][j]] = True

        toKill = np.insert(toKill, 1, False)

        dataframeMod.loc[toKill, (scorer, partName, params)] = np.nan

    return dataframeMod




# ---------------------------------------------------------------------
# DEEPLCUT functions
# ---------------------------------------------------------------------
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



