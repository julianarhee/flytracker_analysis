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
import scipy.signal as signal

# ---------------------------------------------------------------------
# general
# ---------------------------------------------------------------------

def lpfilter(input_signal, win):
    # Low-pass linear Filter
    # (2*win)+1 is the size of the window that determines the values that influence 
    # the filtered result, centred over the current measurement
    from scipy import ndimage
    kernel = np.lib.pad(np.linspace(1,3,win), (0,win-1), 'reflect') 
    kernel = np.divide(kernel,np.sum(kernel)) # normalise
    output_signal = ndimage.convolve(input_signal, kernel) 
    return output_signal

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


#
def convert_df_units(flydf, mm_per_pix):

    flydf['dist_to_other_mm'] = flydf['dist_to_other']*mm_per_pix
    flydf['lin_speed_mms'] = flydf['lin_speed']*mm_per_pix
    # flydf['rel_vel'] = flydf['dist_to_other']/(5/fps)
    flydf['dist_to_other_mm_diff'] =  flydf['dist_to_other_mm'].diff().fillna(0) # if dist incr, will be pos, if distance decr, will be neg
    flydf['time_diff'] =  flydf['time'].diff().fillna(0) # if dist incr, will be pos, if distance decr, will be neg
    flydf['rel_vel_mms'] = flydf['dist_to_other_mm_diff'].abs() / (5*flydf['time_diff']) #(5/fps) # if neg, objects are getting 

    # flydf['centroid_x_mm'] = flydf['centroid_x']*mm_per_pix
    # flydf['centroid_y_mm'] = flydf['centroid_y']*mm_per_pix
    # fly_ctr = flydf[['centroid_x_mm', 'centroid_y_mm']].values
    # cop_ix=len(flydf)
    # flydf['lin_speed_mms'] = np.concatenate((np.zeros(1),
    #                     np.sqrt(np.sum(np.square(np.diff(fly_ctr[:cop_ix, ], axis=0)),
    #                     axis=1)))) / (5/60)
    #
    # dotdf['centroid_x_mm'] = dotdf['centroid_x']*mm_per_pix
    # dotdf['centroid_y_mm'] = dotdf['centroid_y']*mm_per_pix
    # dot_ctr = dotdf[['centroid_x_mm', 'centroid_y_mm']].values
    # IFD_mm = np.sqrt(np.sum(np.square(dot_ctr - fly_ctr),axis=1))
    # flydf['dist_to_other_mm'] = IFD_mm[:cop_ix]

    return flydf
 
def get_fly_params(flypos, cop_ix=None, win=5, fps=60):
    if cop_ix is None:
        cop_ix = len(flypos)
    fly_ctr = get_animal_centroid(flypos)
    # get some more female parameters
    flydf = pd.DataFrame(get_bodypart_angle(flypos.iloc[:cop_ix], 'head', 'thorax'),
                                columns=['heading'])
    flydf['centroid_x'] = fly_ctr[:cop_ix, 0]
    flydf['centroid_y'] = fly_ctr[:cop_ix, 1]
    flydf['lin_speed'] = np.concatenate(
                            (np.zeros(1), 
                            np.sqrt(np.sum(np.square(np.diff(fly_ctr[:cop_ix, ], axis=0)), 
                            axis=1)))) / (win/fps)
    leftw = get_bodypart_angle(flypos, 'thorax', 'wingL')
    rightw = get_bodypart_angle(flypos, 'thorax', 'wingR')
    flydf['left_wing_angle'] = wrap2pi(circular_distance(flydf['heading'].interpolate(), leftw)) - np.pi
    flydf['right_wing_angle'] = wrap2pi(circular_distance(flydf['heading'].interpolate(), rightw)) - np.pi
    flydf['inter_wing_dist'] = get_bodypart_distance(flypos, 'wingR', 'wingL')

    return flydf

def get_dot_params(dotpos, cop_ix=None):
    if cop_ix is None:
        cop_ix = len(dotpos)
    dot_ctr = get_animal_centroid(dotpos)
    dotdf = pd.DataFrame(get_bodypart_angle(dotpos.iloc[:cop_ix], 'center', 'top'),
                                columns=['heading'])
    dotdf['centroid_x'] = dot_ctr[:cop_ix, 0]
    dotdf['centroid_y'] = dot_ctr[:cop_ix, 1]
    dotdf['lin_speed'] = np.concatenate(
                            (np.zeros(1), 
                            np.sqrt(np.sum(np.square(np.diff(dot_ctr[:cop_ix, ], axis=0)), 
                            axis=1))))
    leftw_d = get_bodypart_angle(dotpos, 'center', 'left')
    rightw_d = get_bodypart_angle(dotpos, 'center', 'right')
    dotdf['left_wing_angle'] = wrap2pi(circular_distance(dotdf['heading'].interpolate(), leftw_d)) - np.pi
    dotdf['right_wing_angle'] = wrap2pi(circular_distance(dotdf['heading'].interpolate(), rightw_d)) - np.pi
    dotdf['inter_wing_dist'] = get_bodypart_distance(dotpos, 'left', 'right')

    return dotdf


def get_interfly_params(flydf, dotdf, cop_ix=None):
    if cop_ix is None:
        cop_ix = len(flydf)   
    # inter-fly stuff
    # get inter-fly distance based on centroid
    fly_ctr = flydf[['centroid_x', 'centroid_y']].values 
    dot_ctr = dotdf[['centroid_x', 'centroid_y']].values 

    IFD = np.sqrt(np.sum(np.square(dot_ctr - fly_ctr),axis=1))
    flydf['dist_to_other'] = IFD[:cop_ix]
    flydf['facing_angle'], flydf['ang_between'] = get_relative_orientations(
                                                            flydf, dotdf)
    dotdf['dist_to_other'] = IFD[:cop_ix]
    dotdf['facing_angle'], dotdf['ang_between'] = get_relative_orientations(dotdf, flydf)
    return flydf, dotdf


def load_trk_df(fpath, flyid='fly', fps=60, max_jump=6, cop_ix=None):
    
    trk = pd.read_hdf(fpath)
    tstamp = np.linspace(0, len(trk) * 1 / fps, len(trk))
    flypos = trk.xs(flyid, level='individuals', axis=1)
    flypos = remove_jumps(flypos, max_jump)
    
    if cop_ix is None:
        cop_ix = len(flypos)
    if 'fly' in flyid:
        flydf = get_fly_params(flypos, cop_ix=cop_ix)
    else:
        flydf = get_dot_params(flypos, cop_ix=cop_ix)
    flydf['time'] = tstamp
    
    return flydf

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
    nFrames = len(dataframe)
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
    wrappedAng = ang % (2 * np.pi)
    wrappedAng[ang == 0 & positiveValues] = 2 * np.pi

    return wrappedAng


def get_relative_orientations(ani1, ani2, xvar='centroid_x', yvar='centroid_y'):

    normPos = ani2[[xvar, yvar]] - ani1[[xvar, yvar]]
    absoluteAngle = np.arctan2(normPos[yvar], normPos[xvar])
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


def remove_jumps(dataframe, maxJumpLength):
    # removes large jumps in the x/y position of bodyparts, usually resulting from swaps between animals

    # get all column names
    scorer = dataframe.columns.get_level_values(0)[0]
    bps = list(dataframe.columns.get_level_values(1).unique()) #list(dataframe.columns.levels[1])
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

# GET EPOCHS:
def split_speed_epochs(dotdf, return_stepdict=True, 
                        win=13, cop_ix=None, speed_var='lin_speed_filt',
                        t_start=20, increment=40, n_levels=10):
    '''
    Parse speed-varying (diffspeeds2.csv) epochs for DLC-extracted tracks.
    Smooths x, y to calculate lin_speed_filt, gets indices of each step in a dict, then adds epochs to df. 
    '''
    if cop_ix is None:
        cop_ix = len(dotdf)

    # smooth pos and speed
    dotdf = smooth_speed_steps(dotdf, win=win, cop_ix=cop_ix)
    # get step dict
    step_dict = get_step_indices(dotdf, speed_var=speed_var, 
                        t_start=t_start, increment=increment, n_levels=n_levels)

    # add epochs
    dotdf = add_speed_epoch(dotdf, step_dict)

    if return_stepdict:
        return dotdf, step_dict
    else:
        return dotdf


def smooth_speed_steps(dotdf, win=13, cop_ix=None):
    if cop_ix is None:
        cop_ix = len(dotdf)
    smoothed_x = lpfilter(dotdf['centroid_x'], win)
    smoothed_y = lpfilter(dotdf['centroid_y'], win)

    dot_ctr_sm = np.dstack([smoothed_x, smoothed_y]).squeeze()
    # dotdf['lin_speed_filt'] = smoothed_speed
    dotdf['lin_speed_filt'] = np.concatenate(
                            (np.zeros(1), 
                            np.sqrt(np.sum(np.square(np.diff(dot_ctr_sm[:cop_ix, ], axis=0)), 
                            axis=1)))).round(2)
    dotdf['centroid_x_filt'] = smoothed_x
    dotdf['centroid_y_filt'] = smoothed_y

    return dotdf

def get_step_shift_index(dary, find_stepup=True, plot=False):
    '''
    get index of shift in a noisy step function
    '''
    dary -= np.average(dary)
    step = np.hstack((np.ones(len(dary)), -1*np.ones(len(dary))))
    dary_step = np.convolve(dary, step, mode='valid')
    # Get the peaks of the convolution
    # peaks = signal.find_peaks(dary_step, width=1500) #[0]
    if find_stepup:
        step_indx = np.argmax(dary_step)  # 
    else:
        step_indx = np.argmin(dary_step)  # 
    # for plotting:
    if plot:
        pl.figure()
        pl.plot(dary)
        pl.plot(dary_step/100)
        pl.plot((step_indx, step_indx), (dary_step[step_indx]/100, 0), 'r')
    return step_indx

# get chunks

def get_step_indices(dotdf, speed_var='lin_speed_filt', t_start=20, 
                     increment=40, n_levels=10):
    '''
    Fix DLC tracked dot trajectories with diffspeeds2.csv
    Smooths dot positions, finds indices of steps in velocity. Use these indices to divide trajectories into epochs.
    '''
    # speed_var = 'lin_speed_filt'
    # t_start = 20
    # n_epochs = 9
    if speed_var not in dotdf.columns:
        dotdf = smooth_speed_steps(dotdf)

    tmpdf = dotdf.copy() #loc[motion_start_ix:].copy()
    step_dict={}
    for i in range(n_levels):
        t_stop = t_start + increment
        curr_chunk = tmpdf[ (tmpdf['time']>=t_start) & (tmpdf['time']<=t_stop)].copy().interpolate()
        #if i==(n_levels-1):
        find_stepup = i < (n_levels-1)
        # check in case speed does not actually drop at end:
        if i==(n_levels-1) and tmpdf.iloc[-20:][speed_var].mean()<5:
            find_stepup = False
        else:
            find_stepup = True
        tmp_step_ix = get_step_shift_index(np.array(curr_chunk[speed_var].values),
                                          find_stepup=find_stepup)
        step_ix = curr_chunk.iloc[tmp_step_ix].name
        step_dict.update({i: step_ix})
        t_start = t_stop 
    return step_dict

def add_speed_epoch(dotdf, step_dict):
    '''
    Use step indices found with get_step_indices() to split speed-varying trajectory df into epochs
    '''
    last_ix = step_dict[0]
    dotdf.loc[:last_ix, 'epoch'] = 0
    step_dict_values = list(step_dict.values())
    for i, v in enumerate(step_dict_values):
        if v == step_dict_values[-1]:
            dotdf.loc[last_ix:, 'epoch'] = i+1
            #flydf.loc[last_ix:, 'epoch'] = i+1
        else:
            next_ix = step_dict_values[i+1]
            dotdf.loc[last_ix:next_ix, 'epoch'] = i+1
            #flydf.loc[last_ix:next_ix, 'epoch'] = i+1
        last_ix = next_ix
    return dotdf

def check_speed_steps(dotdf, step_dict):
    '''
    Check speed steps and found indices
    '''
    fig, ax = pl.subplots()
    ax.scatter(dotdf['time'], dotdf['lin_speed_filt'], s=2)
    for i, v in step_dict.items():
        dotdf.loc[v]['time']
        ax.plot(dotdf.loc[v]['time'], dotdf.loc[v]['lin_speed_filt'], 'r*')
    return fig

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


## filter

def get_filtered_pos(df, bp, pcutoff=0.9, return_df=True):
    '''
    df (pd.DataFrame): multi-level df loaded form .h5 file
    bp (str): name of bodypart to get coords for
    '''
    prob = df.xs(
        (bp, "likelihood"), level=(-2, -1), axis=1).values.squeeze()
    mask = prob < pcutoff
    x_filt = np.ma.array(
        df.xs((bp, "x"), level=(-2, -1), axis=1).values.squeeze(),
        mask=mask)
    y_filt = np.ma.array(
        df.xs((bp, "y"), level=(-2, -1), axis=1).values.squeeze(),
        mask=mask)
    if return_df:
        df_ = pd.DataFrame({'x': x_filt, 'y': y_filt})
        return df_
    else:
        return x_filt, y_filt

## Cowley et al.

def dlc_to_multipos_array(df, bps=['head', 'thorax', 'abdomentip'], pcutoff=0.95):
    '''
    Takes multi-index DLC df of 1 fly's coords, and converts to (2,3,T)
    for x/y, head/body/tail, timepoints.
    '''
    xy_list=[]
    df_ = df[bps].copy()
    for bp in bps:
        prob = df_.xs(
            (bp, "likelihood"), level=(-2, -1), axis=1
        ).values.squeeze()
        mask = prob < pcutoff
        temp_x = np.ma.array(
            df_.xs((bp, "x"), level=(-2, -1), axis=1).values.squeeze(),
            mask=mask)
        temp_y = np.ma.array(
            df_.xs((bp, "y"), level=(-2, -1), axis=1).values.squeeze(),
            mask=mask)
        xy_ = np.array([temp_x, temp_y])
        xy_list.append(xy_)
    xy = np.dstack(xy_list)
    positions_male = np.swapaxes(xy, 2, 1)

    return positions_male
