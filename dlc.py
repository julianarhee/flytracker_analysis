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
import os
import numpy as np
import pandas as pd
import scipy.stats as spstats
import matplotlib as mpl
import pylab as pl
import seaborn as sns

import utils as util
import relative_metrics as rem
import theta_error as the
import cv2

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

# filepaht stuff
def get_acq_from_dlc_fpath(fpath):
    return '_'.join(os.path.split(fpath.split('DLC')[0])[-1].split('_')[0:-1])

#
def convert_df_units(flydf, mm_per_pix, win=1):

    flydf['dist_to_other_mm'] = flydf['dist_to_other']*mm_per_pix
    flydf['lin_speed_mms'] = flydf['lin_speed']*mm_per_pix
    # flydf['rel_vel'] = flydf['dist_to_other']/(5/fps)
    flydf['dist_to_other_mm_diff'] =  flydf['dist_to_other_mm'].interpolate().diff()# if dist incr, will be pos, if distance decr, will be neg
    flydf['time_diff'] =  flydf['time'].interpolate().diff() # if dist incr, will be pos, if distance decr, will be neg

    #flydf['dist_to_other_mm_diff'] =  flydf['dist_to_other_mm'].diff().fillna(0) # if dist incr, will be pos, if distance decr, will be neg
    #flydf['time_diff'] =  flydf['time'].diff().fillna(0) # if dist incr, will be pos, if distance decr, will be neg
    flydf['rel_vel_mms'] = flydf['dist_to_other_mm_diff'].abs() / (win*flydf['time_diff']) #(5/fps) # if neg, objects are getting 

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
    '''
    Convert tracked DLC coords to flytracker params.
    TODO: change 'heading' to 'ori'

    Arguments:
        flypos -- _description
_

    Keyword Arguments:
        cop_ix -- _description_ (default: {None})
        win -- _description_ (default: {5})
        fps -- _description_ (default: {60})

    Returns:
        _description_
    '''
    if cop_ix is None:
        cop_ix = len(flypos)
    flypos = flypos.iloc[:cop_ix]

    fly_ctr = get_animal_centroid(flypos)
    # get some more female parameters
    flydf = pd.DataFrame(get_bodypart_angle(flypos, 'abdomentip', 'head'),
                                columns=['ori'])
    flydf['centroid_x'] = fly_ctr[:cop_ix, 0]
    flydf['centroid_y'] = fly_ctr[:cop_ix, 1]
    flydf['lin_speed'] = np.concatenate(
                            (np.zeros(1), 
                            np.sqrt(np.sum(np.square(np.diff(fly_ctr[:cop_ix, ], axis=0)), 
                            axis=1)))) / (win/fps)
    leftw = get_bodypart_angle(flypos, 'thorax', 'wingL')
    rightw = get_bodypart_angle(flypos, 'thorax', 'wingR')
    flydf['left_wing_angle'] = wrap2pi(circular_distance(flydf['ori'].interpolate(), leftw)) - np.pi
    flydf['right_wing_angle'] = wrap2pi(circular_distance(flydf['ori'].interpolate(), rightw)) - np.pi
    flydf['inter_wing_dist'] = get_bodypart_distance(flypos, 'wingR', 'wingL')
    flydf['body_length'] = get_bodypart_distance(flypos, 'head', 'abdomentip')
    flydf['frame'] = np.arange(len(flydf))

    return flydf

def get_dot_params(dotpos, cop_ix=None):
    if cop_ix is None:
        cop_ix = len(dotpos)
    dotpos = dotpos.iloc[:cop_ix]
    dot_ctr = get_animal_centroid(dotpos)
    dotdf = pd.DataFrame(get_bodypart_angle(dotpos.iloc[:cop_ix], 'center', 'top'),
                                columns=['ori'])
    dotdf['centroid_x'] = dot_ctr[:cop_ix, 0]
    dotdf['centroid_y'] = dot_ctr[:cop_ix, 1]
    dotdf['lin_speed'] = np.concatenate(
                            (np.zeros(1), 
                            np.sqrt(np.sum(np.square(np.diff(dot_ctr[:cop_ix, ], axis=0)), 
                            axis=1))))
    leftw_d = get_bodypart_angle(dotpos, 'center', 'left')
    rightw_d = get_bodypart_angle(dotpos, 'center', 'right')
    dotdf['left_wing_angle'] = wrap2pi(circular_distance(dotdf['ori'].interpolate(), leftw_d)) - np.pi
    dotdf['right_wing_angle'] = wrap2pi(circular_distance(dotdf['ori'].interpolate(), rightw_d)) - np.pi
    dotdf['inter_wing_dist'] = get_bodypart_distance(dotpos, 'left', 'right')
    dotdf['body_length'] = get_bodypart_distance(dotpos, 'top', 'bottom')
    dotdf['frame'] = np.arange(len(dotdf))

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
                                                            flydf, dotdf,
                                                            ori_var='ori')
    dotdf['dist_to_other'] = IFD[:cop_ix]
    dotdf['facing_angle'], dotdf['ang_between'] = get_relative_orientations(dotdf, flydf,
                                                                            ori_var='ori')
    return flydf, dotdf


def load_trk_df(fpath, flyid='fly', fps=60, max_jump=6, 
                cop_ix=None, filter_bad_frames=True, pcutoff=0.9):
    
    trk = pd.read_hdf(fpath)


    #bad_ixs = df[df[df.columns[df.columns.get_level_values(3)=='likelihood']] < 0.99].any(axis=1)
    #bad_ixs = df[df[df[df.columns[df.columns.get_level_values(3)=='likelihood']] < 0.9].any(axis=1)].index.tolist()            

    tstamp = np.linspace(0, len(trk) * 1 / fps, len(trk))
    flypos = trk.xs(flyid, level='individuals', axis=1)
    flypos = remove_jumps(flypos, max_jump)

    if filter_bad_frames:
        if flyid == 'fly':
            bad_ixs = flypos[ flypos[ flypos[flypos.columns[flypos.columns.get_level_values(2)=='likelihood']] < pcutoff].any(axis=1)].index.tolist()
        else:
            bad_ixs = []
        flypos.loc[bad_ixs, :] = np.nan

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


def get_relative_orientations(ani1, ani2, ori_var='heading', xvar='centroid_x', yvar='centroid_y'):

    '''
    returns facing_angle and angle_between -- facing_angle is relative to ani1

    Returns:
        _description_
    '''
    normPos = ani2[[xvar, yvar]] - ani1[[xvar, yvar]]
    absoluteAngle = np.arctan2(normPos[yvar], normPos[xvar])
    fA = circular_distance(absoluteAngle, ani1[ori_var])
    aBetween = circular_distance(ani1[ori_var], ani2[ori_var])

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

def add_speed_epochs(dotdf, flydf, acq, filter=True):
    dotdf = smooth_speed_steps(dotdf)
    # get epochs
    if acq in '20240214-1045_f1_Dele-wt_5do_sh_prj10_sz12x12_2024-02-14-104540-0000':
        n_levels = 8
    elif acq in '20240215-1722_fly1_Dmel_sP1-ChR_3do_sh_6x6_2024-02-15-172443-0000':
        n_levels = 9
    else:
        n_levels = 10
    step_dict = get_step_indices(dotdf, speed_var='lin_speed_filt', 
                                t_start=20, increment=40, n_levels=n_levels)

    dotdf = add_speed_epoch(dotdf, step_dict)
    flydf = add_speed_epoch(flydf, step_dict)
    dotdf['acquisition'] = acq
    flydf['acquisition'] = acq
    if filter:
        return dotdf[dotdf['epoch'] < 10], flydf[flydf['epoch'] < 10]
    else:
        return dotdf, flydf
#

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
# Data loading and formatting
# ---------------------------------------------------------------------

def load_dlc_df(fpath, fly1='fly', fly2='single', fps=60, max_jump=6, pcutoff=0.8,
                diff_speeds=True):
    '''
    From a DLC .h5 file, load fly and dot dataframes, and calculate interfly params.

    Arguments:
        fpath -- _description_

    Keyword Arguments:
        fly1 -- _description_ (default: {'fly'})
        fly2 -- _description_ (default: {'single'})
        fps -- _description_ (default: {60})
        max_jump -- max nframes jump allowed (default: {6})
        pcutoff -- _description_ (default: {0.8})
        diff_speeds -- diff_speeds2 protocol, where dot increasing speed (default: {True})

    Returns:
        flydf, dotdf
    '''
    # get dataframes
    flydf = load_trk_df(fpath, flyid=fly1, fps=fps, 
                            max_jump=max_jump, cop_ix=None,
                            filter_bad_frames=True, pcutoff=pcutoff)
    dotdf = load_trk_df(fpath, flyid=fly2, fps=fps, 
                            max_jump=max_jump, cop_ix=None, 
                            filter_bad_frames=True, pcutoff=pcutoff)
    print(flydf.shape, dotdf.shape)
    # set nans
    nan_rows = flydf.isna().any(axis=1)
    dotdf.loc[nan_rows] = np.nan

    # add ID vars
    flydf['id'] = 0
    dotdf['id'] = 1
    trk_ = pd.concat([flydf, dotdf], axis=0)

    # Get metrics between the two objects
    flydf, dotdf = get_interfly_params(flydf, dotdf, cop_ix=None)

    # Add speed epoch if this is a diffspeeds2 protocol
    if diff_speeds:
        acq = get_acq_from_dlc_fpath(fpath)
        dotdf, flydf = add_speed_epochs(dotdf, flydf, acq, filter=False)

    # Combine
    df = pd.concat([flydf, dotdf], axis=0)

    # Add some meta data
    acq = get_acq_from_dlc_fpath(fpath) #'_'.join(os.path.split(fpath.split('DLC')[0])[-1].split('_')[0:-1])
    #print(df_['id'].unique())
    df['acquisition'] = acq
    if 'ele' in acq:
        sp = 'ele'
    elif 'yak' in acq:
        sp = 'yak'
    elif 'mel' in acq:
        sp = 'mel'
    df['species'] = sp

    return df #flydf, dotdf


def convert_dlc_to_flytracker(df, mm_per_pix=None):
    # assign IDs like FlyTracker DFs

    # convert units from pix to mm
    #mm_per_pix = 3 / trk_['body_length'].mean()
    df = df.rename(columns={'dist_to_other': 'dist_to_other_pix'})

    if mm_per_pix is None:
        arena_size_mm = 38 - 4 # arena size minus 2 body lengths
        max_dist_found = df['dist_to_other_pix'].max()
        mm_per_pix = arena_size_mm/max_dist_found

    # convert units to mm/s and mm (like FlyTracker)
    df['vel'] = df['lin_speed'] * mm_per_pix
    df['dist_to_other'] = df['dist_to_other_pix'] * mm_per_pix
    df['pos_x_mm'] = df['centroid_x'] * mm_per_pix
    df['pos_y_mm'] = df['centroid_y'] * mm_per_pix

    df['mm_to_pix'] = mm_per_pix

    # % rename columns to get RELATIVE pos info
    df = df.rename(columns={'centroid_x': 'pos_x',
                               'centroid_y': 'pos_y',
                               'heading': 'ori', # should rename this
                               'body_length': 'major_axis_len',
                               'inter_wing_dist': 'minor_axis_len',
                               'time': 'sec'
                               }) 
    return df


def transform_dlc_to_relative(df_, video_fpath=None, winsize=3): #heading_var='ori'):
    '''
    Transform variables measured from keypoints to relative positions and angles.

    Returns:
        _description_
    '''
    #% Get video info
    cap = cv2.VideoCapture(video_fpath)

    # get frame info
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    #print(frame_width, frame_height) # array columns x array rows

    # input is DLC
    df_ = rem.do_transformations_on_df(df_, frame_width, frame_height) #, fps=fps)
    df_['ori_deg'] = np.rad2deg(df_['ori'])
    #df['targ_pos_theta'] = -1*df['targ_pos_theta']

    # convert centered cartesian to polar
    rad, th = util.cart2pol(df_['ctr_x'].values, df_['ctr_y'].values)
    df_['pos_radius'] = rad
    df_['pos_theta'] = th

    # angular velocity
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='ori', 
                                    vel_var='ang_vel', time_var='sec', winsize=winsize)
    # df_.loc[ (df_['ang_vel']>200) | (df_['ang_vel']<-200), 'ang_vel' ] = np.nan
    df_['ang_vel_deg'] = np.rad2deg(df_['ang_vel'])
    df_['ang_vel_abs'] = np.abs(df_['ang_vel'])

    # targ_pos_theta
    df_['targ_pos_theta_abs'] = np.abs(df_['targ_pos_theta'])
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='targ_pos_theta', vel_var='targ_ang_vel',
                                  time_var='sec', winsize=winsize)

    #% smooth x, y, 
    df_['pos_x_smoothed'] = df_.groupby('id')['pos_x'].transform(lambda x: x.rolling(winsize, 1).mean())
    #sign = -1 if input_is_flytracker else 1
    sign=1
    df_['pos_y_smoothed'] = sign * df_.groupby('id')['pos_y'].transform(lambda x: x.rolling(winsize, 1).mean())  

    # calculate heading
    for i, d_ in df_.groupby('id'):
        df_.loc[df_['id']==i, 'traveling_dir'] = np.arctan2(d_['pos_y_smoothed'].diff(), d_['pos_x_smoothed'].diff())
    df_['traveling_dir_deg'] = np.rad2deg(df_['traveling_dir']) #np.rad2deg(np.arctan2(df_['pos_y_smoothed'].diff(), df_['pos_x_smoothed'].diff())) 
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='traveling_dir', vel_var='traveling_dir_dt',
                                    time_var='sec', winsize=3)

    df_['heading_travel_diff'] = np.abs( np.rad2deg(df_['ori']) - np.rad2deg(df_['traveling_dir']) ) % 180  #% 180 #np.pi 

    df_['vel_smoothed'] = df_.groupby('id')['vel'].transform(lambda x: x.rolling(winsize, 1).mean())

    # calculate theta_error
    f1 = df_[df_['id']==0].copy().reset_index(drop=True)
    f2 = df_[df_['id']==1].copy().reset_index(drop=True)
    #f1 = pp.calculate_theta_error(f1, f2, heading_var=heading_var)
    #f2 = pp.calculate_theta_error(f2, f1, heading_var=heading_var)
    f1 = the.calculate_theta_error(f1, f2, xvar='pos_x', yvar='pos_y')
    f2 = the.calculate_theta_error(f1, f2, xvar='pos_x', yvar='pos_y')

    df_.loc[df_['id']==0, 'theta_error'] = f1['theta_error']
    df_.loc[df_['id']==1, 'theta_error'] = f2['theta_error']

    return df_



#%%
import yaml 
import glob

def get_dlc_analysis_dir(projectname = 'projector-1dot-jyr-2024-02-18', 
                             rootdir='/Volumes/Juliana'):
    # analyzed files directory
    minerva_base = os.path.join(rootdir, '2d-projector-analysis', 'circle_diffspeeds')
    analyzed_dir = os.path.join(minerva_base, 'DeepLabCut', projectname) #'analyzed')
    #analyzed_files = glob.glob(os.path.join(analyzed_dir, '*_el.h5'))
    #print("Found {} analyzed files".format(len(analyzed_files)))
    return analyzed_dir

def load_yaml(cfg_fpath):
    '''load yaml file from full path'''
    with open(cfg_fpath, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg

def load_dlc_config(projectname='projector-1dot-jyr-2024-02-18', 
                    rootdir='/Users/julianarhee/DeepLabCut'):
    project_dir = os.path.join(rootdir, projectname)
    cfg_fpath = os.path.join(project_dir, 'config.yaml')
    with open(cfg_fpath, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg

def get_fpath_from_acq_prefix(analyzed_dir, acq_prefix):
    match_acq = glob.glob(os.path.join(analyzed_dir, '{}*_el.h5'.format(acq_prefix)))
    assert len(match_acq)==1, "Found {} files matching acquisition prefix {} in {}".format(len(match_acq), acq_prefix, analyzed_dir)
    fpath = match_acq[0]
    return fpath

def get_all_h5_files(analyzed_dir):
    return glob.glob(os.path.join(analyzed_dir, '*.h5'))

def load_and_transform_dlc(fpath, #localroot='/Users/julianarhee/DeepLabCut',
                           video_fpath=None,
                           projectname='projector-1dot-jyr-2024-02-18',
                           assay='2d-projector', heading_var='ori',
                           flyid='fly', dotid='single', fps=60, 
                           max_jump=6, pcutoff=0.8, winsize=10):

    #import parallel_pursuit as pp

    acq = get_acq_from_dlc_fpath(fpath) #'_'.join(os.path.split(fpath.split('DLC')[0])[-1].split('_')[0:-1])
    #project_dir = os.path.join(localroot, projectname)

    # load dataframe
    # df0 = pd.read_hdf(fpath)
    # scorer = df0.columns.get_level_values(0)[0]
    
    # load config file
    # cfg = load_dlc_config(projectname=projectname)

    # get fig id 
    fig_id = os.path.split(fpath.split('DLC')[0])[-1]

    # Load dlc into dataframes

    # load _eh.h5
    df = load_dlc_df(fpath, fly1=flyid, fly2=dotid, fps=fps, 
                            max_jump=max_jump, pcutoff=pcutoff, diff_speeds=True)

    # transform to FlyTracker format
    df_ = convert_dlc_to_flytracker(df)

    #% Get video info
    if video_fpath is None:
        try:
            cap = util.get_video_cap_check_multidir(acq, assay=assay)
            assert cap is not None
        except:
            print("No video found for {}. Returning.".format(acq))
            return df_ 
    else:
        cap = cv2.VideoCapture(video_fpath)

    # get frame info
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    #print(frame_width, frame_height) # array columns x array rows

    # input is DLC
    df_ = rem.do_transformations_on_df(df_, frame_width, frame_height) #, fps=fps)
    df_['ori_deg'] = np.rad2deg(df_['ori'])
    #df['targ_pos_theta'] = -1*df['targ_pos_theta']

    # convert centered cartesian to polar
    rad, th = util.cart2pol(df_['ctr_x'].values, df_['ctr_y'].values)
    df_['pos_radius'] = rad
    df_['pos_theta'] = th

    # angular velocity
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='ori', 
                                    vel_var='ang_vel', time_var='sec', winsize=winsize)
    # df_.loc[ (df_['ang_vel']>200) | (df_['ang_vel']<-200), 'ang_vel' ] = np.nan
    df_['ang_vel_deg'] = np.rad2deg(df_['ang_vel'])
    df_['ang_vel_abs'] = np.abs(df_['ang_vel'])

    # targ_pos_theta
    df_['targ_pos_theta_abs'] = np.abs(df_['targ_pos_theta'])
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='targ_pos_theta', vel_var='targ_ang_vel',
                                  time_var='sec', winsize=winsize)

    #% smooth x, y, 
    df_['pos_x_smoothed'] = df_.groupby('id')['pos_x'].transform(lambda x: x.rolling(winsize, 1).mean())
    #sign = -1 if input_is_flytracker else 1
    sign=1
    df_['pos_y_smoothed'] = sign * df_.groupby('id')['pos_y'].transform(lambda x: x.rolling(winsize, 1).mean())  

    # calculate heading
    for i, d_ in df_.groupby('id'):
        df_.loc[df_['id']==i, 'traveling_dir'] = np.arctan2(d_['pos_y_smoothed'].diff(), d_['pos_x_smoothed'].diff())
    df_['traveling_dir_deg'] = np.rad2deg(df_['traveling_dir']) #np.rad2deg(np.arctan2(df_['pos_y_smoothed'].diff(), df_['pos_x_smoothed'].diff())) 
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='traveling_dir', vel_var='traveling_dir_dt',
                                    time_var='sec', winsize=3)

    df_['heading_travel_diff'] = np.abs( np.rad2deg(df_['ori']) - np.rad2deg(df_['traveling_dir']) ) % 180  #% 180 #np.pi 

    df_['vel_smoothed'] = df_.groupby('id')['vel'].transform(lambda x: x.rolling(winsize, 1).mean())

    # calculate theta_error
    f1 = df_[df_['id']==0].copy().reset_index(drop=True)
    f2 = df_[df_['id']==1].copy().reset_index(drop=True)
    f1 = pp.calculate_theta_error(f1, f2, heading_var=heading_var)
    f2 = pp.calculate_theta_error(f2, f1, heading_var=heading_var)
    df_.loc[df_['id']==0, 'theta_error'] = f1['theta_error']
    df_.loc[df_['id']==1, 'theta_error'] = f2['theta_error']

    return df_


#%%

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


def plot_skeleton_on_ax(ix, df0, cap, cfg, ax=None,
                        pcutoff=0.01, animal_colors={'fly': 'm', 'single': 'c'},
                        alphavalue=1, skeleton_color='w', skeleton_color0='w',
                        markersize=3, skeleton_lw=0.5, lw=1):
    '''
    Plot skeleton of animals/bodyparts on ax.

    Arguments:
        ix -- _description_
        df0 -- _description_
        cap -- _description_
        cfg -- _description_

    Keyword Arguments:
        ax -- _description_ (default: {None})
        pcutoff -- _description_ (default: {0.01})
        animal_colors -- _description_ (default: {{'fly': 'm', 'single': 'c'}})
        alphavalue -- _description_ (default: {1})
        skeleton_color -- _description_ (default: {'w'})
        skeleton_color0 -- _description_ (default: {'w'})
        markersize -- _description_ (default: {3})
        skeleton_lw -- _description_ (default: {0.5})
        lw -- _description_ (default: {1})
    '''
    if ax is None:
        fig, ax =pl.subplots()

    scorer = df0.columns.get_level_values(0)[0]

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
        n_bodyparts = len(np.unique(df.columns.get_level_values("bodyparts")[::3]))
        #print(n_bodyparts)
        all_bpts = df.columns.get_level_values("bodyparts")[::3][0:n_bodyparts]

        bodyparts2connect = [v for v in cfg['skeleton'] if v[0] in bodyparts2plot]
        inds  = dlc.get_segment_indices(bodyparts2connect, all_bpts)
        skeleton_edges=bodyparts2connect
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


def plot_skeleton_on_image(ixs2plot, df0, cap, cfg, pcutoff=0.01, animal_colors={'fly': 'm', 'single': 'c'},
                            alphavalue=1, skeleton_color='w', skeleton_color0='w',
                            markersize=3, skeleton_lw=0.5, lw=1):
    '''
    Plot ixs2plot (frames) of DLC data on image. Calls plot_skeleton_on_ax()

    Arguments:
        ixs2plot -- _description_
        df0 -- _description_
        cap -- _description_
        cfg -- _description_

    Keyword Arguments:
        pcutoff -- _description_ (default: {0.01})
        animal_colors -- _description_ (default: {{'fly': 'm', 'single': 'c'}})
        alphavalue -- _description_ (default: {1})
        skeleton_color -- _description_ (default: {'w'})
        skeleton_color0 -- _description_ (default: {'w'})
        markersize -- _description_ (default: {3})
        skeleton_lw -- _description_ (default: {0.5})
        lw -- _description_ (default: {1})

    Returns:
        _description_
    '''
    #fig, axn = pl.subplots(1, len(ixs2plot))
    scorer = df0.columns.get_level_values(0)[0]
    fig, axn = pl.subplots(1, len(ixs2plot))

    #ix=3043
    for ai, ix in enumerate(ixs2plot):
        ax=axn[ai]
        dlc.plot_skeleton_on_ax(ix, df0, cap, cfg, ax=ax,
                        pcutoff=pcutoff, animal_colors=animal_colors,
                        alphavalue=alphavalue, skeleton_color=skeleton_color, skeleton_color0=skeleton_color0,
                        markersize=markersize, skeleton_lw=skeleton_lw, lw=lw)

    return fig



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
