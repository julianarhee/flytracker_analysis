#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/01/28 13:17:48
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com

'''
import os
import re
import glob
import scipy.io
import cv2

from itertools import groupby
from operator import itemgetter

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# General
# ---------------------------------------------------------------------
natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]


# ---------------------------------------------------------------------
# Data loading and formatting
# ---------------------------------------------------------------------
def get_movie_metadata(curr_movie_path):
    '''
    Get metadata for specified movie.

    Args:
    -----
    curr_movie_path: (str)
        Path to .avi file
        
    Returns
    -------
    minfo: dict
    '''
    vidcap = cv2.VideoCapture(curr_movie_path)
    success, image = vidcap.read()
    framerate = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    minfo = {'framerate': framerate,
             'width': width,
             'height': height,
             'n_frames': n_frames,
             'movie_path': curr_movie_path
    }

    vidcap.release()

    return minfo

def get_mat_paths_for_all_vids(acquisition_dir, ftype='track'):
    '''
    Get all .mat files associated with a given acquisition (experiment)
    
    Args:
    -----
    acquisition_dir: (str) 
        Dir containing all input movies (and FlyTracker results).
    ftype: (str)
        Type of .mat to load. Must be: -bg, -feat, -seg, or -track.
         
    Returns:
    --------
    Sorted list of all .mat files. 
    ''' 
    paths_to_matfiles = sorted(glob.glob(os.path.join(acquisition_dir, '*', '*%s.mat' % ftype)), 
           key=natsort)
      
    return paths_to_matfiles


def get_feature_units(mat_fpath):
    '''
    Load -feat.mat and get units for each of the var. names
    '''
    mat = scipy.io.loadmat(mat_fpath)
    
    mdata = mat.get('feat')
    mdtype = mdata.dtype
    ndata = {n: mdata[n][0, 0] for n in mdtype.names}

    columns = [n[0].replace(' ', '_') for n in ndata['names'][0]]
    units = [n[0] if len(n)>0 else 'NA' for n in ndata['units'][0]]
    
    unit_lut = dict((k, v) for k, v in zip(columns, units))
    
    return unit_lut

def load_calibration(curr_acq):
    '''
    Load calibration.mat from FlyTracker

    Args:
    -----
    curr_acq: (str)
        Dir containing all .avi files for specific acquisition (and calibration.mat files).
        
    Returns:
    --------
    
    '''
    fieldnames = ['n_chambers', 'n_rows', 'n_cols', 'FPS', 'PPM', 'w', 'h', 
                  'centroids', 'rois', 'n_flies']
 
    calib_fpath = os.path.join(curr_acq, 'calibration.mat')
    assert os.path.exists(calib_fpath), "No calibration found: %s" % curr_acq

    mat = scipy.io.loadmat(calib_fpath)
    struct_name = [k for k in mat.keys() if not k.startswith('__')]
    assert len(struct_name)==1, "Did not find unique struct name: %s" % str(struct_name)
    
    mdata = mat.get(struct_name[0])

    # Use fields to create dict
    # 'names' (1, 35) 
    # 'data' (n_flies, n_frames, n_fields)
    # 'flags' (possible switches, check with flytracker/visualizer)
    mdtype = mdata.dtype
    ndata = {n: mdata[n][0, 0] for n in mdtype.names}

    all_fields = dict((k, v[0]) for k, v in ndata.items())
    calib = {}
    for k, v in all_fields.items():
        if k not in fieldnames:
            continue
        if len(v)==1:
            if v.dtype=='object':
                calib[k] = np.array(v[0][0])
            
            else:
                calib[k] = int(v) if v.dtype in ['uint8', 'uint16'] else float(v)
        else:
            calib[k] = np.array(v)
            
    return calib

def load_feat(curr_acq):
    mat_fpaths = get_mat_paths_for_all_vids(curr_acq, ftype='feat')
    feat = load_mat(mat_fpaths)
    return feat

def load_tracks(curr_acq):
    mat_fpaths = get_mat_paths_for_all_vids(curr_acq, ftype='track')
    trk = load_mat(mat_fpaths)
    return trk
 
def load_mat(mat_fpaths): #results_dir):
    '''
    Load track.mat and parse into dataframe.

    Args:
    -----
    mat_fpaths: list
        List of path(s) to -track.mat file (from FlyTracker) 

    Returns:
    -------
    df: (pd.DataFrame)
        Dataframe of all the extracted data from FlyTracker.
        Rows are frames, columns are features (including fly ID, 'id')
    '''
    #ft_outfile = glob.glob(os.path.join(results_dir, '*', '*track.mat'))[0]
    #print(ft_outfile)
    all_dfs=[]
    for mat_fpath in sorted(mat_fpaths, key=natsort):
        mat = scipy.io.loadmat(mat_fpath)
        struct_name = [k for k in mat.keys() if not k.startswith('__')]
        assert len(struct_name)==1, "Did not find unique struct name: %s" % str(struct_name) 
        mdata = mat.get(struct_name[0])

        # Use fields to create dict
        # 'names' (1, 35) 
        # 'data' (n_flies, n_frames, n_fields)
        # 'flags' (possible switches, check with flytracker/visualizer)
        mdtype = mdata.dtype
        ndata = {n: mdata[n][0][0] for n in mdtype.names}

        columns = [n[0].replace(' ', '_') for n in ndata['names'][0]]
        n_flies, n_frames, n_flags = ndata['data'].shape
        d_list=[]
        for fly_ix in range(n_flies):
            tmpdf = pd.DataFrame(data=ndata['data'][fly_ix, :], columns=columns)
            tmpdf['id'] = fly_ix
            d_list.append(tmpdf)
        df_ = pd.concat(d_list, axis=0, ignore_index=True)
        df_['fpath'] = mat_fpath  
        all_dfs.append(df_) 
        
    df = pd.concat(all_dfs, axis=0, ignore_index=True)

    return df


# ---------------------------------------------------------------------
# Calculate courtship metrics
# ---------------------------------------------------------------------

def thresh_courtship_bouts(df, max_dist=5, max_angle=30):
    '''
    Set thresholds for identifying courtship bouts using extracted features
    from FlyTracker. Sets thresholds on 'facing_angle' and 'dist_to_other'.
    
    Args:
    -----
    df: (pd.DataFrame)
        Calculated features from -feat.mat 
    
    max_dist: (float)
        Inter-fly distance at which interaction is considered a bout
        
    max_angle: (float)
        Angle (in degs) that flies are oriented to e/o
    
    Returns:
    --------
    df: (pd.DataFrame)
        Updated dataframe with courtship (bool) as column
                
    '''
    df['facing_angle_deg'] = np.rad2deg(df['facing_angle'])
    df['courtship'] = False
    df.loc[(df['dist_to_other'] < max_dist) & (
        df['facing_angle_deg'] < max_angle), 'courtship'] = True
    
    return df

def get_true_bouts(df, calib):
    courtship_ixs = df[df['courtship']].index.tolist()
    bouts = []
    for k, g in groupby(enumerate(courtship_ixs), lambda ix: ix[0]-ix[1]):
        bouts.append(list(map(itemgetter(1), g)))
        
    fps = calib['FPS']
    print('FPS: %.2f Hz' % fps)

    interbout_sec = np.array([(bouts[i+1][0] - b[-1])/fps for i, b in enumerate(bouts[0:-1])])

    # identify where interbout is "too short"
    ibi_min_sec = 0.5
    ibi_too_short = np.where(interbout_sec < ibi_min_sec)[0]

    # get starting indices for true bouts
    gap_starts = np.where(np.diff(ibi_too_short)>1)[0]
    gap_ixs = [ibi_too_short[0]]
    gap_ixs.extend([ibi_too_short[i+1] for i in gap_starts])

    # concatenate too-short bouts into full bouts
    true_bouts=[]
    for i in gap_ixs:
        curr_ = bouts[i]
        curr_.extend(bouts[i+1])
        true_bouts.append(curr_)
    interbout_sec = np.array([(true_bouts[i+1][0] - b[-1])/fps for i, b in enumerate(true_bouts[0:-1])])
    ibi_too_short = np.where(interbout_sec < ibi_min_sec)[0] 
    assert len(ibi_too_short)==0, "Bad bout concatenation. Found %i too short" % len(ibi_too_short)
    
    return true_bouts

