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

def flatten(t):
    return [item for sublist in t for item in sublist]

def euclidean_dist(df1, df2, cols=['x_coord','y_coord']):
    return np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)

def check_nan(wingR):
    if any(np.isnan(wingR)):
        wingR_ = pd.Series(wingR)
        wingR = wingR_.interpolate()
    return wingR

# ---------------------------------------------------------------------
# Data loading and formatting
# ---------------------------------------------------------------------
def get_acq_dir(sessionid, assay_prefix='single_20mm*', rootdir='/mnt/sda/Videos'):
    '''
    Args:
    -----
    sessionid: (str)
        YYYYMMDD-HHMM prefix for acquisition id
    '''
    acquisition_dirs = sorted([f for f in glob.glob(os.path.join(rootdir, '%s*' % assay_prefix, 
                            '%s*' % sessionid)) if os.path.isdir(f)], key=natsort)
    #print("Found %i acquisitions from %s" % (len(acquisition_dirs), sessionid))
    assert len(acquisition_dirs)==1, "Unable to find unique acq. from session ID: %s" % sessionid
    acq_dir = acquisition_dirs[0]
    acquisition = os.path.split(acq_dir)[-1]

    return acq_dir

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


# ---------------------------------------------------------------------
# FlyTracker functions
# ---------------------------------------------------------------------

def add_frame_nums(trackdf, fps=None):
    '''Add frame index and sec to dataframes
    '''
    frame_ixs = trackdf[trackdf['id']==0].index.tolist()
    trackdf['frame'] = None
    for i, g in trackdf.groupby('id'):
        trackdf.loc[g.index, 'frame'] = frame_ixs
    
    # add sec
    if fps is not None:
        trackdf['sec'] = trackdf['frame']/float(fps)
    
    return trackdf

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
                  'centroids', 'rois', 'n_flies', 'cop_ind']
 
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

def threshold_courtship_bouts(feat0, max_dist_to_other=5, max_facing_angle=30):
    '''
    Set thresholds for identifying courtship bouts using extracted features
    from FlyTracker. Sets thresholds on 'facing_angle' and 'dist_to_other'.
    
    Args:
    -----
    feat: (pd.DataFrame)
        Calculated features from -feat.mat  *assumes 1 fly id only*
    
    max_dist: (float)
        Inter-fly distance at which interaction is considered a bout
        
    max_angle: (float)
        Angle (in degs) that flies are oriented to e/o
    
    Returns:
    --------
    df: (pd.DataFrame)
        Updated dataframe with courtship (bool) as column
                
    '''
    feat = feat0.copy()
    # Convert to deg to make my life easier
    feat['courtship'] = False
    all_nans=[]
    #feat.shape
        
    feat['facing_angle_deg'] = np.rad2deg(feat['facing_angle'])

    # Identify bad flips, where wings flip so fly seems to face opposite dir
    #bad_flips = feat[feat['facing_angle'].diff().abs()>0.5].index.tolist()
    #feat.loc[bad_flips] = np.nan

#        # find all non-consecutive nan indices
#        found_nans = feat[feat['facing_angle'].isna()].index.tolist()
#        non_consecs = np.where(np.diff(found_nans)>1)[0] # each pair of values represents 1 chunk   
#        non_consecs 
#        for i, ix in enumerate(non_consecs[0::2]):
#            curr_ix = list(non_consecs).index(ix)        
#            s_ix = found_nans[ix]         
#            next_ix = non_consecs[curr_ix+1]
#            e_ix = found_nans[next_ix]
#            print(s_ix, e_ix)
#            feat.loc[s_ix:e_ix]=np.nan     
# 
#        # Get all nan indices, and block out in-between frames, too
#        nan_ixs = feat.isna().index.tolist()
#        chunks = []
#        for k, g in groupby(enumerate(nan_ixs), lambda ix: ix[0]-ix[1]):
#            chunks.append(list(map(itemgetter(1), g)))
#        for chunk in chunks:
#            feat.loc[chunk] = np.nan
#
    # Find true facing frames    
    feat.loc[(feat['dist_to_other'] < max_dist_to_other) & (
        feat['facing_angle_deg'] < max_facing_angle), 'courtship'] = True
             
    return feat

def get_true_bouts(feat0, calib, ibi_min_sec=0.5):
    '''
    Group frames that pass threshold for "courtship" into actual bouts.
    Bouts specified arbitrarily.
    
    Args:
    -----
    df: (pd.DataFrame)
        Thresholded dataframe from -feat.mat (output of threshold_courtship_bouts())
    
    calib: (dict)
        From calib.mat output (saved as dict)
    
    ibi_min_sec: (float)
        Min. duration (in sec) to be considered a separate bout.
    '''
    feat = feat0.copy()
    # Get list of all bout chunks
    courtship_ixs = feat[feat['courtship']].index.tolist()
    bouts = []
    for k, g in groupby(enumerate(courtship_ixs), lambda ix: ix[0]-ix[1]):
        bouts.append(list(map(itemgetter(1), g)))
    fps = calib['FPS']

    # Identify likely bout-stop false-alarms (i.e., next "bout" starts immediately after...)
    curr_bout_ix = 0
    combine_these=[curr_bout_ix]
    bouts_to_combine={}
    for i, b in enumerate(bouts[0:-1]):
        ibi = (bouts[i+1][0] - b[-1] ) / fps
        # print(i, ibi)

        if np.round(ibi) <= ibi_min_sec: # check dur of next bout after current one
            if len(combine_these)==0:
                combine_these=[i]
            combine_these.append(i+1)
        else:
            if len(combine_these)==0:
                combine_these=[i]
            bouts_to_combine.update({curr_bout_ix: combine_these})
            curr_bout_ix+=1
            combine_these=[]
#     interbout_sec = np.array([(bouts[i+1][0] - b[-1])/fps for i, b in enumerate(bouts[0:-1])])

#     # identify where interbout is "too short"
#     #ibi_min_sec = 0.5
#     ibi_too_short = np.where(interbout_sec < ibi_min_sec)[0] # indexes into bouts
    
#     # get starting indices for true bouts
#     gap_starts = np.where(np.diff(ibi_too_short)>1)[0]
#     gap_ixs = [ibi_too_short[0]]
#     gap_ixs.extend([ibi_too_short[i+1] for i in gap_starts])

#     # concatenate too-short bouts into full bouts
#     true_bouts=[]
#     for i in gap_ixs:
#         curr_ = bouts[i]
#         curr_.extend(bouts[i+1])
#         true_bouts.append(curr_)
#     interbout_sec = np.array([(true_bouts[i+1][0] - b[-1])/fps for i, b in enumerate(true_bouts[0:-1])])
#     ibi_too_short = np.where(interbout_sec < ibi_min_sec)[0] 
#     assert len(ibi_too_short)==0, "Bad bout concatenation. Found %i too short" % len(ibi_too_short)
    bout_dict={}
    for bout_num, bout_ixs in bouts_to_combine.items():
        curr_ixs = flatten([bouts[i] for i in bout_ixs])
        bout_dict.update({bout_num: curr_ixs})

    # reassign courtship bouts:
    for bout_num, bout_ixs in bout_dict.items():
        start, end = bout_ixs[0], bout_ixs[-1]
        feat.loc[start:end, 'courtship']=True
            
    return feat, bout_dict

