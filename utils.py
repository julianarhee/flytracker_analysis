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
import mat73
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


# matlab analysis output to python (AO)
def mat_data_names_to_df(currmat):
    '''
    Specific func for data output stores in struct with 'data' and names' as dict.

    Returns: df, where data is 'data' and 'names' are columns
    '''
    df = pd.DataFrame(data=currmat['data'], columns=currmat['names']).fillna(0)

    return df


def mat_combine_binary_behaviors(curr_acq_mat):
    '''
    Take as input dict of dicts (from a given acquisition) from custom
    output analysis (quick_ethograms.m):

    Arguments:
        curr_acq_mat (dict): mat[species][acquisition_index] 
                             NOTE: (keys should include 'behavior', 'wings')

    Returns:
        B_df (pd.DataFrame): all aggregated behaviors that were binarized in Matlab function (GetBinaryBehaviors_v4.m and wings.m -- outputs of quick_ethograms.m)
 
    '''
    binary_behaviors = mat_data_names_to_df(curr_acq_mat['behavior'])
    binary_behaviors = binary_behaviors.rename(columns={'All Wing Ext': 'All Wing Extensions'})
    # tseries = pd.DataFrame(data=mat[sp][acq_ix]['wings']['tseries']['data'],
    #                        columns=mat[sp][acq_ix]['wings']['tseries']['names']).fillna(0)
    binary_wings = mat_data_names_to_df(curr_acq_mat['wings']['tseries'])
    assert binary_wings['All Wing Extensions'].equals(binary_behaviors['All Wing Extensions']), "ERROR: 'wings' tseries W.E. not the same as 'behavior' dict"
    B_df = pd.merge(binary_behaviors, binary_wings,  how='left', 
             left_on=['All Wing Extensions', 'Time Vector (s)'], right_on=['All Wing Extensions', 'Time Vector (s)']) # right_on = ['B_c1','c2'])
    return B_df



def mat_get_bout_indices(currmat):
    '''
    Fromat separate bout index (start/end) arrays from custom matlab analysis into one dataframe. NOTE: This is specific to WING events.
    
    Arguments:
        currmat (dict): mat[species][acquisition_index]['wings']

    Returns:
        bouts (pd.DataFrame): dataframe with start/end indices of bouts and wings

    '''

    # Get main BOUTS df
    bouts = pd.DataFrame(data=currmat['wings']['bouts']['data'],
                         columns=currmat['wings']['bouts']['names'])

    # boutname = 'boutsL'
    for boutname, colname in zip(['boutsL', 'boutsR', 'boutsB'], ['Left', 'Right', 'Bilateral']):
        if 'names' not in currmat['wings'][boutname].keys():
            currmat['wings'][boutname]['names'] = np.array(['{} Ext Start Indices'.format(colname), '{} Ext End Indices'.format(colname)])

        if currmat['wings'][boutname]['data'].shape[0]==0:
            continue 

        if len(currmat['wings'][boutname]['data'].shape)==1:
            bouts_ = pd.DataFrame(data=currmat['wings'][boutname]['data'],
                             index=currmat['wings'][boutname]['names']).T
        else:
            bouts_ = pd.DataFrame(data=currmat['wings'][boutname]['data'],
                             columns=currmat['wings'][boutname]['names'])
        for bi, brow in bouts_.iterrows():
            s_ix, e_ix = brow[['{} Ext Start Indices'.format(colname), '{} Ext End Indices'.format(colname)]]
            # Note, sometimes bout indices not exact match for Bilateral case
            bouts.loc[(bouts['All Ext Start Indices']<=s_ix) & (bouts['All Ext End Indices']>=e_ix), 'wing'] = colname 
            
        
    return bouts


def mat_split_courtship_bouts(bin_, bout_marker='Disengaged'):
    '''
    Use binary Disengaged 1 or 0 to find bout starts, assign from 0
    '''
    diff_ = bin_[bout_marker].diff()
    bout_starts = np.where(diff_!=0)[0] # each of these index values is the START ix of a bout
    for i, v in enumerate(bout_starts):
        if i == len(bout_starts)-1:
            bin_.loc[v:, 'boutnum'] = i
        else:
            v2 = bout_starts[i+1]
            bin_.loc[v:v2, 'boutnum'] = i
    return bin_


def get_bout_durs(df, bout_varname='boutnum', return_as_df=False,
                    timevar='Time Vector (s)'):
    '''
    Get duration of parsed bouts. 
    Parse with parse_bouts(count_varname='instrip', bout_varname='boutnum').

    Arguments:
        df -- behavior dataframe, must have 'boutnum' as column (run parse_inout_bouts())  

    Returns:
        dict, keys=boutnum, vals=boutdur (in sec)
    '''
    assert 'boutnum' in df.columns, "Bouts not parse. Run:  df=parse_inout_bouts(df)"

    boutdurs={}
    grouper = ['boutnum']
    for boutnum, df_ in df.groupby(bout_varname):
        boutdur = df_.sort_values(by=timevar).iloc[-1][timevar] - df_.iloc[0][timevar]
        boutdurs.update({boutnum: boutdur})

    if return_as_df:
        durs_ = pd.DataFrame.from_dict(boutdurs, orient='index').reset_index()
        durs_.columns = [bout_varname, 'boutdur']

        return durs_

    return boutdurs



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
    try:
        mat = scipy.io.loadmat(mat_fpath)
    except NotImplementedError as e:
        mat = mat73.loadmat(mat_fpath) #scipy.io.loadmat(mat_fpath)
    
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

    try:
        mat = scipy.io.loadmat(calib_fpath)
    except NotImplementedError as e:
        mat = mat73.loadmat(calib_fpath) #scipy.io.loadmat(mat_fpath)

    struct_name = [k for k in mat.keys() if not k.startswith('__')]
    assert len(struct_name)==1, "Did not find unique struct name: %s" % str(struct_name)
    
    mdata = mat.get(struct_name[0])

    # Use fields to create dict
    # 'names' (1, 35) 
    # 'data' (n_flies, n_frames, n_fields)
    # 'flags' (possible switches, check with flytracker/visualizer)
    mdtype = mdata.dtype
    ndata = {n: mdata[n][0, 0] for n in mdtype.names}

    all_fields = dict((k, v[0]) if len(v)>0 else (k, v) for k, v in ndata.items())
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

def load_mat_frames_and_var(mat_fpath):
    try:
        mat = scipy.io.loadmat(mat_fpath)
    except NotImplementedError as e:
        mat = mat73.loadmat(mat_fpath) #scipy.io.loadmat(mat_fpath)

    struct_name = [k for k in mat.keys() if not k.startswith('__')]
    assert len(struct_name)==1, "Did not find unique struct name: %s" % str(struct_name) 
    mdata = mat.get(struct_name[0])

    # Use fields to create dict
    # 'names' (1, 18) 
    # 'data' (n_frames, n_vars)
    # 'units' (units of variables) 
    mdtype = mdata.dtype
    ndata = {n: mdata[n][0][0] for n in mdtype.names}

    columns = [n[0] for n in ndata['names'][0]]
    n_frames, n_vars = ndata['data'].shape
    # turn into dataframe
    df = pd.DataFrame(data=ndata['data'], columns=columns)

    return df
 
 
def load_mat(mat_fpaths): #results_dir):
    '''
    Load track.mat and parse into dataframe. Assumes data is nflies, nframes, nflags.

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
        try:
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
        except NotImplementedError as e:
            mat = mat73.loadmat(mat_fpath) #scipy.io.loadmat(mat_fpath)
            struct_name = [k for k in mat.keys() if not k.startswith('__')]
            # is already dict
            ndata = mat[struct_name[0]] 
            columns = [n.replace(' ', '_') for n in ndata['names']]

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

def load_binary_evs_from_mat(matlab_src, feat=None,
                behavior_names=['Bilateral Wing Extensions', 'Unilateral Wing Extensions', 'Putative Tap Events', 'Chasing', 'Licking/Proboscis Ext', 'Copulation Attempts', 'Orienting']):

    '''
    Specific to output from matlab using AO's binarization of behaviors for ethograms.
    matlab_src: path to Ddata.mat (output of quick_ethograms.m)

    '''
    try:
        mat = scipy.io.loadmat(matlab_src, simplify_cells=True)
    except NotImplementedError as e:
        mat = mat73.loadmat(mat_fpath) #scipy.io.loadmat(mat_fpath)

    species_list = [k for k in mat.keys() if not k.startswith('__')]
    nonorienting_names = [b for b in behavior_names if b!='Orienting']

    binevs_list=[]
    for sp in species_list:
        if len(mat[sp])==0:
            continue
        if not isinstance(mat[sp], list):
            mat[sp] = [mat[sp]]
        for acq_ix, acq_mat in enumerate(mat[sp]):
            acq = acq_mat['acquisition']
            #if acq in ['20240109-1039_fly1_eleWT_4do_sh_eleWT_4do_gh']:
            #    continue
            print(sp, acq)
            bin_ = mat_combine_binary_behaviors(acq_mat) #mat[sp][acq_ix])
            # Get bout starts
            bin_ = mat_split_courtship_bouts(bin_)
            # wing bouts?
            bin_['Unilateral Wing Extensions'] = [1 if (l==1 or r==1) and (l!=r) else 0 for (l, r) \
                                      in bin_[['Left Wing Extensions', 'Right Wing Extensions']].values]
            ori_only = bin_[(bin_[nonorienting_names].eq(0).all(1)) & (bin_['Orienting'])]
            bin_['Orienting Only'] = 0
            bin_['Orienting Only'].loc[ori_only.index] = 1

            #bouts_ = util.mat_get_bout_indices(acq_mat) #mat[sp][acq_ix])
            # get features mat
            if feat is not None:
                feat_ = feat[(feat['acquisition']==acq) & (feat['sex']=='m')].copy().reset_index(drop=True)
                bin_ = bin_.loc[0:feat_['frame'].iloc[-1]] # Only grab til copulation index
                assert bin_.shape[0]==feat_.shape[0], "Incorrect shapes for merging: binary evs {} and feat {}".format(bin_.shape, feat_.shape)
                evs_ = pd.merge(bin_, feat_, left_index=True, right_index=True)
            else:
                evs_ = bin_.copy()
            # bouts
            #bouts_['acquisition'] = acq_mat['acquisition']
            #bouts_['species'] = sp
            #bouts_['strain'] = feat_['strain'].unique()[0]
            binevs_list.append(evs_)
    events = pd.concat(binevs_list).reset_index()

#    for aq, df_ in events.groupby('acquisition'):
#        dur_dict = get_bout_durs(df_)
#        events.loc[df_.index, 'boutdur'] = [dur_dict[v] for v in df_['boutnum']]

    return events

def add_bout_durations(df):
    # add bout durations
    for aq, df_ in df.groupby('acquisition'):
        dur_dict = get_bout_durs(df_)
        df.loc[df_.index, 'boutdur'] = [dur_dict[v] for v in df_['boutnum']]

    return df

def load_flytracker_data(acq_dir, fps=60):
    '''
    Get calibration info, -feat.mat and -track.mat as DFs.
    Returns:
        calib: 
        trackdf: raw tracking data (e.g., position, orientation, left wing ang)
        featdf: features derived from tracking data (e.g., velocity, dist to x)
    '''
    #%% Get corresponding calibration file
    calib=None; trackdf=None; featdf=None;
    try:
        calib = load_calibration(acq_dir)
    except Exception as e:
        print("No calibration!")
        calib={}
        calib['FPS'] = fps

    #% Load feature mat
    featdf = load_feat(acq_dir)
    trackdf = load_tracks(acq_dir)

    trackdf = add_frame_nums(trackdf, fps=calib['FPS'])
    featdf = add_frame_nums(featdf, fps=calib['FPS'])

    return calib, trackdf, featdf


## aggregate funcs
def aggr_load_feat(savedir, found_sessionpaths=[], create_new=False):

    if not create_new:
        try:
            # try loading saved feat.mat
            pkl_fpath = os.path.join(savedir, 'feat.pkl')
            feat = pd.read_pickle(pkl_fpath)
        except Exception as e:
            traceback.print_exc()
            print("Error loading feat. Creating new.")
            create_new = True
        
    if create_new:
        assert len(found_sessionpaths)>1, "No session paths provided."
        feat = aggr_feat_mats(found_sessionpaths) 

    return feat

def aggr_feat_mats(found_sessionpaths):
    '''
    Cycle through found session paths analyzed with FlyTracker, add some additional meta info, then save as feat DF. Tested and some manual corrections included.

    Args:
    -----
    List of fullpaths to parent dirs of experiments with -feat.mat found.

    Returns:
    --------
    feat (pd.DataFrame)
    '''
    f_list = []; t_list=[];
    for i, acq_dir in enumerate(found_sessionpaths[::-1]):
        acq = os.path.split(acq_dir)[-1]

        if acq == '20240109-1039_fly1_eleWT_4do_sh_eleWT_4do_gh':
            # RERUN, unequal feat and trk sizes
            continue

        try:
            calib_, trk_, feat_ = load_flytracker_data(acq_dir) #os.path.join(videodir, acq))
        except Exception as e:
            print("ERROR: %s" % acq)
            print(e)
            continue
            
        if 'cop_ind' not in calib_.keys():
            print(acq, 'No cop')
        else:
            print(acq, calib_['cop_ind'])
            
        # get species
        if 'mel-' in acq:
            species_abbr = 'mel'
            species_strain = 'na'
        elif 'suz-' in acq:
            species_abbr = 'suz'
            species_strain = 'na'
        else:
            if '_fly' in acq:
                species_abbr = acq.split('_')[2]
            else:
                species_abbr = acq.split('_')[1]
            species_strain = 'na'
            if species_abbr.startswith('mau'):
                species_strain = species_abbr[3:]
                species_abbr = species_abbr[0:3]
            elif species_abbr in ('Canton-S', 'ctns'):
                species_abbr = 'mel'
                species_strain = 'cantons'
        # get age
        if '_fly' in acq:
            age = int(re.sub('\D', '', acq.split('_')[3]))
        elif len(acq.split('_'))<2:
            age = None
        else:
            age = int(re.sub('\D', '', acq.split('_')[2]))

        # get sex
        cop_ix = calib_['cop_ind'] if calib_['cop_ind']>=1 else feat_['frame'].iloc[-1]
        #ix_male = trk_.groupby('id').mean()['body_area'].idxmax() #.unique()
        #ix_female = trk_.groupby('id')['body_area'].mean().idxmax()
        if acq in ['20231222-1149_fly2_eleWT_4do_sh_eleWT_4do_gh', '20231223-1212_fly3_eleWT_5do_sh_eleWT_5do_gh']:
            ix_male = 1
        elif float(trk_[trk_['frame']<=cop_ix].groupby('id')['body_area'].mean().round(1).diff().abs().dropna()) == 0: 
            #float(feat_[feat_['frame']<=cop_ix].groupby('id')['max_wing_ang'].mean().round(1).diff().abs().dropna()) == 0: # difference is super tiny
            #ix_male = trk_[trk_['frame']<=cop_ix].groupby('id')['body_area'].mean().idxmin() # use body size
            ix_male = feat_[feat_['frame']<=cop_ix].groupby('id')['max_wing_ang'].mean().idxmax()
        else:
            #ix_male = feat_[feat_['frame']<=cop_ix].groupby('id')['max_wing_ang'].mean().idxmax()
            ix_male = trk_[trk_['frame']<=cop_ix].groupby('id')['body_area'].mean().idxmin() # use body size
            
    #     if trk_.groupby('id')['minor_axis_len'].mean().round().diff().max() <= 1:
    #         ix_female = trk_.groupby('id')['body_area'].mean().idxmax()
    #     else:
    #         ix_female = trk_.groupby('id')['minor_axis_len'].mean().idxmax() # this seems most reliable
        #assert len(ix_max)==1, "Ambiguous sex based on size: {}".format(trk_.groupby('id').mean()[['major_axis_len', 'body_area', 'minor_axis_len']].idxmax())
        #ix_male = feat_.groupby('id')['max_wing_ang'].mean().idxmax()
        feat_.loc[feat_['id']==ix_male, 'sex'] = 'm'
        feat_.loc[feat_['id']!=ix_male, 'sex'] = 'f'
        
        trk_.loc[trk_['id']==ix_male, 'sex'] = 'm'
        trk_.loc[trk_['id']!=ix_male, 'sex'] = 'f'
       
        print('--', species_abbr, age, feat_['sex'].unique(), 'male ID: {}'.format(feat_[feat_['sex']=='m']['id'].unique()[0]))
        try:
            #male_wg = feat_[(feat_['frame']<=cop_ix) & (feat_['sex']=='m')]['max_wing_ang'].median()
            #female_wg = feat_[(feat_['frame']<=cop_ix) & (feat_['sex']=='f')]['max_wing_ang'].median()
            male_wg = feat_[(feat_['frame']<cop_ix) & (feat_['max_wing_ang']>1.5) & (feat_['sex']=='m')].count()[0]
            female_wg = feat_[(feat_['frame']<cop_ix) & (feat_['max_wing_ang']>1.5) & (feat_['sex']=='f')].count()[0]
            assert male_wg-female_wg > 100, "Male wing is not >> than female. Check: %s" % acq
        except AssertionError as e: # AssertionError as e:
            print(e)   
            
        # update
        feat_['species'] = species_abbr
        feat_['strain'] = species_strain
        feat_['age'] = age
        feat_['acquisition'] = acq
        feat_['copulation_index'] = calib_['cop_ind']
        feat_['copulation'] = calib_['cop_ind']>0
        grab_index = calib_['cop_ind']-1 if calib_['cop_ind']>1 else feat_.iloc[-1].name
        f_ = feat_[feat_['frame']<=grab_index].reset_index(drop=True)
        f_list.append(f_)
        # add trk, too
        trk_['acquisition'] = acq
        t_ = trk_[trk_['frame']<=grab_index].reset_index(drop=True)
        t_list.append(t_)
        #print(f_.shape, t_.shape)
        assert f_.shape[0]==t_.shape[0], "Unequal"
    feat = pd.concat(f_list) #.reset_index(drop=True)
    trk = pd.concat(t_list)

    return feat, trk


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


# projector functions
from numpy.fft import fft, ifft, fftfreq
def get_fft(df_, fft_var='pos_y', time_var='sec', return_pos=True):

    x_fly = df_[fft_var] - df_[fft_var].mean()
    t_fly =  df_[time_var] #np.arange(0,1,ts)
    fft_fly, all_freqs = calculate_fft(x_fly, t_fly)
    if return_pos:
        freq = all_freqs[np.where(all_freqs >= 0)] 
        amp_fly = calculate_fft_amplitude(fft_fly[np.where(all_freqs >= 0)], t_fly)

    else:
        freq = all_freqs
        amp_fly = calculate_fft_amplitude(fft_sly, t_fly)

    return amp_fly, freq


def calculate_fft(x, t, sr=60):

    # sampling interval
    ts = np.mean(np.diff(t)) #1.0/sr
    npnts =  len(t)
    
    X = fft(x)
    N = len(X)
    n = np.arange(N)
    T = N/sr
#     freq = n/T  # freq mirrorin above nyquist
    freq = fftfreq(npnts, ts)
    
    return X, freq #X, freq

def calculate_fft_amplitude(X, t):

    pwr = 2*np.abs(X)/len(t)   
    return pwr


def smooth_timecourse(in_trace, win_size=42):
    '''
    Don't Use this one
    '''
    #smooth trace
    win_half = int(round(win_size/2))
    trace_pad = np.pad(in_trace, ((win_half, win_half)), 'reflect') # 'symmetric') #'edge')

    smooth_trace = np.convolve(trace_pad, np.ones((win_size,))*(1/float(win_size)),'valid')
     
    return smooth_trace[int(win_half/2):-int(win_half/2)]


