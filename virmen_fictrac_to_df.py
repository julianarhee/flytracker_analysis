#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 04 16:58:00 2024

Author: Juliana Rhee
Email:  juliana.rhee@gmail.com

This script imports .mat file and converts it to a pandas dataframe
containing virmen and FicTrac variables.

"""
#%%
import os
import glob
import traceback

import numpy as np
import pandas as pd
#import mat73
import scipy
#%%

rootdir = '/Volumes/juliana/2p-data'
session = '20240905'
# acq = 'example-yak-P1-1'
acqnum = 18
#processed_mats = glob.glob(os.path.join(rootdir, session, 'processed', 
#                                        'matlab-files', '*{:03d}.mat'.format(acqnum)))
#processed_dir = os.path.join(rootdir, session, 'processed')


def virmen_to_df(session, acqnum, virmen_dir=None, rootdir='/Volumes/juliana/2p-data'):
    '''
    Loads .mat file from session virmen folder and converts it to pandas dataframe.
    Saves to session/processed/matlab-files folder as .csv file.

    Arguments:
        session -- YYYYMMDD
        acqnum -- numeric 

    Keyword Arguments:
        rootdir -- _description_ (default: {'/Volumes/juliana/2p-data'})

    Returns:
        df
    '''
    if virmen_dir is None: 
        virmen_dir = os.path.join(rootdir, session, 'virmen')

    mat_files = glob.glob(os.path.join(virmen_dir, '*{:03d}.mat'.format(acqnum)))
    mat_fpath = mat_files[0]
    #mat_fpath = processed_mats[0]
    #print(mat_fpath)

    acq = os.path.splitext(os.path.split(mat_fpath)[1])[0]
    #print(acq)
    #%
    # Load mat
    #mat = mat73.loadmat(mat_fpath)
    df = virmen_to_df_from_fpath(mat_fpath)

    # Save to csv
    #savedir = os.path.join(rootdir, session, 'processed', 'matlab-files')
    #fname = 'virft_{}.csv'.format(acq)
    #out_fpath = os.path.join(savedir, fname)
    #df.to_csv(out_fpath, index=False)
    save_virmen_df_to_csv(df, mat_fpath, sessiondir=os.path.join(rootdir, session))

    return df

def save_virmen_df_to_csv(df, mat_fpath, sessiondir=None):
    '''
    Saves virmen dataframe as csv file to session/processed/matlab-files folder.

    Arguments:
        df -- dataframe, output of virmen_to_df_from_fpath()
        mat_fpath -- path to raw .mat file acquired from ViRMEn

    Keyword Arguments:
        sessiondir -- /rootdir/session (default: {None})
    '''
    acqdir, acqname = os.path.split(mat_fpath)
    acq, ext = os.path.splitext(acqname)

    if sessiondir is None:
        sessiondir = os.path.split(acqdir)[0]

    savedir = os.pathjoin(sessiondir, 'processed', 'matlab-files')

    fname = 'virft_{}.csv'.format(acq)
    out_fpath = os.path.join(savedir, fname)

    df.to_csv(out_fpath, index=False)
    
    return

def virmen_to_df_from_fpath(mat_fpath, out_fpath=None):

    mat = scipy.io.loadmat(mat_fpath)
    mdata = mat['expr']

    columns = [
        'toc', # 1
        'counter',
        'pos_x',
        'pos_y',
        'heading', # 5
        'target_x',
        'target_y',
        'clock_H', # 8
        'clock_M',
        'clock_S',
        'pulse_sent', # 11
        'ang_size',   # Can be arc_angle...
        'moving',
        'visible',
        'frame_counter',                  # 1 (15) (FT PARAMS, 15-38)
        'delta_rotation_vector_cam_x',
        'delta_rotation_vector_cam_y',
        'delta_rotation_vector_cam_z',
        'delta_rotation_error_score',     # 5 (19)
        'delta_rotation_vector_lab_x',
        'delta_rotation_vector_lab_y',
        'delta_rotation_vector_lab_z',
        'absolute_rotation_vector_cam_x', # 9-11
        'absolute_rotation_vector_cam_y', 
        'absolute_rotation_vector_cam_z', 
        'absolute_rotation_vector_lab_x', # 12-14
        'absolute_rotation_vector_lab_y', 
        'absolute_rotation_vector_lab_z', 
        'integrated_pos_lab_x',  # 15 (29): should be same/similar to pos_x
        'integrated_pos_lab_y', # 16 (30): should be same/similar to pos_y
        'animal_heading_lab',    # 17 (31): should be the same/similar to heading 
        'animal_movement_direction_lab', # 18 (32) - maybe this is traveling dir?
        'animal_movement_speed_lab',     # 19 (33)
        'lateral_motion_x', # 20
        'lateral_motion_y', # 21
        'timestamp',        # 22
        'sequence_counter', # 23 (# 37)
        'delta_timestamp',  # 24 (#38)
        #'alt.timestamp',   # 25
        'target_vel' # 39  
        ]

    df = pd.DataFrame(data=mat['expr'], columns=columns)

    return df

#%%
def load_custom_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            #print(key)
            if key in ['exper']:
                #print("Exper")
                d[key] = _exper_todict(d[key]) # matobj #[key])
            else:
                if isinstance(d[key], scipy.io.matlab.mat_struct):
                    d[key] = _todict(d[key])
                elif isinstance(d[key], np.ndarray):
                    d[key] = _toarray(d[key])
        return d

    def _exper_todict(matobj):
        """
        Recursively constructs a dictionary from matlab object
        scipy.io.matlab._mio5_params.mat_struct.

        """
        mdict={}
        for strg in matobj._fieldnames:
            #print(strg)
            if strg in ['codeText']:
                continue
            elem = matobj.__dict__[strg] #if len(matobj.__dict__[strg])==0 else matobj.__dict__[strg][0][0] 
            try:
                if isinstance(elem, scipy.io.matlab.mat_struct):
                    mdict[strg] = _todict(elem)
                elif isinstance(elem, scipy.io.matlab.MatlabOpaque):
                    mdict[strg] = elem
                elif isinstance(elem, scipy.io.matlab.MatlabFunction):
                    mdict[strg] = elem.tolist().__dict__['function_handle'].__dict__['function']
                elif isinstance(elem, np.ndarray):
                    mdict[strg] = _toarray(elem)
                else:
                    mdict[strg] = elem
                #if isinstance(mdict[strg], np.ndarray) and len(mdict[strg])==1:
                #    mdict[strg] = mdict[strg][0]    
            except Exception as e:
                print("Error in ({})".format(strg))
                traceback.print_exc()
        return mdict #d['exper']

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, scipy.io.matlab.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    mat = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)

    #data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    d = _check_vars(mat)

    return d #_check_vars(data)

 