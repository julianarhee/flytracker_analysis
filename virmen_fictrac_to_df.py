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


def virmen_to_df(session, acqnum, rootdir='/Volumes/juliana/2p-data'):
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
        'ang_size',
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
        'intergrated_pos_lab_y', # 16 (30): should be same/similar to pos_y
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

    # Save to csv
    savedir = os.path.join(rootdir, session, 'processed', 'matlab-files')

    fname = 'virft_{}.csv'.format(acq)
    out_fpath = os.path.join(savedir, fname)

    df.to_csv(out_fpath, index=False)

    return df

  