#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
import sys
import glob
import importlib

import numpy as np
import pandas as pd
import pickle as pkl

import libs.utils as util

import argparse

def add_ft_actions(procdir, rawdir, currf, verbose=False):
    '''
    Add FlyTracker actions to processed data.

    Arguments:
        procdir -- Directory to save processed data, ends with processed_mats
        rawdir -- Directory containing the raw video folders 
        currf -- Current FlyTracker folder name (should be same as base file_name)

    Keyword Arguments:
        verbose -- Print verbose output (default: {False})

    Returns:
        df -- Processed data with FlyTracker actions added as binary columns, boutnums are None except for bouts that are in the actions file.
    '''

    currfpath = os.path.join(procdir, f'{currf}_df.parquet')
    assert os.path.exists(currfpath), f'File {currfpath} does not exist'

    # Load processed FlyTracker data (data already transformed, transform_data)
    df = pd.read_parquet(currfpath)
    # %%
    # Get actions path 
    srcdir = os.path.join(rawdir, currf)

    assert os.path.exists(srcdir), f'Directory {srcdir} does not exist'
    actions_fpath = os.path.join(srcdir, currf, f'{currf}-actions.mat')
    assert os.path.exists(actions_fpath), f'File {actions_fpath} does not exist'

    # Load actions
    actions = util.load_ft_actions([actions_fpath], split_end=False)
    # %%
    # Add actions to df
    for (action_name, bout_num), a_df in actions.groupby(['action', 'boutnum']):
        #print(action_name, bout_num)
        if action_name not in df.columns:
            print("Adding new action column: ", action_name)
            df[action_name] = 0
            df[f'{action_name}_boutnum'] = None
        # Assign all frames between start and end in df
        start, end = a_df['start'].item(), a_df['end'].item()
        frame_range = np.arange(start, end + 1)
        df.loc[df['frame'].isin(frame_range), action_name] = 1
        df.loc[df['frame'].isin(frame_range), f'{action_name}_boutnum'] = bout_num

    # %%
    # Save to processed again
    # If no parquet exists, write the parquet file and get rid of the pickle
    if currfpath.endswith('.pkl'):
        writepath = currfpath.replace('.pkl', '.parquet')
        df.to_parquet(writepath, engine='pyarrow', compression='snappy')
        os.remove(currfpath)
    else:
        writepath = currfpath
        df.to_parquet(writepath, engine='pyarrow', compression='snappy')

    if verbose:
        print("Saved to: ", writepath)

    return df
    # %%


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add FlyTracker actions to processed data.')
    parser.add_argument('--procdir', type=str, required=True, help='Directory to save processed data, ends with processed_mats')
    parser.add_argument('--acqdir', type=str, required=True, help='Directory containing acquisition folders')
    parser.add_argument('--currf', type=str, required=True, help='Current FlyTracker folder name (should be same as base file_name)')
    args = parser.parse_args()

    procdir = args.procdir
    acqdir = args.acqdir
    currf = args.currf

    # %%
    #if interactive:
    # Load FlyTracker data
    #procdir = '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_painted_eyes/FlyTracker/processed_mats'
    #acquisitiondir = os.path.join('/Volumes/Juliana/Caitlin_RA_data/Caitlin_projector')
    #currf = '20250703-1510_fly1_Dmel-p1_1do_gh_1dR'

    df = add_ft_actions(procdir, acqdir, currf, verbose=False)
