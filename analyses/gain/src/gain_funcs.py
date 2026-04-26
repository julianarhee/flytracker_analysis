#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#%%
import os
import pandas as pd
import numpy as np

import libs.utils as util
import transform_data.relative_metrics as rel

# %%

def transform_projector_data(acquisition_parentdir, acqs, processedmat_dir, 
                            movie_fmt='.avi',subdir=None, flyid1=0, flyid2=1,
                            create_new=False, reassign_acquisition_name=False):
    """
    Load transformed projector data for specified acquisitions.
    Assumes that acquisitions are in the acquisition_parentdir.
    Assumes that flytracker output is in the processedmat_dir.

    Arguments:
        acquisition_parentdir (str): Parent directory of acquisitions.
        acqs (list): List of acquisitions.
        processedmat_dir (str): Directory to save processed mats.
        movie_fmt (str): Movie format.
        subdir (str): Subdirectory of acquisitions.
        flyid1 (int): Flytracker male index.
        flyid2 (int): Flytracker female index.
        create_new (bool): Create new processed mats.
        reassign_acquisition_name (bool): Reassign acquisition name, TRUE if one data-fly has multiple files (like projector CW/CCW)

    Returns:
        df0 (pd.DataFrame): Processed data.
        errors (list): List of errors.
    """
    d_list = []
    errors = []
    for i, acq in enumerate(acqs):
        if i % 10 == 0:
            print('Processing {} of {}: {}'.format(i, len(acqs), acq))
        acq_dir = os.path.join(acquisition_parentdir, acq)
        try:
            # Load flytracker output
            calib, trk, feat = util.load_flytracker_data(acq_dir, filter_ori=True)
            # Transform data to relative coordinates
            df_ = rel.get_metrics_relative_to_focal_fly(acq_dir,
                                                    savedir=processedmat_dir,
                                                    movie_fmt='.avi',
                                                    mov_is_upstream=None,
                                                    flyid1=0, flyid2=1,
                                                    plot_checks=False,
                                                    create_new=create_new,
                                                    get_relative_sizes=False)
            if 'targ_centered_x' not in df_.columns:
                print(f"{acq} targ_centered_x missing, rerun relative_metrics.py ")
                # Rerun: get_metrics_relative_to_focal_fly
                df_ = rel.get_metrics_relative_to_focal_fly(acq_dir,
                                                    savedir=processedmat_dir,
                                                    movie_fmt='.avi',
                                                    mov_is_upstream=None,
                                                    flyid1=0, flyid2=1,
                                                    plot_checks=False,
                                                    create_new=True,
                                                    get_relative_sizes=False)
                if 'targ_centered_x' not in df_.columns:
                    raise ValueError(f"{acq} targ_centered_x STILL not in df_")
            #assert len(df_['targ_centered_x'].unique()) > 1, "Bad targ calculation"
        except Exception as e:
            errors.append((acq, e))
            print("ERROR: {}".format(e))
            continue
        df_['file_name'] = os.path.split(acq)[-1]
        # Get species from acquisition

        if 'species' not in df_.columns: 
            df_['species'] = util.get_species_from_acquisition_name(acq)
        if reassign_acquisition_name:
            df_['date_fly'] = ['_'.join([f.split('-')[0], f.split('_')[1]]) for f in df_['file_name']]
            df_['acquisition'] = ['_'.join([a, b]) for a, b in df_[['date_fly', 'species']].values]
        else:
            df_['acquisition'] = acq #os.path.split(acq)[-1] 
        
        d_list.append(df_)

    if len(d_list) == 0:
        error_summary = '\n'.join(f'  {acq}: {e}' for acq, e in errors)
        raise ValueError(
            f"No acquisitions loaded successfully ({len(errors)} failed):\n{error_summary}"
        )
    df0 = pd.concat(d_list)

    return df0, errors


def assign_paint_conditions(df0, meta):
    # Add all paint conditions
    for fn, df_ in df0.groupby('file_name'):
        currm = meta[meta['file_name']==fn]
        assert len(currm)>0, 'No meta data for {}'.format(fn)
        assert len(currm)==1, 'Multiple meta data for {}'.format(fn)
        #df0.loc[df0['file_name']==fn, 'stim_direction'] = currm['stim_direction'].values[0]
        stim_dir = currm['stim_direction'].unique()[0] #fn.split('_')[-1]
        df0.loc[df0['file_name']==fn, 'stim_direction'] = stim_dir
        df0.loc[df0['file_name']==fn, 'paint_coverage'] = currm['painted'].values[0]
        manipulation_ = currm['manipulation_male'].values[0]
        if manipulation_.startswith('no '):
            paint_side = 'none'
        elif manipulation_.startswith('left '):
            paint_side = 'left'
        elif manipulation_.startswith('right '):
            paint_side = 'right'
        elif manipulation_.startswith('both '):
            paint_side = 'both'
        df0.loc[df0['file_name']==fn, 'paint_side'] = paint_side 
    df0['date'] = [int(a.split('_')[0]) for a in df0['acquisition']]

    return df0

def load_transformed_data(parquet_path):
    """Thin wrapper: derive acqdir/savedir from a parquet path, delegate to rel.load_processed_df."""
    savedir = os.path.dirname(parquet_path)
    acq = os.path.basename(parquet_path).replace('_df.parquet', '').replace('_df.pkl', '')
    acqdir = os.path.join(savedir, acq)
    df = rel.load_processed_df(acqdir, savedir=savedir)
    if df is None:
        raise FileNotFoundError("No processed df found at {}".format(parquet_path))
    return df

def get_heading_diff(expr, heading_var='integrated_heading', invert_heading=True):
    '''
    Calculate the stepwise wrapped difference in heading.
    Note, is same whether you use integrated_heading or heading_wrapped.

    Parameters:
        expr (pd.DataFrame): DataFrame containing the heading data.
        heading_var (str): Name of the heading column in the DataFrame.

    Returns:
        np.ndarray: Stepwise wrapped difference in heading.
    '''
    # Stepwise warpped difference in heading (new - old), 
    # CCW is positive by convention
    hdiffs = []
    for i, v in zip(expr[heading_var].iloc[0:-1], 
                    expr[heading_var].iloc[1:]):
        hd = util.circ_dist(v, i) #i, v) #* -1 # match matlab
        hdiffs.append(hd)
    heading_diff = np.array(hdiffs)

    # Make CW-positive (custom convention, so left->right is positive)
    if invert_heading:
        heading_diff_cw = -1*heading_diff
    else:
        heading_diff_cw = heading_diff

    # add 0
    heading_diffs = np.concatenate( ([0], heading_diff_cw) )
    
    return heading_diffs

def bin_x(chase_, xvar, start_bin = -180, end_bin=180, bin_size=20):
    '''
    Bin xvar by object position. Convert xvar to degrees.
    '''
    chase_['binned_{}'.format(xvar)] = pd.cut(chase_[xvar],
                                    bins=np.arange(start_bin, end_bin, bin_size),
                                    labels=np.arange(start_bin+bin_size/2, 
                                                     end_bin-bin_size/2, bin_size))
    return chase_


# Group by binned theta errors
def bin_by_object_position(chase_, start_bin = -180, end_bin=180, bin_size=20):
    '''
    Bin theta error by object position. Convert theta error to degrees.
    '''
    chase_['binned_theta_error'] = pd.cut(chase_['theta_error_deg'],
                                    bins=np.arange(start_bin, end_bin, bin_size),
                                    labels=np.arange(start_bin+bin_size/2,
                                            end_bin-bin_size/2, bin_size))    
    return chase_

