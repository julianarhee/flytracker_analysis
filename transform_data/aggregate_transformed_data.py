#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Aggregate individual _df.pkl files into a single processed.pkl file for analysis.
Combines output of relative_metrics.py.

Saves to both local and remote directories. 
In remote: /relative_metrics/processed.pkl.

@author: julianarhee
@email: juliana.rhee@gmail.com  
"""

#%%
import sys
import os
import sys
import glob
import importlib

import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import pylab as pl
import matplotlib as mpl

import matplotlib.gridspec as gridspec

# import some custom funcs
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from relative_metrics import load_processed_data
import utils as util
import plotting as putil
import argparse

#%%
def aggregate_relative_metrics(srcdir, create_new=False, localdir=None):
    """
    Aggregate individual _df.pkl files into a single relative_metrics.pkl file for analysis.    
    Saves to both local and remote directories.
    Individual _df.pkl files are output of relative_metrics.py.

    Parameters
    ----------
    srcdir : str
        Directory containing individual _df.pkl files. Output of relative_metrics.py. 
        Likely /behavior-analysis-rootdir/assay/experiment/FlyTracker/processed_mats/,
        where assay is either '2d-projector' or '38mm_dyad' and experiment is 'circle_diffspeeds' or 'MF'.

    localdir : str
        Local directory to save the processed data. Default is None.
        Likely /Users/julianarhee/Documents/rutalab/projects/courtship/data/2d-projector/circle_diffspeeds/FlyTracker/
        or /Users/julianarhee/Documents/rutalab/projects/courtship/data/MF/38mm-dyad/FlyTracker/
    """
    #%
    # Specify data source directory based on server filetree
#    if assay == '2d-projector':
#        # Set sourcedirs
#        srcdir = os.path.join(basedir, '2d-projector-analysis', experiment, 'FlyTracker/processed_mats') #relative_metrics'
#    elif assay == '38mm_dyad':
#        # src dir of processed .dfs from feat/trk.mat files (from relative_metrics.py)
#        srcdir = os.path.join(basedir, 'free-behavior-analysis', 'MF', 'FlyTracker', assay, 'processed_mats')

    assert os.path.exists(srcdir), "Specified source directory does not exist:\n {}".format(srcdir)
    assert len(os.listdir(srcdir)) > 0, "No files found in source directory:\n {}".format(srcdir)
    
    # set output dir
    destdir = os.path.split(srcdir)[0] 
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    out_fpath = os.path.join(destdir, 'relative_metrics.pkl')
    print("Saving output file to:\n  {}".format(out_fpath))
    
    # get local file for aggregated data
    if localdir is not None:
        out_fpath_local = os.path.join(localdir, 'relative_metrics.pkl')
        print("Also saving to localdir:\n  {}".format(out_fpath_local))

    # try reading if we don't want to create a new one
    if not create_new:
        if os.path.exists(out_fpath_local):
            df = pd.read_pickle(out_fpath_local)
            print("Loaded local processed data.")
        else:
            create_new = True
    print(create_new)
    #%
    # cycle over all the acquisition dfs in srcdir and make an aggregated df
    if create_new:
        df = util.load_aggregate_data_pkl(srcdir, mat_type='df')
        print("Found species: ", df['species'].unique())

        #% save to server loc
        df.to_pickle(out_fpath)
        print("Saved to:\n {}".format(out_fpath))

        # save local, too
        if localdir is not None:
            df.to_pickle(out_fpath_local)
            print("Saved local, too.")

    # summary of what we've got
    print(df[['species', 'acquisition']].drop_duplicates().groupby('species').count())


#%%
#plot_style='dark'
#putil.set_sns_style(plot_style, min_fontsize=24)
#bg_color = [0.7]*3 if plot_style=='dark' else 'w'

#assay = '2d-projector' #38mm-dyad'
#assay = '38mm-dyad'
#create_new = True

#minerva_base = '/Volumes/Julie'


#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process FlyTracker data for relative metrics.')
    parser.add_argument('--srcdir', type=str, help='Dir containing relative metrics mats (e.g., /Volumes/Juliana/.../processed_mats/)', default=None)    
    parser.add_argument('--new', type=bool, default=False, help='Create new processed data (default: False).')
    #parser.add_argument('--assay', type=str, default='2d-projector', help='Assay type (default: 2d-projector; alt: 38mm-dyad).')
    #parser.add_argument('--experiment', type=str, default='circle_diffspeeds', help='Experiment type (default: circle_diffspeeds).')
    parser.add_argument('--localdir', type=str, 
                        default=None,
                        #default='/Users/julianarhee/Documents/rutalab/projects/courtship/data/2d-projector/circle_diffspeeds/FlyTracker',
                        # default1 = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/2d-projector/circle_diffspeeds/FlyTracker'
                        # default2 = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/MF/38mm-dyad/FlyTracker'
                        help='Secondary *local* dir to save large .pkl output to.')
    
    args = parser.parse_args()

    #assay = args.assay
    create_new = args.new
    #basedir = args.basedir
    #experiment = args.experiment
    srcdir = args.srcdir
    localdir = args.localdir

    print(f"Aggregating relative metrics from:\n  {srcdir}")
    print(f"Creating new data: {create_new}")
    #aggregate_relative_metrics(basedir, assay=assay, experiment=experiment, create_new=create_new, localdir=localdir)   
    aggregate_relative_metrics(srcdir, create_new=create_new, localdir=localdir)


