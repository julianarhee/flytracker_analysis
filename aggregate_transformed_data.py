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


from relative_metrics import load_processed_data
import utils as util
import plotting as putil
import argparse

#%%

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
    parser.add_argument('--basedir', type=str, help='Rootdir of src files (default: /Volumes/Julie)', default='/Volumes/Julie')    
    parser.add_argument('--new', type=bool, default=False, help='Create new processed data (default: False).')
    parser.add_argument('--assay', type=str, default='2d-projector', help='Assay type (default: 2d-projector; alt: 38mm-dyad).')
    parser.add_argument('--experiment', type=str, default='circle_diffspeeds', help='Experiment type (default: circle_diffspeeds).')
    parser.add_argument('--localdir', type=str, 
                        default='/Users/julianarhee/Documents/rutalab/projects/courtship/data/2d-projector/circle_diffspeeds/FlyTracker',
                        # default1 = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/2d-projector/circle_diffspeeds/FlyTracker'
                        # default2 = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/MF/38mm-dyad/FlyTracker'
                        help='Secondary *local* dir to save large .pkl output to.')
    
    args = parser.parse_args()

    assay = args.assay
    create_new = args.new
    basedir = args.basedir
    experiment = args.experiment
    localdir = args.localdir

    #%%
    # Specify data source directory based on server filetree
    if assay == '2d-projector':
        # Set sourcedirs
        srcdir = os.path.join(basedir, '2d-projector-analysis', experiment, 'FlyTracker/processed_mats') #relative_metrics'
    elif assay == '38mm_dyad':
        # src dir of processed .dfs from feat/trk.mat files (from relative_metrics.py)
        srcdir = os.path.join(basedir, 'free-behavior-analysis', 'MF', 'FlyTracker', assay, 'processed_mats')

    # set output dir
    destdir = os.path.split(srcdir)[0] 
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    print(destdir)
    out_fpath = os.path.join(destdir, 'relative_metrics.pkl')

    # get local file for aggregated data
    out_fpath_local = os.path.join(localdir, 'relative_metrics.pkl')
    print(out_fpath_local)

    # try reading if we don't want to create a new one
    if not create_new:
        if os.path.exists(out_fpath_local):
            df = pd.read_pickle(out_fpath_local)
            print("Loaded local processed data.")
        else:
            create_new = True

    print(create_new)

    #%%
    # cycle over all the acquisition dfs in srcdir and make an aggregated df
    if create_new:
        df = util.load_aggregate_data_pkl(srcdir, mat_type='df')
        print(df['species'].unique())

        #% save to server loc
        df.to_pickle(out_fpath)
        print(out_fpath)

        # save local, too
        df.to_pickle(out_fpath_local)

    # summary of what we've got
    print(df[['species', 'acquisition']].drop_duplicates().groupby('species').count())

