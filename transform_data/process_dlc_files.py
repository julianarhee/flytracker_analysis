#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Procsses all dlc .h5 files and runs dlc.load_and_transform_dlc() on each.
Saves individual .pkl files to ./2d-projector-analysis/DeepLabCut/processed directory
Saves aggregate processed.pkl file to ./2d-projector-analysis/DeepLabCut directory
Saves a copy to local directory /Users/julianarhee/Documents/rutalab/projects/courtship/data/2d-projector/DLC
"""
#%%
import sys
import os
import sys
import glob

import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import pylab as pl
import matplotlib as mpl

import matplotlib.gridspec as gridspec
import traceback

# import some custom funcs
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
#from relative_metrics import load_processed_data
import utils as util
import plotting as putil
import dlc as dlc
import plot_dlc_frames as pdlc

#%%

# DLC
minerva_base='/Volumes/Julie'
projectname='projector-1dot-jyr-2024-02-18' 
procdir = os.path.join(minerva_base, '2d-projector-analysis/DeepLabCut', projectname)
print(len(os.listdir(procdir)))

#% get src paths
localroot = '/Users/julianarhee/DeepLabCut' # all these functions assume this is the local rootdir
#% Look at 1 data file
analyzed_dir = dlc.get_dlc_analysis_dir(projectname=projectname)

all_fpaths = dlc.get_all_h5_files(analyzed_dir)
print(len(all_fpaths))


#%%
assay = '2d-projector'
flyid = 'fly' # double check in the plots for abdomen lengths
dotid = 'single'
fps = 60  # Hz
max_jump = 6
pcutoff=0.8 #0.99


destdir = os.path.join(procdir.split(projectname)[0], 'processed')
if not os.path.exists(destdir):
    os.makedirs(destdir)
print("Saving processed files to: {}".format(destdir))

#%%

errors = []
d_list = []
for fpath in all_fpaths:
    acq = dlc.get_acq_from_dlc_fpath(fpath)
    print(acq)
    out_f = os.path.join(destdir, '{}.pkl'.format(acq))   

    create_new=False
    if os.path.exists(out_f):
        try:
            df_ = pd.read_pickle(out_f)
            assert df_.shape[0] > 0, "No data: {}".format(acq)
        except Exception as e:
            traceback.print_exc()
            create_new=True

    if create_new:
        try:
            df_ = dlc.load_and_transform_dlc(fpath, winsize=10,
                                     localroot=localroot, projectname=projectname,
                                     assay=assay, flyid=flyid, dotid=dotid, fps=fps, max_jump=max_jump, pcutoff=pcutoff)

            df_.to_pickle(out_f)

        except Exception as e:
            traceback.print_exc()
            errors.append(acq)
            continue 
    #del aq
    #aq = [a for a in df_['acquisition'].unique() if isinstance(a, str) and a.startswith('20')][0]
    df_['acquisition'] = acq

    d_list.append(df_)

#%%    
#drop_weird = [d.fillna(0) for d in d_list if d.shape[0]==48000]
#df = pd.concat(drop_weird, axis=0)
#check_these = ['20240212-1230_fly3_Dmel_sP1-ChR_3do_sh_4x4',
# '20240215-1722_fly1_Dmel_sP1-ChR_3do_sh_6x6']

['20240215-1722_fly1_Dmel_sP1-ChR_3do_sh_6x6']
#df = pd.concat([df, d], ignore_index=True).reset_index(drop=True)

df = pd.concat(d_list, ignore_index=False)

#%%
#dd_list = []
#for i, v in enumerate(d_list):
#    nr = v.shape[0]
#    aq = [a for a in v['acquisition'].unique() if isinstance(a, str) and a.startswith('20')][0]
#    v['acquisition'] = aq
#
#    dd_list.append(v)
#
#df = pd.concat(dd_list, axis=0)
#print(df[df['acquisition'] == acq].shape)


#%% Save
out_fpath = os.path.join(os.path.split(destdir)[0], 'processed.pkl')
df.to_pickle(out_fpath)
print("Saved to: {}".format(out_fpath))

#%%
localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/2d-projector/DLC'
out_fpath_local = os.path.join(localdir, 'processed.pkl')

df.to_pickle(out_fpath_local)
print("Saved to: {}".format(out_fpath_local))


# %%



