#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : dlc_to_flytracker.py
Created        : 2025/12/03 15:53:30
Project        : /Users/julianarhee/Repositories/flytracker_analysis
Author         : jyr
Email          : juliana.rhee@gmail.com
Last Modified  : 
'''
#%%
import sys
import os
import glob
import numpy as np
import pandas as pd

# import some custom funcs
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import utils as util
import plotting as putil
import dlc as dlc

# %%
fpath = '/Volumes/Juliana/free-behavior-analysis/MF/DeepLabCut/38mm-dyad-jyr-2024-02-23/20231213-1122_fly2_eleWT_5do_sh_eleWT_5do_gh_11_32_44MJPG-0001DLC_dlcrnetms5_38mm-dyadFeb23shuffle1_50000_el.h5'
# To read the hdf5 into a dataframe, can use pd.read_hdf:
# trk = pd.read_hdf(fpath)
 
# Load dataframe from .h5 filepath to match FlyTracker output 
df = dlc.load_dlc_df(fpath, fly1='fly1', fly2='fly2', fps=60, max_jump=6,\
            pcutoff=0.9, diff_speeds=False)
# %
# Combine fly data to 1 dataframe, convert to flytracker units
df = dlc.convert_dlc_to_flytracker(df, mm_per_pix=None)

# %%
# Specify path to video file to extract metadata
minerva = '/Volumes/Juliana'
prefix = '20231213-1122_fly2_eleWT_5do_sh_eleWT_5do_gh'
found_video_paths = glob.glob(os.path.join(minerva, 'courtship-videos', '38mm_dyad',
                       '{}*'.format(prefix), '*.avi'))
print(found_video_paths)
#%
video_fpath = found_video_paths[0]

#%% Transform from allocentric to egocentric coords
df = dlc.transform_dlc_to_relative(df, video_fpath=video_fpath, winsize=3)

# %%
# Load FlyTracker data
acquisition_dir = os.path.join(minerva, 'courtship-videos', '38mm_dyad', 
                               '{}*'.format(prefix))
ft_calib, ft_track, ft_feat = util.load_flytracker_data(acquisition_dir, calib_is_upstream=True, 
                               fps=60, subfolder='*', filter_ori=True)

# %%
