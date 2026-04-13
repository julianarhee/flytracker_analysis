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

import matplotlib.pyplot as plt
import seaborn as sns

import libs.utils as util
import libs.plotting as putil
import libs.theta_error as terr
import transform_data.relative_metrics as rem

from analyses.preprocessing.src.add_ft_actions import add_ft_actions
import gain_funcs as gf

#%%
plot_style='dark'
min_fontsize=12
putil.set_sns_style(style=plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

# %%
#species_colors = ['plum', 'mediumseagreen']
species_palette = {'Dmel': 'plum', 
                   'Dyak': 'mediumseagreen',
                   'Dele': 'aquamarine'}

# %%
from datetime import datetime
#today = datetime.now().strftime('%Y%m%d')
#print(today)
#%%
# Acquisition dir:
acqdir = '/Volumes/Juliana/Caitlin_RA_data/Caitlin_projector'
#has_meta_csv = True

# Processed data directory (after running transform_data.py)
procdir = '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_painted_eyes/FlyTracker/processed_mats'
#aggr_mat_fname ='transformed_projector_data.pkl' 

# local_fpath = None

# Basedir, for figures
basedir = os.path.split(procdir)[0]

# figdir 
figdir = os.path.join(basedir, 'plot_gain_projector')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print("saving figures to {}".format(figdir))

# %%
# Get metadata
meta_fpath = os.path.join(basedir, 'courtship_free_behavior_data - 38mm_projector.csv')
assert os.path.exists(meta_fpath), "Meta file does not exist: {}".format(meta_fpath)
# Load meta data
meta0 = pd.read_csv(meta_fpath)
meta = meta0[(meta0['tracked in matlab and checked for swaps']==1)
           & (meta0['exclude']==0) 
           & (meta0['annotated']==1)
           & (meta0['manipulation_male']=='no paint')
           & (meta0['painted']=='none')
           & (meta0['genotype_male']=='P1a-CsChR')
           & (meta0['calibration']==0)].copy()
# date-fly
meta['date_fly'] = ['_'.join( [f.split('-')[0], \
                        f.split('_')[1]] ) for f in meta['file_name']]
print(meta.shape)
# %
n_species = meta.groupby('species_male', as_index=False)['date_fly'].nunique()
print(n_species)

#%%
create_new = False
create_aggregate = False

# Check for existing aggregate:
found_aggr_fpaths = glob.glob(os.path.join(basedir, f'2026*_transformed_yak_mel_controls_chasing.parquet'))
for fpath in sorted(found_aggr_fpaths):
    print(fpath)

if len(found_aggr_fpaths) > 0:
    try:
        df0 = pd.read_parquet(found_aggr_fpaths[0])
        print("Loaded existing aggregate from {}".format(found_aggr_fpaths[0]))
        assert 'stim_direction' in df0.columns, "stim_direction column not found in dataframe, remaking"
    except Exception as e:
        print(f"Error loading aggregate from {found_aggr_fpaths[0]}: {e}")
        print("Remaking aggregate")
        create_aggregate = True
else:
    print("No existing aggregate found, transforming data")
    create_aggregate = True

if create_aggregate:
    today = datetime.now().strftime('%Y%m%d')
    aggr_fpath = os.path.join(procdir, f'{today}_transformed_yak_mel_controls_chasing.parquet')

#%%
# Transform data 
all_acqs = meta['file_name'].unique()

if create_aggregate:
    #%
    # Add actions
    df_list = []; errs=[];
    for acq in all_acqs: #curracqs:
        try:
            df_ = add_ft_actions(procdir, acqdir, acq, verbose=False)

            # Assign stimulus direction from meta
            currm = meta[meta['file_name']==acq]
            assert len(currm)>0, 'No meta data for {}'.format(acq)
            assert len(currm)==1, 'Multiple meta data for {}'.format(acq)
            stim_dir = currm['stim_direction'].unique()[0] #fn.split('_')[-1]
            df_['stim_direction'] = stim_dir

            # Add file info 
            df_['file_name'] = os.path.split(acq)[-1]
            if 'species' not in df_.columns: 
                df_['species'] = util.get_species_from_acquisition_name(acq)
            #if reassign_acquisition_name:
            df_['date_fly'] = ['_'.join([f.split('-')[0], f.split('_')[1]]) for f in df_['file_name']]
            df_['acquisition'] = ['_'.join([a, b]) for a, b in df_[['date_fly', 'species']].values]

            df_list.append(df_)
        except Exception as e:
            print(f"Error adding actions for {acq}: {e}")
            errs.append((acq, e))
            continue
        # df.to_parquet(os.path.join(procdir, f'{acq}_df.parquet'))
    df0 = pd.concat(df_list)

    #%
    # Save transformed data
    #today_str = datetime.now().strftime('%Y%m%d')
    #aggr_fpath = os.path.join(procdir, f'{today}_transformed_yak_mel_controls_chasing.parquet')
    df0.to_parquet(aggr_fpath, engine='pyarrow', compression='snappy')
    print("Transformed data saved to {}".format(aggr_fpath)) 

#%%
if 'ang_vel_abs' not in df0.columns:
    df0['ang_vel_abs'] = np.abs(df0['ang_vel'])
#    df0, errors = gf.transform_projector_data(acqdir, all_acqs,
#                            procdir, movie_fmt='.avi',
#                            subdir=None, flyid1=0, flyid2=1,
#                            create_new=create_new, 
#                            reassign_acquisition_name=True)

#%%
# errors - no actions:
'20250514-1051_fly1_Dyak-p1_5do_gh_2dR_ccw'
'20250710-1505_fly1_Dyak-p1_3do_gh_2dR'
'20250910-1338_fly3_Dyak-p1_5do_gh_2dR' 
'20250910-1409_fly5_Dyak-p1_5do_gh_2dR'


# %%
# Assigm stimulus direction from meta
f1 = df0[(df0['id']==0)].copy()
#%%
# Assign "pr_direction" progressive or regressive:
# stim_direction is CCW, and target_position < 0: progressive
# stim_direciton is CCW, and target_position > 0: regressive
# is CW and > 0: progressive
# is CW and < 0: regressive
f1['pr_direction'] = None
f1.loc[(f1['stim_direction']=='CCW') & (f1['targ_pos_theta']<0), 'pr_direction'] = 'progressive'
f1.loc[(f1['stim_direction']=='CCW') & (f1['targ_pos_theta']>0), 'pr_direction'] = 'regressive'
f1.loc[(f1['stim_direction']=='CW') & (f1['targ_pos_theta']>0), 'pr_direction'] = 'progressive'
f1.loc[(f1['stim_direction']=='CW') & (f1['targ_pos_theta']<0), 'pr_direction'] = 'regressive'
#%%
f1 = terr.shift_variables_by_lag(f1, lag=12, file_grouper='file_name')
f1['ang_vel_fly_shifted_abs'] = np.abs(f1['ang_vel_fly_shifted'])
f1['ang_vel_fly_shifted_deg'] = np.rad2deg(f1['ang_vel_fly_shifted'])
f1['ang_vel_fly_deg'] = np.rad2deg(f1['ang_vel_fly'])

#%%
f1['theta_error_deg'] = np.rad2deg(f1['theta_error'])
f1 = gf.bin_by_object_position(f1, start_bin=-180, end_bin=180, bin_size=20)
f1['binned_theta_error_num'] = pd.to_numeric(f1['binned_theta_error'], errors='coerce')

f1.reset_index(drop=True, inplace=True)

#%%

# filter chase
chasedf = f1[f1['chasing']==0].copy()

chasedf = f1[(f1['theta_error_deg']<135)
           & (f1['theta_error_deg']>-135)
           & (f1['vel']>5)
           ].copy()
#chasedf = gf.bin_by_object_position(chasedf, start_bin=-180, end_bin=180, bin_size=20)
chasedf.reset_index(drop=True, inplace=True)

#chasedf['binned_theta_error_num'] = chasedf['binned_theta_error'].astype(int)
chasedf['binned_theta_error_num'] = pd.to_numeric(chasedf['binned_theta_error'], errors='coerce')

#%%a
# PLOT ALL
hue_var = 'pr_direction'
pr_palette = {'progressive': 'darkgreen', 'regressive': 'purple'}
cw_palette = {'CW': 'cyan', 'CCW': 'magenta'}
stim_palette = pr_palette if hue_var == 'pr_direction' else cw_palette
hue_str = 'PRO-REG' if hue_var == 'pr_direction' else 'CW-CCW'
start_bin = -180
end_bin = 180
lw = 2

figsize = (8, 4)
n_species = chasedf['species'].nunique()

# Get average ang vel across bins
grouper = ['species', 'acquisition', 'binned_theta_error',
           'binned_theta_error_num']
#if assay == '38mm_projector':
grouper.append(hue_var) 


yvar = 'ang_vel_fly_shifted_deg'

# Get average ang vel across individuals
mean_no_levels = chasedf.groupby(grouper)[yvar].mean().reset_index()
#avg_ang_vel_no_levels = chase_.copy() #groupby(grouper)[yvar].mean().reset_index()
err = 'se'
print(mean_no_levels.groupby('species')['acquisition'].nunique())

fig, axn = plt.subplots(1, n_species, figsize=figsize,
                        sharex=True, sharey=True)
for si, (sp, plotd) in enumerate(mean_no_levels.groupby('species')):
    #plotd = avg_ang_vel[avg_ang_vel['species']=='Dyak'].copy()
    if n_species==1:
        ax=axn
    else:
        ax=axn[si]

    sns.lineplot(data=plotd[plotd['binned_theta_error_num']<0], 
                x='binned_theta_error', y=yvar, ax=ax,
                hue=hue_var, palette=stim_palette, 
                errorbar=err, marker='o', 
                markersize=0, markeredgewidth=0,
                err_style='bars', legend=0, lw=lw, err_kws={'linewidth': lw})
    sns.lineplot(data=plotd[plotd['binned_theta_error_num']>0], 
                x='binned_theta_error', y=yvar, ax=ax,
                hue=hue_var, palette=stim_palette, 
                errorbar=err, marker='o', 
                markersize=0, markeredgewidth=0,
                err_style='bars', lw=lw, err_kws={'linewidth': lw},
                legend=si==1)
    # if legend exist, move it
    if si==1:
        sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1),
                        frameon=False, title='movement dir', 
                        fontsize=min_fontsize-2)
    ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
    ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
    ax.set_xticks(np.linspace(start_bin, end_bin, 5))
    ax.set_title(sp)

#%%
currfly = '20250703_fly1'

curracqs = meta[meta['date_fly']==currfly]['file_name'].unique()
assert len(curracqs) > 0, f"No acquisitions found for fly '{currfly}' — check currfly and meta filters."
#currf = '20250703-1510_fly1_Dmel-p1_1do_gh_1dR'

