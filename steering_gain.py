#/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
 # @ Author: Juliana Rhee
 # @ Filename:
 # @ Create Time: 2025-07-03 15:12:50
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-07-03 15:12:56
 # @ Description:
 '''

#%%
import os
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotting as putil
import utils as util
import theta_error as the

import transform_data.relative_metrics as rel

def transform_projector_data(acquisition_parentdir, acqs, processedmat_dir, 
                            movie_fmt='.avi',subdir=None, flyid1=0, flyid2=1,
                            create_new=False, reassign_acquisition=False):
    """
    Load transformed projector data for specified acquisitions.
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
        except Exception as e:
            errors.append((acq, e))
            print("ERROR: {}".format(e))
            continue
        df_['file_name'] = os.path.split(acq)[-1]
        df_['species'] = 'Dmel' if 'mel' in acq else 'Dyak'
        if reassign_acquisition:
            df_['date_fly'] = ['_'.join([f.split('-')[0], f.split('_')[1]]) for f in df_['file_name']]
            df_['acquisition'] = ['_'.join([a, b]) for a, b in df_[['date_fly', 'species']].values]
        else:
            df_['acquisition'] = acq 
        d_list.append(df_)

    df0 = pd.concat(d_list)
    
    return df0, errors

def assign_paint_conditions(df0, meta):
    # Add all paint conditions
    for fn, df_ in df0.groupby('file_name'):
        currm = meta[meta['file_name']==fn]
        assert len(currm)>0, 'No meta data for {}'.format(fn)
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
#%%
plot_style='dark'
min_fontsize=18
putil.set_sns_style(style=plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

# Set directories
# Main assay containing all the acquisitions
#assay = '38mm_dyad' #'38mm_projector'
assay = '38mm_projector'

if assay == '38mm_projector':
    # Dropbox/source directory:
    acquisition_parentdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data/38mm_projector'

    # Processed data directory (after running transform_data.py)
    processedmat_dir = '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_painted_eyes/FlyTracker/processed_mats'

    # Output filepath for transformed data 
    output_fpath = os.path.join(os.path.split(processedmat_dir)[0], 
                                'transformed_projector_data.pkl')

elif assay == '38mm_dyad':

    acquisition_parentdir = '/Volumes/Giacomo/JAABA_classifiers/free_behavior'
    
    # Processed data directory (after running transform_data.py)
    processedmat_dir = '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker/processed_mats'

    # Output filepath for transformed data 
    # Can load older file, from aggregate_relative_metrics.py
    output_fpath = os.path.join(os.path.split(processedmat_dir)[0], 
                                'transformed_data_GG.pkl')     
    
#%%
if assay == '38mm_projector':
    # Get metadata from .csv since not all data processed/annotated
    # Load meta data: .csv file
    meta_fpath = glob.glob(os.path.join(acquisition_parentdir, '*.csv'))[0]
    print("Loading meta data from {}".format(meta_fpath))
    meta0 = pd.read_csv(meta_fpath)
    meta1 = meta0[(meta0['tracked in matlab and checked for swaps']==1)
                & (meta0['exclude']==0) 
                & (meta0['genotype_male']=='P1a-CsChR')].copy()
    meta1[(meta1['manipulation_male']=='no paint ')].groupby(['species_male', 'stim_direction'])['file_name'].nunique()

    # Get list of acquisitions from meta data
    acqs = meta1['file_name'].values
elif assay == '38mm_dyad':
#     found_meta_fpaths = glob.glob(os.path.join(acquisition_parentdir, '*.csv'))
#     if len(found_meta_fpaths) > 1:
#         print("WARNING: Multiple meta files found, using the first one.")
#     meta_fpath = found_meta_fpaths[0]
#     print("Loading meta data from {}".format(meta_fpath))
#     meta1 = pd.read_csv(meta_fpath)
    
    found_acqs = glob.glob(os.path.join(acquisition_parentdir, '202*'))
    # Get list of acquisitions    
    # Check if is directory
    acqs = [a for a in found_acqs if os.path.isdir(a)]
    print(len(acqs))
     
    
#%%
# Set output directories: set it upstream of processedmat_dir     
figdir = os.path.join(os.path.split(processedmat_dir)[0], 'steering_gain')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print("saving figures to {}".format(figdir))

#%% 
create_new = False
reassign_acquisition = assay == '38mm_projector'
if create_new:
    # Transform data 
    df0, errors = transform_projector_data(acquisition_parentdir, acqs,
                                        processedmat_dir, movie_fmt='.avi',
                                        subdir=None, flyid1=0, flyid2=1,
                                        create_new=False, 
                                        reassign_acquisition=reassign_acquisition)
    if assay == '38mm_projector':
        df0 = assign_paint_conditions(df0, meta0)
        
    # Save transformed data
    df0.to_pickle(output_fpath)
    print("Transformed data saved to {}".format(output_fpath))  
else:
    # Load existing data
    df0 = pd.read_pickle(output_fpath)
    print("Loaded transformed data from {}".format(output_fpath))

df0[['species', 'acquisition']].drop_duplicates().groupby('species')['acquisition'].count()

            
#%% Check
if assay == '38mm_projector':
    # Acquisitions loaded by .csv metadata, check that we have everything
    df0_fns = sorted(df0['file_name'].unique())
    meta_fns = sorted(meta1['file_name'].unique())

    if df0_fns != meta_fns:
        print("WARNING: File names in df0 and meta1 do not match!")
        missing_from_df0 = [f for f in meta_fns if f not in df0_fns]
        missing_from_meta = [f for f in df0_fns if f not in meta_fns]
        print("Missing from df0: {}".format(len(missing_from_df0)))
        print("Missing from meta1: {}".format(len(missing_from_meta))) 

# %% 
# Split by STIM_HZ
file_grouper = 'file_name' if assay == '38mm_projector' else 'acquisition'
df0 = util.add_stim_hz(df0, n_frames=24000, n_epochs=10, 
                       file_grouper=file_grouper)

grouper = ['species', 'acquisition'] 
if assay == '38mm_projector':
    f1 = df0[(df0['id']==0) & (df0['file_name'].isin(meta1['file_name']))].copy()
else:
    f1 = df0[df0['id']==0].copy()
#%%
# Add additional metrics
f1 = rel.calculate_angle_metrics_focal_fly(f1, winsize=5, grouper=grouper)

#%%
f1 = the.shift_variables_by_lag(f1, lag=2)
f1['ang_vel_fly_shifted_abs'] = np.abs(f1['ang_vel_fly_shifted'])

# %%
#species_colors = ['plum', 'mediumseagreen']
species_palette = {'Dmel': 'plum', 'Dyak': 'mediumseagreen'}

#%%
#chase_ = f1.copy()

min_vel = 5
max_targ_pos_theta = 180
min_targ_pos_theta = -180
max_facing_angle = 45
min_wing_ang = 0
max_dist_to_other = 20

chase_ = f1[ #(f1['sex']=='m') #(tdf['id']%2==0)
         (f1['vel'] >= min_vel)
        & (f1['targ_pos_theta'] <= max_targ_pos_theta)
        & (f1['targ_pos_theta'] >= min_targ_pos_theta)
        & (f1['facing_angle'] <= max_facing_angle)
        & (f1['max_wing_ang'] >= min_wing_ang)
        & (f1['dist_to_other'] <= max_dist_to_other)
        ].copy()
        
#%%
yvar = 'ang_vel_fly_shifted'
chase_['theta_error_deg'] = np.rad2deg(chase_['theta_error'])
start_bin = -180
end_bin = 180
bin_size = 20
chase_['binned_theta_error'] = pd.cut(chase_['theta_error_deg'],
                                bins=np.arange(start_bin, end_bin, bin_size),
                                labels=np.arange(start_bin+bin_size/2,
                                                    end_bin-bin_size/2, bin_size))    
# Get average ang vel across bins
grouper = ['species', 'acquisition', 'binned_theta_error' ]
if assay == '38mm_projector':
    grouper.append('stim_direction') 
avg_ang_vel = chase_.groupby(grouper)[yvar].mean().reset_index()

#%%
#avg_ang_vel = avg_ang_vel.dropna() 
fig, axn = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
for si, (sdir, plotd) in enumerate(avg_ang_vel.groupby('stim_direction')):
#plotd = avg_ang_vel[avg_ang_vel['stim_direction']=='CW'].copy()
    ax=axn[si]
    sns.lineplot(data=plotd, x='binned_theta_error', y=yvar,
                    hue='species', palette=species_palette, ax=ax,
                    errorbar='se', marker='o') #errwidth=0.5)
    ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
    ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
    ax.set_xticks(np.linspace(start_bin, end_bin, 5))
    ax.set_title(sdir)
    if si==0:
        ax.legend_.remove()
    else:
        sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1),
                        frameon=False, title='species', fontsize=min_fontsize-2)
    ax.set_xlabel('Object position (deg)')
    ax.set_ylabel('Ang. vel. shifted (rad/s)')
  
figname = 'turns_by_objectpos_{}_by_stimdir'.format(yvar)
print(figdir, figname)
#plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))

#%%
stim_palette = {'CW': 'cyan', 'CCW': 'magenta'}

fig, axn = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
for si, (sp, plotd) in enumerate(avg_ang_vel.groupby('species')):
    #plotd = avg_ang_vel[avg_ang_vel['species']=='Dyak'].copy()
    ax=axn[si]
    if assay == '38mm_projector':
        sns.lineplot(data=plotd, x='binned_theta_error', y=yvar, ax=ax,
                    hue='stim_direction', palette=stim_palette, 
                    errorbar='se', marker='o') #errwidth=0.5)
    else:
        sns.lineplot(data=plotd, x='binned_theta_error', y=yvar, ax=ax,
                    #hue='stim_direction', palette=stim_palette, ax=ax,
                    errorbar='se', marker='o') #errwidth=0.5)
    ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
    ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
    ax.set_xticks(np.linspace(start_bin, end_bin, 5))
    #ax.set_xticklabels(np.arange(start_bin+bin_size/2, end_bin-bin_size/2+1, bin_size))
    #ax.set_box_aspect(1)
    ax.set_title(sp)
    if assay == '38mm_projector':
        if si==0:
            ax.legend_.remove()
        else:
            sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1),
                            frameon=False, title='movement dir', fontsize=min_fontsize-2)
    ax.set_xlabel('Object position (deg)')
    ax.set_ylabel('Ang. vel. shifted (rad/s)')
    #putil.label_figure(fig, figid) 
    
figname = 'turns_by_objectpos_{}_CCW-CW_{}'.format(yvar, sp)
print(figdir, figname)
#plt.savefig(os.path.join(figdir, '{}.png'.format(figname))) 
    
# %%
