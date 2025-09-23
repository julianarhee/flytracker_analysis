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
from numpy._typing import _128Bit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotting as putil
import utils as util
import theta_error as the

import transform_data.relative_metrics as rel

def transform_projector_data(acquisition_parentdir, acqs, processedmat_dir, 
                            movie_fmt='.avi',subdir=None, flyid1=0, flyid2=1,
                            create_new=False, reassign_acquisition_name=False):
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
        if reassign_acquisition_name:
            df_['date_fly'] = ['_'.join([f.split('-')[0], f.split('_')[1]]) for f in df_['file_name']]
            df_['acquisition'] = ['_'.join([a, b]) for a, b in df_[['date_fly', 'species']].values]
        else:
            df_['acquisition'] = acq #os.path.split(acq)[-1] 
        d_list.append(df_)

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
#%%
plot_style='white'
min_fontsize=18
putil.set_sns_style(style=plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

# Set directories
# Main assay containing all the acquisitions
assay = '38mm_dyad' #'38mm_projector'
#assay = '38mm_projector'

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
                & (meta0['genotype_male']=='P1a-CsChR')
                & (meta0['manipulation_male']=='no paint')].copy()
    
    # Check counts of NON painted
    meta1[(meta1['manipulation_male']=='no paint')].groupby(['species_male', 'stim_direction'])['file_name'].nunique()
    # Get list of acquisitions from meta data
    acqs = meta1['file_name'].unique()
    
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
    acqs = [os.path.split(a)[-1] for a in found_acqs if os.path.isdir(a)]
    print(len(acqs))
     
    
#%%
# Set output directories: set it upstream of processedmat_dir     
figdir = os.path.join(os.path.split(processedmat_dir)[0], 'steering_gain')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print("saving figures to {}".format(figdir))

#%% 
create_new = True #False
reassign_acquisition_name = False

if create_new:
    # Transform data 
    df0, errors = transform_projector_data(acquisition_parentdir, acqs,
                                        processedmat_dir, movie_fmt='.avi',
                                        subdir=None, flyid1=0, flyid2=1,
                                        create_new=False, 
                                        reassign_acquisition_name=reassign_acquisition_name)
    if assay == '38mm_projector':
        df0 = assign_paint_conditions(df0, meta1)
        
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
#%%
grouper = ['species', 'acquisition'] 
if assay == '38mm_projector':
    f1 = df0[(df0['id']==0) 
             & (df0['file_name'].isin(meta1['file_name']))
             & (df0['paint_coverage']=='none')].copy()
else:
    f1 = df0[df0['id']==0].copy()
#%%
# Add additional metrics
f1 = rel.calculate_angle_metrics_focal_fly(f1, winsize=5, grouper=grouper)

#%%
import matplotlib as mpl
# Check targ_ang_vel, is negative CW or CCW?
# Plot on polar plot
sp = 'Dyak'
curr_acqs = f1[f1['species']==sp]['acquisition'].unique()
acq = curr_acqs[4]
print(acq)
#plotd = f1[(f1['species']=='Dmel') & (f1['acquisition']==acq)].iloc[6855:6863] # target on fly's RIGHT (plots on upper-right hand side) | (starts y+ then gets smaller/down, angvel targ is neg)
#plotd = f1[(f1['species']=='Dmel') & (f1['acquisition']==acq)].iloc[7952:7974] #7930] # target on fly's RIGHT, then center (plots on upper-right hand side | starts y+ then to 0, angvel targ is neg))
s_ = 0
e_ = 15

#s_ = 120
#e_ = 150

s_ = 20598
e_ = 20598+15

f_ = f1[(f1['species']==sp) & (f1['acquisition']==acq)].copy()
plotd = f_.iloc[s_:e_] # target on fly's LEFT, then crosses ot right, target is moving rightward (starts on bottom-left, goes to upper-right | starts negative y, then positive y | angvel targ is positive )
#plotd = f1[(f1['species']==sp) & (f1['acquisition']==acq)].iloc[14588:14606] # target on fly's LEFT, then slightly on right, target is moving rightward (starts on bottom-left, goes to upper-right | starts negative y, then positive y | angvel targ is positive )

#% find where f_['targ_ang_vel'] meets threshold condition for consecutive frames
def find_consecutive_frames(series, threshold=0, condition='less', min_consecutive=10):
    """
    Find indices where a series meets threshold condition for at least min_consecutive consecutive frames.
    
    Parameters:
    - series: pandas Series to analyze
    - threshold: value to compare against
    - condition: 'less', 'greater', 'less_equal', 'greater_equal', 'equal'
    - min_consecutive: minimum number of consecutive frames required
    
    Returns a list of (start_idx, end_idx) tuples for each consecutive sequence.
    """
    if condition == 'less':
        is_condition = series < threshold
    elif condition == 'greater':
        is_condition = series > threshold
    elif condition == 'less_equal':
        is_condition = series <= threshold
    elif condition == 'greater_equal':
        is_condition = series >= threshold
    elif condition == 'equal':
        is_condition = series == threshold
    else:
        raise ValueError("condition must be one of: 'less', 'greater', 'less_equal', 'greater_equal', 'equal'")
    
    # Find transitions from False to True and True to False
    transitions = np.diff(np.concatenate([[False], is_condition, [False]]).astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    # Filter for sequences that are at least min_consecutive long
    consecutive_sequences = []
    for start, end in zip(starts, ends):
        if end - start >= min_consecutive:
            consecutive_sequences.append((start, end))
    
    return consecutive_sequences

# Find consecutive frames below threshold
threshold_low = -5  # Adjust this value as needed
min_consecutive = 10
condition = 'less'
low_sequences = find_consecutive_frames(f_['targ_ang_vel'], threshold=threshold_low, 
                                        condition='less', 
                                        min_consecutive=min_consecutive)
print(f"Found {len(low_sequences)} sequences where targ_ang_vel < {threshold_low} for {min_consecutive}+ consecutive frames:")
for i, (start, end) in enumerate(low_sequences):
    print(f"  Sequence {i+1}: frames {start} to {end-1} (length: {end-start})")
    print(f"    Values: {f_['targ_ang_vel'].iloc[start:end].values}")

# Find consecutive frames above threshold
threshold_high = 5  # Adjust this value as needed
condition = 'greater'
high_sequences = find_consecutive_frames(f_['targ_ang_vel'], threshold=threshold_high, 
                                         condition='greater', min_consecutive=min_consecutive)
print(f"\nFound {len(high_sequences)} sequences where targ_ang_vel > {threshold_high} for {min_consecutive}+ consecutive frames:")
for i, (start, end) in enumerate(high_sequences):
    print(f"  Sequence {i+1}: frames {start} to {end-1} (length: {end-start})")
    print(f"    Values: {f_['targ_ang_vel'].iloc[start:end].values}")

seq = high_sequences[4]
s_ = seq[0]#215284
e_ = seq[1] #215294
print("FRAMES: ", s_, e_)
plotd = f_.iloc[s_:e_]

plot_polar = True

if plot_polar:
    fig, axn = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
    # split colorbar at 0
    ax=axn[0]
    sns.scatterplot(data=plotd, x='targ_pos_theta', y='targ_pos_radius', 
                    hue='sec', ax=ax,
                    palette='viridis', edgecolor='none', alpha=1, legend=0)
    ax=axn[1]
    hue_norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
    sns.scatterplot(data=plotd, x='targ_pos_theta', y='targ_pos_radius', 
                    hue='targ_ang_vel', ax=ax,
                    palette='coolwarm', edgecolor='none', alpha=1, legend=0,
                    hue_norm=hue_norm)
    ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
    ax.set_xlabel('Targ. ang. vel. (rad/s)')
    ax.set_ylabel('Stim. direction')
else:
    fig, axn = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
    ax=axn[0]
    sns.scatterplot(data=plotd, x='targ_rel_pos_y', y='targ_rel_pos_x', 
                    hue='sec', ax=ax,
                    palette='viridis', edgecolor='none', alpha=1, legend=0)
    ax=axn[1]
    hue_norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)
    sns.scatterplot(data=plotd, x='targ_rel_pos_y', y='targ_rel_pos_x', 
                    hue='targ_ang_vel', ax=ax,
                    palette='coolwarm', edgecolor='none', alpha=1, legend=0,
                    hue_norm=hue_norm)
    for ax in axn:
        ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
        ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
    #ax.invert_yaxis()

fig.suptitle('{}: {} ({} - {})'.format(sp, acq, s_, e_), fontsize=8)

#%%


seq = low_sequences[2]
s_ = seq[0]#215284
e_ = seq[1] #215294
print("FRAMES: ", s_, e_)
plotd = f_.iloc[s_:e_]

hue_norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)

# Calculate 
fig, axn = plt.subplots(1, 3, subplot_kw={'projection': 'polar'})
ax=axn[0]
sns.scatterplot(data=plotd, x='targ_pos_theta', y='targ_pos_radius', 
                hue='sec', ax=ax,
                palette='viridis', edgecolor='none', alpha=1, legend=0)
ax=axn[1]

ax.set_title('hue: theta_error_dt_deg')
sns.scatterplot(data=plotd, x='theta_error', y='dist_to_other', ax=ax, 
                #hue='targ_ang_vel',
                hue='theta_error_dt_deg', hue_norm=hue_norm,
                palette='coolwarm', edgecolor='none', alpha=1, legend=0)
ax=axn[2]
ax.set_title('hue: targ_ang_vel')
sns.scatterplot(data=plotd, x='theta_error', y='dist_to_other', ax=ax, 
                    hue='targ_ang_vel', 
                    palette='coolwarm', edgecolor='none', alpha=1, legend=0,
                    hue_norm=hue_norm)
fig.suptitle('{}: {} ({} - {})'.format(sp, acq, s_, e_), fontsize=6)

fig, ax = plt.subplots()
sns.scatterplot(data=plotd, x='theta_error_deg', y='dist_to_other', 
                hue='sec', ax=ax,
                palette='viridis', edgecolor='none', alpha=1, legend=0)


#%%

fig, ax= plt.subplots()
sns.scatterplot(data=plotd, x='theta_error_dt_deg', y='targ_ang_vel_deg', ax=ax,
                palette='viridis', edgecolor='none', alpha=1, legend=0)
ax.set_xlabel('Theta error (deg)')
ax.set_ylabel('Targ. ang. vel. (rad/s)')
ax.set_title('{}: {} ({} - {})'.format(sp, acq, s_, e_))
ax.set_aspect(1)


#%%





#%%
f1 = the.shift_variables_by_lag(f1, lag=2)
f1['ang_vel_fly_shifted_abs'] = np.abs(f1['ang_vel_fly_shifted'])

# %%
#species_colors = ['plum', 'mediumseagreen']
species_palette = {'Dmel': 'plum', 'Dyak': 'mediumseagreen'}

#%%

filter_chase = True
f1['chasing_heuristic'] = False

if filter_chase:
    min_vel = 5
    max_targ_pos_theta = 180
    min_targ_pos_theta = -180
    max_facing_angle = 60
    min_wing_ang = 0
    max_dist_to_other = 20

    f1.loc[ #(f1['sex']=='m') #(tdf['id']%2==0)
            (f1['vel'] >= min_vel)
            & (f1['targ_pos_theta'] <= max_targ_pos_theta)
            & (f1['targ_pos_theta'] >= min_targ_pos_theta)
            & (f1['facing_angle'] <= max_facing_angle)
            & (f1['max_wing_ang'] >= min_wing_ang)
            & (f1['dist_to_other'] <= max_dist_to_other)
            , 'chasing_heuristic'] = True
    chase_ = f1[f1['chasing_heuristic']==True].copy()
else: 
    f1['chasing_heuristic'] = True
    chase_ = f1.copy()
     
#%% 
# Group by binned theta errors
def bin_by_object_position(chase_, start_bin = -180, end_bin=180, bin_size=20):
    '''
    Bin theta error by object position. Convert theta error to degrees.
    '''
    chase_['theta_error_deg'] = np.rad2deg(chase_['theta_error'])
    #start_bin = -180
    #end_bin = 180
    #bin_size = 20
    chase_['binned_theta_error'] = pd.cut(chase_['theta_error_deg'],
                                    bins=np.arange(start_bin, end_bin, bin_size),
                                    labels=np.arange(start_bin+bin_size/2,
                                            end_bin-bin_size/2, bin_size))    
    return chase_

#%%

# Count N frames total and N frames chasing_heuristic==True
if assay == '38mm_projector':
    grouper = ['species', 'acquisition', 'stim_direction']
else:
    grouper = ['species', 'acquisition']
chase_counts = the.count_chasing_frames(f1, 
                        grouper=grouper,
                        chase_var='chasing_heuristic')
print(chase_counts)
#%%
curr_cols = [c for c in chase_counts.columns if c != 'stim_direction']
mean_counts = chase_counts[curr_cols].groupby(['species', 'acquisition']).mean().reset_index()

#%%

fig, ax = plt.subplots()
sns.histplot(data=mean_counts, x='frac_frames_chasing',
             hue='species', bins=5,
             )

#%%
#top_mel = mean_counts[mean_counts['species']=='Dmel']\
#            .sort_values('frac_frames_chasing', ascending=False).head(4)
#top_yak = mean_counts[mean_counts['species']=='Dyak']\
#            .sort_values('frac_frames_chasing', ascending=False).head(4)
top_mel = mean_counts[(mean_counts['species']=='Dmel')\
                         & (mean_counts['frac_frames_chasing']>=0.5)].copy()
top_yak = mean_counts[(mean_counts['species']=='Dyak')\
                         & (mean_counts['frac_frames_chasing']>=0.5)].copy()
top_acqs = pd.concat([top_mel, top_yak])

tmp1 = chase_[chase_['acquisition'].isin(top_acqs['acquisition'].unique())].copy()
tmp1['courtship_level'] = 'high'

#bottom_mel = mean_counts[mean_counts['species']=='Dmel']\
#            .sort_values('frac_frames_chasing', ascending=True).iloc[5:10] #head(4)
bottom_mel = mean_counts[(mean_counts['species']=='Dmel')\
                         & (mean_counts['frac_frames_chasing']<0.3)
                         & (mean_counts['frac_frames_chasing']>=0.1)].copy()
bottom_yak = mean_counts[(mean_counts['species']=='Dyak')\
                         & (mean_counts['frac_frames_chasing']<0.3)
                         & (mean_counts['frac_frames_chasing']>=0.1)].copy()
#bottom_yak = mean_counts[mean_counts['species']=='Dyak']\
#            .sort_values('frac_frames_chasing', ascending=True).iloc[5:10] #head(4)
bottom_acqs = pd.concat([bottom_mel, bottom_yak])
bottom_acqs

tmp2 = chase_[chase_['acquisition'].isin(bottom_acqs['acquisition'].unique())].copy()
tmp2['courtship_level'] = 'low'

chase_ = pd.concat([tmp1, tmp2])

#%%
chase_ = bin_by_object_position(chase_, start_bin=-180, end_bin=180, bin_size=20)

# Get average ang vel across bins
grouper = ['species', 'acquisition', 'binned_theta_error', 'courtship_level']
if assay == '38mm_projector':
    grouper.append('stim_direction') 
    
yvar = 'ang_vel_fly_shifted'
avg_ang_vel = chase_.groupby(grouper)[yvar].mean().reset_index()

#%%

stim_palette = {'CW': 'cyan', 'CCW': 'magenta'}
start_bin = -180
end_bin = 180

# Get average ang vel across bins
grouper = ['species', 'acquisition', 'binned_theta_error']
if assay == '38mm_projector':
    grouper.append('stim_direction') 
yvar = 'ang_vel_fly_shifted'
avg_ang_vel_no_levels = chase_.groupby(grouper)[yvar].mean().reset_index()

fig, axn = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
for si, (sp, plotd) in enumerate(avg_ang_vel_no_levels.groupby('species')):
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
fig.suptitle('all courtship levels')    

figname = 'turns_by_objectpos_{}_CCW-CW_{}'.format(yvar, sp)
print(figdir, figname)


#%%
stim_palette = {'CW': 'cyan', 'CCW': 'magenta'}
start_bin = -180
end_bin = 180

for lvl, avg_ang_vel_lvl in avg_ang_vel.groupby('courtship_level'):
    fig, axn = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    for si, (sp, plotd) in enumerate(avg_ang_vel_lvl.groupby('species')):
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
    fig.suptitle('courtship level: {}'.format(lvl))    
    
figname = 'turns_by_objectpos_{}_CCW-CW_{}'.format(yvar, sp)
print(figdir, figname)
#plt.savefig(os.path.join(figdir, '{}.png'.format(figname))) 

#%% 
#%%
#avg_ang_vel = avg_ang_vel.dropna() 
start_bin = -180
end_bin = 180
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

   
# %%
fig, axn = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
for si, (sdir, plotd) in enumerate(avg_ang_vel.groupby('species')):
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