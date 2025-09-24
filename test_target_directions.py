#/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
 # @ Author: Juliana Rhee
 # @ Filename: test_target_directions.py
 # @ Create Time: 2025-01-27
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-01-27
 # @ Description: Test script for analyzing target directions and angular velocities
 '''

#%%
import os
import glob
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import plotting as putil
import utils as util
import theta_error as the

import transform_data.relative_metrics as rel

def get_species_from_acquisition_name(acq: str) -> str:
    """
    Extract 4-letter species code starting with 'D' from acquisition name.
    Example: '..._Dele-HK_...' -> 'Dele'. Falls back to known substrings.
    """
    m = re.search(r'_(D[a-zA-Z]{3})', acq)
    if m:
        return m.group(1)
    if 'mel' in acq:
        return 'Dmel'
    if 'yak' in acq:
        return 'Dyak'
    if 'ele' in acq:
        return 'Dele'
    return 'Unknown'

def wrap_pi(a):  # (-pi, pi]
    return (a + np.pi) % (2*np.pi) - np.pi

def circ_dist2(x, y):
    '''
    % r = circ_dist(alpha, beta)
    %   All pairwise difference x_i-y_j around the circle computed efficiently.
    %
    %   Input:
    %     alpha       sample of linear random variable
    %     beta       sample of linear random variable
    %
    %   Output:
    %     r       matrix with pairwise differences
    %
    % References:
    %     Biostatistical Analysis, J. H. Zar, p. 651
    %
    % PHB 3/19/2009
    %
    % Circular Statistics Toolbox for Matlab

    % By Philipp Berens, 2009
    % berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html

    np.angle:  Calculates the phase angle of a complex number or 
               array of complex numbers, returning the angle in radians by default
               Uses the atan2 function internally to compute the angle between 
               the positive real axis and the complex number on the complex plane
               Returns in range (-pi, pi].
    ''' 

    #np.tile(A, (m, n)) like repmat(A, m, n) in MATLAB

    at = np.angle( np.tile( np.exp(1j*x), (1, np.size(y)) ) / np.tile( np.exp(1j*y), (np.size(x), 1) ) )

    if len(at) == 1:
        return float(at)
    else:
        return at #np.exp(1j*x) / np.exp(1j*y) )


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
        hd = circ_dist2(v, i) #i, v) #* -1 # match matlab
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

#%%
plot_style='white'
min_fontsize=18
putil.set_sns_style(style=plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

# Set directories
# Main assay containing all the acquisitions
assay = '38mm_projector'

if assay == '38mm_projector':
    # Dropbox/source directory:
    acquisition_parentdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data/38mm_projector'
    has_meta_csv = True

    # Processed data directory (after running transform_data.py)
    processedmat_dir = '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_painted_eyes/FlyTracker/processed_mats'
    aggr_mat_fname ='transformed_projector_data.pkl' 

    local_fpath = None

# Output filepath for transformed data: saved in parent dir of processed_mats
aggr_mat_savedir = os.path.split(processedmat_dir)[0]
output_fpath = os.path.join(aggr_mat_savedir, aggr_mat_fname)
print("Output filepath: {}".format(output_fpath))
    
#%%
if has_meta_csv:
    if assay == '38mm_projector':
        meta_fpath = os.path.join(aggr_mat_savedir, 'courtship_free_behavior_data - 38 mm projector mel and yak controls.csv')

    print("Loading meta data from {}".format(meta_fpath))
    meta0 = pd.read_csv(meta_fpath)

    if assay == '38mm_projector':
        meta1 = meta0[(meta0['tracked in matlab and checked for swaps']==1)
                    & (meta0['exclude']==0) 
                    & (meta0['genotype_male']=='P1a-CsChR')
                    & (meta0['manipulation_male']=='no paint')].copy()     
        # Check counts of NON painted
        meta1[(meta1['manipulation_male']=='no paint')].groupby(['species_male', 'stim_direction'])['file_name'].nunique()
        # Get list of acquisitions from meta data
        acqs = meta1['file_name'].unique()

print("Found {} acqs.".format(len(acqs)))

#%%
# LOAD DATA
# --------------------------------
create_new = False
reassign_acquisition_name = assay=='38mm_projector' 
load_local = False

if load_local:
    assert local_fpath is not None, "Local file path is not set"

if create_new:
    # Transform data 
    df0, errors = transform_projector_data(acquisition_parentdir, acqs,
                                        processedmat_dir, movie_fmt='.avi',
                                        subdir=None, flyid1=0, flyid2=1,
                                        create_new=create_new, 
                                        reassign_acquisition_name=reassign_acquisition_name)
    if assay == '38mm_projector':
        df0 = assign_paint_conditions(df0, meta1)
        
    # Save transformed data
    df0.to_pickle(output_fpath)
    print("Transformed data saved to {}".format(output_fpath))  

else:
    # Load existing data
    if load_local and os.path.exists(local_fpath):
        print("Loading LOCAL data from: {}".format(local_fpath))
        df0 = pd.read_pickle(local_fpath)
    else:
        print("Loading existing data from: {}".format(output_fpath))
        df0 = pd.read_pickle(output_fpath)
    print("Loaded transformed.")

df0[['species', 'acquisition']].drop_duplicates().groupby('species')['acquisition'].count()
        
# Reset index 
df0.reset_index(drop=True, inplace=True)

#%%
# Select only focal fly
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

seq = high_sequences[0]
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

# DEBUGGING

#seq = low_sequences[2]
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

# DEBUG TARGET TURNING DIRECTION

acq_dir = os.path.join(acquisition_parentdir, acq)
# Load flytracker output
calib, trk, feat = util.load_flytracker_data(acq_dir, filter_ori=True)
# Transform data to relative coordinates
df_ = rel.get_metrics_relative_to_focal_fly(acq_dir,
                                        savedir=processedmat_dir,
                                        movie_fmt='.avi',
                                        mov_is_upstream=None,
                                        flyid1=0, flyid2=1,
                                        plot_checks=False,
                                        create_new=True,
                                        get_relative_sizes=False)

#%%
#f1_ = df_[df_['id']==1].iloc[0:5000]
#%%

# DEBUGGING

acq = '20240119-1517-fly7-yakWT_3do_sh_yakWT_3do_gh'
f1_ = f1[f1['acquisition']==acq].copy()

# Plot ang_vel from x and y
f1_['target_angular_position'] = -1*wrap_pi(np.arctan2(f1_['targ_centered_x'], f1_['targ_centered_y']))

f1_['target_angle_diff'] = get_heading_diff(f1_, 'target_angular_position',
                                                 invert_heading=False)
f1_['ang_vel_target'] = f1_['target_angle_diff'] / f1_['sec'].diff().mean()

#f1_['target_dir_deg'] = np.rad2deg(-1*f1_['target_dir'])

s_ix = 1500
e_ix = 3000
fig, axn = plt.subplots(1, 2, sharex=True, sharey=True)
ax=axn[0]
sns.scatterplot(data=f1_.iloc[s_ix:e_ix], ax=ax,
        x='targ_centered_x', y='targ_centered_y', #ax=ax,
        hue='sec', palette='viridis', edgecolor='none')
ax=axn[1]
hue_norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
sns.scatterplot(data=f1_.iloc[s_ix:e_ix], ax=ax,
        x='targ_centered_x', y='targ_centered_y', #ax=ax,
        hue='ang_vel_target', palette='coolwarm', 
        edgecolor='none', hue_norm=hue_norm)
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1), frameon=False)
ax.invert_yaxis()
