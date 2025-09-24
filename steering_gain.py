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
import re

import numpy as np
from numpy._typing import _128Bit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
            assert len(df_['targ_centered_x'].unique()) > 1, "Bad targ calculation"
        except Exception as e:
            errors.append((acq, e))
            print("ERROR: {}".format(e))
            continue
        df_['file_name'] = os.path.split(acq)[-1]
        # Get species from acquisition
        
        df_['species'] = get_species_from_acquisition_name(acq)
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
    #chase_['theta_error_deg'] = np.rad2deg(chase_['theta_error'])
    #start_bin = -180
    #end_bin = 180
    #bin_size = 20
    chase_['binned_theta_error'] = pd.cut(chase_['theta_error_deg'],
                                    bins=np.arange(start_bin, end_bin, bin_size),
                                    labels=np.arange(start_bin+bin_size/2,
                                            end_bin-bin_size/2, bin_size))    
    return chase_

#%%
plot_style='white'
min_fontsize=24
putil.set_sns_style(style=plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

# Set directories
# Main assay containing all the acquisitions
#assay = '38mm_dyad_GG' #'38mm_projector'
#assay = '38mm_dyad_Dele'
assay = '38mm_projector'

if assay == '38mm_projector':
    # Dropbox/source directory:
    acquisition_parentdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data/38mm_projector'
    has_meta_csv = True

    # Processed data directory (after running transform_data.py)
    processedmat_dir = '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_painted_eyes/FlyTracker/processed_mats'
    aggr_mat_fname ='transformed_projector_data.pkl' 

    local_fpath = None

elif assay == '38mm_dyad_GG':
    # If using Giacomo's mel/yak data
    acquisition_parentdir = '/Volumes/Giacomo/JAABA_classifiers/free_behavior'
    has_meta_csv = False # is True, but for now leave it

    # Processed data directory (after running transform_data.py)
    processedmat_dir = '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker/processed_mats'
    aggr_mat_fname ='transformed_data_GG.pkl' 

    # set local
    localdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/free_behavior_analysis/38mm_dyad/MF/FlyTracker'
    local_fpath = os.path.join(localdir, aggr_mat_fname)

elif assay == '38mm_dyad_Dele':
    # If using Caitlin's elehk data
    acquisition_parentdir = os.path.join('/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee', 
                                'caitlin_data', 'Caitlin_elehk_38mm')
    has_meta_csv = True

    # Processed data directory (after running transform_data.py)
    processedmat_dir = '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker/processed_mats'
    aggr_mat_fname ='transformed_data_Dele-HK.pkl' 

    local_fpath = None

# Output filepath for transformed data: saved in parent dir of processed_mats
# Can load older file, from aggregate_relative_metrics.py
aggr_mat_savedir = os.path.split(processedmat_dir)[0]
output_fpath = os.path.join(aggr_mat_savedir, aggr_mat_fname)
print("Output filepath: {}".format(output_fpath))
    
#%%
if has_meta_csv:

    if assay == '38mm_projector':
        meta_fpath = os.path.join(aggr_mat_savedir, 'courtship_free_behavior_data - 38 mm projector mel and yak controls.csv')

    else:
        # Get metadata from .csv since not all data processed/annotated
        # Load meta data: .csv file
        meta_fpath = glob.glob(os.path.join(acquisition_parentdir, '*.csv'))[0]
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

    elif assay == '38mm_dyad_Dele':
        meta1 = meta0[(meta0['tracked']==1)
                    & (meta0['annotated']==1) 
                    & (meta0['courtship']==1) ].copy()     
        # Get list of acquisitions from meta data
        acqs = meta1['acquisition'].unique()
    
elif assay == '38mm_dyad_GG':    
    found_acqs = glob.glob(os.path.join(acquisition_parentdir, '202*'))
    # Get list of acquisitions    
    # Check if is directory
    acqs = [os.path.split(a)[-1] for a in found_acqs if os.path.isdir(a)]

print("Found {} acqs.".format(len(acqs)))
    
#%%
# Set output directories: set it upstream of processedmat_dir     
figdir = os.path.join(os.path.split(processedmat_dir)[0], 'steering_gain')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print("saving figures to {}".format(figdir))

figid = acquisition_parentdir

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
if assay == '38mm_projector':
    file_grouper = 'file_name' if assay == '38mm_projector' else 'acquisition'
    df0 = util.add_stim_hz(df0, n_frames=24000, n_epochs=10, 
                        file_grouper=file_grouper)

#%% 
# TMP: ADD TARGET_VEL
if 'target_vel' not in df0.columns:
    df0['target_vel'] = np.nan
    df0['target_ang_vel'] = np.nan

    for acq, df_ in df0.groupby('acquisition'):
        f1_ = df_[df_['id']==0].copy()
        f2_ = df_[df_['id']==1].copy()
        # Add target info for fly1
        df_.loc[df_['id']==0, 'target_vel'] = f2_['vel'].values
        df_.loc[df_['id']==0, 'target_ang_vel'] = f2_['ang_vel'].values
        # Add target info for fly2
        df_.loc[df_['id']==1, 'target_vel'] = f1_['vel'].values
        df_.loc[df_['id']==1, 'target_ang_vel'] = f1_['ang_vel'].values
        # Add to main df0
        df0.loc[df0['acquisition']==acq, 'target_vel'] = df_['target_vel'].values
        df0.loc[df0['acquisition']==acq, 'target_ang_vel'] = df_['target_ang_vel'].values

#%%
# Select only focal fle
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
# Calculate TARGET's STIMULUS DIRECTION
# =====================================
calculate_target_direction = assay != '38mm_projector'

if calculate_target_direction:
    print("Calculating target direction for {}".format(assay))

    f_list = []
    for a, f1_ in f1.groupby('acquisition'):
        #print(a, f1_.shape)
        # Plot ang_vel from x and y
        f1_['target_angular_position'] = -1*wrap_pi(np.arctan2(f1_['targ_centered_x'], f1_['targ_centered_y']))

        f1_['target_angle_diff'] = get_heading_diff(f1_, 'target_angular_position',
                                                    invert_heading=False)
        f1_['ang_vel_target'] = f1_['target_angle_diff'] / f1_['sec'].diff() #.mean()
        f_list.append(f1_)

    f1 = pd.concat(f_list, ignore_index=True)

    # Plot ang_vel from x and y
    #f1['target_dir'] = wrap_pi(np.arctan2(f1['targ_centered_y'], f1['targ_centered_x']))
    #f1['target_dir_deg'] = np.rad2deg(f1['target_dir'])

    f1.loc[f1['ang_vel_target']>0, 'stim_direction'] = 'CW'
    f1.loc[f1['ang_vel_target']<0, 'stim_direction'] = 'CCW'

    # Calculate target direction
    f1.loc[f1['targ_ang_vel']>0, 'rel_stim_direction'] = 'CW'
    f1.loc[f1['targ_ang_vel']<0, 'rel_stim_direction'] = 'CCW'


#%%
f1 = the.shift_variables_by_lag(f1, lag=2)
f1['ang_vel_fly_shifted_abs'] = np.abs(f1['ang_vel_fly_shifted'])
f1['ang_vel_fly_shifted_deg'] = np.rad2deg(f1['ang_vel_fly_shifted'])
f1['ang_vel_fly_deg'] = np.rad2deg(f1['ang_vel_fly'])

# %%
#species_colors = ['plum', 'mediumseagreen']
species_palette = {'Dmel': 'plum', 
                   'Dyak': 'mediumseagreen',
                   'Dele': 'aquamarine'}

#%%

filter_chase = True
# --------------------------------

f1['chasing_heuristic'] = False
if filter_chase:
    min_vel = 3
    max_targ_pos_theta = 250 #180
    min_targ_pos_theta = -250 #180
    max_facing_angle = 120
    min_wing_ang = 0
    max_dist_to_other = 30

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
# Count N frames total and N frames chasing_heuristic==True
#if assay == '38mm_projector':
grouper = ['species', 'acquisition'] #, 'stim_direction']
#else:
#    grouper = ['species', 'acquisition']
chase_counts = the.count_chasing_frames(f1, 
                        grouper=grouper,
                        chase_var='chasing_heuristic')
#print(chase_counts)
#%
curr_cols = [c for c in chase_counts.columns if c != 'stim_direction']
mean_counts = chase_counts[curr_cols].groupby(['species', 'acquisition']).mean().reset_index()

print(mean_counts)

#%%

fig, ax = plt.subplots()
sns.histplot(data=mean_counts, x='frac_frames_chasing',
             hue='species', bins=20, 
             )

#%%
#top_mel = mean_counts[mean_counts['species']=='Dmel']\
#            .sort_values('frac_frames_chasing', ascending=False).head(4)
#top_yak = mean_counts[mean_counts['species']=='Dyak']\
#            .sort_values('frac_frames_chasing', ascending=False).head(4)
#for sp, means_ in mean_counts.groupby('species'):

chase_['courtship_level']  = 'neither'
split_by_acquisition = True
if split_by_acquisition and filter_chase:
    top_acqs = mean_counts[ mean_counts['frac_frames_chasing']>=0.5].copy()
    tmp1 = chase_[chase_['acquisition'].isin(top_acqs['acquisition'].unique())].copy()

    bottom_acqs = mean_counts[ (mean_counts['frac_frames_chasing']<=0.3) 
                            & (mean_counts['frac_frames_chasing']>=0.05)].copy()
    tmp2 = chase_[chase_['acquisition'].isin(bottom_acqs['acquisition'].unique())].copy()

    chase_.loc[tmp1.index, 'courtship_level'] = 'high' 
    chase_.loc[tmp2.index, 'courtship_level'] = 'low'
#else:
    # SPlit by courtship bout
    # actually, can't do this because there is no
    # "tracking_index" in a given bout
chase_['theta_error_deg'] = np.rad2deg(chase_['theta_error'])
chase_ = bin_by_object_position(chase_, start_bin=-180, end_bin=180, bin_size=20)
chase_.reset_index(drop=True, inplace=True)

#%%
# PLOT ALL
stim_palette = {'CW': 'cyan', 'CCW': 'magenta'}
start_bin = -180
end_bin = 180

# Get average ang vel across bins
grouper = ['species', 'acquisition', 'binned_theta_error']
#if assay == '38mm_projector':
grouper.append('stim_direction') 

yvar = 'ang_vel_fly_shifted'

avg_ang_vel_no_levels = chase_.groupby(grouper)[yvar].mean().reset_index()
n_species = chase_['species'].nunique()
species_str = '_'.join(chase_['species'].unique())

if min_fontsize==6:
    figsize = (n_species*3, 4)
else:
    figsize = (n_species*3.5, 6)

fig, axn = plt.subplots(1, n_species, figsize=figsize,
                        sharex=True, sharey=True)
for si, (sp, plotd) in enumerate(avg_ang_vel_no_levels.groupby('species')):
    #plotd = avg_ang_vel[avg_ang_vel['species']=='Dyak'].copy()
    if n_species==1:
        ax=axn
    else:
        ax=axn[si]

    # if assay == '38mm_projector':
    sns.lineplot(data=plotd, x='binned_theta_error', y=yvar, ax=ax,
                    hue='stim_direction', palette=stim_palette, 
                    errorbar='ci', marker='o', 
                    markersize=6, markeredgewidth=0) #errwidth=0.5)
    #else:
#         sns.lineplot(data=plotd, x='binned_theta_error', y=yvar, ax=ax,
#                     #hue='stim_direction', palette=stim_palette, ax=ax,
#                     errorbar='se', marker='o') #errwidth=0.5)
    ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
    ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
    ax.set_xticks(np.linspace(start_bin, end_bin, 5))
    #ax.set_xticklabels(np.arange(start_bin+bin_size/2, end_bin-bin_size/2+1, bin_size))
    #ax.set_box_aspect(1)
    ax.set_title(sp)
    #if assay == '38mm_projector':
    if si==0 and n_species>1:
        ax.legend_.remove()
    else:
        sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1),
                        frameon=False, title='movement dir', fontsize=min_fontsize-2)
    # label shoudl use degree symbol
    ax.set_xlabel('Target position (°)')
    ax.set_ylabel('Ang. vel. (rad/s)')
    ax.set_box_aspect(1)
    #ax.set_ylim([-7.5, 7.5])
    #putil.label_figure(fig, figid) 
    # Remove seaborn styling and use pure matplotlib
    
    # Set tick positions explicitly - only 3 points centered at 0
    ax.set_xticks([-180, 0, 180])
    ax.set_yticks([-6, 0, 6])
    
    # Only show ticks on bottom and left axes
    ax.tick_params(axis='x', which='major', 
                   length=4, width=1.0, direction='out',
                   colors='black', zorder=10, pad=3,
                   bottom=True, top=False, labelbottom=True, labeltop=False)
    
    ax.tick_params(axis='y', which='major', 
                   length=4, width=1.0, direction='out',
                   colors='black', zorder=10, pad=3,
                   left=True, right=False, labelleft=True, labelright=False)
    
    # Turn off minor ticks
    ax.minorticks_off()

    sns.despine(offset=2, ax=ax, trim=True)

fig.suptitle('all frames (filter-chase={})'.format(filter_chase), fontsize=8)    
plt.subplots_adjust(wspace=0.4)

putil.label_figure(fig, figid)

# Save
figname = 'gain_{}_filter-chase-{}_CCW-CW_{}'.format(yvar, filter_chase, species_str)
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
plt.savefig(os.path.join(figdir, '{}.svg'.format(figname)))
print(figdir, figname)

#%%
# Get average ang vel across bins
grouper = ['species', 'acquisition', 'binned_theta_error', 'courtship_level']
#if assay == '38mm_projector':
grouper.append('stim_direction') 
    
yvar = 'ang_vel_fly_shifted'
avg_ang_vel = chase_.groupby(grouper)[yvar].mean().reset_index()

#%%
stim_palette = {'CW': 'cyan', 'CCW': 'magenta'}
start_bin = -180
end_bin = 180
yvar = 'ang_vel_fly_shifted'

if filter_chase:
    # Don't use avg_ang_vel, bec some acquisitions are not represnted in each courtship level
    for lvl, chase_lvl in chase_.groupby('courtship_level'):
        fig, axn = plt.subplots(1, n_species, figsize=(8, 4), sharex=True, sharey=True)
        for si, (sp, plotd) in enumerate(chase_lvl.groupby('species')):
            #plotd = avg_ang_vel[avg_ang_vel['species']=='Dyak'].copy()
            if n_species==1:
                ax=axn
            else:
                ax=axn[si]

            #if assay == '38mm_projector':
            sns.lineplot(data=plotd, ax=ax,
                        x='binned_theta_error', y=yvar,
                        hue='stim_direction', 
                        palette=stim_palette, 
                        errorbar='se', marker='o', legend=0) #errwidth=0.5)
            #else:
            #    sns.lineplot(data=plotd, x='binned_theta_error', y=yvar, ax=ax,
            #                #hue='stim_direction', palette=stim_palette, ax=ax,
            #                errorbar='se', marker='o') #errwidth=0.5)
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
            ax.set_box_aspect(1)
            ax.set_ylim([-5.5, 5.5])

            #putil.label_figure(fig, figid) 
        fig.suptitle('courtship level: {}'.format(lvl))    
        

#%%
if filter_chase:
    n_species = chase_['species'].nunique()

    fig, axn = plt.subplots(1, n_species, figsize=(n_species*3.5, 4), sharex=True, sharey=True)
    for si, (sp, plotd) in enumerate(chase_.groupby('species')):

        if n_species==1:
            ax=axn
        else:
            ax=axn[si]
        #if assay == '38mm_projector':
        ax.set_title(sp)
        sns.lineplot(data=plotd[plotd['courtship_level']!='neither'], x='binned_theta_error', y=yvar, ax=ax,
                        hue='courtship_level', 
                        palette={'low': 'blue', 'high': 'red',
                        'neither': 'gray'},
                        style='stim_direction', 
                        #line={'CW': '-', 'CCW': ':'},
                        markers=['none'], 
                        errorbar='se', legend=si==0)
        if si==0:
            sns.move_legend(ax, 'upper left', 
                    bbox_to_anchor=(n_species+0.6, 1),
                        frameon=False, fontsize=min_fontsize-2)
        ax.set_box_aspect(1) 

        ax.set_ylim([-3.5, 3.5])
        ax.set_xticks([-180, -90, 0, 90, 180])
        #ax.set_xlim([-195, 195])

        ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
        ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)

    plt.subplots_adjust(wspace=0.5, right=0.8)
    fig.suptitle('filter-chase={}'.format(filter_chase), fontsize=8)     

    putil.label_figure(fig, figid)

    # Save
    figname = 'split-by-courtship-level_{}_filter-chase-{}_CCW-CW'.format(yvar, filter_chase)
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname))) 
    print(figdir, figname)


#%%
min_vel = 10
max_vel = 50

fig, ax = plt.subplots()
# chase_[chase_['target_vel']>80] = np.nan

sns.histplot(data=chase_, x='target_vel', ax=ax, bins=10)
ax.axvline(x=min_vel, color='r', linestyle='--', lw=0.5)
ax.axvline(x=max_vel, color='r', linestyle='--', lw=0.5)

#%%
# Divide target_vel into bins
min_vel = round(chase_['target_vel'].min())
max_vel = round(chase_['target_vel'].max())

min_vel = 0
max_vel = 50
chase_ = bin_x(chase_, 'target_vel', start_bin=min_vel, 
               end_bin=max_vel, bin_size=5)

chase_['stim_hz'] = chase_['binned_target_vel'].copy()
 
# %%
mean_vel_by_stimhz = None
#if 'stim_hz' in df0.columns:
mean_vel_by_stimhz = chase_.groupby('stim_hz')['target_vel'].mean().round(1).reset_index()
print(mean_vel_by_stimhz)

#%%
# Split by stim_hz?
n_stim_levels = chase_['stim_hz'].nunique()

fig, axn = plt.subplots(2, n_stim_levels, 
                figsize=(n_stim_levels*3, 8), 
                sharex=True, sharey=True)

for sp, (curr_species, curr_df) in enumerate(chase_.groupby('species')):

    for si, (stim_hz, plotd) in enumerate(curr_df.groupby('stim_hz')):
        ax=axn[sp, si]
        mean_vel = mean_vel_by_stimhz[mean_vel_by_stimhz['stim_hz']==stim_hz]['target_vel'].values[0]

        sns.lineplot(data=plotd, x='binned_theta_error', y=yvar, ax=ax,
                    hue='stim_direction', palette=stim_palette,
                    errorbar='se', marker='o', 
                    legend= (si==n_stim_levels-1 and sp==0)) #errwidth=0.5)
        ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
        ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
        ax.set_title('{}mm/s'.format(mean_vel))
        ax.set_xlabel('')
        if si==0:
            ax.set_ylabel(curr_species)
        else:
            ax.set_ylabel('')
        ax.set_box_aspect(1)

    ax.set_xlim([-180, 180])
    ax.set_ylim([-8, 8])
    #ax.set_ylim([-15, 15])


    if si==n_stim_levels-1 and sp==0:
        sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1),
                        frameon=False, title='movement dir', 
                        fontsize=min_fontsize-2)

fig.text(0.5, 0.1, 'Object position (deg)', rotation=0, 
         ha='center', va='center', fontsize=min_fontsize)
fig.text(0.05, 0.5, 'Ang. vel. shifted (rad/s)', rotation=90, 
         ha='center', va='center', fontsize=min_fontsize)

fig.suptitle('filter-chase={}'.format(filter_chase))
plt.subplots_adjust(left=0.1, right=0.8, wspace=0.3, bottom=0.2)

putil.label_figure(fig, figid)

# Save
figname = 'split-by-stimhz_{}_filter-chase-{}_CCW-CW'.format(yvar, filter_chase)
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
plt.savefig(os.path.join(figdir, '{}.svg'.format(figname)))
print(figdir, figname)

#%%

# For each stim_hz epoch, caluclate the average vel of the target

fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
sns.pointplot(data=df0, x='stim_hz', y='target_vel', 
                ax=ax, hue='species', palette=species_palette,
                errorbar='sd', marker='o')


#%%