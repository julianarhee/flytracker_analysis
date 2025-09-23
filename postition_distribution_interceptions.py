#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : postition_distribution_interceptions.py
Created        : 2025/09/23 13:34:55
Project        : /Users/julianarhee/Repositories/flytracker_analysis
Author         : jyr
Last Modified  : 
'''
#%%
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

#%%
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


def get_targ_pos_epochs(curr_df, fps=60, nsec_pre=1, nsec_peri=0.5):
    #fps = 60
    #n_frames_start = int(.25*fps)

    n_frames_peri = int(nsec_peri*fps)

    n_frames_pre = int(nsec_pre*fps)
    n_frames_pre_end = int(nsec_pre*fps) - 1

    curr_df['epoch'] = None
    p_list = []
    #targ_pos_peri = []
    #targ_pos_pre = []
    for i, b_ in curr_df.groupby('interception_bout'):
        peri_ = b_.iloc[0:n_frames_peri].copy() #['theta_error']

        pre_start = b_.iloc[0].name-n_frames_pre
        pre_end = pre_start + n_frames_pre_end
        pre_ = curr_df.loc[pre_start:pre_end].copy() #['theta_error']

        peri_['epoch'] = 'peri'
        pre_['epoch'] = 'pre'
        pp = pd.concat([peri_, pre_])

        pp['interception_bout'] = i
        pp['acquisition'] = acq
        
        p_list.append(pp)

    targpos = pd.concat(p_list).reset_index(drop=True)
    return targpos



#%%
plot_style='white'
min_fontsize=18
putil.set_sns_style(style=plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%%
# If using Caitlin's elehk data
acquisition_parentdir = os.path.join('/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee', 
                            'caitlin_data', 'Caitlin_elehk_38mm')
has_meta_csv = True

# Processed data directory (after running transform_data.py)
processedmat_dir = '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker/processed_mats'
aggr_mat_fname ='transformed_data_Dele-HK.pkl' 

# set local
#localdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/free_behavior_analysis/38mm_dyad/MF/FlyTracker'
#local_fpath = os.path.join(localdir, aggr_mat_fname)
local_fpath = None

# Output filepath for transformed data: saved in parent dir of processed_mats
# Can load older file, from aggregate_relative_metrics.py
aggr_mat_savedir = os.path.split(processedmat_dir)[0]
output_fpath = os.path.join(aggr_mat_savedir, aggr_mat_fname)
print("Output filepath: {}".format(output_fpath))

#%%a

# Get metadata from .csv since not all data processed/annotated
# Load meta data: .csv file
meta_fpath = glob.glob(os.path.join(acquisition_parentdir, '*.csv'))[0]
print("Loading meta data from {}".format(meta_fpath))
meta0 = pd.read_csv(meta_fpath)

meta1 = meta0[(meta0['tracked']==1)
            & (meta0['annotated']==1) 
            & (meta0['courtship']==1) ].copy()     
# Get list of acquisitions from meta data
acqs = meta1['acquisition'].unique()

#%%
# Set output directories: set it upstream of processedmat_dir     
figdir = os.path.join(os.path.split(processedmat_dir)[0], 'position_distribution_interceptions')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print("saving figures to {}".format(figdir))

figid = acquisition_parentdir

#%% 
# LOAD DATA
# --------------------------------
create_new = False
reassign_acquisition_name = False #assay=='38mm_projector' 
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

df0 = df0.reset_index(drop=True)

#%%
# Calculate circular difference between ori and heading on each row
df0['ori_heading_diff'] = np.angle(np.exp(1j * df0['ori']) / np.exp(1j * df0['heading']))
df0['ori_heading_diff_abs'] = np.abs(df0['ori_heading_diff'])

#%%
# Load ACTIONS
found_actions_paths = glob.glob(os.path.join(acquisition_parentdir, 
                        '20*', '*ele*', '*actions.mat'))
print('Found {} actions files'.format(len(found_actions_paths)))

all_actions = util.load_ft_actions(found_actions_paths)
all_actions.head()
# %%
int_bouts = all_actions[all_actions['action']=='interception'].copy()
int_bouts[(int_bouts['likelihood']>=0)].groupby('acquisition').count()

#%%
# 

#%%
df0['interception'] = 0
df0['interception_bout'] = None

for acq, curr_df in df0.groupby('acquisition'):

    curr_bouts= int_bouts[int_bouts['acquisition']==acq].copy()
    for i, bout_ in curr_bouts.iterrows():
        start_, end_ = bout_[['start', 'end']]
        curr_df.loc[curr_df['frame'].isin(np.arange(start_, end_+1)), 'interception'] = 1
        curr_df.loc[curr_df['frame'].isin(np.arange(start_, end_+1)), 'interception_bout'] = i

    df0.loc[curr_df.index, 'interception'] = curr_df['interception']
    df0.loc[curr_df.index, 'interception_bout'] = curr_df['interception_bout']

#%%

p_list = []
for acq, curr_df in df0.groupby('acquisition'):
    if curr_df['interception'].max() == 1:
        pos_ = get_targ_pos_epochs(curr_df, fps=60, 
                            nsec_pre=0.5, nsec_peri=0.25)
        pos_['acquisition'] = acq
        p_list.append(pos_)
    else:
        print("No interceptions for {}".format(acq))
all_targpos = pd.concat(p_list).reset_index(drop=True)

#%
all_targpos['theta_error_deg'] = np.rad2deg(all_targpos['theta_error'])
all_targpos['theta_error_deg_abs'] = np.abs(all_targpos['theta_error_deg'])
all_targpos['theta_error_abs'] = np.abs(all_targpos['theta_error'])

#%%
# Look at dist of targ_pos
fig, ax = plt.subplots()
sns.histplot(data=all_targpos, x='theta_error_deg', ax=ax, 
            hue='epoch', edgecolor='none', bins=50,
             stat='density', common_norm=False, legend=1)
sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', 
                frameon=False, fontsize=min_fontsize-2)
ax.set_xlim([-180, 180])
ax.set_xlabel('rel. target position (deg)')

ax.set_box_aspect(1)
ax.set_title('Target position before/during interceptions', 
            fontsize=min_fontsize)

#%% Plot average +/- se of theta_error by epocih for each acquisition

show_legend = False

fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(data=all_targpos, x='epoch', y='theta_error_deg_abs',
             hue='acquisition', palette='viridis',
             errorbar='se', marker='o', legend=show_legend)
if show_legend:
    sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', 
                frameon=False, fontsize=min_fontsize-2)
ax.set_box_aspect(1)
ax.set_ylabel('Dev. from center (deg)')
ax.set_xlabel('Epoch peri/pre interception')

ax.set_title('Target pos. before/during interceptions', 
            fontsize=min_fontsize)
putil.label_figure(fig, figid)

#%%

fig, ax = plt.subplots()
ax.set_title('Heading vs. Traveling Direction diff.')
sns.histplot(data=df0, x='ori_heading_diff_abs', ax=ax, 
            hue='interception', palette='viridis',
            stat='density', common_norm=False, legend=1,
            fill=False)
ax.set_box_aspect(1)
sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', 
                frameon=False, fontsize=min_fontsize-2)

#%%

# Plot average ori_heading_diff_abs by interecption for each acquisition

fig, ax = plt.subplots()
sns.pointplot(data=df0, 
             x='interception', 
             y='ori_heading_diff_abs', dodge=True,
             hue='acquisition', palette='viridis',
             estimator=np.median) #marker='o', legend=1)
sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', 
                frameon=False, fontsize=min_fontsize-2)
ax.set_box_aspect(1)
ax.set_title('Heading vs. Traveling Direction diff.', fontsize=min_fontsize)
putil.label_figure(fig, figid)









#%%

#acq = '20250918-1019_fly2_Dele-HK_3do_gh'
acq = '20240814-1023_fly2_Dele-HK_WT_6do_gh'
#acq = '20250919-1022_fly2_Dele-HK_3do_gh'
curr_ints = int_bouts[int_bouts['acquisition']==acq].copy()
curr_df = df0[df0['acquisition']==acq].copy()

#%%
# Assign interception actions to df
curr_df['interception'] = 0
curr_df['interception_bout'] = None

for i, b_ in curr_ints.iterrows():
    start_, end_ = b_[['start', 'end']]
    curr_df.loc[curr_df['frame'].isin(np.arange(start_, end_+1)), 'interception'] = 1
    curr_df.loc[curr_df['frame'].isin(np.arange(start_, end_+1)), 'interception_bout'] = i

int_df = curr_df[curr_df['interception']==1].copy()

#%%

fig, axn = plt.subplots(2, 1, figsize=(10, 5))
ax=axn[0]
ax.set_title('Non-interception')
sns.scatterplot(data=curr_df[curr_df['interception']==0], ax=ax, 
                x='ori', y='heading', 
                hue='vel', palette='viridis',
                alpha=0.5, edgecolor='none', s=5, legend=0)
ax=axn[1]
ax.set_title('Interception')
sns.scatterplot(data=curr_df[curr_df['interception']==1], ax=ax, 
                x='ori', y='heading', 
                hue='vel', palette='viridis',
                alpha=0.5, edgecolor='none', s=5, legend=0)

#%%
curr_df['ori_heading_diff_abs'] = np.abs(curr_df['ori_heading_diff'])
fig, ax = plt.subplots()
ax.set_title('Heading vs. Traveling Direction diff.')
sns.histplot(data=curr_df, x='ori_heading_diff_abs', ax=ax, 
            hue='interception', palette='viridis',
            stat='density', common_norm=False, legend=1,
            fill=False)
ax.set_box_aspect(1)
sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', 
                frameon=False, fontsize=min_fontsize-2)



# Include median value of histograms
medians = curr_df.groupby('interception')['ori_heading_diff_abs'].mean().reset_index()

# Include median value of histograms
curr_df.groupby('interception')['ori_heading_diff'].describe()


# %%
# Plot distribution of theta_error in first N frames of each
# interception bout
targpos = get_targ_pos_epochs(curr_df, fps=60, nsec_pre=1, nsec_peri=0.5)
#%%

# Plot distribution of peri and pre theta_error on polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
sns.histplot(data=targpos, x='theta_error',  ax=ax,
            hue='epoch', edgecolor='none',
             stat='density', common_norm=False, legend=1)


ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False,
          fontsize=min_fontsize-2)

# Set 0 to North
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

#%%
# Plot the distribution of peri/pre theta_error on a
# linear plot, centered at 0, from -180 to 180.
# Convert to degrees if needed

# Bin theta_error into 10 bins
curr_df['theta_error_deg'] = np.rad2deg(curr_df['theta_error'])
curr_df = bin_by_object_position(curr_df)

targpos['theta_error_deg'] = np.rad2deg(targpos['theta_error'])
targpos = bin_by_object_position(targpos)


fig, ax = plt.subplots()
sns.lineplot(data=targpos, ax=ax,
            x='binned_theta_error', y='ori_heading_diff',
            hue='epoch', palette='viridis',
            errorbar='se', marker='o')


#%%
fig, ax = plt.subplots()
sns.histplot(data=targpos, x='theta_error_deg', ax=ax, 
            hue='epoch', edgecolor='none', bins=50,
             stat='density', common_norm=False, legend=1)
sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', 
                frameon=False, fontsize=min_fontsize-2)
ax.set_box_aspect(1)
ax.set_title('Target position before/during interceptions', 
            fontsize=min_fontsize)
#ax.legend()


#%%
# Plot scatterplot of theta_error on polar
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
sns.scatterplot(data=targpos, ax=ax,
            x='theta_error', y='dist_to_other',
            hue='epoch', edgecolor='none',
            legend=1, s=5, alpha=0.5)
# Face north
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)



# %%


