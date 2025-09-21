#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
'''
 # @ Author: Juliana Rhee
 # @ Filename: plot_relative_position.py
 # @ Create Time: 2025-02-07 20:02:14
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-06-23 10:12:06
 # @ Description:
 Read projector data and plot target position relative to focal fly.
 '''

#%%
import os
import sys
import glob
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import utils as util

import transform_data.relative_metrics as rel
import plotting as putil

#%%

def transform_projector_data(rootdir, assay, acqs, processedmat_dir, 
                            movie_fmt='.avi',subdir=None, flyid1=0, flyid2=1,
                            create_new=False):
    """
    Load transformed projector data for specified acquisitions.
    """
    d_list = []
    errors = []
    for i, acq in enumerate(acqs):
        if i % 10 == 0:
            print('Processing {} of {}: {}'.format(i, len(acqs), acq))
        acq_dir = os.path.join(rootdir, assay, acq)
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
        df_['date_fly'] = ['_'.join([f.split('-')[0], f.split('_')[1]]) for f in df_['file_name']]
        df_['species'] = 'Dmel' if 'mel' in acq else 'Dyak'
        df_['acquisition'] = ['_'.join([a, b]) for a, b in df_[['date_fly', 'species']].values]
       
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
        df0.loc[df0['file_name']==fn, 'genotype'] = currm['genotype_male']

    df0['date'] = [int(a.split('_')[0]) for a in df0['acquisition']]
    print("Adding genotype: {}".format(currm['genotype_male']))
    return df0


#%%


#     #acq = df0['file_name'].unique()[0] # '20241213-1103_fly1_Dmel-p1_2do_gh_2dR'
#     acq = '20250522-1511_fly1_Dmel-p1-backhalf_2do_gh_2dR_cw'
#     dotdf = df0[(df0['file_name']==acq) & (df0['id']==1)].copy().reset_index(drop=True)
#     import dlc as dlc 
#     # get step dict
#     step_dict = dlc.get_step_indices(dotdf, speed_var='vel', time_var='sec',
#                         t_start=0, increment=40, n_levels=10)
# 
#     # add epochs
#     dotdf = add_speed_epoch(dotdf, step_dict)
 
#%    
    

def filter_court(df,ftjaaba=None, min_vel=10, 
                 max_facing_angle=np.deg2rad(45), 
                 max_dist_to_other=20, 
                 max_targ_pos_theta=np.deg2rad(270), 
                 min_targ_pos_theta=np.deg2rad(-270),
                 min_wing_ang=np.deg2rad(30), use_jaaba=False, manual=False):
    if use_jaaba:
        court_ = ftjaaba[(ftjaaba['id']==0) & (ftjaaba['chasing']==1)].copy() 
        court_filter_str = 'jaaba'
    elif manual:
        court_ = df[ (df['id']==0) & (df['chasing_manual']==1)]
    else:
        court_ = df[(df['id']==0) #& (ftjaaba['chasing']==1)
                    & (df['vel']> min_vel)
                    & (df['targ_pos_theta'] <= max_targ_pos_theta)
                    & (df['targ_pos_theta'] >= min_targ_pos_theta)
                    & (df['facing_angle'] <= max_facing_angle)
                    & (df['max_wing_ang'] >= min_wing_ang)
                    & (df['dist_to_other'] <= max_dist_to_other)].copy()
        court_filter_str = 'vel-targpostheta-facingangle-disttoother-minwingang{}'.format(min_wing_ang_deg)

    return court_
#
#court_ = filter_court(df)
#meanbouts = court_.groupby(['acquisition', 'file_name', 'id', 
#                        'stim_direction', 'paint_side', 
#                        'subboutnum']).mean().reset_index()
#meanbouts.head()    
#meanbouts['stim_hz'] = meanbouts['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))  

def plot_egocentric_hue(df_, ax=None, xvar='targ_rel_pos_x', yvar='targ_rel_pos_y', 
                    hue_var='stim_hz', hue_norm=None, markersize=5, 
                    edgecolor='none', lw=0.5, bg_color='k',
                    plot_com=False,
                    cmap='viridis', com_markersize=60, 
                    com_edgecolor='none', com_lw=0, alpha=0.5):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
     
    sns.scatterplot(data=df_, x=xvar, y=yvar, ax=ax,
                    s=markersize, alpha=alpha, palette=cmap, hue=hue_var,
                    legend=0, hue_norm=hue_norm, edgecolor=edgecolor, lw=lw)
    ax.plot(0, 0, '>', color=bg_color, markersize=3)
    ax.set_aspect(1)
    ax.axis('off')
    if plot_com:
        for hueval, f_ in df_.groupby(hue_var):
            cm_theta = pd.Series(np.unwrap(f_[xvar])).mean()
            cm_radius = f_[yvar].mean()
            ax.scatter(cm_theta, cm_radius, s=com_markersize, c=stimhz_palette[hueval],
                    marker='o', edgecolor=com_edgecolor, lw=com_lw,
                    label='COM: {:.2f}'.format(hueval))
    ax.set_aspect(1)
    ax.axis('off')
    return ax


#%%
plot_style='dark'
putil.set_sns_style(style=plot_style, min_fontsize=18)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'
#%%
# Set directories
# Dropbox/source directory:
rootdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data'
# Main assay containing all the acquisitions
assay = '38mm_projector'
# Processed data directory (after running transform_data.py)
processedmat_dir = '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_painted_eyes/FlyTracker/processed_mats'
if not os.path.exists(processedmat_dir):
    os.makedirs(processedmat_dir)

# Set output directories: set it upstream of processedmat_dir     
figdir = os.path.join(os.path.split(processedmat_dir)[0], 'relative_position')
if not os.path.exists(figdir):
    os.makedirs(figdir)
if plot_style=='white':
    figdir = os.path.join(figdir, 'white')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

figid = '{}|{}'.format(assay, processedmat_dir)

#%%
# Load meta data: .csv file
# Assign fly numbers across acquisitions bec multiple acquisitions per fly
src = os.path.join(rootdir, assay)
meta_fpath = glob.glob(os.path.join(src, '*.csv'))[0]
print("Loading meta data from {}".format(meta_fpath))
meta0 = pd.read_csv(meta_fpath)
meta = meta0[(meta0['tracked in matlab and checked for swaps']==1)
             & (meta0['exclude']==0) & (meta0['annotated']==1)]
meta['acquisition'] = ['_'.join( [f.split('-')[0], f.split('_')[1]] ) for f in meta['file_name']]
meta.head()

#meta1 = meta0[(meta0['tracked in matlab and checked for swaps']==1)
#             & (meta0['exclude']==0) ].copy()
#meta1[(meta1['manipulation_male']=='no paint ')].groupby(['species_male', 'stim_direction'])['file_name'].nunique()

#%%
# Found all video directories for this experiment
acq_dirs = glob.glob(os.path.join(src, '20*'))
found_acqs = [os.path.split(a)[-1] for a in acq_dirs]
print("Found {} {} acqs.".format(len(found_acqs), assay))

# Get list of acquisitions from meta data
acqs = meta['file_name'].values

# Check that all acquisitions specified in meta are found in the source directory
not_in_meta = [a for a in found_acqs if a not in meta['file_name'].values]
not_in_src = [a for a in found_acqs if not os.path.exists(os.path.join(src, a))]
if len(not_in_src)>0:
    print("Not in src: {}".format(len(not_in_src)))
    for i in not_in_src:
        print(i)
else:
    print("All acqs in src")    

#%%
# Load transformed data
create_new=True
retransform_data = False
# --------------------------------
# Output filepath for transformed data 
output_fpath = os.path.join(os.path.split(processedmat_dir)[0], 'transformed_projector_data.pkl')
# ---------------------------------
if create_new:
    df0, errors = transform_projector_data(rootdir, assay, acqs,
                                        processedmat_dir, movie_fmt='.avi',
                                        subdir=None, flyid1=0, flyid2=1,
                                        create_new=retransform_data)
    try:
        assert len(acqs) - len(errors) == df0['file_name'].nunique(), 'Not all acqs processed'  
    except AssertionError as e:
        error_acqs = [l[0] for l in errors]
        loaded_acqs = df0['file_name'].unique()
        accounted = list(set(error_acqs + list(loaded_acqs)))
        found_is = []
        for i, a in enumerate(acqs):
            found_i = accounted.index(a)
            found_is.append(found_i)
        # Count the number of each occurrence in found_is
        from collections import Counter
        counter = Counter(found_is)
        print(counter)        
        
    # Assign paint conditions
    print("Assigning paint conditions to DataFrame")
    df0 = assign_paint_conditions(df0, meta0)
    
    # Save the DataFrame to a pickle file
    print("Saving transformed data to {}".format(output_fpath))
    df0.to_pickle(output_fpath)

else:
    # Load existing data
    df0 = pd.read_pickle(output_fpath)
    print("Loaded transformed data from {}".format(output_fpath))

print(df0['genotype'].unique())
  
#%%
# Check counts
def print_condition_counts(df0):
    print(df0.groupby(['species', 'paint_side', 'paint_coverage'])['acquisition'].nunique())

print_condition_counts(df0)

# %% 
# Split by STIM_HZ
df0 = util.add_stim_hz(df0, n_frames=24000, n_epochs=10)

#%%
# subdivide into smaller boutsa
bout_dur = 0.20
df0 = util.subdivide_into_subbouts(df0, bout_dur=bout_dur, grouper='file_name')


#%%

fn = '20250701-1613_fly3_Dmel-LC10aS-KIR_1do_gh_1dR'
#lots of NaNs in first half of array, but correct dur (48000, i.e., 24000 frames per ID)


#%%
# Assign manual chasing labels
# --------------------------------
df0['chasing_manual'] = 0
df0['chasing_manual'] = df0['chasing_manual'].astype('int8')
# once, up front:
df0['file_name'] = df0['file_name'].astype('category')

chase_meta = []
has_jaaba = []
for fn, df_ in df0.groupby('file_name'):
    fp = [i for i in df_['fpath'].unique() if isinstance(i, str)][0]
    # Find actions path
    action_fpath = fp.replace('-track', '-actions')
    if os.path.exists(action_fpath):
        # Load actions dataframe
        fp_actions = util.ft_actions_to_bout_df(action_fpath)
        curr_actions = fp_actions[fp_actions['action']=='chasing']
        if len(curr_actions) > 0:
            has_jaaba.append(fn)
            print("Found {} chasing actions in {}".format(len(curr_actions), fn))
            for ri, row in curr_actions.iterrows():
                chase_meta.append((fn, row['start'], row['end'])) 
#%
# grab the numpy arrays
frames = df0['frame'].to_numpy()
files  = df0['file_name'].to_numpy()
chase  = df0['chasing_manual'].to_numpy()

# one-time: set a two-level index and sort it
df0 = df0.set_index(['file_name','frame']).sort_index()

# then for each triplet, pandas can jump straight to the block
for fn, s, e in chase_meta:
    print(fn, s, e)
    df0.loc[(fn, slice(s-1, e-1)), 'chasing_manual'] = 1
#%
# (if you need your old columns back:)
df0 = df0.reset_index()
df0['file_name'] = df0['file_name'].astype(str)

#%% 
check_bout_split = False
if check_bout_split:
    # Use df_['chasing_manual'].diff() to find start and end of bouts
    fn = '20250514-1255_fly3_Dyak-p1-right_5do_gh_2dR_cw'
    df_ = df0[df0['file_name']==fn].copy()
    df_['chasing_manual'] = df_['chasing_manual'].astype('int8')
    diffs = df_['chasing_manual'].diff()
    # find the start and end of bouts
    starts = df_.loc[diffs==1, 'frame']
    ends = df_.loc[diffs==-1, 'frame']
    for s, e in zip(starts, ends):
        print(s, e)
    # check
    fp = df_['fpath'].unique()[0]
    action_fpath = fp.replace('-track', '-actions')
    fp_actions = util.ft_actions_to_bout_df(action_fpath)
    curr_actions = fp_actions[fp_actions['action']=='chasing']
    curr_actions        

#%%
# Look at multiple examples, each condition:
# ------------------------------------------
# paint_coverage: 
#      'whole eye ', 'back 1/2', 'front 1/2', 'front 1/3', 
#      'back 2/3', 'back 1/3'

# paint_side: 'left', 'right', 'none', 'bone'
# ------------------------------------------
annotated = [
    '20250205-1113_fly3_Dmel-p1-left-pwder_2do-gh_2dR_cw',
    '20250205-1538_fly4_Dmel-p1-right-pwder_2do_gh_2dR_ccw',
    '20250205-1121_fly3_Dmel-p1-left-pwder_2do-gh_2dR_ccw'
    
    '20250512-1637_fly5_Dyak-p1-left_3do_gh_3dR_cw',
    '20250512-1629_fly5_Dyak-p1-left_3do_gh_3dR_ccw',
    
    '20250512-1732_fly7_Dyak-p1-right_3do_gh_3dR_ccw',
    # '20250512-1742_fly7_Dyak-p1-right_3do_gh_3dR_cw',
    
    '20250512-1747_fly8_Dyak-p1-left_3do_gh_3dR_cw',
    '20250514-1240_fly3_Dyak-p1-right_5do_gh_2dR_ccw',
    # '20250514-1255_fly3_Dyak-p1-right_5do_gh_2dR_cw',
    
    '20250512-1438_fly1_Dyak-p1_3do_gh_3dR_ccw',
]

#df1 = df0[df0['file_name'].isin(annotated)].copy()
#df1.groupby(['paint_side', 'paint_coverage'])['acquisition'].nunique()

# species  paint_side  paint_coverage    
# Dmel     both        front 1/4 vertical     6
#          left        whole eye              1
#          none        none                   8
#          right       whole eye              1
# Dyak     both        back 1/2 vertical      6
#                      front 1/2 vertical     5
#                      front 1/4 parallel     3
#                      front 1/4 vertical     4
#          left        whole eye              7
#          none        none                  13
#          right       whole eye              8
# Name: acquisition, dtype: int64
 
#%%
coverage = 'front 1/4 parallel' # 'whole eye ' # 'front 1/2 vertical' #'whole eye '
side = 'both'
df = df0[(df0['paint_side']==side)
       & (df0['paint_coverage']==coverage)].copy()
#       & (df0['date']>=20250512)].copy()
print(df['acquisition'].unique())
cond_str = '{}_{}'.format(coverage, side)
#%
print(cond_str)

for fn, df_ in df.groupby('file_name'):
    if len(df_)>0:
        print(fn, df_.shape)

#%%
# Add additional metrics
import theta_error as the
f1 = df[df['id']==0].copy()
f1 = rel.calculate_angle_metrics_focal_fly(f1, winsize=5, grouper='file_name')
f1['targ_ang_vel_abs'] = np.abs(f1['targ_ang_vel'])
f1 = the.shift_variables_by_lag(f1, lag=2)
    
#%%
xvar = 'targ_rel_pos_x'
yvar = 'targ_rel_pos_y'
hue_var = 'stim_hz' #'epoch'
cmap='viridis'
#stimhz_palette = putil.get_palette_dict(df[df[hue_var]>=0], hue_var, cmap=cmap)
stimhz_palette = putil.get_palette_dict(f1, hue_var, cmap=cmap)

exclude_cols = ['fpath', 'stim_dir']
incl_cols_for_mean = [c for c in df0.columns if c not in exclude_cols]

# 
chasing_var = 'chasing_manual'
groupby_cols = ['species', 'acquisition', 'file_name', 'id', 
                'stim_direction', 'paint_side', 'paint_coverage', 'stim_hz', 
                'subboutnum', chasing_var]
meanbouts = util.groupby_aggr_if_numeric(f1, groupby_cols, aggr_type='mean')
#meanbouts = f1[incl_cols_for_mean].groupby().mean().reset_index()   
meanbouts['stim_hz'] = meanbouts['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))  

plot_com = False
use_bouts = False
use_frames=True

if use_bouts:
    plotd = meanbouts.copy()
elif use_frames:
    plotd = f1.copy()
else:
    plotd = df[df['id']==0].copy() #f1.copy()
print(plotd['file_name'].dropna().nunique()) 

#%%
nr=plotd['acquisition'].nunique()
nc=plotd.groupby('acquisition')['file_name'].nunique().max()
print(nr, nc)

rows_are_conds=False
if nc<=2:
    rows_are_conds=True
    nc=nr
    nr=plotd.groupby('acquisition')['file_name'].nunique().max()
print(nr, nc)
print("Rows are conds: {}".format(rows_are_conds))

#%%
classifier_type = 'manual'
# -------------------------------------------
min_vel = 5
max_facing_angle = np.deg2rad(80)
max_dist_to_other = 20
max_targ_pos_theta = np.deg2rad(180) #270) #270 #160
min_targ_pos_theta = np.deg2rad(-180) #-270) # -160
min_wing_ang_deg = 5
min_wing_ang = np.deg2rad(min_wing_ang_deg)

use_jaaba = True if classifier_type=='jaaba' else False
use_manual = True if classifier_type=='manual' else False
clf_str = 'annot-{}'.format(classifier_type) if use_manual else 'annot-None'
court_ = filter_court(plotd, min_vel=min_vel, max_facing_angle=max_facing_angle, 
                      max_dist_to_other=max_dist_to_other, 
                      max_targ_pos_theta=max_targ_pos_theta, 
                      min_targ_pos_theta=min_targ_pos_theta,
                      min_wing_ang=min_wing_ang, use_jaaba=use_jaaba, manual=use_manual)
court_ = court_.reset_index(drop=True)
print(clf_str)

#%%
markersize=1 if classifier_type=='none' else 5
cmap='viridis'

# Make hue_norm for stim_hz
hue_min = 0 
hue_max = 1
hue_norm = plt.Normalize(vmin=hue_min, vmax=hue_max)
# PLOT, egocentric, color by stim Hz
min_courted_stimhz = court_['stim_hz'].min()

for sp, tmpdf in plotd.groupby('species'):
    nc=tmpdf['acquisition'].nunique()
    nr=tmpdf.groupby('acquisition')['stim_direction'].nunique().max()
    print(nr, nc)
    rows_are_conds=True
    #if nc<=2:
    #    rows_are_conds=True
    #j    nc=nr
    #    nr=curr_court.groupby('acquisition')['file_name'].nunique().max()
    fig, axn = plt.subplots(nr, nc, sharex=True, sharey=True, 
                            figsize=(nc*2, nr*2))
    for ai, (acq, currdf) in enumerate(tmpdf.groupby('acquisition')):
        #acq = currdf['acquisition'].unique()[0]
        #for ci, (fn, df_) in enumerate(currdf.groupby('file_name')):
        for ci, (stim_dir, df_) in enumerate(currdf.groupby('stim_direction')):
            curr_court = court_[(court_['acquisition']==acq)
                                & (court_['stim_direction']==stim_dir)].copy() 

            if stim_dir == 'CW':
                col_ix = 1
            else:
                col_ix = 0
            fname = df_['file_name'].unique()[0] 
            #stim_dir = df_['stim_direction'].unique()[0] 
            #ci = 0 if stim_dir=='ccw' else 1
            if nr==1 and nc==1:
                ax=axn#[ai]
            elif nr==2 and nc==1:
                ax=axn[col_ix]
            else:
                ax=axn[col_ix, ai]
            ax = plot_egocentric_hue(curr_court, ax=ax, xvar=xvar, yvar=yvar, 
                                     hue_var=hue_var, hue_norm=hue_norm,
                                markersize=markersize, plot_com=plot_com,
                                bg_color=bg_color, cmap=cmap, alpha=0.75,
                                edgecolor=bg_color, lw=0.1)
            # plot grid
            ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
            ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
           
            paint_coverage = df_['paint_coverage'].unique()[0]
            paint_side = df_['paint_side'].unique()[0]
            cond_str = '{}_{}_{}'.format(sp, paint_coverage, paint_side) 
            title = '{} ({})\n{}'.format(cond_str, stim_dir, acq)
            ax.set_title(title, loc='left', fontsize=6)
    if nr > 1 and nc >= 1:
        for ax in axn.flat:
            ax.axis('off')
    ax.invert_yaxis() # to match video POV
     
    # Add colorbar
    cbar_ax = fig.add_axes([0.93, 0.4, 0.01, 0.3])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=hue_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Stimulus Hz', fontsize=6)
    cbar.ax.tick_params(labelsize=4) 
    plt.subplots_adjust(hspace=0.2, wspace=0.5, top=0.8, left=0.1)
    fig.text(0.1, 0.96, 
            '{}: Chasing frames, min stim_hz: {:.3f}'.format(cond_str, min_courted_stimhz), fontsize=8)
      
    # save 
    putil.label_figure(fig, figid, fontsize=4, y=1.0)
    figname = '{}_{}_egoc_hue-stimhz_{}'.format(cond_str.replace('/', '-'), sp, clf_str)
    print(figname)
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    #plt.savefig(os.path.join(figdir, '{}.svg'.format(figname)))


#%%


#%%
# -- FILTERING PARAMS --
min_vel = 5
max_facing_angle = np.deg2rad(45)
max_dist_to_other = 30
max_targ_pos_theta = np.deg2rad(160) #270 #160
min_targ_pos_theta = np.deg2rad(-160) # -160
min_wing_ang_deg = 30
min_wing_ang = np.deg2rad(min_wing_ang_deg)

court_ = filter_court(f1, min_vel=min_vel, max_facing_angle=max_facing_angle, 
                      max_dist_to_other=max_dist_to_other, 
                      max_targ_pos_theta=max_targ_pos_theta, 
                      min_targ_pos_theta=min_targ_pos_theta,
                      min_wing_ang=min_wing_ang, use_jaaba=False)
court_ = court_.reset_index(drop=True)

# Split theta_error into big/small, look at distribution
# ------------------------------------------------
# Compare small vs. large theta -error
# -------------------------------------------------
var1 = 'vel_shifted'
var2 = 'ang_vel_shifted'
err_palette={'small': 'r', 'large': 'cornflowerblue'}
min_frac_bout = 0.
use_bouts = True
data_type = 'BOUTS' if use_bouts else 'FRAMES'

theta_error_small_deg = 10
theta_error_large_deg = 25
theta_error_small = np.deg2rad(theta_error_small_deg)
theta_error_large = np.deg2rad(theta_error_large_deg)

chase_ = the.split_theta_error(court_, theta_error_small=theta_error_small, theta_error_large=theta_error_large)

fig = the.plot_ang_v_fwd_vel_by_theta_error_size(chase_, 
            var1=var1, var2=var2, err_palette=err_palette, lw=2)
fig.text(0.1, 0.9, cond_str, fontsize=12)

fig.axes[1].set_xlim([-10, 50])
fig.axes[2].set_xlim([-10, 50])

putil.label_figure(fig, figid)
figname = '{}_split_theta_error'.format(cond_str.replace('/', '-'))
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
            
# %%
