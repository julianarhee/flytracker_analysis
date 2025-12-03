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
plot_style='dark'
min_fontsize=18
putil.set_sns_style(style=plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'
#%%
src_minerva = True
if src_minerva: 
    rootdir = '/Volumes/Juliana/Caitlin_RA_data'
    assay = 'Caitlin_projector'
    
    src = os.path.join(rootdir, assay)

    alt_rootdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data'
    alt_assay = '38mm_projector'
    alt_src = os.path.join(alt_rootdir, alt_assay)

else:
    rootdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data'
    assay = '38mm_projector'
    src = os.path.join(rootdir, assay)
   
    alt_rootdir = '/Volumes/Juliana/Caitlin_RA_data'
    alt_assay = 'Caitlin_projector'
    alt_src = os.path.join(alt_rootdir, alt_assay)  
    

#%% Set save directories
processedmat_dir = '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_painted_eyes/FlyTracker/processed_mats'

if not os.path.exists(processedmat_dir):
    os.makedirs(processedmat_dir)
     
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
# Load meta data -- assign fly numbers across acquisitions 
# Multiple acquisitions per fly
#%
src = os.path.join(rootdir, assay)

meta_fpath = glob.glob(os.path.join(src, '*.csv'))[0]
meta0 = pd.read_csv(meta_fpath)
meta = meta0[(meta0['tracked in matlab and checked for swaps']==1)
           & (meta0['exclude']==0) ]
           #& (meta0['annotated']==1)]
meta['acquisition'] = ['_'.join( [f.split('-')[0], \
                        f.split('_')[1]] ) for f in meta['file_name']]
print(meta.shape)
meta.head()

#%% Transform data

movie_fmt = '.avi'
flyid1=0
flyid2=1
subdir=None

#%%
# session = '20250205-11'
# flynum = 3
# acqs = glob.glob(os.path.join(src, '{}*_fly{}_*'.format(session, flynum)))
# #acq = '20250205-1113_fly3_Dmel-p1-left-pwder_2do-gh_2dR_cw'
# 
# session = '20250205-15'
# flynum = 1
# acqs2 = glob.glob(os.path.join(src, '{}*_fly{}_*'.format(session, flynum)))
# for a in acqs2:
#     acqs.append(a)
#     
# for a in acqs:
#     print(a)
 
#%%
acq_dirs = glob.glob(os.path.join(src, '20*'))
found_acqs = [os.path.split(a)[-1] for a in acq_dirs]
print("Found {} {} acqs.".format(len(found_acqs), assay))

meta_acqs = meta['file_name'].values

not_in_meta = [a for a in found_acqs if a not in meta_acqs]
not_in_src = [a for a in meta_acqs if not os.path.exists(os.path.join(src, a))]
if len(not_in_src)>0:
    print("Not in src: {}".format(len(not_in_src)))
    for i in not_in_src:
        print(i)
else:
    print("All acqs in src")

#%%
#% Check processed
processed_acqs = [a.split('_df.pkl')[0] for a in os.listdir(processedmat_dir)]
print('Found {} processed acqs'.format(len(processed_acqs)))

# Meta includes subset, for ex. annotated/tracked/excluded
proc_not_in_meta = [a for a in processed_acqs if a not in meta_acqs]
print('Found {} processed acqs not in meta'.format(len(proc_not_in_meta)))

not_processed_acqs = [a for a in meta_acqs if a not in processed_acqs]
print('Found {} meta acqs not processed'.format(len(not_processed_acqs)))
if len(not_processed_acqs)>0:
    for a in not_processed_acqs:
        print(a)
else:
    print('All meta acqs processed')

#%%

# '20250106-1522_fly2_Dmel-LC10aS_3do_gh_3dR': only 1 fly ID?
# '20250116-1021_fly4_Dmel-p1_gh_2do_2dR':  not tracked?
# '20250218-1531_fly4_Dmel-p1-right-back-half_gh_1do_1dR_ccw' - not tracked
# '20250218-1602_fly5_Dmel-p1-right-front-half_gh_1do_1dR_cw' - not trk
# '20250220-1537_fly2_Dmel-p1-front-left-third-slant_1do_gh_1dR_ccw' - 
# '20250220-1545_fly2_Dmel-p1-front-left-third-slant_1do_gh_1dR_cw' - 
create_new=False
d_list = []
errors = []
for i, acq in enumerate(meta_acqs):
    if i%10==0:
        print('Processing {} of {}: {}'.format(i, len(meta_acqs), acq))
    #% Load mats
    acq_dir = os.path.join(src, acq)
    
    try:
        #calib, trk, feat = util.load_flytracker_data(acq_dir, filter_ori=True)
        #print(feat.shape, trk.shape)
        # Transform data to relative coordinates
        df_ = rel.get_metrics_relative_to_focal_fly(acq_dir,
                                                savedir=processedmat_dir,
                                                movie_fmt=movie_fmt, 
                                                mov_is_upstream=subdir is not None,
                                                flyid1=flyid1, flyid2=flyid2,
                                                plot_checks=False, 
                                                create_new=create_new,
                                                get_relative_sizes=False)
    except Exception as e:
        alt_acq_dir = os.path.join(alt_src, acq)
        try:
            df_ = rel.get_metrics_relative_to_focal_fly(alt_acq_dir,
                                                savedir=processedmat_dir,
                                                movie_fmt=movie_fmt, 
                                                mov_is_upstream=subdir is not None,
                                                flyid1=flyid1, flyid2=flyid2,
                                                plot_checks=False, 
                                                create_new=create_new,
                                                get_relative_sizes=False)
            print("Loaded from alt src: {}".format(alt_acq_dir))    
        except Exception as e: 
            errors.append((acq, e))
            print("ERROR: {}".format(e))
            continue
    df_['file_name'] = os.path.split(acq)[-1]
    df_['acquisition'] = ['_'.join( [f.split('-')[0], f.split('_')[1]] ) for f in df_['file_name']]    
    df_['species'] = 'Dmel' if 'mel' in acq else 'Dyak' 
    df_['date'] = [int(a.split('_')[0]) for a in df_['acquisition']]
    df_['annotated'] = meta[meta['file_name']==acq]['annotated'].values[0]
    d_list.append(df_)
    
df0 = pd.concat(d_list)
print(df0.shape)
#
assert len(meta_acqs) - len(errors) == df0['file_name'].nunique(), 'Not all acqs processed'  

#%%
for err in errors:
    print(err)
    

#%%

#df0['file_name'] = df0['acquisition'] #os.path.split(acq)[-1]
#df0['acquisition'] = ['_'.join( [f.split('-')[0], f.split('_')[1]] ) for f in df0['file_name']]
#df0.groupby('acquisition')['file_name'].nunique()


#%%
# Add all paint conditions
# OPTIMIZED: Create mapping dictionaries and use vectorized operations instead of repeated .loc calls
def extract_paint_side(manipulation):
    """Extract paint side from manipulation string."""
    if pd.isna(manipulation):
        return 'none'
    manipulation_str = str(manipulation)
    if manipulation_str.startswith('no '):
        return 'none'
    elif manipulation_str.startswith('left '):
        return 'left'
    elif manipulation_str.startswith('right '):
        return 'right'
    elif manipulation_str.startswith('both '):
        return 'both'
    return 'none'

def extract_stim_direction(file_name):
    if '_ccw' in file_name:
        return 'ccw'
    else:
        return 'cw'
    
# Create mappings from meta dataframe (one-time computation)
# Handle duplicate file_name values by keeping the first occurrence
meta_unique = meta.drop_duplicates(subset='file_name', keep='first')

# Check for duplicates and warn if found
if len(meta_unique) < len(meta):
    duplicates = meta[meta.duplicated(subset='file_name', keep=False)]['file_name'].unique()
    print(f"Warning: Found {len(duplicates)} duplicate file_name(s) in meta. Keeping first occurrence.")
    print("Duplicate filenames:")
    for dup_fn in duplicates:
        dup_count = len(meta[meta['file_name'] == dup_fn])
        print(f"  - {dup_fn} (appears {dup_count} times)")

meta_dict = meta_unique.set_index('file_name').to_dict('index')

# Verify all file_names have metadata (assertion from original code)
for fn in df0['file_name'].unique():
    assert fn in meta_dict, 'No meta data for {}'.format(fn)

# Build mapping dictionaries for vectorized assignment
stim_dir_map = {fn: meta_dict[fn]['stim_direction'] for fn in df0['file_name'].unique()}
paint_coverage_map = {fn: meta_dict[fn]['painted'] for fn in df0['file_name'].unique()}
paint_side_map = {fn: extract_paint_side(meta_dict[fn]['manipulation_male']) 
                  for fn in df0['file_name'].unique()}

# Vectorized assignment (much faster than .loc in loop)
df0['stim_direction'] = df0['file_name'].map(stim_dir_map)
df0['paint_coverage'] = df0['file_name'].map(paint_coverage_map)
df0['paint_side'] = df0['file_name'].map(paint_side_map) 

#%%
#df0['species'] = [ 'Dmel' if 'mel' in a else 'Dyak' for a in df0['file_name'] ]

df0.groupby(['species', 'paint_side', 'paint_coverage'])['acquisition'].nunique()

#%%

df0[df0['paint_side']=='both'].groupby(['paint_coverage', 'species'])['acquisition'].nunique()

#%%
# Save total?
aggregate_fpath = os.path.join(os.path.split(processedmat_dir)[0], 'transformed_projector_data.parquet')
# Save parquet
df0.to_parquet(aggregate_fpath)
print(f"Saved to: {aggregate_fpath}")

#%%
def plot_egocentric_scatter(df_, ax=None, xvar='targ_rel_pos_x', yvar='targ_rel_pos_y', 
                            plot_hue=True, hue_var='stim_hz', marker_size=5, 
                            plot_com=False, fly_color='r', hue_norm=None,
                            bg_color='k', cmap='viridis', com_markersize=60, 
                            com_edgecolor='none', com_lw=0, alpha=0.5,
                            edgecolor='none', lw=0):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    if plot_hue: 
        if hue_norm is None:
            vmin, vmax = df_[hue_var].min(), df_[hue_var].max()
            hue_norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sns.scatterplot(data=df_, x=xvar, y=yvar, ax=ax,
                        s=marker_size, alpha=alpha, palette=cmap, 
                        hue=hue_var, hue_norm=hue_norm,
                        legend=0, edgecolor=edgecolor, linewidth=lw)
    else:
        sns.scatterplot(data=df_, x=xvar, y=yvar, 
                        ax=ax, s=marker_size, color=bg_color)

    fly_marker = '>' if xvar=='targ_rel_pos_x' else '^'
    ax.plot(0, 0, fly_marker, color=fly_color, markersize=3)
    ax.set_aspect(1)
    ax.axis('off')
    if plot_com:
        for hueval, f_ in df_.groupby(hue_var):
            cm_theta = pd.Series(np.unwrap(f_[xvar])).mean()
            cm_radius = f_[yvar].mean()
            ax.scatter(cm_theta, cm_radius, s=60, c=stimhz_palette[hueval],
                    marker='o', edgecolor='none', lw=0,
                    label='COM: {:.2f}'.format(hueval))
    ax.set_aspect(1)
    ax.axis('off')
    return ax

def add_stimhz_colorbar(fig, cmap, max_hz=11, axes=[0.92, 0.6, 0.015, 0.25]):
    # add custom colorbar for stimhz
    cbar_ax = fig.add_axes(axes)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_hz))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Stimulus Frequency (Hz)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    return cbar


#%%
# --------------------------------------------------------
# EXAMPLE: WHOLE EYE painted, each side
# --------------------------------------------------------
cond_str = 'whole_eye'
xvar = 'targ_rel_pos_y'
yvar = 'targ_rel_pos_x'
if cond_str == 'whole_eye':
    # Look at 1
    session = '20250205'
    flynum1 = 3
    acq1 = '{}_fly{}'.format(session, flynum1)

    flynum2 = 4
    acq2 = '{}_fly{}'.format(session, flynum2)
    #currdf = df0[df0['acquisition']==acq].copy() 
# % PLOT
# Egocentric, no STIM_HZ
fig, axn = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10,7))
for ri, (acq, currdf) in enumerate(df0[df0['acquisition'].isin([acq1, acq2])].groupby('acquisition')): #[acq1, acq2]):
    #currdf = df0[df0['acquisition']==acq].copy()
    plotd = currdf[currdf['id']==0].copy()

    for ci, (fn, df_) in enumerate(plotd.groupby('file_name')):
        print(fn)
        ax = axn[ri, ci]
        #sns.scatterplot(data=df_, x='targ_rel_pos_y', y='targ_rel_pos_x', 
        #                ax=ax, 
        #                s=0.1, color=bg_color)
        ax = plot_egocentric_scatter(df_, ax=ax, marker_size=0.1,
                                     xvar=xvar, yvar=yvar,
                                     plot_hue=False, bg_color=bg_color)
        #ax.plot(0, 0, '>', color='r', markersize=3)
        #ax.set_aspect(1)
        #ax.invert_yaxis() # to match video POV
        paint_cond = '{}, {}'.format(df_['paint_side'].unique()[0], df_['paint_coverage'].unique()[0])
        stim_dir = df_['stim_direction'].unique()[0] 
        title = '{} ({})'.format(paint_cond, stim_dir)
        ax.set_title(title, loc='left')    
    
for ax in axn.flat:
    ax.axis('off')
    
fig.text(0.1, 0.95, 'Left: {} and Right: {}'.format(acq1, acq2), fontsize=12)
putil.label_figure(fig, figid)
plt.subplots_adjust(hspace=0.3)

figname = 'example_targ_rel_pos-{}_{}_{}'.format(cond_str, acq1, acq2)
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
#plt.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

# %% 
# Split by STIM_HZ
df0 = util.add_stim_hz(df0, n_frames=24000, n_epochs=10)
# n_frames = 24000
# n_epochs = 10
# n_frames_per_epoch = n_frames/n_epochs
# epoch_starts = np.arange(0, n_frames, n_frames_per_epoch)
# 
# #d_['epoch'] = 0
# wstim= []
# for fn, df_ in df0.groupby('file_name'):
#     df_['stim_hz']=0
#     for i, v in enumerate(epoch_starts):
#         df_.loc[(df_['frame']>=v) & (df_['frame']<v+n_frames_per_epoch), 'stim_hz'] = i 
#     wstim.append(df_)
# df0 = pd.concat(wstim)
 
#%%
# subdivide into smaller boutsa
#bout_dur = 0.20
#df0 = util.subdivide_into_subbouts(df0, bout_dur=bout_dur, grouper='file_name')

#%%
#%
exclude_cols = ['fpath', 'stim_dir']
incl_cols_for_mean = [c for c in df0.columns if c not in exclude_cols]
#if 'fpath' in df0.columns:
#    df0.drop(columns=['fpath'], inplace=True) 
#if 'stim_dir' in df0.columns:
#    df0.drop(columns=['stim_dir'], inplace=True)
#if 'paint_coverage' in df0.columns:
#    df0.drop(columns=['paint_coverage'], inplace=True)

#%%
def filter_court(df,ftjaaba=None, min_vel=10, 
                 max_facing_angle=np.deg2rad(45), 
                 max_dist_to_other=20, 
                 max_targ_pos_theta=np.deg2rad(270), 
                 min_targ_pos_theta=np.deg2rad(-270),
                 min_wing_ang=np.deg2rad(30), use_jaaba=False):
    if use_jaaba:
        court_ = ftjaaba[(ftjaaba['id']==0) & (ftjaaba['chasing']==1)].copy() 
        court_filter_str = 'jaaba'
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

# %%
# FILTER CHASING FRAMES
# --------------------------
plot_com= False#True 
use_bouts = False
marker_size = 5 if use_bouts else 2

bout_str = 'bouts' if use_bouts else 'frames'
com_str = 'com' if plot_com else 'no-com'
hue_var = 'stim_hz'

fig_str = 'hue-{}_{}_{}'.format(hue_var, bout_str, com_str)

xvar = 'targ_rel_pos_y'
yvar = 'targ_rel_pos_x'
cmap = 'viridis'
stimhz_palette = putil.get_palette_dict(df0[df0[hue_var]>=0], hue_var, cmap=cmap)
acqd = df0[ (df0['id']==0) & (df0['acquisition'].isin([acq1, acq2]))
           & (df0['stim_hz']>0)].copy()

hue_norm = plt.Normalize(vmin=0, vmax=1) #vmax=n_epochs-1)

# -- FILTERING PARAMS --
min_vel = 10
max_facing_angle = np.deg2rad(180)
max_dist_to_other = 20
max_targ_pos_theta = np.deg2rad(180) #270 #160
min_targ_pos_theta = np.deg2rad(-180) # -160
min_wing_ang_deg = 10
min_wing_ang = np.deg2rad(min_wing_ang_deg)

court_ = filter_court(acqd, min_vel=min_vel, max_facing_angle=max_facing_angle, 
                      max_dist_to_other=max_dist_to_other, 
                      max_targ_pos_theta=max_targ_pos_theta, 
                      min_targ_pos_theta=min_targ_pos_theta,
                      min_wing_ang=min_wing_ang, use_jaaba=False)
court_ = court_.reset_index(drop=True)

if use_bouts:
    plotd = court_[incl_cols_for_mean].groupby(['acquisition', 'file_name', 'id', 
                           'stim_direction', 'paint_side', 'paint_coverage', 'stim_hz', 
                           'subboutnum']).mean().reset_index()   
else:
    plotd = court_.copy()

fig, axn = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10,7))
for ri, (acq, currdf) in enumerate(plotd.groupby('acquisition')):
    for ci, (fn, df_) in enumerate(currdf.groupby('file_name')):
        ax = axn[ri, ci]
        ax = plot_egocentric_scatter(df_, ax=ax, xvar=xvar, yvar=yvar, hue_var=hue_var,
                                     plot_hue=True, marker_size=2, 
                                     plot_com=plot_com,
                                    bg_color=bg_color, cmap=cmap)
        paint_cond = '{}, {}'.format(df_['paint_side'].unique()[0], df_['paint_coverage'].unique()[0])
        stim_dir = df_['stim_direction'].unique()[0] 
        title = '{} ({})'.format(paint_cond, stim_dir)
        ax.set_title(title, loc='left')     
        #ax.invert_yaxis() # to match video POV
        
for ax in axn.flat:
    ax.axis('off')

# add custom colorbar for stimhz
cbar = add_stimhz_colorbar(fig, cmap, max_hz=1)

plt.subplots_adjust(hspace=0.3)

fig.text(0.1, 0.95, 'Left: {} and Right: {}'.format(acq1, acq2), fontsize=12)
putil.label_figure(fig, figid)
plt.subplots_adjust(hspace=0.3)

figname = 'example_targ_rel_pos_{}_{}_{}_stimhz'.format(cond_str, acq1, acq2)
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
#plt.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

#%%
# Look at individual examples 
# ---------------------------
cond_str = 'LC10a_silenced'
currmeta = meta[meta['genotype_male']=='SS1-LC10a>GtACR1; P1a-CsChR']
df = df0[(df0['file_name'].isin(currmeta['file_name']))
         & (df0['stim_hz']>0)].copy()
#%
court_ = filter_court(df, min_vel=min_vel, max_facing_angle=max_facing_angle, 
                      max_dist_to_other=max_dist_to_other, 
                      max_targ_pos_theta=max_targ_pos_theta, 
                      min_targ_pos_theta=min_targ_pos_theta,
                      min_wing_ang=min_wing_ang, use_jaaba=False)
court_ = court_.reset_index(drop=True)

ncols = court_['acquisition'].nunique()

fig, axn = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10,7))
for ci, (fn, df_) in enumerate(court_.groupby('file_name')):
    
    ax = axn.flat[ci]
    #for ri, currdir in enumerate(['cw', 'ccw']):
    ax = plot_egocentric_scatter(df_, ax=ax, xvar=xvar, yvar=yvar, hue_var=hue_var,
                                 plot_hue=True, marker_size=2, 
                                 hue_norm=hue_norm,
                                 plot_com=plot_com,
                                bg_color=bg_color, cmap=cmap)
    currdir = df_['stim_direction'].unique()[0]
    title = currdir #'{}: ({})'.format(currdir, fn)
    ax.set_title(title, loc='left', fontsize=8)
#ax.invert_yaxis() # to match video POV #don't invert if plotting y vs. x
plt.subplots_adjust(hspace=0.3)

# add custom colorbar for stimhz
cbar = add_stimhz_colorbar(fig, cmap, 1, 
                           axes=[0.92, 0.68, 0.015, 0.2])

fig.text(0.1, 0.95, 'LC10a-GtACR1; P1-CsChrimson', fontsize=12)
putil.label_figure(fig, figid)
plt.subplots_adjust(hspace=0.3)

figname = 'LC10a-GtACR1_egocentric_{}_{}'.format(cond_str,  fig_str)
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
#plt.savefig(os.path.join(figdir, '{}.svg'.format(figname)))


#%%
lc10a_silenced_court_ = court_.copy() #[court_['file_name'].isin(currmeta['file_name'])]
# Create a histogram of the targ_rel_pos_x and targ_rel_pos_y converted to 
# theta (and radius).
lc10a_silenced_court_['theta'] = np.arctan2(lc10a_silenced_court_['targ_rel_pos_y'], lc10a_silenced_court_['targ_rel_pos_x'])
lc10a_silenced_court_['radius'] = np.sqrt(lc10a_silenced_court_['targ_rel_pos_y']**2 + lc10a_silenced_court_['targ_rel_pos_x']**2)

fig, ax = plt.subplots(figsize=(5, 5))
sns.histplot(lc10a_silenced_court_, x='theta', ax=ax, bins=100)

#%%
# look at control

control_df = df0[(df0['paint_coverage']=='none')
                & (df0['paint_side']=='none')].copy()
control_df.groupby('species')['acquisition'].nunique()

#%%
control_court_ = filter_court(control_df, min_vel=10, 
                              max_facing_angle=max_facing_angle, 
                      max_dist_to_other=max_dist_to_other, 
                      max_targ_pos_theta=max_targ_pos_theta, 
                      min_targ_pos_theta=min_targ_pos_theta,
                      min_wing_ang=min_wing_ang, use_jaaba=False)
control_court_ = control_court_.reset_index(drop=True)

#%%
stim_dir = 'CCW'

ccw_control_court_ = control_court_[(control_court_['stim_hz']>0)
                                & (control_court_['stim_direction']==stim_dir)].copy()
n_mel = ccw_control_court_[ccw_control_court_['species']=='Dmel']['acquisition'].nunique()
n_dyak = ccw_control_court_[ccw_control_court_['species']=='Dyak']['acquisition'].nunique()

print(f"Dmel: {n_mel}, Dyak: {n_dyak}")

#%%
mel_controls = ccw_control_court_[ccw_control_court_['species']=='Dmel']['acquisition'].unique()#[0::2]
dyak_controls = ccw_control_court_[ccw_control_court_['species']=='Dyak']['acquisition'].unique()#[0::2]
all_controls = np.concatenate([mel_controls, dyak_controls])

#stim_dir = 'CW'
plotd = ccw_control_court_[(ccw_control_court_['acquisition'].isin(all_controls))].reset_index(drop=True)

#%%
nr=4
nc=6
for ri, (sp, curr_court) in enumerate(plotd.groupby('species')):
    fig, axn = plt.subplots(nr, nc, sharex=True, sharey=True, figsize=(10,5))
    for ci, (acq, df_) in enumerate(curr_court.groupby('acquisition')):
        if ci>(nr*nc-1):
            break
        ax = axn.flat[ci]
        ax = plot_egocentric_scatter(df_, ax=ax, xvar=xvar, yvar=yvar, hue_var=hue_var,
                                     plot_hue=True, marker_size=0.5,
                                     plot_com=plot_com,
                                    bg_color=bg_color, cmap=cmap)
        ax.set_title(f'{ci}\n{acq}', loc='left', fontsize=8)
    fig.suptitle(f'{sp}, {stim_dir}', fontsize=12)

    for ax in axn.flat:
        ax.axis('off')
    # Save
    figname = f'{sp}_controls_{stim_dir}'
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))

#%%
# Plot example
#fn_mel = control_court_[control_court_['species']=='Dmel']['file_name'].unique()[1]
#fn_dyak = control_court_[control_court_['species']=='Dyak']['file_name'].unique()[0]
if stim_dir == 'CW':
    acq_mel = '20250214_fly1'
    acq_dyak = '20250912_fly2'
else:
    acq_mel = '20241217_fly1'
    acq_dyak = '20250312_fly1'
#acq_dyak = '20250702_fly1'
marker_size=5
mel_control_ex = plotd[(plotd['acquisition']==acq_mel)
                       & (plotd['species']=='Dmel')].copy()
dyak_control_ex = plotd[(plotd['acquisition']==acq_dyak)
                       & (plotd['species']=='Dyak')].copy()

fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
ax = axn[0]
ax = plot_egocentric_scatter(mel_control_ex, ax=ax, xvar=xvar, yvar=yvar, hue_var=hue_var,
                                 plot_hue=True, marker_size=marker_size, plot_com=plot_com,
                                bg_color=bg_color, cmap=cmap)
ax.set_title(f'Dmel: {acq_mel}', loc='left', fontsize=8)
ax = axn[1]
ax = plot_egocentric_scatter(dyak_control_ex, ax=ax, xvar=xvar, yvar=yvar, hue_var=hue_var,
                                 plot_hue=True, marker_size=marker_size, plot_com=plot_com,
                                bg_color=bg_color, cmap=cmap)
ax.set_title(f'Dyak: {acq_dyak}', loc='left', fontsize=8)

figname  = f'example_Dmel_Dyak_controls_{stim_dir}'
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))


#%%
control_court_['theta'] = np.arctan2(control_court_['targ_rel_pos_y'], control_court_['targ_rel_pos_x'])
control_court_['radius'] = np.sqrt(control_court_['targ_rel_pos_y']**2 + control_court_['targ_rel_pos_x']**2)

fig, ax = plt.subplots(figsize=(5, 5))
sns.histplot(control_court_, x='theta', hue='species', ax=ax, 
             bins=100, stat='density', common_norm=False)

figname = f'{stim_dir}_theta_distribution_control'
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))

#%%
control_court_['genotype'] = 'P1-CsChR'
lc10a_silenced_court_['genotype'] = 'LC10a-GtACR1'
mel_control = control_court_[control_court_['species']=='Dmel']

stim_dir = 'CCW'
tmp_combined = pd.concat([lc10a_silenced_court_[lc10a_silenced_court_['stim_direction']==stim_dir], 
                          mel_control[mel_control['stim_direction']==stim_dir]])
fig, ax = plt.subplots(figsize=(5, 5))
sns.histplot(tmp_combined, x='theta', ax=ax, bins=100, hue='genotype', 
             stat='density', common_norm=False)
ax.axvline(x=0, color='w', linestyle='--', lw=0.5)
sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

figname = f'LC10a-GtACR1_vs_control_theta_distribution_{stim_dir}'
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))

#%%
# Look at multiple examples, each condition:
# ------------------------------------------
# paint_coverage: 
#      'whole eye ', 'back 1/2', 'front 1/2', 'front 1/3', 
#      'back 2/3', 'back 1/3'

# paint_side: 'left', 'right', 'none', 'bone'
# ------------------------------------------

#cond_str = 'back_twothird_painted'
#cond_str = 'front_third_painted'
# f cond_str == 'back_twothird_painted':
#    # Look at 1
#    session = '20250304'
#    flynum1 = 4
#    acq = '{}_fly{}'.format(session, flynum1)
#    currmeta = meta[meta['acquisition']==acq].copy()
# lif cond_str == 'front_third_painted':
#    # Look at 1
#    session = '20250304'
#    flynum1 = 6 #5
#    acq = '{}_fly{}'.format(session, flynum1)
#    currmeta = meta[meta['acquisition']==acq].copy()

#print(df0[['paint_coverage', 'paint_side']].drop_duplicates())
df0[df0['paint_side']=='both'].groupby(['paint_coverage', 'species'])['acquisition'].nunique()
# paint_coverage      species
# back 1/2            Dmel        3
# back 1/2 vertical   Dmel        3
#                     Dyak        8
# back 1/4 vertical   Dmel        5
#                     Dyak        1
# back 2/3            Dmel        5
# back 3/4            Dmel        6
#                     Dyak        3
# front 1/2           Dmel        7
# front 1/2 vertical  Dyak        8
# front 1/3           Dmel        4
# front 1/4 parallel  Dyak        3
# front 1/4 vertical  Dmel       15
#                     Dyak        7
# Name: acquisition, dtype: int64
  
#%%
coverage = 'front 1/3' #2/3' #1/2 vertical'
side = 'both'
df = df0[(df0['paint_side']==side)
       & (df0['paint_coverage']==coverage)].copy()
       #& (df0['date']>=20250512)].copy()
print(df['acquisition'].unique())

cond_str = '{}_{}'.format(coverage, side)
print(cond_str)
#%
print(df['file_name'].unique())
#df['stim_hz'] = df['epoch']
# subdivide into smaller boutsa
bout_dur = 0.20
df = util.subdivide_into_subbouts(df, bout_dur=bout_dur, grouper='file_name')

#%%
import theta_error as the
importlib.reload(the)

f1 = df[df['id']==0].copy()
f1 = the.calculate_angle_metrics_focal_fly(f1, winsize=5, grouper='file_name')
                                           #has_size=False)
f1['targ_ang_vel_abs'] = np.abs(f1['targ_ang_vel'])
f1 = the.shift_variables_by_lag(f1, lag=2)

#%%
xvar = 'targ_rel_pos_y'
yvar = 'targ_rel_pos_x'
hue_var = 'stim_hz' #'epoch'
cmap='viridis'
stimhz_palette = putil.get_palette_dict(df[df[hue_var]>=0], hue_var, cmap=cmap)

#meanbouts = f1[incl_cols_for_mean].groupby(['species', 'acquisition', 'file_name', 'id', 
#                        'stim_direction', 'paint_side', 'paint_coverage', 'stim_hz', 
#                        'subboutnum']).mean().reset_index()   
#meanbouts['stim_hz'] = meanbouts['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))  

plot_com = False

use_bouts = False
use_frames=True

#if use_bouts:
#    plotd = meanbouts.copy()
if use_frames:
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

min_vel = 10
max_facing_angle = np.deg2rad(180)
max_dist_to_other = 25 #0
max_targ_pos_theta = np.deg2rad(270) #270 #160
min_targ_pos_theta = np.deg2rad(-270) # -160
min_wing_ang_deg = 5
min_wing_ang = np.deg2rad(min_wing_ang_deg)

court_ = filter_court(plotd, min_vel=min_vel, max_facing_angle=max_facing_angle, 
                      max_dist_to_other=max_dist_to_other, 
                      max_targ_pos_theta=max_targ_pos_theta, 
                      min_targ_pos_theta=min_targ_pos_theta,
                      min_wing_ang=min_wing_ang, use_jaaba=False)
court_ = court_.reset_index(drop=True)

#%%
marker_size=1
cmap='viridis'
# PLOT, egocentric, color by stim Hz



for sp, curr_court in court_.groupby('species'):
    nacq=curr_court['acquisition'].nunique()
    
    #nc=curr_court.groupby('acquisition')['file_name'].nunique().max()
    if nacq > 3:
        nc = 3
        nr = int(np.ceil(nacq/nc)) #acq // nc
    else:
        nc = nacq
        nr = 1
    fig, axn = plt.subplots(nr, nc, sharex=True, sharey=True, 
                            figsize=(10, 10))
    fig.suptitle(f'{sp}: {cond_str}', fontsize=12)
    for ai, (acq, currdf) in enumerate(curr_court.groupby('acquisition')):
        #acq = currdf['acquisition'].unique()[0]
#         for fi, (fn, df_) in enumerate(currdf.groupby('file_name')):
#             stim_dir = df_['stim_direction'].unique()[0]
#             #if stim_dir == 'CCW':
#             ci = 0 if stim_dir == 'CCW' else 1
#             if rows_are_conds:
#                 ax=axn[ci, ai]
#             else:
#                 ax=axn[ai, ci]
#             stim_dir = df_['stim_direction'].unique()[0] 
        if nr==1:
            ax=axn[ai]
        else:
            ax = axn.flat[ai] 
        ax = plot_egocentric_scatter(currdf, ax=ax, xvar=xvar, yvar=yvar, hue_var=hue_var,
                            plot_hue=True, marker_size=marker_size, plot_com=plot_com,
                            bg_color=bg_color, cmap=cmap)
                
        title = '{}\n{}'.format(cond_str, acq)
        ax.set_title(title, loc='left', fontsize=10) 

    for ax in axn.flat:
        ax.axis('off')

    plt.subplots_adjust(hspace=0.2, wspace=0.5)

    putil.label_figure(fig, figid)
    figname = '{}_{}_egocentric_hue-stimhz'.format(cond_str.replace('/', '-'), sp)
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
