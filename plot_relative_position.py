#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
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

plot_style='dark'
putil.set_sns_style(style=plot_style, min_fontsize=12)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'
#%%

rootdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data'
assay = '38mm_projector'


#%% Set save directories
processedmat_dir = '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_painted_eyes/FlyTracker/processed_mats'
if not os.path.exists(processedmat_dir):
    os.makedirs(processedmat_dir)
     
figdir = os.path.join(os.path.split(processedmat_dir)[0], 'relative_position')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

figid = '{}|{}'.format(assay, processedmat_dir)

#%% Transform data

movie_fmt = '.avi'
flyid1=0
flyid2=1
subdir=None

#%
src = os.path.join(rootdir, assay)

session = '20250205-11'
flynum = 3
acqs = glob.glob(os.path.join(src, '{}*_fly{}_*'.format(session, flynum)))
#acq = '20250205-1113_fly3_Dmel-p1-left-pwder_2do-gh_2dR_cw'

session = '20250205-15'
flynum = 1
acqs2 = glob.glob(os.path.join(src, '{}*_fly{}_*'.format(session, flynum)))
for a in acqs2:
    acqs.append(a)
    
for a in acqs:
    print(a)

#%%

acq_dirs = glob.glob(os.path.join(src, '20*'))
acqs = [os.path.split(a)[-1] for a in acq_dirs]
len(acqs)
acqs[0:5]

#%%
create_new=False

d_list = []
errors = []
for acq in acqs:
    print(acq)
    #% Load mats
    acq_dir = os.path.join(rootdir, assay, acq)
    
    try:
        calib, feat, trk = util.load_flytracker_data(acq_dir, filter_ori=True)
        print(feat.shape, trk.shape)
        # Transform data to relative coordinates
        df_ = rel.get_metrics_relative_to_focal_fly(acq_dir,
                                                savedir=processedmat_dir,
                                                movie_fmt=movie_fmt, 
                                                mov_is_upstream=subdir is not None,
                                                flyid1=flyid1, flyid2=flyid2,
                                                plot_checks=False, create_new=create_new,
                                                get_relative_sizes=False)
    except Exception as e:
        errors.append(acq)
        continue
    df_['file_name'] = os.path.split(acq)[-1]
    df_['acquisition'] = ['_'.join( [f.split('-')[0], f.split('_')[1]] ) for f in df_['file_name']]

    #stimdir = acq.split('_')[-1]
    #print(stimdir)    
    #df_['stim_dir'] = stimdir    
    
    d_list.append(df_)
    
df0 = pd.concat(d_list)

print(df0.shape)
#%%

#df0['file_name'] = df0['acquisition'] #os.path.split(acq)[-1]
#df0['acquisition'] = ['_'.join( [f.split('-')[0], f.split('_')[1]] ) for f in df0['file_name']]
#df0.groupby('acquisition')['file_name'].nunique()

#%% 

# Load meta data -- assign fly numbers across acquisitions 
# Multiple acquisitions per fly

meta_fpath = glob.glob(os.path.join(src, '*.csv'))[0]
meta = pd.read_csv(meta_fpath)

meta['acquisition'] = ['_'.join( [f.split('-')[0], f.split('_')[1]] ) for f in meta['file_name']]

meta.head()

#%%
# Add all paint conditions

for fn, df_ in df0.groupby('file_name'):
    currm = meta[meta['file_name']==fn]
    assert len(currm)>0, 'No meta data for {}'.format(fn)
    #df0.loc[df0['file_name']==fn, 'stim_direction'] = currm['stim_direction'].values[0]
    stim_dir = fn.split('_')[-1]
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

#%%
# WHOLE EYE painted, each side
cond_str = 'whole_eye'

if cond_str == 'whole_eye':
    # Look at 1
    session = '20250205'
    flynum1 = 3
    acq1 = '{}_fly{}'.format(session, flynum1)

    flynum2 = 4
    acq2 = '{}_fly{}'.format(session, flynum2)

    #currdf = df0[df0['acquisition']==acq].copy()
 
# %% PLOT

# Egocentric, no STIM_HZ
fig, axn = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10,7))
for ri, (acq, currdf) in enumerate(df0[df0['acquisition'].isin([acq1, acq2])].groupby('acquisition')): #[acq1, acq2]):
    #currdf = df0[df0['acquisition']==acq].copy()
    plotd = currdf[currdf['id']==0].copy()

    for ci, (fn, df_) in enumerate(plotd.groupby('file_name')):
        ax = axn[ri, ci]
        sns.scatterplot(data=df_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
                        s=0.1, color='w')
        ax.plot(0, 0, '>', color='r', markersize=3)
        ax.set_aspect(1)
        paint_cond = '{}, {}'.format(df_['paint_side'].unique()[0], df_['paint_coverage'].unique()[0])
        stim_dir = df_['stim_direction'].unique()[0] 
        title = '{} ({})'.format(paint_cond, stim_dir)
        ax.set_title(title, loc='left')    
    ax.invert_yaxis() # to match video POV

fig.text(0.1, 0.95, 'Left: {} and Right: {}'.format(acq1, acq2), fontsize=12)
putil.label_figure(fig, figid)
plt.subplots_adjust(hspace=0.3)

figname = 'targ_rel_pos_EX-{}_{}_{}'.format(cond_str, acq1, acq2)
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))

# %% Split by STIM_HZ
n_frames = 24000
n_epochs = 10
n_frames_per_epoch = n_frames/n_epochs
epoch_starts = np.arange(0, n_frames, n_frames_per_epoch)

#d_['epoch'] = 0
wstim= []
for fn, df_ in df0.groupby('file_name'):
    df_['stim_hz']=0
    for i, v in enumerate(epoch_starts):
        df_.loc[(df_['frame']>=v) & (df_['frame']<v+n_frames_per_epoch), 'stim_hz'] = i 
    wstim.append(df_)
df0 = pd.concat(wstim)

#%%
# subdivide into smaller boutsa
bout_dur = 0.20
df0 = util.subdivide_into_subbouts(df0, bout_dur=bout_dur, grouper='file_name')

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

# %%
hue_var = 'stim_hz'
xvar = 'targ_rel_pos_x'
yvar = 'targ_rel_pos_y'
cmap = 'viridis'
stimhz_palette = putil.get_palette_dict(df0[df0[hue_var]>=0], hue_var, cmap=cmap)

acqd = df0[ (df0['id']==0) & (df0['acquisition'].isin([acq1, acq2]))].copy()

# -- FILTERING PARAMS --
min_vel = 8
max_facing_angle = np.deg2rad(180)
max_dist_to_other = 35
max_targ_pos_theta = np.deg2rad(160) #270 #160
min_targ_pos_theta = np.deg2rad(-160) # -160
min_wing_ang_deg = 30
min_wing_ang = np.deg2rad(min_wing_ang_deg)

court_ = filter_court(acqd, min_vel=min_vel, max_facing_angle=max_facing_angle, 
                      max_dist_to_other=max_dist_to_other, 
                      max_targ_pos_theta=max_targ_pos_theta, 
                      min_targ_pos_theta=min_targ_pos_theta,
                      min_wing_ang=min_wing_ang, use_jaaba=False)
court_ = court_.reset_index(drop=True)

use_bouts = False
if use_bouts:
    plotd = court_[incl_cols_for_mean].groupby(['acquisition', 'file_name', 'id', 
                           'stim_direction', 'paint_side', 'paint_coverage', 'stim_hz', 
                           'subboutnum']).mean().reset_index()   
else:
    plotd = court_.copy()
    
#currd = plotd[plotd['epoch']!=0].copy()
fig, axn = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10,7))
for ri, (acq, currdf) in enumerate(plotd.groupby('acquisition')):
    #currdf = df0[df0['acquisition']==acq].copy()
    #plotd = currd[(currd['acquisition']==acq)
    #              & currd['id']==0].copy()    
    for ci, (fn, df_) in enumerate(currdf.groupby('file_name')):
        ax = axn[ri, ci]
        sns.scatterplot(data=df_, x=xvar, y=yvar, ax=ax, 
                        s=1, alpha=0.5, palette='viridis', hue=hue_var, legend=0)
        ax.plot(0, 0, '>', color='r', markersize=3)

        # plot CoM         
        for hueval, f_ in df_.groupby(hue_var):
            cm_theta = pd.Series(np.unwrap(f_[xvar])).mean()
            cm_radius = f_[yvar].mean()
            ax.scatter(cm_theta, cm_radius, s=60, c=stimhz_palette[hueval],
                    marker='o', edgecolor='w', lw=0.5,
                    label='COM: {:.2f}'.format(hueval))

        ax.set_aspect(1)
        paint_cond = '{}, {}'.format(df_['paint_side'].unique()[0], df_['paint_coverage'].unique()[0])
        stim_dir = df_['stim_direction'].unique()[0] 
        title = '{} ({})'.format(paint_cond, stim_dir)
        ax.set_title(title, loc='left')    
    
    ax.invert_yaxis() # to match video POV

plt.subplots_adjust(hspace=0.3)

fig.text(0.1, 0.95, 'Left: {} and Right: {}'.format(acq1, acq2), fontsize=12)
putil.label_figure(fig, figid)
plt.subplots_adjust(hspace=0.3)

figname = 'targ_rel_pos_hue-stimhz_EX-{}_{}_{}'.format(cond_str, acq1, acq2)
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))

# %%
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

# Look at conditions

#%%
# Look at individual examples 
# ---------------------------
cond_str = 'LC10a_silenced'
currmeta = meta[meta['genotype_male']=='SS1-LC10a>GtACR1; P1a-CsChR']

#cond_str = 'back_twothird_painted'
#cond_str = 'front_third_painted'
if cond_str == 'back_twothird_painted':
    # Look at 1
    session = '20250304'
    flynum1 = 4
    acq = '{}_fly{}'.format(session, flynum1)
    currmeta = meta[meta['acquisition']==acq].copy()
elif cond_str == 'front_third_painted':
    # Look at 1
    session = '20250304'
    flynum1 = 6 #5
    acq = '{}_fly{}'.format(session, flynum1)
    currmeta = meta[meta['acquisition']==acq].copy()

df = df0[df0['file_name'].isin(currmeta['file_name'])].copy()

#%%
# Look at multiple examples, each condition:
# ------------------------------------------
# paint_coverage: 
#      'whole eye ', 'back 1/2', 'front 1/2', 'front 1/3', 
#      'back 2/3', 'back 1/3'

# paint_side: 'left', 'right', 'none', 'bone'
# ------------------------------------------
print(df0[['paint_coverage', 'paint_side']].drop_duplicates())

#%%
coverage = 'front 1/3'
side = 'both'
df = df0[(df0['paint_side']==side)
       & (df0['paint_coverage']==coverage)].copy()
print(df['acquisition'].unique())

cond_str = '{}_{}'.format(coverage, side)
#%%
print(cond_str)


#%%
print(df['file_name'].unique())
#df['stim_hz'] = df['epoch']

# subdivide into smaller boutsa
bout_dur = 0.20
df = util.subdivide_into_subbouts(df, bout_dur=bout_dur, grouper='file_name')

#%%
import theta_error as the

importlib.reload(the)

f1 = df[df['id']==0].copy()
f1 = the.calculate_angle_metrics_focal_fly(f1, winsize=5, grouper='file_name',
                                           has_size=False)
f1['targ_ang_vel_abs'] = np.abs(f1['targ_ang_vel'])
f1 = the.shift_variables_by_lag(f1, lag=2)

#%%
xvar = 'targ_rel_pos_x'
yvar = 'targ_rel_pos_y'
hue_var = 'stim_hz' #'epoch'
cmap='viridis'
stimhz_palette = putil.get_palette_dict(df[df[hue_var]>=0], hue_var, cmap=cmap)

meanbouts = f1[incl_cols_for_mean].groupby(['acquisition', 'file_name', 'id', 
                        'stim_direction', 'paint_side', 'paint_coverage', 'stim_hz', 
                        'subboutnum']).mean().reset_index()   
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
# PLOT, egocentric, color by stim Hz
fig, axn = plt.subplots(nr, nc, sharex=True, sharey=True, figsize=(nc*3, nr*2.5))

for ai, (acq, currdf) in enumerate(plotd.groupby('acquisition')):
    #acq = currdf['acquisition'].unique()[0]
    for ci, (fn, df_) in enumerate(currdf.groupby('file_name')):
        stim_dir = df_['stim_direction'].unique()[0] 
        #ci = 0 if stim_dir=='ccw' else 1
        if rows_are_conds:
            if nr==1:
                ax=axn[ai]
            else:
                ax=axn[ci, ai]
        elif nr==1:
            ax=axn[ci]
        else:
            ax = axn[ai, ci]
        sns.scatterplot(data=df_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
                        s=0.5, alpha=0.5, palette='viridis', hue=hue_var, legend=0)
        ax.plot(0, 0, '>', color='r', markersize=3)
        ax.set_aspect(1)
        #paint_cond = '{}, {}'.format(df_['paint_side'].unique()[0], df_['paint_coverage'].unique()[0])
        title = '{} ({})\n{}'.format(cond_str, stim_dir, acq)
        ax.set_title(title, loc='left', fontsize=10) 
        #ax.axvline(0, color=[0.5]*3, lw=2, linestyle=':')
        #ax.axhline(0, color=[0.5]*3, lw=2, linestyle=':')
       
        # plot CoM         
        if plot_com:
            for hueval, f_ in df_.groupby(hue_var):
                cm_theta = pd.Series(np.unwrap(f_[xvar])).mean()
                cm_radius = f_[yvar].mean()
                ax.scatter(cm_theta, cm_radius, s=60, c=stimhz_palette[hueval],
                        marker='o', edgecolor='w', lw=0.5,
                        label='COM: {:.2f}'.format(hueval))
        #ai+=1
    #if ci<nc-1:
    #    for i in range(ci+1, nc):
    #        axn[ai, i].axis('off')
for ax in axn.flat:
    ax.axis('off')
ax.invert_yaxis() # to match video POV
plt.subplots_adjust(hspace=0.2, wspace=0.5)

putil.label_figure(fig, figid)
figname = '{}_targ_rel_pos_hue-stimhz'.format(cond_str.replace('/', '-'))
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))

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
