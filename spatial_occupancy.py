#!/usr/bin/env python3
# -*- coding:  -*
"""
Created on Mon Apr  1 15:11:00 2024

"""

#%%
import os
import glob
import numpy as np
import pandas as pd

import importlib
import matplotlib as mpl
import pylab as pl
import seaborn as sns

# %%
import matplotlib.gridspec as gridspec

from transform_data.relative_metrics import load_processed_data
import utils as util
import plotting as putil

#%%
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=12)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%%
create_new = False

# Set sourcedirs
# srcdir = '/Volumes/Juliana/2d-projector-analysis/FlyTracker/processed_mats' #relative_metrics'
srcdir = '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker/processed'

if plot_style == 'white':
    figdir = os.path.join(os.path.split(srcdir)[0], 'spatial_occupancy', 'figures', 'white')
else:
    figdir = os.path.join(os.path.split(srcdir)[0], 'spatial_occupancy', 'figures')

if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

fig_id = srcdir

# LOCAL savedir 
#localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/2d-projector/FlyTracker'
#localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/MF/38mm-dyad/FlyTracker'
localdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/free_behavior/38mm_dyad/MF/FlyTracker'
out_fpath_local = os.path.join(localdir, 'processed.pkl')
print(out_fpath_local)

#print("There are {} processed files".format( len(os.listdir(srcdir))))

#found_ = glob.glob(os.path.join(srcdir, '*{}.pkl'.format('df')))
#len(found_)


#%%
importlib.reload(util)
create_new = False

if not create_new:
    if os.path.exists(out_fpath_local):
        df = pd.read_pickle(out_fpath_local)
        print("Loaded local processed data.")
    else:
        create_new = True

if create_new:
    print("Creating new.")
    df = util.load_aggregate_data_pkl(srcdir, mat_type='df')
    print(df['species'].unique())

    #% save
    out_fpath = os.path.join(os.path.split(figdir)[0], 'processed.pkl')
    df.to_pickle(out_fpath)
    print(out_fpath)

    # save local, too
    df.to_pickle(out_fpath_local)

print(df[['species', 'acquisition']].drop_duplicates().groupby('species').count())

# %%

# acq = '20240322-1105_f5_eleWT_4do_gh'
# acq = '20231223-1212_fly3_eleWT_5do_sh_eleWT_5do_gh'
# acq = '20231223-1117_fly1_eleWT_5do_sh_eleWT_5do_gh'
# acq = '20231223-1212_fly3_eleWT_5do_sh_eleWT_5do_gh'
# acq = '20231223-1136_fly2_eleWT_5do_sh_eleWT_5do_gh'

# acq = '20240109-1545_fly4_yakWT_4do_sh_yakWT_4do_gh'
# acq = '20240119-1517-fly7-yakWT_3do_sh_yakWT_3do_gh'
# acq = '20240116-1015-fly1-yakWT_4do_sh_yakWT_4do_gh'
# acq = '20240116-1500-fly3-yakWT_4do_sh_yakWT_4do_gh' # little or no courtship

# acq = '20240115-1100-fly2-melWT_3do_sh_melWT_3do_gh'
# acq = '20240126-1122-fly5-melWT_4do_sh_melWT_4do_gh'
# acq = '20240119-1149-fly5-melWT_4do_sh_melWT_4do_gh' # circling 13min in
acq = '20240118-1425-fly3-melWT_3do_sh_melWT_3do_gh'

mat_type = 'df'

single_fly_srcdir = os.path.join(os.path.split(srcdir)[0], 'processed_mats')
found_fns = glob.glob(os.path.join(single_fly_srcdir, '{}*{}.pkl'.format(acq, mat_type)))
print(found_fns)
fp = found_fns[0]
df_ = pd.read_pickle(fp)
acq = os.path.split(fp)[1].split('_{}'.format(mat_type))[0] 
df_['acquisition'] = acq

df_['id'].unique()

#%%

#import relative_metrics as rem
#importlib.reload(rem)
#df_ = rem.get_metrics_relative_to_focal_fly(viddir)

#%%

# %%
# cap = get_video_cap_check_multidir(acq, assay='2d-projector'):

# %% Get manually annotated actions -- annoted with FlyTracker

if acq == '20240322-1105_f5_eleWT_4do_gh':
    vidbase = '/Volumes/Juliana/courtship-videos/38mm_dyad'
    viddir = glob.glob(os.path.join(vidbase, '{}*'.format(acq)))[0]

    # get path to actions file for current acquisition
    action_fpaths = glob.glob(os.path.join(viddir, '{}*'.format(acq), '*actions.mat'))
    print(action_fpaths)
    action_fpath = action_fpaths[0]

    # load actions to df
    boutdf = util.ft_actions_to_bout_df(action_fpath)

    #% Look at bilateral front wiggle
    frontwiggle = boutdf[boutdf['action']=='bilateral front wiggle']

    start_ix = frontwiggle['start'].values[0]-1
    stop_ix = frontwiggle['end'].values[0]-1

    print(start_ix, stop_ix)

    curr_frames = np.arange(start_ix, stop_ix+1)
    currdf = df_[df_['frame'].isin(curr_frames)].copy()

    #%%
    plot_vars = ['vel', 'facing_angle', 'angle_between', 'dist_to_other', 'max_wing_ang', 'min_wing_ang']

    fig, axn = pl.subplots(len(plot_vars), 1, figsize=(10, len(plot_vars)*2)) #, sharex=True)

    for i, yvar in enumerate(plot_vars):
        ax=axn[i]
        sns.histplot(data=currdf, x=yvar, ax=ax, bins=20)

    pl.subplots_adjust(hspace=0.5)

# %%
min_wing_ang = np.deg2rad(45)

wing_ext = df_[df_['min_wing_ang'] >= min_wing_ang].copy()

min_vel = 5
max_facing_angle = np.deg2rad(25)
max_dist_to_other = 25
max_targ_pos_theta = np.deg2rad(160)
min_targ_pos_theta = np.deg2rad(-160)

court_ = df_[ (df_['id']==0)
             & (df_['vel'] > min_vel)
             & (df_['targ_pos_theta'] <= max_targ_pos_theta)
             & (df_['targ_pos_theta'] >= min_targ_pos_theta)
             & (df_['facing_angle'] <= max_facing_angle)
             & (df_['min_wing_ang'] >= min_wing_ang)
             & (df_['dist_to_other'] <= max_dist_to_other)].copy()

f2_ = df_[ (df_['frame'].isin(court_['frame']))
          & (df_['id']==1)].copy() #wing_ext[wing_ext['id']==1].copy()

fig, ax = pl.subplots()
sns.histplot(data=f2_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
             cmap='magma', vmin=0, vmax=20) # %%
ax.plot(0, 0, 'w', markersize=3, marker='o')
# ax.set_xlim([])
ax.set_aspect(1)
ax.set_xlim([-800, 800])
ax.set_ylim([-800, 800])

# %%

# =============================================================================
# AGGREGATE DATA
# =============================================================================

#wing_ext = df_[df_['min_wing_ang'] >= min_wing_ang].copy()
importlib.reload(util)
jaaba = util.load_jaaba(assay='38mm_dyad', experiment='MF',
                        fname='jaaba_20240403')

if 'filename' in jaaba.columns:
    jaaba = jaaba.rename(columns={'filename': 'acquisition'})

print(jaaba[['species', 'acquisition']].drop_duplicates().groupby('species').count())
#%
d_list = []
for acq, d_ in jaaba.groupby('acquisition'):
    match_ = df[ (df['acquisition']==acq) & (df['id']==0)].copy().reset_index(drop=True)
    assert match_.shape[0] == d_.shape[0], "Bad shape: {}, jaaba {}, df {}".format(acq, d_.shape, match_.shape)
    ja_cols = [c for c in jaaba.columns if c not in match_.columns]
    currdf1 = pd.concat([match_, d_[ja_cols]], axis=1)
   
    # Repeat for Fly2 
    match2 =  df[ (df['acquisition']==acq) & (df['id']==1)].copy().reset_index(drop=True)
    assert match2.shape[0] == d_.shape[0], "Bad shape: {}, jaaba {}, df {}".format(acq, d_.shape, match_.shape)
    currdf2 = pd.concat([match2, d_[ja_cols]], axis=1)

    currdf = pd.concat([currdf1, currdf2])
    d_list.append(currdf)

ftjaaba = pd.concat(d_list)

#%%
 
use_jaaba = True

# -- FILTERING PARAMS --
min_vel = 10
max_facing_angle = np.deg2rad(90)
max_dist_to_other = 20
max_targ_pos_theta = np.deg2rad(270) #160
min_targ_pos_theta = np.deg2rad(-270) # -160
min_wing_ang_deg = 30
min_wing_ang = np.deg2rad(min_wing_ang)

#%
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
# Get female-centered frames
f_list = []
for acq, curr_court in court_.groupby('acquisition'):
   
    # NOTE: BEFORE, this was using df, instead of df_!! 
    f2_ = df[ (df['frame'].isin(curr_court['frame']))
             & (df['id']==1)
             & (df['acquisition']==acq)].copy() #wing_ext[wing_ext['id']==1].copy()
    f_list.append(f2_)
f2 = pd.concat(f_list)
#f_list = []
#for acq, df_ in sing_.groupby('acquisition'):
#    
#    f2_ = df[ (df['frame'].isin(df_['frame']))
#             & (df['id']==1)].copy() #wing_ext[wing_ext['id']==1].copy()
#    f_list.append(f2_)
#f2_sing = pd.concat(f_list)

#%% CHECK ONE
cmap = 'magma' #'YlOrBr'
stat = 'probability' #'count' #count'
vmax=0.0015 if stat=='probability' else 250

acq = f2[f2['species']=='Dyak']['acquisition'].unique()[8]
print(acq)

df_ = f2[f2['acquisition']==acq].copy()

fig, ax = pl.subplots()
sns.histplot(data=df_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
             cmap=cmap, stat=stat, vmin=0, vmax=vmax, bins=100)
ax.set_aspect(1)

#%% 
# Plot ALL data -- SPATIAL OCUPANCY
common_bins = True

cmap = 'magma' #'YlOrBr'
stat = 'probability' #'count' #count'
vmax=0.0015 if stat=='probability' else 250
#vmax = 0.002 if stat=='probability' else 250
bins=100 if common_bins else 'auto'
axlim = 500

norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
fig, axn = pl.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
for ai, sp in enumerate(['Dmel', 'Dyak', 'Dele']): #(sp, df_) in enumerate(f2_.groupby('species')):

    ax=axn[ai]
    df_ = f2[f2['species']==sp].copy().reset_index(drop=True)
    sns.histplot(data=df_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
             cmap=cmap, stat=stat, vmin=0, vmax=vmax, bins=bins) # %%
    ax.plot(0, 0, 'w', markersize=3, marker='>')
    ax.set_title(sp)
    # ax.set_xlim([])
    ax.set_aspect(1)
    # if sp=='Dmel': #or sp=='Dyak':
    #     ax.set_xlim([-250, 250])
    #     ax.set_ylim([-250, 250])
    # else:
    ax.set_xlim([-axlim, axlim])
    ax.set_ylim([-axlim, axlim])

    ax.axvline(0, color=bg_color, linestyle=':', lw=0.5)
    ax.axhline(0, color=bg_color, linestyle=':', lw=0.5)

    ax.set_xlabel('x-pos (pix)')
    ax.set_ylabel('y-pos (pix)')

putil.colorbar_from_mappable(ax, norm=norm, cmap=cmap, axes=[0.92, 0.3, 0.01, 0.4],
                             hue_title=stat)

pl.subplots_adjust(wspace=0.4, top=0.9)
fig.suptitle('Male position relative to female (bins-{})\n{}'.format(bins, court_filter_str), 
             x=0.1, y=0.9, fontsize=10, horizontalalignment='left')
putil.label_figure(fig, fig_id)

fig_str = '{}_court_{}'.format(stat, court_filter_str)
figname = 'male-pos-relative-to-fem_bins-{}_{}'.format(bins, fig_str)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)), dpi=300, bbox_inches='tight')
#pl.savefig(os.path.join(figdir, '{}.svg'.format(figname))) #, dpi=300, bbox_inches='tight')
print(figdir)
print(figname)

#%% What is the range of variables used for filtering ?

params = ['targ_pos_theta', 'facing_angle', 'dist_to_other', 'vel']
fig, axn = pl.subplots(1, len(params), sharey=True, 
                       figsize=(8,4))
for ai, par in enumerate(params):
    ax=axn.flat[ai]
    sns.histplot(court_[par], bins=20, stat='probability', ax=ax)
    ax.set_box_aspect(1)
    if par=='targ_pos_theta':
        ax.axvline(min_targ_pos_theta, color='r', linestyle='--', lw=0.5)
        ax.axvline(max_targ_pos_theta, color='r', linestyle='--', lw=0.5)
    elif par=='facing_angle':
        ax.axvline(max_facing_angle, color='r', linestyle='--', lw=0.5)
    elif par=='dist_to_other':
        ax.axvline(max_dist_to_other, color='r', linestyle='--', lw=0.5)
fig.suptitle('Filtering params, for courtship (jaaba={})'.format(use_jaaba),
             x=0.2, y=0.9, fontsize=10, horizontalalignment='left')
putil.label_figure(fig, fig_id)

figname = 'courtship_filter_params_{}'.format(court_filter_str)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

#%%           

# Check if true by acquisition
#sp = 'Dele'
#species_df = f2[f2['species']==sp].copy().reset_index(drop=True)
#species_df['acquisition'].nunique()
bins_for_each=50
vmax = 0.002
for sp, species_df in f2.groupby('species'):

    if sp=='Dyak':
        nr=4
        nc=6
    elif sp=='Dmel':
        nr=4
        nc=4
    else: # sp=='Dele':
        nr=5
        nc=6
         
    fig, axn = pl.subplots(nr, nc, figsize=(nc*1.5, nr*1.5), sharex=True, sharey=True) 
    for ai, (acq, df_) in enumerate(species_df.groupby('acquisition')):

        ax=axn.flat[ai] 
        sns.histplot(data=df_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
                cmap=cmap, stat=stat, vmin=0, vmax=vmax, bins=bins_for_each) # %%
        ax.plot(0, 0, 'w', markersize=1, marker='o')
        #ax.set_title(sp)
        # ax.set_xlim([])
        ax.set_aspect(1)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('{}, {}'.format(sp, ai), fontsize=4, loc='left')
    for ax in axn.flat[ai+1:]:
        ax.axis('off')
        
    fig.text(0.5, 0.02, 'x-pos (pix)', ha='center')
    fig.text(0.01, 0.5, 'y-pos (pix)', va='center', rotation='vertical')
    fig.suptitle('{}: male position rel. to female, {}'.format(sp, court_filter_str), 
                fontsize=10, horizontalalignment='left', x=0.1, y=0.93)

    putil.label_figure(fig, fig_id)

    figname = 'spatial_occupancy_per_pair_{}_{}'.format(sp, court_filter_str)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)), dpi=300, bbox_inches='tight')
#%%
#xvar = 'rel_vel_abs'
#yvar = 'targ_ang_size_deg'
#joint_type = 'kde'
#
#plotdf = ftjaaba[(ftjaaba['chasing']==1)]
#sns.jointplot(data=plotdf, x=xvar, y=yvar, hue='species', 
#              palette=species_palette, kind=joint_type)

# %%

print(ftjaaba[['species', 'acquisition']].drop_duplicates().groupby('species').count())

# %%

# Plot from MALE's POV 
common_bins = False

cmap = 'magma' #'YlOrBr'
stat = 'probability' #'count' #count'
vmax=0.001 if stat=='probability' else 250
#vmax = 0.002 if stat=='probability' else 250
bins=100 if common_bins else 'auto'
axlim = 500

norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
fig, axn = pl.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
for ai, sp in enumerate(['Dmel', 'Dyak', 'Dele']): #(sp, df_) in enumerate(f2_.groupby('species')):

    ax=axn[ai]
    df_ = court_[court_['species']==sp].copy().reset_index(drop=True)
    sns.histplot(data=df_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
            cmap=cmap, stat=stat, vmin=0, vmax=vmax, bins=bins) # %%
    ax.plot(0, 0, 'w', markersize=3, marker='>')
    ax.set_title(sp)
    # ax.set_xlim([])
    ax.set_aspect(1)
    # if sp=='Dmel': #or sp=='Dyak':
    #     ax.set_xlim([-250, 250])
    #     ax.set_ylim([-250, 250])
    # else:
    ax.set_xlim([-axlim, axlim])
    ax.set_ylim([-axlim, axlim])

    ax.axvline(0, color=bg_color, linestyle=':', lw=0.5)
    ax.axhline(0, color=bg_color, linestyle=':', lw=0.5)

    ax.set_xlabel('x-pos (pix)')
    ax.set_ylabel('y-pos (pix)')

putil.colorbar_from_mappable(ax, norm=norm, cmap=cmap, axes=[0.92, 0.3, 0.01, 0.4],
                             hue_title=stat)

pl.subplots_adjust(wspace=0.4, top=0.9)
fig.suptitle('Female position from male POV (bins-{})\n{}'.format(bins, court_filter_str), 
             x=0.1, y=0.9, fontsize=10, horizontalalignment='left')
putil.label_figure(fig, fig_id)

fig_str = '{}_court_{}'.format(stat, court_filter_str)
figname = 'female-pos-relative-to-male-view_bins-{}_{}'.format(bins, fig_str)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)), dpi=300, bbox_inches='tight')
#pl.savefig(os.path.join(figdir, '{}.svg'.format(figname))) #, dpi=300, bbox_inches='tight')
print(figdir)
print(figname)
# %%
