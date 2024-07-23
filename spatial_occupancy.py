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

from relative_metrics import load_processed_data
import utils as util
import plotting as putil

#%%
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=12)
bg_color = [0.7]*3 if plot_style=='dark' else 'w'

#%%
create_new = False

# Set sourcedirs
# srcdir = '/Volumes/Julie/2d-projector-analysis/FlyTracker/processed_mats' #relative_metrics'
srcdir = '/Volumes/Julie/free-behavior-analysis/FlyTracker/38mm_dyad/processed'
figdir = os.path.join(os.path.split(srcdir)[0], 'relative_metrics', 'figures')

if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

# LOCAL savedir 
#localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/2d-projector/FlyTracker'
localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/38mm-dyad/FlyTracker'
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

found_fns = glob.glob(os.path.join(srcdir, '{}*{}.pkl'.format(acq, mat_type)))
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
    vidbase = '/Volumes/Julie/courtship-videos/38mm_dyad'
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

#wing_ext = df_[df_['min_wing_ang'] >= min_wing_ang].copy()

min_vel = 5
max_facing_angle = np.deg2rad(20)
max_dist_to_other = 20
max_targ_pos_theta = np.deg2rad(160)
min_targ_pos_theta = np.deg2rad(-160)
min_wing_ang = np.deg2rad(45)

court_ = df[ (df['id']==0)
             & (df['vel'] > min_vel)
             & (df['targ_pos_theta'] <= max_targ_pos_theta)
             & (df['targ_pos_theta'] >= min_targ_pos_theta)
             & (df['facing_angle'] <= max_facing_angle)
             & (df['min_wing_ang'] >= min_wing_ang)
             & (df['dist_to_other'] <= max_dist_to_other)].copy()

#sing_ = df[ (df['id']==0)
#             & (df['vel'] > min_vel)
#             & (df['targ_pos_theta'] <= max_targ_pos_theta)
#             & (df['targ_pos_theta'] >= min_targ_pos_theta)
#             & (df['facing_angle'] <= max_facing_angle)
#             & (df['min_wing_ang'] >= min_wing_ang)
#             & (df['dist_to_other'] <= max_dist_to_other)].copy()


# Get female-centered frames
f_list = []
for acq, df_ in court_.groupby('acquisition'):
    
    f2_ = df[ (df['frame'].isin(df_['frame']))
             & (df['id']==1)].copy() #wing_ext[wing_ext['id']==1].copy()
    f_list.append(f2_)
f2 = pd.concat(f_list)


#f_list = []
#for acq, df_ in sing_.groupby('acquisition'):
#    
#    f2_ = df[ (df['frame'].isin(df_['frame']))
#             & (df['id']==1)].copy() #wing_ext[wing_ext['id']==1].copy()
#    f_list.append(f2_)
#f2_sing = pd.concat(f_list)
#
#%%

cmap = 'magma'
stat = 'count'
vmax=0.0015 if stat=='probability' else 250

norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
fig, axn = pl.subplots(1, 3, figsize=(10, 5))
for ai, sp in enumerate(['Dmel', 'Dyak', 'Dele']): #(sp, df_) in enumerate(f2_.groupby('species')):

    ax=axn[ai]
    df_ = f2[f2['species']==sp].copy().reset_index(drop=True)
    sns.histplot(data=df_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
             cmap='magma', vmin=0, vmax=vmax, stat=stat) # %%
    ax.plot(0, 0, 'w', markersize=1, marker='o')
    ax.set_title(sp)
    # ax.set_xlim([])
    ax.set_aspect(1)
    if sp=='Dmel':
        ax.set_xlim([-250, 250])
        ax.set_ylim([-250, 250])
    else:
        ax.set_xlim([-1000, 1000])
        ax.set_ylim([-1000, 1000])
putil.colorbar_from_mappable(ax, norm=norm, cmap=cmap, axes=[0.92, 0.3, 0.01, 0.4])

pl.subplots_adjust(wspace=0.6)
fig.suptitle('Courting frames')


figname = 'male-pos-relative-to-fem_{}-vmax{:.3f}_all-frames_min-wing-ang-{}_zoom'.format(stat, vmax, np.rad2deg(min_wing_ang))
print(figname)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)), dpi=300, bbox_inches='tight')

print(figdir, figname)


 # %%

acq = '20240116-1601-fly5-yakWT_4do_sh_yakWT_4do_gh'
df_ = f2[f2['acquisition']==acq]

fig, ax = pl.subplots()
sns.histplot(data=df_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
            cmap='magma', vmin=0, vmax=100) # %%
ax.plot(0, 0, 'w', markersize=3, marker='o')
ax.set_title(sp)
ax.set_aspect(1)
ax.set_xlim([-1000, 1000])
ax.set_ylim([-1000, 1000])

# %%


#%%

importlib.reload(util)
jaaba = util.load_jaaba('38mm-dyad')

if 'filename' in jaaba.columns:
    jaaba = jaaba.rename(columns={'filename': 'acquisition'})

print(jaaba[['species', 'acquisition']].drop_duplicates().groupby('species').count())

#%%

d_list = []
for acq, d_ in jaaba.groupby('acquisition'):
    match_ = df[ (df['acquisition']==acq)
                & (df['id']==0)].copy()
    assert match_.shape[0] == d_.shape[0], "Bad shape: {}, jaaba {}, df {}".format(acq, d_.shape, match_.shape)

    ja_cols = [c for c in jaaba.columns if c not in match_.columns]
    currdf = pd.concat([match_, d_[ja_cols]], axis=1)
    d_list.append(currdf)

ftjaaba = pd.concat(d_list)

#%% plotting settings
curr_species = ['Dele', 'Dmau', 'Dmel', 'Dsant', 'Dyak']
species_cmap = sns.color_palette('colorblind', n_colors=len(curr_species))
print(curr_species)
species_palette = dict((sp, col) for sp, col in zip(curr_species, species_cmap))

#%%
xvar = 'rel_vel_abs'
yvar = 'targ_ang_size_deg'
joint_type = 'kde'


plotdf = ftjaaba[(ftjaaba['chasing']==1)]
sns.jointplot(data=plotdf, x=xvar, y=yvar, hue='species', 
              palette=species_palette, kind=joint_type)


# %%


print(ftjaaba[['species', 'acquisition']].drop_duplicates().groupby('species').count())

# %%
