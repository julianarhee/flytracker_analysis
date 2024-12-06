#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#%%
import os
import sys
import glob
import importlib

import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import pylab as pl
import matplotlib as mpl

import matplotlib.gridspec as gridspec

#%%
# import some custom funcs
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from relative_metrics import load_processed_data
import utils as util
import plotting as putil

#%%
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=24)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%% LOAD ALL THE DATA
#savedir = '/Volumes/Julie/free-behavior-analysis/FlyTracker/38mm_dyad/processed'
#figdir = os.path.join(os.path.split(savedir)[0], 'figures', 'relative_metrics')
importlib.reload(util)

assay = '2d-projector' # '38mm-dyad'
create_new = False
minerva_base = '/Volumes/Juliana'

#%%
if assay == '2d-projector':
    # Set sourcedirs
    srcdir = os.path.join(minerva_base, '2d-projector-analysis/circle_diffspeeds/FlyTracker/processed_mats') #relative_metrics'
    # LOCAL savedir 
    localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/2d-projector/circle_diffspeeds/FlyTracker'
elif assay == '38mm-dyad':
    # src dir of processed .dfs from feat/trk.mat files (from relative_metrics.py)
    srcdir = os.path.join(minerva_base, 'free-behavior-analysis/FlyTracker/38mm_dyad/processed')
    # local savedir for giant pkl
    localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/38mm-dyad/FlyTracker'

# Specify path to local file for aggregated data 
out_fpath_local = os.path.join(localdir, 'relative_metrics.pkl') #'processed.pkl')
print(out_fpath_local)

# Set figdir to be in parent directory as the source data on server
if plot_style == 'white':
    figdir = os.path.join(os.path.split(srcdir)[0], 'relative_metrics', 'figures', 'white')
else: 
    figdir = os.path.join(os.path.split(srcdir)[0], 'relative_metrics', 'figures')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

# try reading if we don't want to create a new one
if not create_new:
    if os.path.exists(out_fpath_local):
        df = pd.read_pickle(out_fpath_local)
        print("Loaded local processed data.")
    else:
        create_new = True
assert not(create_new), "No new data to create, and no existing data to load."

#%%
# cycle over all the acquisition dfs in srcdir and make an aggregated df
if create_new:
    df = util.load_aggregate_data_pkl(srcdir, mat_type='df')
    print(df['species'].unique())

    #% save to server
    out_fpath = os.path.join(os.path.split(figdir)[0], 'relative_metrics.pkl')
    df.to_pickle(out_fpath)
    print(out_fpath)

    # save local, too
    df.to_pickle(out_fpath_local)

# df['acquisition'] = ['_'.join(f.split('_')[0:-1]) for f in df['acquisition']]
# summary of what we've got
print(df[['species', 'acquisition']].drop_duplicates().groupby('species').count())


#%% plotting settings
curr_species = ['Dele', 'Dmau', 'Dmel', 'Dsant', 'Dyak']
species_cmap = sns.color_palette('colorblind', n_colors=len(curr_species))
print(curr_species)
species_palette = dict((sp, col) for sp, col in zip(curr_species, species_cmap))

#%% Set fig id
figid = srcdir  


#%% load jaaba data
importlib.reload(util)

fname = 'free_behavior_data_mel_yak_20240403' if assay=='38mm-dyad' else None
jaaba = util.load_jaaba(assay, fname=fname)

#%%
if 'filename' in jaaba.columns and 'acquisition' not in jaaba.columns:
    jaaba = jaaba.rename(columns={'filename': 'acquisition'})

print(jaaba[['species', 'acquisition']].drop_duplicates().groupby('species').count())

#%% merge jaaba and processed data
c_list = []
for acq, ja_ in jaaba.groupby('acquisition'):
    df_ = df[(df['acquisition']==acq) & (df['id']==0)].reset_index(drop=True)
    try:
        if len(df_)>0:
            if ja_.shape[0] < df_.shape[0]:
                last_frame = ja_['frame'].max()
                df_ = df_[df_['frame']<=last_frame]
            else:
                assert ja_.shape[0] == df_.shape[0], "Mismatch in number of flies between jaaba {} and processed data {}.".format(ja_.shape, df_.shape) 
            drop_cols = [c for c in ja_.columns if c in df_.columns]
            combined_ = pd.concat([df_, ja_.drop(columns=drop_cols)], axis=1)
            assert combined_.shape[0] == df_.shape[0], "Bad merge: {}".format(acq)
            c_list.append(combined_)
    except Exception as e:
        print(acq)
        print(e)
        continue

ftjaaba = pd.concat(c_list, axis=0).reset_index(drop=True)
# summarize what we got
ftjaaba[['species', 'acquisition']].drop_duplicates().groupby('species').count()

#%%
if 'courtship' in ftjaaba.columns:
    ftjaaba = ftjaaba.rename(columns={'courtship': 'courting'})

#%% add bouts
if 'fpath' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['fpath'])
if 'name' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['name'])

#%  split into bouts of courtship
d_list = []
for acq, df_ in ftjaaba.groupby('acquisition'):
    df_ = df_.reset_index(drop=True)
    df_ = util.mat_split_courtship_bouts(df_, bout_marker='courting')
    dur_ = util.get_bout_durs(df_, bout_varname='boutnum', return_as_df=True,
                    timevar='sec')
    d_list.append(df_.merge(dur_, on=['boutnum']))
ftjaaba = pd.concat(d_list)


#%%
# Calculate a bunch of additional params
winsize=5
#print(acq)

#df_ = ftjaaba[ftjaaba['acquisition']==acq]
ftjaaba['targ_pos_theta_deg'] = np.rad2deg(ftjaaba['targ_pos_theta'])
ftjaaba['facing_angle_deg'] = np.rad2deg(ftjaaba['facing_angle'])

ftjaaba['rel_vel'] = np.nan
for acq, df_ in ftjaaba.groupby('acquisition'):
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='targ_pos_theta', vel_var='targ_ang_vel',
                                  time_var='sec', winsize=winsize)
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='targ_pos_theta_deg', vel_var='targ_ang_vel_deg',
                                  time_var='sec', winsize=winsize)

    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='facing_angle', vel_var='facing_angle_vel',)
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='facing_angle_deg', vel_var='facing_angle_vel_deg',)

    tmp_d = []
    for i, d_ in df_.groupby('id'):    
        df_['rel_vel'] = d_['dist_to_other'].interpolate().diff() / d_['sec'].diff().mean()
        tmp_d.append(df_)
    tmp_ = pd.concat(tmp_d)
    ftjaaba.loc[ftjaaba['acquisition']==acq, 'rel_vel'] = tmp_['rel_vel']

    ftjaaba.loc[ftjaaba['acquisition']==acq, 'targ_ang_vel'] = df_['targ_ang_vel']
    ftjaaba.loc[ftjaaba['acquisition']==acq, 'targ_ang_vel_deg'] = df_['targ_ang_vel_deg']

    ftjaaba.loc[ftjaaba['acquisition']==acq, 'facing_angle_vel'] = df_['facing_angle_vel']
    ftjaaba.loc[ftjaaba['acquisition']==acq, 'facing_angle_vel_deg'] = df_['facing_angle_vel_deg']  

#%%
ftjaaba['rel_vel_abs'] = np.abs(ftjaaba['rel_vel']) 

ftjaaba['targ_ang_vel_abs'] = np.abs(ftjaaba['targ_ang_vel'])
ftjaaba['targ_pos_theta_abs'] = np.abs(ftjaaba['targ_pos_theta'])
ftjaaba['targ_ang_size_deg'] = np.rad2deg(ftjaaba['targ_ang_size'])
ftjaaba['targ_ang_vel_deg_abs'] = np.abs(ftjaaba['targ_ang_vel_deg'])
#ftjaaba['targ_pos_theta_abs_deg'] = np.rad2deg(ftjaaba['targ_pos_theta_abs'])  
ftjaaba['facing_angle_deg'] = np.rad2deg(ftjaaba['facing_angle'])
ftjaaba['ang_vel_abs'] = np.abs(ftjaaba['ang_vel'])
if 'dovas' in ftjaaba.columns:
    ftjaaba['dovas_deg'] = np.rad2deg(ftjaaba['dovas'])

ftjaaba['facing_angle_vel_abs'] = np.abs(ftjaaba['facing_angle_vel'])
ftjaaba['facing_angle_vel_deg_abs'] = np.abs(ftjaaba['facing_angle_vel_deg'])

if 'good_frames' not in ftjaaba.columns:
    ftjaaba['good_frames'] = 1


#%% get means by BOUT
groupcols = [ 'species', 'acquisition', 'boutnum']

min_pos_theta = np.deg2rad(-160)
max_pos_theta = np.deg2rad(160)
min_dist_to_other = 1

filtdf = ftjaaba[(ftjaaba['id']==0)
            & (ftjaaba['targ_pos_theta']>=min_pos_theta) 
            & (ftjaaba['targ_pos_theta']<=max_pos_theta)
            & (ftjaaba['good_frames']==1)
            ].copy().reset_index(drop=True)

#meandf = ftjaaba_filt.groupby(groupcols).mean().reset_index()

#%% DEBUG
#meandf = meandf[meandf['boutdur']>=min_boutdur] 

# ------------------------------
# PLOT
# ------------------------------
#%% boutdurs
importlib.reload(util)
min_boutdur = 0.5

if filtdf['chasing'].max() == 1:
    jaaba_thresh_dict = {'orienting': 0, 
                        'chasing': 0,
                        'singing': 0}
else:
    jaaba_thresh_dict = {'orienting': 10,
                        'chasing': 10,
                        'singing': 5}

filtdf = util.binarize_behaviors(filtdf, jaaba_thresh_dict=jaaba_thresh_dict)

plotdf = filtdf[filtdf['boutdur']>=min_boutdur]

xvar = 'dist_to_other'
varname = 'singing'


fig, axn = pl.subplots(1, 3, figsize=(10,4))#, sharex=True, sharey=True)
ax=axn[0]
ax.set_title('not courting')
sns.histplot(data=plotdf[plotdf['courting']==0], x=xvar,  ax=ax,
             hue='species', alpha=0.7, palette=species_palette, stat='probability',
             cumulative=False, common_norm=False, bins=40, legend=0)
ax=axn[1]
ax.set_title('courting')
sns.histplot(data=plotdf[plotdf['courting']==1], x=xvar,  ax=ax,
             hue='species', alpha=0.7, palette=species_palette, stat='probability',
            cumulative=False, common_norm=False, bins=40, legend=0)
ax=axn[2]
ax.set_title(varname)
sns.histplot(data=plotdf[plotdf['{}_binary'.format(varname)]==1], x=xvar,  ax=ax,
             hue='species', alpha=0.7, palette=species_palette, stat='probability',
            cumulative=False, common_norm=False, bins=40)
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
for ax in axn:
    ax.set_box_aspect(1)

fig.text(0.05, 0.9, '{} (min bout dur: {:.2f}s)'.format(xvar, min_boutdur), fontsize=12)

pl.subplots_adjust(wspace=0.75)
putil.label_figure(fig, figid) 

figname = 'hist_{}_nocourt-v-court-v-{}bouts_minboutdur-{}_mel-v-yak'.format(xvar, varname, min_boutdur)
pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)
print(figdir, figname)


#%% JOINT DISTS
min_boutdur = 1
plotdf = filtdf[filtdf['boutdur']>=min_boutdur]

varname = 'singing'

joint_kind = 'kde' #'hist'
if joint_kind=='kde':
    joint_kws={'bw_adjust': 1.0}
elif joint_kind=='scatter':
    joint_kws={'s': 10, 'alpha': 0.8, 'n_levels': 30}
else:
    joint_kws={}
    #joint_kind == 'hist':

x = 'abs_rel_vel' #'targ_ang_vel_deg_abs' #'abs_rel_ang_vel'
y = 'dovas_deg' #'targ_ang_size_deg' #'dovas'
if varname=='notcourting':
    g = sns.jointplot(data=plotdf[plotdf['courting']==0].reset_index(drop=True), 
                x=x, y=y, 
                hue='species', palette=species_palette, #palette=species_cdict,
                kind=joint_kind, joint_kws=joint_kws)
                #, )
else:
    g = sns.jointplot(data=plotdf[plotdf['{}_binary'.format(varname)]>0].reset_index(drop=True), 
                x=x, y=y, 
                hue='species', palette=species_palette, #palette=species_cdict,
                kind=joint_kind, joint_kws=joint_kws) #joint_kws={'bw_adjust': 0.75}) 
pl.xlim([-2, 20])
#pl.ylim([-5, 50]) 
g.fig.suptitle(varname)
pl.subplots_adjust(top=0.9)

putil.label_figure(g.fig, figid) 

figname = 'jointdist-{}_{}_{}-v-{}_minboutdur-{}_mel-v-yak'.format(joint_kind, varname, x, y, min_boutdur)
pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)
print(figdir, figname)

#%%

# -------------------------------------
# PLOT INDIVIDUALS
# -------------------------------------

def plot_polar_pos_with_hists(plotdf, 
            min_pos_theta=np.deg2rad(-160), max_pos_theta=np.deg2rad(160)):
    '''
    Plot polar with relative locations, color by rel_vel_abs.
    Also plot 3 histograms below: targ_ang_size_deg, rel_vel_abs, dist_to_other.

    Arguments:
        plotdf -- _description_

    Keyword Arguments:
        min_pos_theta -- _description_ (default: {np.deg2rad(-160)})
        max_pos_theta -- _description_ (default: {np.deg2rad(160)})

    Returns:
        fig
    '''
    fig = pl.figure(figsize=(12,10))
    spec = gridspec.GridSpec(ncols=3, nrows=3)

    ax = fig.add_subplot(spec[-1, 0])
    sns.histplot(data=plotdf, x='targ_ang_size_deg', color=[0.7]*3, ax=ax,
                stat='probability')
    ax.set_box_aspect(1)

    ax = fig.add_subplot(spec[-1, 1])
    sns.histplot(data=plotdf, x='rel_vel_abs', color=[0.7]*3, ax=ax,
                stat='probability')
    ax.set_box_aspect(1)

    ax = fig.add_subplot(spec[-1, 2])
    sns.histplot(data=plotdf, x='dist_to_other', color=[0.7]*3, ax=ax,
                stat='probability')
    ax.set_box_aspect(1)

    ax = fig.add_subplot(spec[0:2, 0:2], projection='polar') #=pl.subplots(subplot_kw={'projection': 'polar'})
    sns.scatterplot(data=plotdf, x='targ_pos_theta', y='targ_pos_radius', ax=ax,
                    #size='ang_size_deg_hue',
                hue='rel_vel_abs', #hue_norm=mpl.colors.Normalize(vmin=15, vmax=60),
                palette='magma', edgecolor='w', alpha=0.8)
    ax.plot([0, min_pos_theta], [0, ax.get_ylim()[-1]], 'r')
    ax.plot([0, max_pos_theta], [0, ax.get_ylim()[-1]], 'r')
    sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1.05,1))

    return fig

#def assign_jaaba_behaviors(plotdf, jaaba_thresh_dict, courtvar='courting', min_thresh=5):
#    plotdf.loc[plotdf[courtvar]==0, 'behavior'] = 'disengaged'
#    for b, thr in jaaba_thresh_dict.items():
#        plotdf.loc[plotdf[b]>thr, 'behavior'] = b
#    #plotdf.loc[plotdf['chasing']>, 'behavior'] = 'chasing'
#    #plotdf.loc[plotdf['singing']>0, 'behavior'] = 'singing'
#    #plotdf.loc[((plotdf['chasing']>0) & (plotdf['singing']==0)), 'behavior'] = 'chasing only'
#    return plotdf

def plot_2d_hist_by_behavior(plotdf, binwidth=10,
            plot_behavs = ['disengaged', 'orienting', 'chasing', 'singing'],
            behavior_colors = [[0.3]*3, 'mediumaquamarine', 'aqua', 'violet']):
    '''
    Plot 2D histograms of target position, color by behavior.

    Arguments:
        plotdf -- _description_

    Keyword Arguments:
        plot_behavs -- _description_ (default: {['disengaged', 'orienting', 'chasing', 'singing']})
        behavior_colors -- _description_ (default: {[[0.3]*3, 'mediumaquamarine', 'aqua', 'violet']})

    Returns:
        _description_
    '''
    behavior_palette = dict((b, c) for b, c in zip(plot_behavs, behavior_colors))
    g = sns.displot(plotdf[plotdf['behavior'].isin(plot_behavs)], 
                x='targ_rel_pos_x', y='targ_rel_pos_y',
                hue='behavior', hue_order=plot_behavs, binwidth=10,
            palette=behavior_palette, kind='hist', common_norm=False)
    #pl.ylim([-100, 100])
    #pl.xlim([-100, 500])

    pl.plot(0, 0, 'w*')
    pl.gca().set_aspect(1)

    return g.fig

def plot_2d_hist_courting_vs_not(plotdf, plot_behavs, behavior_palette,
                                 nbins=25, binwidth=10):
    '''
    Plot two 2D histograms of target position, side by side: one for courting, one for not courting.
    
    Arguments:
        plotdf -- _description_
        plot_behavs -- _description_
        behavior_palette -- _description_

    Keyword Arguments:
        nbins -- _description_ (default: {25})

    Returns:
        _description_
    '''
    fig, axn = pl.subplots(1, 2, sharex=True, sharey=True)
    ax = axn[0]
    g = sns.histplot(plotdf[plotdf['behavior']=='disengaged'], ax=ax,
                x='targ_rel_pos_x', y='targ_rel_pos_y',
                hue='behavior', hue_order=plot_behavs,legend=False,
            palette=behavior_palette, common_norm=False, 
            bins=nbins, binwidth=binwidth)
    ax.set_title('not courting')
    ax=axn[1]
    sns.histplot(plotdf[plotdf['courting']>0], ax=ax,
                x='targ_rel_pos_x', y='targ_rel_pos_y',
                hue='behavior', hue_order=plot_behavs,legend=False,
            palette=behavior_palette, common_norm=False, 
            bins=nbins, binwidth=binwidth)
    ax.set_title('courting')
    for ax in axn:
        ax.set_aspect(1)

    return fig

def plot_2d_hist_by_behavior_subplots(plotdf,
                        plot_behavs, behavior_palette,focal_marker='*', markersize=10,
                        nbins=25, binwidth=None, discrete=None, stat='count', 
                        ylim=[-250, 250], xlim=[-100, 600]):
    '''
    Plot 2D histograms of target position, color by behavior, in subplots.

    Arguments:
        plotdf -- _description_
        plot_behavs -- _description_
        behavior_palette -- _description_

    Keyword Arguments:
        nbins -- _description_ (default: {25})
        binwidth -- width of each bin, overrides bins
        discrete -- True/False, if True binwidth=1, center bar on data points (default: {None})
        stat -- aggr stat for each bin, can be:
                    count: num per bin, 
                    frequency: num/bin width,
                    probability: normalize so bars sum to 1.
                    percent: normalize so bars sum to 100,
                    density: total area of hist = 1 (default: {'count'})
        ylim -- _description_ (default: {[-250, 250]})
        xlim -- _description_ (default: {[-100, 600]})

    Returns:
        _description_
    '''
    fig, axn = pl.subplots(1, len(plot_behavs[1:]), sharex=True, sharey=True)
    for ai, behav in enumerate(plot_behavs[1:]):
        print(behav)
        ax=axn[ai]
        g = sns.histplot(plotdf[plotdf['behavior']==behav], ax=ax,
                    x='targ_rel_pos_x', y='targ_rel_pos_y',
                    hue='behavior', hue_order=plot_behavs[1:], legend=False,
                palette=behavior_palette, common_norm=False, 
                bins=nbins, binwidth=binwidth, discrete=discrete)
        ax.set_aspect(1)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.plot(0, 0, 'w', marker=focal_marker, markersize=markersize)
        ax.set_title(behav)
    #pl.ylim([-100, 100])
    return fig


def plot_2d_hist(plotdf, xvar='targ_rel_pos_x', yvar='targ_rel_pos_y', ax=None,
                 color='b', nbins=25, xlim=[-100, 600], ylim=[250, 250], stat='count'):
    if ax is None:
        fig, ax =pl.subplots()

    sns.histplot(plotdf, ax=ax, 
                     x=xvar, y=yvar, color=color, bins=nbins, stat=stat)
    ax.set_aspect(1)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    #ax.plot(0, 0, 'w', marker=focal_marker, markersize=markersize)
    #ax.set_title(behav)

    return

#%% plot 1 individual - trajectories and hists
acq = ftjaaba['acquisition'].unique()[9]
print(acq)

min_pos_theta = np.deg2rad(-160)
max_pos_theta = np.deg2rad(160)
min_dist_to_other = 1

for acq in ftjaaba['acquisition'].unique():
    print(acq)
    # plot info for courting vs. all
    for split in ['COURT', 'ALL']:
        if split=='COURT':
            plotdf = ftjaaba[(ftjaaba['acquisition']==acq) & (ftjaaba['id']==0)
                    & (ftjaaba['courting']>0)
                    & (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                    & (ftjaaba['targ_pos_theta']<=max_pos_theta)
                    & (ftjaaba['dist_to_other']>=min_dist_to_other)][::50]
        else:
            plotdf = ftjaaba[(ftjaaba['acquisition']==acq) & (ftjaaba['id']==0)
                    & (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                    & (ftjaaba['targ_pos_theta']<=max_pos_theta)
                    & (ftjaaba['dist_to_other']>=min_dist_to_other)][::50]

        fig = plot_polar_pos_with_hists(plotdf)
        pl.subplots_adjust(left=0.1, right=0.9, wspace=0.4, hspace=0.4)
        putil.label_figure(fig, acq)
        fig.suptitle(split)

        figname = 'polar-with-hists_{}_{}'.format(split, acq)
        pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)
        print(figdir, figname)

#%% 2D Hists, color by BEHAVIOR TYPE
importlib.reload(util)
courtvar='courting'

jaaba_thresh_dict = {'orienting': 10,
                     'chasing': 10,
                     'singing': 5}

binwidth=20
for acq in ftjaaba['acquisition'].unique():
    print(acq)
    # plot info for courting vs. all

    #% Plot 2D hists for all behavior classes
    plotdf = ftjaaba[(ftjaaba['acquisition']==acq) & (ftjaaba['id']==0)
                & (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                & (ftjaaba['targ_pos_theta']<=max_pos_theta)
                & (ftjaaba['dist_to_other']>=min_dist_to_other)][::50]
    plotdf = util.assign_jaaba_behaviors(plotdf, courtvar=courtvar, jaaba_thresh_dict=jaaba_thresh_dict)

    plot_behavs = ['disengaged', 'orienting', 'chasing', 'singing']
    behavior_colors = [[0.3]*3, 'mediumaquamarine', 'aqua', 'violet']
    behavior_palette = dict((b, c) for b, c in zip(plot_behavs, behavior_colors))

    fig = plot_2d_hist_by_behavior(plotdf, plot_behavs=plot_behavs,
                                behavior_colors=behavior_colors, binwidth=binwidth)
    putil.label_figure(fig, acq)
    figname = 'rel-pos_all-behaviors_{}'.format(acq)
    pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)
    print(figdir, figname)

    #% Plot COURTING vs. NOT-COURTING
    fig = plot_2d_hist_courting_vs_not(plotdf, plot_behavs, 
                                    behavior_palette, binwidth=binwidth) #nbins=25)
    putil.label_figure(fig, acq)
    figname = 'rel-pos_notcourt-v-court_{}'.format(acq)
    pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)
    print(figdir, figname)

#%% SPLIT SUBPLOTS
binwidth=20 # in pixels
for acq in ftjaaba['acquisition'].unique():
    print(acq)
    plotdf = ftjaaba[(ftjaaba['acquisition']==acq) & (ftjaaba['id']==0)
                & (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                & (ftjaaba['targ_pos_theta']<=max_pos_theta)
                & (ftjaaba['dist_to_other']>=min_dist_to_other)][::50]
    plotdf = util.assign_jaaba_behaviors(plotdf, courtvar=courtvar, jaaba_thresh_dict=jaaba_thresh_dict)
    #% Plot each COURTING BEHAV
    plot_behavs = ['disengaged', 'orienting', 'chasing', 'singing']
    behavior_colors = [[0.3]*3, 'mediumaquamarine', 'aqua', 'violet']
    behavior_palette = dict((b, c) for b, c in zip(plot_behavs[1:], behavior_colors[1:]))
    fig = plot_2d_hist_by_behavior_subplots(plotdf, plot_behavs,
                                            behavior_palette, binwidth=binwidth) #nbins=25)
    putil.label_figure(fig, acq)
    figname = 'rel-pos_by-courting-behavior_{}'.format(acq)
    pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)
    print(figdir, figname)



#%% ---------------------------------------------
# AGGREGATE -- 2D hists by species
# -----------------------------------------------
min_boutdur = .25
courtvar = 'courting'
filtdf = ftjaaba[(ftjaaba['id']==0)
                & (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                & (ftjaaba['targ_pos_theta']<=max_pos_theta)
                & (ftjaaba['dist_to_other']>=min_dist_to_other)
                & (ftjaaba['boutdur']>=min_boutdur)
                & (ftjaaba['good_frames']==1)
                ].copy().reset_index(drop=True).copy()

#meanbins = filtdf.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()    

# means are averaged over bout, so threshold is now 0
jaaba_thresh_dict = {'orienting': 10,
                     'chasing': 10,
                     'singing': 5}

# PLOT POSITION
# ------------------------------
filtdf = util.assign_jaaba_behaviors(filtdf, courtvar=courtvar, jaaba_thresh_dict=jaaba_thresh_dict)
plot_behavs = ['disengaged', 'orienting', 'chasing', 'singing']
behavior_colors = [[0.3]*3, 'mediumaquamarine', 'aqua', 'violet']
behavior_palette = dict((b, c) for b, c in zip(plot_behavs[1:], behavior_colors[1:]))

#for sp in filtdf['species'].unique():
for sp, plotdf in filtdf.groupby('species'):
    fig, axn = pl.subplots(1, 3, figsize=(10, 5))
    for ai, (behav, thr) in enumerate(jaaba_thresh_dict.items()):
        ax=axn[ai]
        #fig = plot_2d_hist_by_behavior_subplots(plotdf, plot_behavs,
        #                                    behavior_palette, binwidth=binwidth,
        #                                    ylim=[-900, 900], xlim=[-250, 900]) #nbins=25)

        plotdf = filtdf[ (filtdf['species']==sp) & (filtdf[behav]> thr)]
        plot_2d_hist(plotdf, ax=ax, color=behavior_palette[behav], nbins=30,
                                    ylim=[-900, 900], xlim=(-500, 900))
        ax.set_title(behav)
    #fig.suptitle(sp)
    fig.text(0.05, 0.85, 
        '{}: Relative target pos. (all frames, min bout dur={:.2}s)'.format(sp, min_boutdur), fontsize=12)
    
    pl.subplots_adjust(wspace=0.6)
    putil.label_figure(fig, figid)

    figname = 'rel-pos_frames_mindur-{}_by-courting-behavior_{}'.format(min_boutdur, sp)
    #pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)

#%%
    
# negatively correlated -- 
# targ_ang_vel: negative values, rightward rotation (smaller val - larger val)
# rel_ang_vel: positive values, rightward rotation (smaller val - larger val)?
fig, ax = pl.subplots()

behav = 'chasing'
sns.scatterplot(data=filtdf[filtdf[behav]>jaaba_thresh_dict[behav]],
                x='rel_ang_vel', y='targ_ang_vel', ax=ax)
ax.set_box_aspect(1)


#%% plot heatmaps


#ftjaaba.loc[ftjaaba['targ_ang_vel_abs_deg']>500, 'targ_ang_vel_abs_deg'] = np.nan

min_pos_theta = np.deg2rad(-160)
max_pos_theta = np.deg2rad(160)
min_dist_to_other = 2
   
min_boutdur = 0.25 #0.25

filtdf = ftjaaba[(ftjaaba['id']==0)
                #& (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                #& (ftjaaba['targ_pos_theta']<=max_pos_theta)
                & (ftjaaba['dist_to_other']>=min_dist_to_other)
                & (ftjaaba['boutdur']>=min_boutdur)
                & (ftjaaba['good_frames']==1)
                ].copy().reset_index(drop=True)
#meanbins = filtdf.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()    


#%%
frames_ = filtdf[filtdf['species']=='Dmel']

#xvar = 'targ_ang_size_deg'
#yvar = 'targ_ang_vel_abs_deg'
xvar = 'targ_pos_theta_abs' #'dovas' #'targ_ang_size_deg'
yvar = 'abs_rel_ang_vel' #'targ_ang_vel_abs_deg'

jaaba_thresh_dict = {'orienting': 10,
                     'chasing': 10,
                     'singing': 5}

g = sns.jointplot(data=frames_[frames_['chasing']>jaaba_thresh_dict['chasing']], ax=ax,
             x = xvar, y = yvar, 
           kind='hist', bins=40, palette='magma') 
g.fig.suptitle('Dmel: chasing')


frames_ = filtdf[filtdf['species']=='Dyak']
g2 = sns.jointplot(data=frames_[frames_['chasing']>jaaba_thresh_dict['chasing']], ax=ax,
             x = xvar, y = yvar, 
           kind='hist', bins=20, palette='magma') 
g2.fig.suptitle('Dyak: chasing')


#%% ---------------------------------------------
# Look at joint hists by FRAME
# -----------------------------------------------

min_boutdur = 0.25 #0.25
min_dist_to_other=2

filtdf = ftjaaba[(ftjaaba['id']==0)
                #& (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                #& (ftjaaba['targ_pos_theta']<=max_pos_theta)
                & (ftjaaba['dist_to_other']>=min_dist_to_other)
                & (ftjaaba['boutdur']>=min_boutdur)
                & (ftjaaba['good_frames']==1)
                & (ftjaaba['led_level']>0)
                ].copy().reset_index(drop=True)

#xvar = 'targ_rel_pos_x' # 'targ_ang_size'
#yvar = 'targ_rel_pos_y' #'targ_ang_vel'

xvar = 'facing_angle_deg' #'targ_ang_size_deg'
yvar = 'ang_vel_abs' #targ_ang_vel_abs_deg'
#xmin, xmax = meanbins[xvar].min(), 0.5 #meanbins[xvar].max()
#ymin, ymax = -2, 2 #meanbins[yvar].min(), meanbins[yvar].max()

xmin, xmax = filtdf[xvar].min(), filtdf[xvar].max()
ymin, ymax = filtdf[yvar].min(), filtdf[yvar].max()

plot_behavs = ['orienting', 'chasing', 'singing']
fig, axn =pl.subplots(2, len(plot_behavs), figsize=(10,8))

for ci, behav in enumerate(plot_behavs):
    for ri, (sp, frames_) in enumerate(filtdf.groupby('species')):
        ax = axn[ri, ci]
        ax.set_title(behav)
        x_data = frames_[frames_[behav]>jaaba_thresh_dict[behav]][xvar].dropna()
        y_data = frames_[frames_[behav]>jaaba_thresh_dict[behav]][yvar].dropna()

        if len(x_data) != len(y_data):
            if len(x_data) < len(y_data):
                # Randomly sample y_data to match the length of x_data
                y_data = np.random.choice(y_data, size=len(x_data), replace=False)
            else:
                x_data = np.random.choice(x_data, size=len(y_data), replace=False)

        # Create 2D histogram using np.histogram2d()
        hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=25)

        # Normalize
        total_counts = np.sum(hist)
        hist_normalized = hist / total_counts

        # Create meshgrid from edges
        x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)
        pcm = ax.pcolormesh(x_mesh, y_mesh, hist_normalized.T, cmap='magma') #, vmin=0, vmax=0.05)

        if ci==0:
            ax.set_title('{}: {}'.format(sp, behav))
        # ax.set_xlim([xmin, xmax])
        #ax.set_ylim([ymin, 300])

        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)
        ax.set_box_aspect(1)

#%%
# look at hist of 1 variable
plotvar = 'targ_ang_vel_abs'
plot_behavs = ['orienting', 'chasing', 'singing']
fig, axn = pl.subplots(1, 3, figsize=(10,4)) #sharex=True, sharey=True)
for ai, behav in enumerate(plot_behavs):
    ax=axn[ai]
    sns.histplot(data=filtdf[filtdf[behav]>jaaba_thresh_dict[behav]], x=plotvar, ax=ax,
             hue='species')
    ax.set_title(behav)
    ax.set_box_aspect(1)
    #ax.set_xlim([-50, 100])
pl.subplots_adjust(wspace=0.6)

#%%
importlib.reload(util)

#%%  =====================================================

# --------------------------------------------------------
# split into small bouts
# --------------------------------------------------------
bout_dur = 0.25 
min_boutdur = 0.25
min_dist_to_other = 2

if filtdf['chasing'].max() == 1:
    jaaba_thresh_dict = {'orienting': 0, 
                        'chasing': 0,
                        'singing': 0}
else:
    jaaba_thresh_dict = {'orienting': 10,
                        'chasing': 10,
                        'singing': 5}

filtdf = ftjaaba[(ftjaaba['id']==0)
                #& (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                #& (ftjaaba['targ_pos_theta']<=max_pos_theta)
                & (ftjaaba['dist_to_other']>=min_dist_to_other)
                & (ftjaaba['boutdur']>=min_boutdur)
                & (ftjaaba['good_frames']==1)
                & (ftjaaba['led_level']>0)
                ].copy() #.reset_index(drop=True)

# binarize behavs
filtdf = util.binarize_behaviors(filtdf, jaaba_thresh_dict=jaaba_thresh_dict)

# subdivide into smaller boutsa
# bout_dur = 0.5
filtdf = util.subdivide_into_subbouts(filtdf, bout_dur=bout_dur)

#%% Get mean value of small bouts

if 'filename' in filtdf.columns:
    filtdf.drop('filename', axis=1, inplace=True)
if 'strain' in filtdf.columns:
    filtdf.drop('strain', axis=1, inplace=True)

meanbouts = filtdf.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()
meanbouts.head()

cmap='viridis'
stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)

# find the closest matching value to one of the keys in stimhz_palette:
meanbouts['stim_hz'] = meanbouts['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))

#%%
#xvar = 'facing_angle_deg'
#yvar = 'abs_rel_ang_vel'

yvar = 'abs_rel_ang_vel'
xvar = 'dovas'
xmin, xmax = meanbouts[xvar].min(), meanbouts[xvar].max()
ymin, ymax = meanbouts[yvar].min(), meanbouts[yvar].max()

min_frac_bout = 0.
plot_behavs = ['orienting', 'chasing', 'singing']
fig, axn =pl.subplots(2, len(plot_behavs), figsize=(10,8), sharex=True, sharey=True)

vmin=0
vmax=0.05
cmap='magma'
for ci, behav in enumerate(plot_behavs):
    for ri, (sp, means_) in enumerate(meanbouts.groupby('species')):
        ax = axn[ri, ci]
        ax.set_title(behav)
        x_data = means_[means_['{}_binary'.format(behav)]>min_frac_bout][xvar].dropna()
        y_data = means_[means_['{}_binary'.format(behav)]>min_frac_bout][yvar].dropna()

        if len(x_data) != len(y_data):
            if len(x_data) < len(y_data):
                # Randomly sample y_data to match the length of x_data
                y_data = np.random.choice(y_data, size=len(x_data), replace=False)
            else:
                x_data = np.random.choice(x_data, size=len(y_data), replace=False)

        # Create 2D histogram using np.histogram2d()
        hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=25)

        # Normalize
        total_counts = np.sum(hist)
        hist_normalized = hist / total_counts

        # Create meshgrid from edges
        x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)
        pcm = ax.pcolormesh(x_mesh, y_mesh, hist_normalized.T, cmap=cmap, 
                            vmin=vmin, vmax=vmax)  #, vmin=0, vmax=0.05) #, vmin=0, vmax=0.05)

        if ci==0:
            ax.set_title('{}: {}'.format(sp, behav))
        # ax.set_xlim([xmin, xmax])
        #ax.set_ylim([ymin, 300])

        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)
        ax.set_box_aspect(1)

putil.colorbar_from_mappable(ax, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                             cmap=cmap, axes=[0.92, 0.3, 0.01, 0.4])
fig.suptitle('Behavior tuning profile, min frac of bout={}'.format(min_frac_bout))


putil.label_figure(fig, figid)
figname = 'hist2d-behavior-tuning_{}_v-{}_min-frac-bout-{}_mel-v-yak'.format(xvar, yvar, min_frac_bout)

pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)
print(os.path.join(figdir, figname+'.png'))

#%% ---------------------------------------------
# KDE joints
# -----------------------------------------------
putil.set_sns_style(plot_style, min_fontsize=24)

#xvar = 'rel_vel' # 'abs_rel_vel'
#yvar = 'dovas_deg'
xvar = 'dovas'
yvar = 'abs_rel_ang_vel'

joint_type='kde'

xlabel = xvar
ylabel = yvar
if xvar == 'rel_vel':
    xlim = [-50, 50]
elif xvar == 'dovas':
    xlim = [-0.1, 2]
    xlabel = 'relative size (deg.vis.ang.)'
if yvar == 'dovas_deg':
    ylim = [-50, 100]
elif yvar == 'abs_rel_ang_vel':
    ylim = [-5, 300]

if joint_type=='kde':
    joint_kws={'bw_adjust': 1, 'levels': 20}
else:
    joint_kws={}
plot_str = '-'.join(['-'.join([str(i), str(v)]) for i, v in joint_kws.items()])

min_frac_bout = 0.5

for behav in ['chasing',  'singing']:
    g = sns.jointplot(data=meanbouts[meanbouts['{}_binary'.format(behav)]>min_frac_bout], ax=ax,
                x = xvar, y = yvar, hue='species', palette=species_palette,
            kind=joint_type, joint_kws=joint_kws, legend=1) 
    sns.move_legend(g.ax_joint, bbox_to_anchor=(1,1), loc='upper left', frameon=False)
    g.fig.suptitle('{}: min frac of bout={:.2f}'.format(behav, min_frac_bout))
    pl.subplots_adjust(wspace=0.5, hspace=0.5, top=0.9)
    pl.xlim(xlim)
    pl.ylim(ylim)
    putil.label_figure(g.fig, figid)
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)

    figname = 'joint-{}_{}_{}-v-{}_min-frac-bout-{}_mel-v-yak_{}'.format(joint_type, behav, xvar, yvar, min_frac_bout, plot_str)
    pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)
    print(figdir, figname)


#%% 
# Prob(singing) vs. DIST TO OTHER
nbins = 5
min_frac_bout=0.5

dist_bins =  np.linspace(meanbouts['dist_to_other'].min(), meanbouts['dist_to_other'].max(), nbins, endpoint=False)
meanbouts['dist_bin'] = pd.cut(meanbouts['dist_to_other'], bins=dist_bins, labels=dist_bins[0:-1])

#%
# Plot BAR 
fig, ax = pl.subplots() #1, 2, figsize=(10,4))
behav = 'singing'
sns.barplot(data=meanbouts, x='dist_bin', y='{}_binary'.format(behav), ax=ax,
            hue='species', palette=species_palette, edgecolor='w', errcolor=[0.7]*3)
sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', frameon=False)
ax.set_xticklabels([np.round(v) for v in dist_bins[0:-1]])
ax.set_box_aspect(1)
sns.despine( offset=4, trim=True)
fig.suptitle('Prob of singing, min frac of bout > {:.2f}'.format(min_frac_bout))

putil.label_figure(fig, figid)
figname = 'bar-prob-singing_v-dist-to-other_min-frac-bout-{}_mel-v-yak'.format(min_frac_bout)
pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)

# Plot as HIST
fig, ax = pl.subplots()
sns.histplot(data=meanbouts[meanbouts['{}_binary'.format(behav)]>min_frac_bout], x='dist_to_other', ax=ax,
            hue='species', palette=species_palette, fill=False, element='poly',
            common_norm=False, stat='probability')
sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', frameon=False)
ax.set_box_aspect(1)
fig.suptitle('Prob of singing, min frac of bout > {:.2f}'.format(min_frac_bout))

putil.label_figure(fig, figid)
figname = 'prob-singing_v-dist-to-other_min-frac-bout-{}_mel-v-yak'.format(min_frac_bout)
#pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)

#%% ------------------------------------------------
# ANG_VEL vs. THETA_ERROR
# -------------------------------------------------
cmap='viridis'
stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)

# Compare ang vel vs. theta-error? Plot as REGR.

behav = 'chasing'
min_frac_bout = 0.8
chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>=min_frac_bout].copy()

fig, axn = pl.subplots(1, 2, sharex=True, sharey=True)
for ai, (sp, df_) in enumerate(chase_.groupby('species')):
    ax = axn[ai]
    sns.regplot(data=df_, x='facing_angle_deg', y='ang_vel_abs', ax=ax,
                color=species_palette[sp])
    ax.set_title(sp)
    ax.set_box_aspect(1)

fig.suptitle('{} bouts, where min fract of bout >= {:.2f}'.format(behav, min_frac_bout))

putil.label_figure(fig, figid)


#%%

import regplot as rpl

#%% 
# Scatterplot:  ANG_VEL vs. THETA_ERROR -- color coded by STIM_HZ
xvar = 'facing_angle_deg'
yvar = 'ang_vel_abs_deg'

meanbouts['ang_vel_deg'] = np.rad2deg(meanbouts['ang_vel'])
meanbouts['ang_vel_abs_deg'] = np.rad2deg(meanbouts['ang_vel_abs'])

#importlib.reload(putil)
min_frac_bout = 0.5
chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>min_frac_bout].copy()

cmap='viridis'
# stimhz_palette = putil.get_palette_dict(chase_[chase_['stim_hz']>0], 'stim_hz', cmap=cmap)
#stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>0], 'stim_hz', cmap=cmap)
stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)

vmin = min(list(stimhz_palette.keys()))
vmax = max(list(stimhz_palette.keys()))

fig, axn = pl.subplots(1, 2, sharex=True, sharey=True)
for ai, (sp, df_) in enumerate(chase_[chase_['stim_hz']>0].groupby('species')):
    ax = axn[ai]
    sns.scatterplot(data=df_, x=xvar, y=yvar, ax=ax,
                 hue='stim_hz', palette=stimhz_palette, legend=0, edgecolor='none', alpha=0.7)
    ax.set_title(sp)
    sns.regplot(data=df_, x=xvar, y=yvar, ax=ax,
                color=bg_color, scatter=False)
    ax.set_box_aspect(1)
    
    # set xlabel to be theta subscript E
    ax.set_xlabel(r'$\theta_{E}$')
    ax.set_ylabel('$\omega_{f}$ (deg/s)')

    # annotate regr
    putil.annotate_regr(df_, ax, x=xvar, y=yvar, 
                        xloc=0.05, yloc=0.9, fontsize=8)
    # Do fit
    res = rpl.regplot(data=df_, ax=ax, x=xvar, y=yvar,
                color=bg_color, scatter=False) #, ax=ax)
    # res.params: [intercept, slope]
    ax.set_box_aspect(1)
    fit_str = 'OLS: y = {:.2f}x + {:.2f}'.format(res.params[1], 
                                                 res.params[0])
    ax.text(0.05, 0.95, fit_str, fontsize=8, 
            transform=ax.transAxes)

putil.colorbar_from_mappable(ax, cmap=cmap, norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                             axes=[0.92, 0.3, 0.01, 0.4], hue_title='stim. freq. (Hz)', fontsize=18)

fig.suptitle('{} bouts, where min fract of bout >= {:.2f}'.format(behav, min_frac_bout))

putil.label_figure(fig, figid)
figname = 'sct_angvel_v_thetaerr_stimhz_mel-v-yak_min-frac-bout-{}'.format(min_frac_bout)
#pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir, figname)

#%% Fit REGR to each stim_hz level

behav = 'chasing'
min_frac_bout = 0.5
chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>min_frac_bout].copy()

fig, ax =pl.subplots()
for ai, (sp, df_) in enumerate(chase_[chase_['stim_hz']>0].groupby('species')):
    g = sns.lmplot(data=df_, x='facing_angle_deg', y='ang_vel_abs', 
                 hue='stim_hz', palette=stimhz_palette, legend=0)
    g.fig.axes[0].set_title(sp)
    g.fig.axes[0].set_box_aspect(1)
    
    # set xlabel to be theta subscript E
    g.fig.axes[0].set_xlabel(r'$\theta_{E}$')
    g.fig.axes[0].set_ylabel('$\omega_{f}$ (deg/s)')

    pl.xlim([0, 80])
    pl.ylim([0, 10])

    putil.colorbar_from_mappable(g.fig.axes[0], cmap=cmap, norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                             axes=[0.92, 0.3, 0.01, 0.4])
    pl.subplots_adjust(top=0.9)

    putil.label_figure(g.fig, figid)
    figname = 'sct-regr_angvel_v_thetaerr_stimhz_{}_min-frac-bout-{}'.format(sp, min_frac_bout)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    print(figdir, figname)


#%% IS THIS FAKE? look at 1 animal
# 

acqs = meanbouts['acquisition'].unique()
acq = acqs[20] #'20240212-1215_fly3_Dmel_sP1-ChR_3do_sh_8x8'a
behav = 'chasing'
frames_ = filtdf[ (filtdf['acquisition']==acq) & (filtdf['{}_binary'.format(behav)]>0)].copy()

mean_ = frames_.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()
mean_['stim_hz'] = mean_['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))
mean1 = meanbouts[ (meanbouts['acquisition']==acq) & (meanbouts['{}_binary'.format(behav)]>0)].copy()

fig, axn =pl.subplots(1, 2, figsize=(10,4))

ax=axn[0]
sns.scatterplot(data=mean_, x='facing_angle', y='ang_vel_abs', ax=ax, hue='stim_hz', palette=stimhz_palette,
                legend=0)
ax.set_title('{:.2f}s bouts'.format(bout_dur))
ax=axn[1]
sns.scatterplot(data=frames_, x='facing_angle', y='ang_vel_abs', ax=ax, hue='stim_hz', palette=stimhz_palette,
                legend=0)
ax.set_title('frames')

pl.subplots_adjust(top=0.8)
fig.suptitle(acq, fontsize=18)

pl.subplots_adjust(wspace=0.5)

#%%
xvar = 'facing_angle'
yvar = 'ang_vel_abs'
n_stim = filtdf['stim_hz'].nunique() # no zero

do_bouts = False

plotd_ = mean1.copy() if do_bouts else frames_.copy()
data_type = 'BOUTS' if do_bouts else 'FRAMES'

if 'vel' in xvar:
    xlabel = r'$\omega_{\theta}$'
else:
    xlabel = r'$\theta_{E}$'
ylabel = '$\omega_{f}$'

fig, axn = pl.subplots(n_stim, 1, sharex=True, sharey=True, figsize=(4, n_stim*3))
all_stims = sorted(list(stimhz_palette.keys()))
start_i = all_stims.index(plotd_['stim_hz'].min())

for i, (stim, sd_) in enumerate(plotd_.groupby('stim_hz')):  
    ax=axn[i+start_i]
    ax.set_title(stim, loc='left', fontsize=12)
    sns.scatterplot(data=sd_, x=xvar, y=yvar, ax=ax, hue='stim_hz', palette=stimhz_palette,
                legend=0)
    ax.set_ylabel(ylabel)

ax.set_xlabel(xlabel)
putil.label_figure(fig, acq)

fig.suptitle('{:.2f}s {}'.format(bout_dur, data_type), fontsize=24)


#%% delta theta-error?

#meanbouts['facing_angle_vel_abs'] = meanbouts['facing_angle_vel'].abs()

xvar = 'facing_angle_vel_deg_abs'
yvar = 'ang_vel_abs'

behav = 'chasing'
min_frac_bout = 0.5
do_bouts = False

chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>min_frac_bout].copy()

fig, axn = pl.subplots(1, 2, sharex=True, sharey=True)
for ai, (sp, df_) in enumerate(chase_.groupby('species')):
    ax = axn[ai]
    sns.regplot(data=df_, x=xvar, y=yvar, ax=ax,
                color=species_palette[sp])
    ax.set_title(sp)
    ax.set_box_aspect(1)

fig.suptitle('{} bouts, where min fract of bout >= {:.2f}'.format(behav, min_frac_bout))

putil.label_figure(fig, figid)

#%%

xvar = 'facing_angle_vel_deg_abs'
yvar = 'ang_vel_abs'

behav = 'chasing'
min_frac_bout = 0.8
do_bouts = True

if do_bouts:
    chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>min_frac_bout].copy()
else:
    chase_ = filtdf[( filtdf['chasing']>jaaba_thresh_dict['chasing'])
              & ( filtdf['stim_hz']>0 )].copy()


fig, axn = pl.subplots(1, 2, sharex=True, sharey=True)
for ai, (sp, df_) in enumerate(chase_[chase_['stim_hz']>0].groupby('species')):
    ax = axn[ai]
    sns.scatterplot(data=df_, x=xvar, y=yvar, ax=ax,
                 hue='stim_hz', palette=stimhz_palette, legend=0, edgecolor='none', alpha=0.7)
    ax.set_title(sp)
    ax.set_box_aspect(1)
    
    # set xlabel to be theta subscript E
    ax.set_xlabel(r'$\omega_{\theta}}$')
    ax.set_ylabel('$\omega_{f}$ (deg/s)')
putil.colorbar_from_mappable(ax, cmap=cmap, norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                             axes=[0.92, 0.3, 0.01, 0.4])

fig.suptitle('{} bouts, where min fract of bout >= {:.2f}'.format(behav, min_frac_bout))








#%% ---------------------------------------------
# PLOT ALLO vs. EGO

import parallel_pursuit as pp
importlib.reload(pp)

behav = 'chasing'
min_frac_bout = 0.8
do_bouts=False

markersize=5
huevar='stim_hz'
cmap='viridis'
plot_com=True

xvar= 'facing_angle'
yvar = 'targ_pos_radius'

is_flytracker=True
sign = -1 if is_flytracker else 1

chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>=min_frac_bout].copy()

data_type = 'BOUTS' if do_bouts else 'FRAMES' 
if not do_bouts:
    # plot frames
    palette_dict = putil.get_palette_dict(filtdf[filtdf['stim_hz']>0], huevar, cmap='viridis')
    plotdf = filtdf[( filtdf['chasing']>jaaba_thresh_dict['chasing'])
              & ( filtdf['stim_hz']>0 )].copy()
else:
    palette_dict = stimhz_palette
    plotdf = chase_[chase_['stim_hz']>0].copy()

plotdf['facing_angle'] = sign * plotdf['facing_angle']


vmin = min(list(palette_dict.keys()))
vmax = max(list(palette_dict.keys()))
hue_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

for sp, p_ in plotdf.groupby('species'):
    fig, axn = pl.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True,
                                subplot_kw={'projection': 'polar'})
    pp.plot_allo_vs_egocentric_pos(p_, axn=axn, xvar=xvar, yvar=yvar, huevar=huevar,
                                palette_dict=palette_dict, hue_norm=hue_norm, markersize=5,
                                com_markersize=40, com_lw=1)
    for ax in axn:
        yl = ax.get_yticklabels()
        ax.set_yticklabels([v if i%2==0 else '' for i, v in enumerate(yl)])
    pl.subplots_adjust(wspace=0.6, top=0.8, right=0.8)

    putil.label_figure(fig, figid)

    fig.suptitle('{}: {} {}, where min fract of bout >= {:.2f}'.format(sp, behav, data_type, min_frac_bout))

    figname = 'allo-v-ego-{}-{}_{}-v-{}_min-frac-bout-{}_{}'.format(behav, data_type, xvar, yvar, min_frac_bout, sp)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

    print(figdir, figname)



#%% Plot as 2D hists

xvar = 'facing_angle_deg'
yvar = 'ang_vel_abs' #'ang_vel_abs'

plot_behavs = ['orienting', 'chasing', 'singing']
fig, axn =pl.subplots(2, len(plot_behavs), figsize=(10,8), sharex=True, sharey=True)

nbins=25
x_bins = np.linspace(0, meanbouts[xvar].max(), nbins, endpoint=False)
y_bins = np.linspace(0, meanbouts[yvar].max(), nbins, endpoint=False)

meanbouts['x_bin'] = pd.cut(meanbouts[xvar], bins=x_bins, labels=x_bins[0:-1])
meanbouts['y_bin'] = pd.cut(meanbouts[yvar], bins=y_bins, labels=y_bins[0:-1])


vmin = 0
vmax = 1.0
cmap='magma'
for ri, (sp, means_) in enumerate(meanbouts.groupby('species')):

    #xmin, xmax = means_[xvar].min(), means_[xvar].max()
    #ymin, ymax = means_[yvar].min(), means_[yvar].max()
     
    for ci, behav in enumerate(plot_behavs):
        ax = axn[ri, ci]
        ax.set_title(behav)
        
        probs = means_.groupby(['x_bin', 'y_bin'])['{}_binary'.format(behav)].mean().reset_index()
        arr_ = pd.pivot_table(probs, columns='x_bin', index='y_bin').values

        ax.imshow(arr_, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        print(np.nanmax(arr_))        
        
        ax.set_xticks(np.arange(len(x_bins)))
        ax.set_xticklabels([np.round(v) if i%10==0 else '' for i, v in enumerate(x_bins)]) #probs['x_bin'])
        ax.set_yticks(np.arange(len(y_bins)))
        ax.set_yticklabels([np.round(v) if i%10==0 else '' for i, v in enumerate(y_bins)]) #probs['x_bin'])

        # Create 2D histogram using np.histogram2d()
        #hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=25)

        # Normalize
        #total_counts = np.sum(hist)
        #hist_normalized = hist / total_counts

        # Create meshgrid from edges
        #x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)
        #pcm = ax.pcolormesh(x_mesh, y_mesh, hist_normalized.T, cmap='magma') #, vmin=0, vmax=0.05) #, vmin=0, vmax=0.05)

        if ci==0:
            ax.set_title('{}: {}'.format(sp, behav))
        # ax.set_xlim([xmin, xmax])
        #ax.set_ylim([ymin, 300])

        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)
        ax.set_box_aspect(1)

putil.colorbar_from_mappable(ax, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                             cmap=cmap, axes=[0.92, 0.3, 0.01, 0.4])

fig.suptitle('Prob of each behavior, bin x and y values (n={} bins)'.format(nbins))

putil.label_figure(fig, figid)


#%%


#%%



#%%

nbins=20
#bins = np.arange(0, nbins)
binlabels = np.arange(0, nbins)
plotdf['targ_rel_pos_x_binned'] = pd.cut(plotdf['targ_rel_pos_x'], bins=nbins)
plotdf['targ_rel_pos_y_binned'] = pd.cut(plotdf['targ_rel_pos_y'], bins=nbins)
plotdf['targ_rel_pos_x_left'] = [v.left if isinstance(v, pd.Interval) else v for v in plotdf['targ_rel_pos_x_binned']]
plotdf['targ_rel_pos_y_left'] = [v.left if isinstance(v, pd.Interval) else v for v in plotdf['targ_rel_pos_y_binned']]



#%%


#%%
import SeabornFig2Grid as sfg
import matplotlib.gridspec as gridspec

gdict = dict((sp, []) for sp in plotdf['species'].unique())
for (sp, acq), df_ in plotdf.groupby(['species', 'acquisition']):
    g1 = sns.jointplot(data=df_, x=xvar, y=yvar, hue='species', kind='kde',
                    palette=species_palette)
    gdict[sp].append(g1)

max_n = int(plotdf[['species', 'acquisition']].drop_duplicates().groupby('species').count().max())
fig = pl.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, max_n)

for ri, sp in enumerate(gdict.keys()):
    for ci, g1 in enumerate(gdict[sp]):
        mg0 = sfg.SeabornFig2Grid(g1, fig, gs[ri, ci])




# %%
