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


from relative_metrics import load_processed_data
import utils as util
import plotting as putil

#%%
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=12)
bg_color = [0.7]*3 if plot_style=='dark' else 'w'


#%%
def load_aggregate_data(savedir, mat_type='df'):
    '''
    Find all *feat.pkl (or *trk.pkl) files in savedir and load them into a single dataframe.

    Arguments:
        savedir -- Full path to dir containing processed *feat.pkl files.

    Keyword Arguments:
        mat_type -- feat or trk (default: {'feat'})

    Returns:
        feat -- pandas dataframe containing all processed data.
    '''
    found_fns = glob.glob(os.path.join(savedir, '*{}.pkl'.format(mat_type)))
    print("Found {} processed *_{}.pkl files".format(len(found_fns), mat_type))
    f_list=[]
    for fp in found_fns:
        if 'BADTRACKING' in fp:
            continue
        if 'ele' in fp: # ignore ele for now
            continue
        #fp = found_fns[0]
        #acq = os.path.split(acq_viddir)[0]
        print(os.path.split(fp)[-1])
        with open(fp, 'rb') as f:
            feat_ = pkl.load(f)
        acq = os.path.split(fp)[1].split('_{}'.format(mat_type))[0] 
        feat_['acquisition'] = acq 

        if 'yak' in acq:
            feat_['species'] = 'Dyak'
        elif 'mel' in acq:
            feat_['species'] = 'Dmel'
        else:
            feat_['species'] = 'Dele'

        f_list.append(feat_)

    feat = pd.concat(f_list, axis=0).reset_index(drop=True) 

    return feat

#%% LOAD ALL THE DATA
#savedir = '/Volumes/Julie/free-behavior-analysis/FlyTracker/38mm_dyad/processed'
#figdir = os.path.join(os.path.split(savedir)[0], 'figures', 'relative_metrics')

create_new = False

# Set sourcedirs
srcdir = '/Volumes/Julie/2d-projector-analysis/FlyTracker/processed_mats' #relative_metrics'
figdir = os.path.join(os.path.split(srcdir)[0], 'relative_metrics', 'figures')

if not os.path.exists(figdir):
    os.makedirs(figdir)

# LOCAL savedir 
localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/2d-projector/FlyTracker'
out_fpath_local = os.path.join(localdir, 'processed.pkl')
print(out_fpath_local)

if not create_new:
    if os.path.exists(out_fpath_local):
        df = pd.read_pickle(out_fpath_local)
        print("Loaded local processed data.")
    else:
        create_new = True

if create_new:
    df = load_aggregate_data(srcdir, mat_type='df')
    print(df['species'].unique())

    #% save
    out_fpath = os.path.join(os.path.split(figdir)[0], 'processed.pkl')
    df.to_pickle(out_fpath)
    print(out_fpath)

    # save local, too
    df.to_pickle(out_fpath_local)


print(df[['species', 'acquisition']].drop_duplicates().groupby('species').count())

#%%

#f = df['acquisition'].iloc[0]
#df['acquisition'] = ['_'.join(f.split('_')[0:-1]) for f in df['acquisition']]


#%% plotting settings
curr_species = ['Dele', 'Dmau', 'Dmel', 'Dsant', 'Dyak']
species_cmap = sns.color_palette('colorblind', n_colors=len(curr_species))
print(curr_species)
species_palette = dict((sp, col) for sp, col in zip(curr_species, species_cmap))

#%% load jaaba data
importlib.reload(util)
jaaba = util.load_jaaba('2d-projector')

print(jaaba[['species', 'filename']].drop_duplicates().groupby('species').count())

jaaba = jaaba.rename(columns={'filename': 'acquisition'})

#%% Set fig id
figid = srcdir  

#%% merge jaaba and processed data
c_list = []
for acq, ja_ in jaaba.groupby('acquisition'):
    df_ = df[(df['acquisition']==acq) & (df['id']==0)].reset_index(drop=True)
    if len(df_)>0:
        assert ja_.shape[0] == df_.shape[0], "Mismatch in number of flies between jaaba {} and processed data {}.".format(ja_.shape, df_.shape) 
        drop_cols = [c for c in ja_.columns if c in df_.columns]
        combined_ = pd.concat([df_, ja_.drop(columns=drop_cols)], axis=1)
        assert combined_.shape[0] == df_.shape[0], "Bad merge: {}".format(acq)
        c_list.append(combined_)

ftjaaba = pd.concat(c_list, axis=0).reset_index(drop=True)
# unsmooth
#ftjaaba['rel_vel_abs_raw'] = ftjaaba['rel_vel_abs'] * 5.

ftjaaba[['species', 'acquisition']].drop_duplicates().groupby('species').count()
#%%

ftjaaba = ftjaaba.rename(columns={'courtship': 'courting'})
#%% add bouts
if 'fpath' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['fpath'])
if 'name' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['name'])
#% 
d_list = []
for acq, df_ in ftjaaba.groupby('acquisition'):
    df_ = df_.reset_index(drop=True)
    df_ = util.mat_split_courtship_bouts(df_, bout_marker='courting')
    dur_ = util.get_bout_durs(df_, bout_varname='boutnum', return_as_df=True,
                    timevar='sec')
    d_list.append(df_.merge(dur_, on=['boutnum']))
ftjaaba = pd.concat(d_list)
#%%
winsize=5
#print(acq)

#df_ = ftjaaba[ftjaaba['acquisition']==acq]
ftjaaba['targ_pos_theta_deg'] = np.rad2deg(ftjaaba['targ_pos_theta'])

for acq, df_ in ftjaaba.groupby('acquisition'):
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='targ_pos_theta', vel_var='targ_ang_vel',
                                  time_var='sec', winsize=winsize)
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='targ_pos_theta_deg', vel_var='targ_ang_vel_deg',
                                  time_var='sec', winsize=winsize)

    ftjaaba.loc[ftjaaba['acquisition']==acq, 'targ_ang_vel'] = df_['targ_ang_vel']
    ftjaaba.loc[ftjaaba['acquisition']==acq, 'targ_ang_vel_deg'] = df_['targ_ang_vel_deg']

#%%
ftjaaba['targ_ang_vel_abs'] = np.abs(ftjaaba['targ_ang_vel'])
ftjaaba['targ_pos_theta_abs'] = np.abs(ftjaaba['targ_pos_theta'])
ftjaaba['targ_ang_size_deg'] = np.rad2deg(ftjaaba['targ_ang_size'])
ftjaaba['targ_ang_vel_deg_abs'] = np.abs(ftjaaba['targ_ang_vel_deg'])
#ftjaaba['targ_pos_theta_abs_deg'] = np.rad2deg(ftjaaba['targ_pos_theta_abs'])  
ftjaaba['facing_angle_deg'] = np.rad2deg(ftjaaba['facing_angle'])
ftjaaba['ang_vel_abs'] = np.abs(ftjaaba['ang_vel'])

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

pl.subplots_adjust(wspace=0.5)
putil.label_figure(fig, figid) 

figname = 'hist_{}_nocourt-v-court-v-{}bouts_minboutdur-{}_mel-v-yak'.format(xvar, varname, min_boutdur)
pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)
print(figdir, figname)


#%% JOINT DISTS
min_boutdur = 0.5
plotdf = filtdf[filtdf['boutdur']>=min_boutdur]

varname = 'singing'

joint_kind = 'kde'
x = 'abs_rel_ang_vel'
y = 'dovas'
if varname=='notcourting':
    g = sns.jointplot(data=plotdf[plotdf['courting']==0].reset_index(drop=True), 
                x=x, y=y, 
                hue='species', palette=species_palette, #palette=species_cdict,
                kind=joint_kind) #, joint_kws={'s': 10, 'alpha': 0.8, 'n_levels': 30})
else:
    g = sns.jointplot(data=plotdf[plotdf['{}_binary'.format(varname)]>0].reset_index(drop=True), 
                x=x, y=y, 
                hue='species', palette=species_palette, #palette=species_cdict,
                kind=joint_kind) #, joint_kws={'s': 10, 'alpha': 0.8, 'n_levels': 20})
#pl.xlim([-2, 20])
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


def plot_2d_hist(plotdf, xvar='targ_rel_pos_x', yvar='targ_rel_pos_y', ax=ax,
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
min_boutdur = 1.0
filtdf = ftjaaba[(ftjaaba['id']==0)
                #& (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                #& (ftjaaba['targ_pos_theta']<=max_pos_theta)
                & (ftjaaba['dist_to_other']>=min_dist_to_other)
                & (ftjaaba['boutdur']>=min_boutdur)
                & (ftjaaba['good_frames']==1)
                ].copy().reset_index(drop=True)
#meanbins = filtdf.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()    

# means are averaged over bout, so threshold is now 0
jaaba_thresh_dict = {'orienting': 10,
                     'chasing': 10,
                     'singing': 5}

filtdf = util.assign_jaaba_behaviors(filtdf, courtvar=courtvar, jaaba_thresh_dict=jaaba_thresh_dict)
plot_behavs = ['disengaged', 'orienting', 'chasing', 'singing']
behavior_colors = [[0.3]*3, 'mediumaquamarine', 'aqua', 'violet']
behavior_palette = dict((b, c) for b, c in zip(plot_behavs[1:], behavior_colors[1:]))

for sp in filtdf['species'].unique():
    fig, axn = pl.subplots(1, 3, figsize=(10, 5))
    for ai, (behav, thr) in enumerate(jaaba_thresh_dict.items()):
        ax=axn[ai]
        plotdf = filtdf[ (filtdf['species']==sp) & (filtdf[behav]> thr)]
        plot_2d_hist(plotdf, ax=ax, color=behavior_palette[behav], nbins=30,
                                    ylim=[-900, 900], xlim=(-500, 900))
        ax.set_title(behav)
    #fig.suptitle(sp)
    fig.text(0.05, 0.85, 
        '{}: Relative target pos. (all frames, min bout dur={:.12}s)'.format(sp, min_boutdur), fontsize=12)
    
    pl.subplots_adjust(wspace=0.6)
    putil.label_figure(fig, figid)

    figname = 'rel-pos_frames_mindur-{}_by-courting-behavior_{}'.format(min_boutdur, sp)
    pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)

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
min_dist_to_other = 1
   
min_boutdur = 0.25 #0.25

filtdf = ftjaaba[(ftjaaba['id']==0)
                #& (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                #& (ftjaaba['targ_pos_theta']<=max_pos_theta)
                #& (ftjaaba['dist_to_other']>=min_dist_to_other)
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


#%%

min_boutdur = 0.25 #0.25

filtdf = ftjaaba[(ftjaaba['id']==0)
                #& (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                #& (ftjaaba['targ_pos_theta']<=max_pos_theta)
                #& (ftjaaba['dist_to_other']>=min_dist_to_other)
                & (ftjaaba['boutdur']>=min_boutdur)
                & (ftjaaba['good_frames']==1)
                & (ftjaaba['led_level']>0)
                ].copy().reset_index(drop=True)

#xvar = 'targ_rel_pos_x' # 'targ_ang_size'
#yvar = 'targ_rel_pos_y' #'targ_ang_vel'

xvar = 'facing_angle_deg' #'targ_ang_size_deg'
yvar = 'abs_rel_ang_vel' #targ_ang_vel_abs_deg'
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
        ax.set_ylim([ymin, 300])

        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)
        ax.set_box_aspect(1)

#%%
# look at hist of 1 variable
plotvar = 'targ_ang_vel_abs_deg'
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



#%% cut up into mini-bouts

def subdivide_into_bouts(filtdf, bout_dur=1.0):

    # assign grouping based on row index -- filtdf should have the original indices (frames) as ftjaaba
    consec_bouts = util.get_indices_of_consecutive_rows(filtdf)

    b_list = []
    boutnum = 0
    for g, df_ in filtdf.groupby('group'):
        group_dur_sec = df_.iloc[-1]['sec'] - df_.iloc[0]['sec']
        #print(group_dur_sec)
        if group_dur_sec / bout_dur < 2:
            df_['boutnum'] = boutnum
            #filtdf.loc[filtdf['group'] == g, 'boutnum'] = boutnum
            bin_labels = [boutnum]
        else:
            # subdivide into mini-bouts of bout_dur length
            group_dur_sec = df_.iloc[-1]['sec'] - df_.iloc[0]['sec']
            #t0 = df_.iloc[0]['sec']
            n_mini_bouts = int(group_dur_sec / bout_dur)
            #t1_values = np.linspace(t0 + bout_dur, t0 + group_dur_sec, n_mini_bouts, endpoint=False, )

            bins = np.linspace(df_.iloc[0]['sec'], df_.iloc[-1]['sec'], n_mini_bouts, endpoint=False)
            bin_labels = np.arange(boutnum, boutnum + len(bins)-1)
            #print(bin_labels)
            df_['boutnum'] = pd.cut(df_['sec'], bins=bins, labels=bin_labels)

            #filtdf.loc[filtdf['group']==g, 'boutnum'] = df_['bin_sec']

        boutnum += len(bin_labels)
        b_list.append(df_)
    filtdf = pd.concat(b_list)

    return filtdf

#%%
min_bout_dur = 0.25
min_dist_to_other = 2

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

# subdivide into smaller bouts
filtdf = subdivide_into_bouts(filtdf, bout_dur=1.5)

#%%

meanbouts = filtdf.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()
meanbouts.head()


#%%
#xvar = 'facing_angle_deg'
#yvar = 'abs_rel_ang_vel'

yvar = 'dovas'
xvar = 'abs_rel_vel' #'abs_rel_ang_vel'
xmin, xmax = meanbouts[xvar].min(), meanbouts[xvar].max()
ymin, ymax = meanbouts[yvar].min(), meanbouts[yvar].max()

min_frac_bout = 0.3
plot_behavs = ['orienting', 'chasing', 'singing']
fig, axn =pl.subplots(2, len(plot_behavs), figsize=(10,8), sharex=True, sharey=True)

vmin=0
vmax=0.04
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

#%%
# KDE joints

xvar = 'abs_rel_vel'
yvar = 'dovas'

min_frac_bout = 0.5

for behav in ['chasing',  'singing']:
    g = sns.jointplot(data=meanbouts[meanbouts['{}_binary'.format(behav)]>min_frac_bout], ax=ax,
                x = xvar, y = yvar, hue='species', palette=species_palette,
            kind='kde') 
    g.fig.suptitle('{}: min frac of bout={:.2f}'.format(behav, min_frac_bout))

#%%
# Prob(singing) vs. DIST TO OTHER
nbins = 10
dist_bins =  np.linspace(meanbouts['dist_to_other'].min(), meanbouts['dist_to_other'].max(), nbins, endpoint=False)
meanbouts['dist_bin'] = pd.cut(meanbouts['dist_to_other'], bins=dist_bins, labels=dist_bins[0:-1])

fig, axn = pl.subplots(1, 2, figsize=(10,4))
ax=axn[0]
behav = 'singing'
sns.barplot(data=meanbouts, x='dist_bin', y='{}_binary'.format(behav), ax=ax,
            hue='species', palette=species_palette, edgecolor='none')
ax.legend_.remove()
ax.set_xticklabels([np.round(v) for v in dist_bins[0:-1]])

ax = axn[1]#
min_frac_bout=0
sns.histplot(data=meanbouts[meanbouts['{}_binary'.format(behav)]>min_frac_bout], x='dist_to_other', ax=ax,
            hue='species', palette=species_palette, fill=False, element='poly',
            common_norm=False, stat='probability')
sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', frameon=False)

for ax in axn:
    ax.set_box_aspect(1)

putil.label_figure(fig, figid)


#%%

behav = 'chasing'
min_frac_bout = 0.2
chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>min_frac_bout].copy()
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

xvar = 'facing_angle_deg'
yvar = 'ang_vel_abs'

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
