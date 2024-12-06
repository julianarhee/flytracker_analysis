#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#%%
import os
import sys
import glob
import traceback

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
#%%

import dlc as dlc
import plot_dlc_frames as pdlc

#%%
# src paths
minerva_base='/Volumes/Julie'
projectname='projector-1dot-jyr-2024-02-18' 
procdir = os.path.join(minerva_base, '2d-projector-analysis/DeepLabCut', projectname)
#destdir = os.path.join(procdir.split(projectname)[0], 'processed')

# Load
localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/2d-projector/DLC'
out_fpath = os.path.join(localdir, 'processed.pkl')
df = pd.read_pickle(out_fpath)

#%% Set output dirs
destdir = '/Volumes/Julie/2d-projector-analysis/DeepLabCut/relative_metrics'
figdir = os.path.join(destdir, 'figures')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

#%%
# Create colormap for velocities
stimhz_vals = np.array([0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.625, 0.8, 1])
stimhz_dict = dict((k, v) for k, v in zip(np.arange(0, len(stimhz_vals)), stimhz_vals))
df['stim_hz'] = [stimhz_dict[v] if v in stimhz_dict.keys() else None for v in df['epoch']]

#%% Load JAABA

jaaba = util.load_jaaba('2d-projector')
#
#srcdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/2d-projector/JAABA'
##% Load jaaba-traansformed data
#jaaba_fpath = os.path.join(srcdir, 'jaaba_transformed_data_transf.pkl')
#assert os.path.exists(jaaba_fpath), "File not found: {}".format(jaaba_fpath)
#jaaba = pd.read_pickle(jaaba_fpath)   
print(jaaba['species'].unique())

# Load FT
#ft_fpath = os.path.join(srcdir, 'flytracker.pkl')
#assert os.path.exists(ft_fpath), "File not found: {}".format(ft_fpath)
#ft = pd.read_pickle(ft_fpath)   

#%% Check matches
print('JAABA: {} acqs.'.format(jaaba['name'].nunique()))
print(jaaba[['species', 'name']].drop_duplicates().groupby('species').count())

print('DLC: {} acqs.'.format(df['acquisition'].nunique()))
print(df[['species', 'acquisition']].drop_duplicates().groupby('species').count())

#%% plotting settings
#curr_species = ['Dele', 'Dmau', 'Dmel', 'Dsant', 'Dyak']
curr_species = ['Dele', 'Dmau', 'mel', 'Dsant', 'yak']

species_cmap = sns.color_palette('colorblind', n_colors=len(curr_species))
print(curr_species)
species_palette = dict((sp, col) for sp, col in zip(curr_species, species_cmap))

#%% Set fig id
figid = '2d-projector-analysis/DeepLabCut/processed - {}'.format(os.path.split(jaaba_fpath))
#%%

has_both = [c for c in df['acquisition'].unique() if c in jaaba['name'].unique()]
print(len(has_both))

#%% Create combined DF with both JAABA and FT data

errors = []
c_list = []
for acq, df_ in df[df['id']==0].groupby('acquisition'):
    
    if acq not in jaaba['name'].unique():
        continue
    #df_ = df[df['acquisition']==acq]
    try:
        ja_ = jaaba[jaaba['name']==acq]
        assert df_.shape[0] == ja_.shape[0], "Mismatch in number of flies between jaaba {} and processed data {}.".format(ja_.shape, df_.shape)
        drop_cols = [c for c in ja_.columns if c in df_.columns]
        combined_ = pd.concat([df_, ja_.drop(columns=drop_cols)], axis=1)
        c_list.append(combined_)
    except Exception as e:
        traceback.print_exc()
        errors.append(acq)
        continue

ftjaaba = pd.concat(c_list, axis=0).reset_index(drop=True)

ftjaaba[['species', 'name']].drop_duplicates().groupby('species').count()


#%%
fp = '/Volumes/Julie/2d-projector-analysis/DeepLabCut/processed/20240212-1130_fly2_Dyak_sP1-ChR_3do_sh_6x6.pkl'

df_ = pd.read_pickle(fp)
df_.shape



#%% add bouts
if 'acquisition' not in ftjaaba.columns:
    ftjaaba['acquisition'] = ftjaaba['filename']
if 'fpath' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['fpath'])
#% 
d_list = []
for acq, df_ in ftjaaba.groupby('acquisition'):
    df_ = df_.reset_index(drop=True)
    df_ = util.mat_split_courtship_bouts(df_, bout_marker='courtship')
    dur_ = util.get_bout_durs(df_, bout_varname='boutnum', return_as_df=True,
                    timevar='sec')
    d_list.append(df_.merge(dur_, on=['boutnum']))
ftjaaba = pd.concat(d_list)

#%% get means by BOUT
groupcols = [ 'species', 'acquisition', 'boutnum']

min_pos_theta = np.deg2rad(-160)
max_pos_theta = np.deg2rad(160)
min_dist_to_other = 1

ftjaaba.drop(columns=['name', 'filename'], inplace=True)

ftjaaba_filt = ftjaaba[(ftjaaba['id']==0)
            & (ftjaaba['targ_pos_theta']>=min_pos_theta) 
            & (ftjaaba['targ_pos_theta']<=max_pos_theta)]


ftjaaba_filt['acquisition'] = ftjaaba_filt['acquisition'].astype(str)

meandf = ftjaaba_filt.groupby(groupcols).mean().reset_index()

#%% DEBUG
#meandf = meandf[meandf['boutdur']>=min_boutdur] 

# ------------------------------
# PLOT
# ------------------------------
#%% boutdurs
min_boutdur = 0.5
court_var = 'courtship'
plotdf = meandf[meandf['boutdur']>=min_boutdur]

xvar = 'dist_to_other'
varname = 'singing'

jaaba_thresh = 5
plotdf['{}_binary'.format(varname)] = 0
plotdf.loc[plotdf[varname]>jaaba_thresh, '{}_binary'.format(varname)] = 1

fig, axn = pl.subplots(1, 3, figsize=(10,4))#, sharex=True, sharey=True)
ax=axn[0]
ax.set_title('not courting')
sns.histplot(data=plotdf[plotdf[court_var]==0], x=xvar,  ax=ax,
             hue='species', alpha=0.7, palette=species_palette, stat='probability',
             cumulative=False, common_norm=False, bins=40, legend=0)
ax=axn[1]
ax.set_title('courting')
sns.histplot(data=plotdf[plotdf[court_var]==1], x=xvar,  ax=ax,
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
min_boutdur = 1
plotdf = meandf[meandf['boutdur']>=min_boutdur]

varname = 'chasing'
x = 'targ_ang_size_deg' #'rel_vel_abs'
y = 'targ_ang_vel' #'targ_ang_size_deg'
if varname=='notcourting':
    g = sns.jointplot(data=plotdf[plotdf[court_var]==0].reset_index(drop=True), 
                x=x, y=y, 
                hue='species', palette=species_palette, #palette=species_cdict,
                kind='kde', joint_kws={'s': 10, 'alpha': 0.8, 'n_levels': 20})
else:
    g = sns.jointplot(data=plotdf[plotdf[varname]>0].reset_index(drop=True), 
                x=x, y=y, 
                hue='species', palette=species_palette, #palette=species_cdict,
                kind='kde', joint_kws={'s': 10, 'alpha': 0.8, 'n_levels': 20})
#pl.xlim([-2, 20])
#pl.ylim([-5, 50]) 
pl.xlim([-2, 40])
pl.ylim([-3, 3]) 

g.fig.suptitle(varname)
pl.subplots_adjust(top=0.9)

putil.label_figure(g.fig, figid) 

figname = 'jointdist_{}_{}-v-{}_minboutdur-{}_mel-v-yak'.format(varname, x, y, min_boutdur)
pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)
print(figdir, figname)


#%%
# ------------------------------------
# COMPARE WITH DLC
# ------------------------------------

#%%
dlc_file = '/Volumes/Julie/free-behavior-analysis/38mm-dyad/dlc.pkl'
with open(dlc_file, 'rb') as f:
    dlc = pkl.load(f)

d_list = []
no_jaaba = []
for acq, df_ in dlc[dlc['species'].isin(['Dyak', 'Dmel'])].groupby('acquisition'):
    last_ix = df_.iloc[-1].name
    if acq not in jaaba['acquisition'].unique():
        no_jaaba.append(acq)
        continue
    j_ = jaaba[jaaba['acquisition']==acq].loc[:last_ix]
    assert j_.shape[0] ==  df_.shape[0], "Bad match: {}".format(acq)
    df_ = pd.concat([df_, j_.drop(columns=['species', 'acquisition', 'age'])], axis=1)
    d_list.append(df_)

fulldf = pd.concat(d_list)

#%%

acq = '20240126-1023-fly2-melWT_4do_sh_melWT_4do_gh'
df1 = ftjaaba[ftjaaba['acquisition']==acq].reset_index(drop=True)
df2 = fulldf[fulldf['acquisition']==acq].reset_index(drop=True)

fig, ax = pl.subplots()
sns.histplot(data=df1[df1['courting']==1], x='dist_to_other', ax=ax)
sns.histplot(data=df2[df2['courting']==1], x='dist_to_other_mm', ax=ax)

fig, ax =pl.subplots()
sns.histplot(data=df1[df1['courting']==1], x='rel_vel_abs', ax=ax)
sns.histplot(data=df2[df2['courting']==1], x='rel_vel_mms', ax=ax)


#%%
df1 = util.mat_split_courtship_bouts(df1, bout_marker='courting')
df2 = util.mat_split_courtship_bouts(df2, bout_marker='courting')

# df1['rel_vel_abs_raw'] = df1['rel_vel_abs'] * 5

m1 = df1.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()
m2 = df2.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()


fig, axn =pl.subplots(1, 2)
ax=axn[0]
sns.histplot(data=m1[m1['courting']==1], x='rel_vel_abs', ax=ax)
sns.histplot(data=m2[m2['courting']==1], x='rel_vel_mms', ax=ax)
ax=axn[1]
sns.histplot(data=m1[m1['courting']==1], x='targ_ang_size_deg', ax=ax)
sns.histplot(data=m2[m2['courting']==1], x='rel_ang_size_deg', ax=ax)
#%%

fig, axn = pl.subplots(1, 2)
plot_vars = {'targ_ang_size_deg': 'rel_ang_size_deg',
             'rel_vel_abs': 'rel_vel_mms'}


for ai, (v1, v2) in enumerate(plot_vars.items()):
    ax=axn[ai]
    sns.scatterplot(ax=ax, x=m1[m1['courting']==1][v1],
                    y=m2[m2['courting']==1][v2])
    ax.set_aspect(1)
    ax.set_xlabel('FlyTracker')
    ax.set_ylabel('DLC')
    ax.set_title(v1)


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

#def assign_jaaba_behaviors(plotdf, courtvar='courting'):
#    plotdf.loc[plotdf[courtvar]==0, 'behavior'] = 'disengaged'
#    plotdf.loc[plotdf[courtvar]>0, 'behavior'] = 'orienting'
#    plotdf.loc[plotdf['chasing']>0, 'behavior'] = 'chasing'
#    plotdf.loc[plotdf['singing']>0, 'behavior'] = 'singing'
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

def plot_2d_hist_courting_vs_not(plotdf, plot_behavs, behavior_palette, courtvar='courting',
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
    sns.histplot(plotdf[plotdf[courtvar]>0], ax=ax,
                x='targ_rel_pos_x', y='targ_rel_pos_y',
                hue='behavior', hue_order=plot_behavs,legend=False,
            palette=behavior_palette, common_norm=False, 
            bins=nbins, binwidth=binwidth)
    ax.set_title('courting')
    for ax in axn:
        ax.set_aspect(1)

    return fig

def plot_2d_hist_by_behavior_subplots(plotdf,
                        plot_behavs, behavior_palette, 
                        marker='*', markercolor='w', markersize=10,
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
        ax.plot(0, 0, marker=marker, color=markercolor, markersize=markersize)
    
        ax.set_title(behav)
    #pl.ylim([-100, 100])
    return fig



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
                    & (ftjaaba[court_var]>0)
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

#%%
importlib.reload(util)
#%% 2D Hists, color by BEHAVIOR TYPE
courtvar='courtship'

jaaba_thresh_dict = {'orienting': 20,
                     'chasing': 10,
                     'singing': 5}

binwidth=10
for acq in ftjaaba['acquisition'].unique():
    print(acq)
    # plot info for courting vs. all

    #% Plot 2D hists for all behavior classes
    plotdf = ftjaaba[(ftjaaba['acquisition']==acq) & (ftjaaba['id']==0)
                & (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                & (ftjaaba['targ_pos_theta']<=max_pos_theta)
                & (ftjaaba['dist_to_other']>=min_dist_to_other)] #[::50]
    plotdf = util.assign_jaaba_behaviors(plotdf, jaaba_thresh_dict=jaaba_thresh_dict, courtvar=courtvar)

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
    fig = plot_2d_hist_courting_vs_not(plotdf, plot_behavs, behavior_palette, 
                                       courtvar=courtvar, binwidth=binwidth) #nbins=25)
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
    fig = plot_2d_hist_by_behavior_subplots(plotdf, plot_behavs, behavior_palette, 
                                            binwidth=binwidth) #nbins=25)
    putil.label_figure(fig, acq)
    figname = 'rel-pos_by-courting-behavior_{}'.format(acq)
    pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)
    print(figdir, figname)


#%%
ftjaaba['targ_pos_theta_abs'] = np.abs(ftjaaba['targ_pos_theta'])
#ftjaaba['ang_vel_abs'] = np.abs(ftjaaba['ang_vel'])

min_pos_theta = np.deg2rad(-160)
max_pos_theta = np.deg2rad(160)

min_boutdur = 0.5
filtdf = ftjaaba[(ftjaaba['id']==0)
                & (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                & (ftjaaba['targ_pos_theta']<=max_pos_theta)
                & (ftjaaba['dist_to_other']>min_dist_to_other)
                & (ftjaaba['boutdur']>=min_boutdur)
                & (ftjaaba['good_frames']==1)
                ].copy()
#%
#meanbins = plotdf.groupby(['species', 'acquisition', 'boutnum', 'targ_rel_pos_x_binned', 'targ_rel_pos_y_binned']).mean().reset_index()    
# meanbins = plotdf.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()    

#%%

meanbins = util.assign_jaaba_behaviors(filtdf, courtvar=courtvar, jaaba_thresh_dict=jaaba_thresh_dict)
plot_behavs = ['disengaged', 'orienting', 'chasing', 'singing']
behavior_colors = [[0.3]*3, 'mediumaquamarine', 'aqua', 'violet']
behavior_palette = dict((b, c) for b, c in zip(plot_behavs[1:], behavior_colors[1:]))

for sp in meanbins['species'].unique():
    plotdf = meanbins[meanbins['species']==sp]
    fig = plot_2d_hist_by_behavior_subplots(plotdf, plot_behavs, 
                                    behavior_palette, binwidth=20, #nbins=20,
                                    discrete=False, marker='o', markercolor='w', markersize=2,
                                    ylim=[-800, 800], xlim=(-500,900))
    #fig.suptitle(sp)
    fig.text(0.05, 0.85, 
        '{}: Relative target pos. (all frames, min bout dur={:.12}s)'.format(sp, min_boutdur), fontsize=12)
    putil.label_figure(fig, figid)

    figname = 'rel-pos_frames_mindur-{}_by-courting-behavior_{}'.format(min_boutdur, sp)
    pl.savefig(os.path.join(figdir, figname+'.png'), dpi=300)

#%%
plotdf = filtdf[ (filtdf['behavior']=='chasing') & (filtdf['good_frames']==1)]
   
fig, ax =pl.subplots()
sns.regplot(data=plotdf, x='targ_pos_theta_abs', y='ang_vel_abs', ax=ax)

# convert to polar from cartesian
rad, th = util.cart2pol(plotdf['dot_x_transformed'], plotdf['dot_y_transformed'])

pl.figure()
pl.scatter(plotdf['targ_rel_pos_x'], plotdf['dot_x_transformed'])

pl.figure()

pl.scatter(plotdf['targ_pos_theta'], th)

pl.figure()
targ_ang_vel = np.diff(th) / plotdf['sec_diff'].mean()
pl.scatter(targ_ang_vel, plotdf['abs_rel_ang_vel'].iloc[1:])




#%% plot centered and rotated with orientation vectors
import cv2
import importlib
import relative_metrics as rem
#%%
importlib.reload(rem)
assay = '2d-projector'

cap = util.get_video_cap_check_multidir(acq, assay=assay)

# get frame info
n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
print(frame_width, frame_height) # array columns x array rows

#%%
acq = '20240222-1145_fly3_Dyak_sP1-ChR_2do_sh_6x6'
filtdf = df[(df['targ_pos_theta']>=min_pos_theta) 
                & (df['targ_pos_theta']<=max_pos_theta)
                & (df['dist_to_other']>min_dist_to_other)
                #& (ftjaaba['good_frames']==1)
                & (df['acquisition']==acq)
                & (df['targ_pos_theta']<-0.7)
].copy()


ix = int(filtdf.iloc[1]['frame'])

fly1 = filtdf[filtdf['id']==0].copy().reset_index(drop=True)
fly2 = filtdf[filtdf['id']==1].copy().reset_index(drop=True)


fig = rem.plot_frame_check_affines(ix, fly1, fly2, cap, frame_width, frame_height)

#% double-check rotation matrix
fig = rem.check_rotation_transform(ix, filtdf, cap) # frame_width, frame_height)





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
