#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sept 25 16:44:00 2024

Author: Juliana Rhee
Email:  juliana.rhee@gmail.com

This script is used to plot FlyTracker variables for free behavior.
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
#%
import matplotlib.gridspec as gridspec

#from relative_metrics import load_processed_data
import transform_data.relative_metrics as rel
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
#srcdir = '/Volumes/Juliana/free_behavior_analysis/MF/FlyTracker/38mm_dyad/processed'
srcdir = '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker/processed'

if plot_style == 'white':
    figdir = os.path.join(os.path.split(srcdir)[0], 'plot_flytracker_vars', 'white')
else:
    figdir = os.path.join(os.path.split(srcdir)[0], 'plot_flytracker_vars')

if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

# LOCAL savedir 
#localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/2d-projector/FlyTracker'
#localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/MF/38mm-dyad/FlyTracker'
localdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/free_behavior/38mm_dyad/MF/FlyTracker'
out_fpath_local = os.path.join(srcdir, 'processed.pkl')

out_fpath_local = os.path.join(localdir, 'processed.pkl')
print(out_fpath_local)
assert os.path.exists(out_fpath_local)

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

#print(df[['species', 'acquisition']].drop_duplicates().groupby('species').count())
print(df.groupby('species')['acquisition'].nunique())

# Take subset
curr_species = ['Dmel', 'Dyak']
df = df[df['species'].isin(curr_species)]
df.groupby('species')['acquisition'].nunique()

# %%

jaaba_fpath = glob.glob(os.path.join(localdir, 'free_behavior_*_jaaba.pkl'))[0]
print(jaaba_fpath)

jaaba = pd.read_pickle(jaaba_fpath)
jaaba.head()

#%%
figid = jaaba_fpath.split('.pkl')[0]
print(figid)

dataid = os.path.split(figid)[1]


# %%

#% Check missing
missing = [c for c in jaaba['acquisition'].unique() if c not in df['acquisition'].unique()]
print("Missing {} acquisitions in found jaaba.".format(len(missing)))

missing_jaaba = [c for c in df['acquisition'].unique() if c not in jaaba['acquisition'].unique()]
print("Missing {} acquisitions in found processed files.".format(len(missing_jaaba)))

#%%
# merge jaaba --------------------------------------------------------
print("Merging flydf and jaaba...")
ftjaaba = util.combine_jaaba_and_processed_df(df, jaaba)

print(ftjaaba.groupby('species')['acquisition'].nunique())

#%%

if jaaba['chasing'].max() == 1:
    jaaba_thresh_dict = {'orienting': 0, 
                        'chasing': 0,
                        'singing': 0}
else:
    jaaba_thresh_dict = {'orienting': 10,
                        'chasing': 10,
                        'singing': 5}
# binarize behavs
ftjaaba = util.binarize_behaviors(ftjaaba, jaaba_thresh_dict=jaaba_thresh_dict)

ftjaaba.loc[(ftjaaba['chasing_binary']==1) | (ftjaaba['singing_binary']==1) | (ftjaaba['orienting_binary']==1), 'courtship'] = 1   

#%%
# Compare velocity during singing and chasing vs. chasing only bouts

#plotd = ftjaaba[(ftjaaba['chasing_binary']==1)
#                         | (ftjaaba['singing_binary']==1)].copy()
plotd = ftjaaba[ftjaaba['courtship']==1].copy()
plotd.loc[plotd['chasing_binary']==1, 'behavior'] = 'chasing'
plotd.loc[plotd['singing_binary']==1, 'behavior'] = 'singing'

#%%
# Plot Dmel & Dyak, compare chasing vs. singing

cumhist=False
plot_type = 'cumhist' if cumhist else 'hist'
cmap = {'chasing': 'royalblue', 
        'singing': 'magenta'}
fig, axn = pl.subplots(1, 2, sharex=True, sharey=True)
sns.histplot(data=plotd[plotd['species']=='Dmel'], x='vel', ax=axn[0], 
             label='Dmel', hue='behavior', palette=cmap,
             common_norm=False, stat='probability', cumulative=cumhist,
             fill=not(cumhist), element='step')
axn[0].legend_.remove()
axn[0].set_title('Dmel')
axn[0].set_xlim([-5, 40])
if not cumhist:
    axn[0].set_ylim([0, 0.12])

sns.histplot(data=plotd[plotd['species']=='Dyak'], x='vel', ax=axn[1], 
             label='Dyak', hue='behavior', palette=cmap,
             common_norm=False, stat='probability', cumulative=cumhist,
             fill=not(cumhist), element='step')
axn[1].set_title('Dyak')
for ax in axn:
    ax.set_xlabel('velocity (mm/s)')

sns.move_legend(axn[1], bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
sns.despine(offset=4)

for ax in axn:
    ax.set_box_aspect(1)

putil.label_figure(fig, figid)

figname = 'compare_behaviors_vel_sing_vs_chase_{}__{}'.format(plot_type, dataid)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir, figname)
#%%

# Plot CHASING vs. SINGING by species
cumhist=True
plot_type = 'cumhist' if cumhist else 'hist'
species_palette = {'Dmel': 'lavender', 
                   'Dyak': 'mediumorchid'}
fig, axn = pl.subplots(1, 2, sharex=True, sharey=True)
sns.histplot(data=plotd[plotd['behavior']=='chasing'], x='vel', ax=axn[0], 
             hue='species', palette=species_palette,
             common_norm=False, stat='probability', cumulative=cumhist,
             fill=not(cumhist), element='step')
axn[0].legend_.remove()
axn[0].set_title('Chasing')
axn[0].set_xlim([-5, 40])
axn[0].set_xlabel('velocity (mm/s)')

if not cumhist:
    axn[0].set_ylim([0, 0.12])

sns.histplot(data=plotd[plotd['behavior']=='singing'], x='vel', ax=axn[1], 
             hue='species', palette=species_palette,
             common_norm=False, stat='probability', cumulative=cumhist,
             fill=not(cumhist), element='step')
axn[1].set_title('Singing')
axn[1].set_xlabel('velocity (mm/s)')

sns.move_legend(axn[1], bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
sns.despine(offset=4)

for ax in axn:
    ax.set_box_aspect(1)

putil.label_figure(fig, figid)

figname = 'compare_species_vel_sing_vs_chase_{}__{}'.format(plot_type, dataid)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir, figname)

# %%
#% Split into bouts of courtship
d_list = []
for acq, df_ in ftjaaba.groupby('acquisition'):
    df_ = df_.reset_index(drop=True)
    df_ = util.mat_split_courtship_bouts(df_, bout_marker='courtship')
    dur_ = util.get_bout_durs(df_, bout_varname='boutnum', return_as_df=True,
                    timevar='sec')
    d_list.append(df_.merge(dur_, on=['boutnum']))
ftjaaba = pd.concat(d_list)

# Subdivide into mini bouts
subbout_dur = 0.20
ftjaaba = util.subdivide_into_subbouts(ftjaaba, bout_dur=subbout_dur)

#%%

if 'fpath' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['fpath'])
if 'filename' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['filename'])
if 'name' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['name'])
if 'strain' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['strain'])

#%%
bin_size = 5
max_dist = 30 #np.ceil(ftjaaba['dist_to_other'].max())
dist_bins = np.arange(0, max_dist+bin_size, bin_size)

# Cut dist_to_other into bins and assign label to new columns:
ftjaaba['binned_dist_to_other'] = pd.cut(ftjaaba['dist_to_other'], 
                                    bins=dist_bins, 
                                    labels=dist_bins[:-1])   
ftjaaba['binned_dist_to_other'] = ftjaaba['binned_dist_to_other'].astype(float)
#n_bins=5
#max_vel = ftjaaba['female_velocity'].max()
#vel_bins = np.linspace(0, max_vel, n_bins)
#ftjaaba['binned_female_velocity'] =pd.cut(ftjaaba['female_velocity'], 
#                                       bins=vel_bins, labels=vel_bins[:-1])   

#%%
#plotd = ftjaaba[(ftjaaba['chasing_binary']==1)
#                         | (ftjaaba['singing_binary']==1)].copy()
courting = ftjaaba[ftjaaba['courtship']==1].copy()
#courting.loc[courting['chasing_binary']==1, 'behavior'] = 'chasing'
#courting.loc[courting['singing_binary']==1, 'behavior'] = 'singing'
#%%
# average over subbout

bout_type = 'frames' #'subboutnum'
if bout_type == 'subboutnum':
    meanbouts_courting = courting.groupby(['species', 'acquisition', #'behavior', 
                        'binned_dist_to_other',
                         bout_type]).mean().reset_index()
    bout_type = 'subboutnum-{}'.format(subbout_dur)

else:
    meanbouts_courting = courting.groupby(['species', 'acquisition', #'behavior', 
                        'binned_dist_to_other']).mean().reset_index()
    bout_type = 'frames'
meanbouts_courting.head()


#%%
# Bin dist_to_other during chasing and singing
species_palette = {'Dmel': 'lavender', 
                   'Dyak': 'mediumorchid'}
error_type = 'ci'

#max_dist = np.ceil(meanbouts_courting['dist_to_other'].max())
#bin_size=5
#bins = np.arange(0, max_dist+bin_size, bin_size)
#meanbouts_courting['binned_dist_to_other'] = pd.cut(meanbouts_courting['dist_to_other'], 
#                                       bins=bins, labels=bins[:-1])  

fig, axn = pl.subplots(1, 2, sharex=True, sharey=False)
sns.barplot(data=meanbouts_courting,
             x='binned_dist_to_other', 
             y='chasing_binary', ax=axn[0], 
             errorbar=error_type, errcolor=bg_color,
             hue='species', palette=species_palette, 
             edgecolor='none')
axn[0].legend_.remove()
axn[0].set_ylabel("p(chasing|courtship)")

sns.barplot(data=meanbouts_courting,
             x='binned_dist_to_other', 
             y='singing_binary', ax=axn[1], 
             errorbar=error_type, errcolor=bg_color,
             hue='species', palette=species_palette, 
             edgecolor='none')
for ax in axn:
    ax.set_box_aspect(1)
    ax.set_xlabel('distance to other (mm)')
# format xticks to single digit numbers:
bin_edges = [str(int(x)) for x in dist_bins[:-1]]
bin_edges[0] = '<{}'.format(bin_size)
bin_edges[-1] = '>{}'.format(int(dist_bins[-2]))
axn[0].set_xticks(range(len(bin_edges)))
axn[0].set_xticklabels([str(x) for x in bin_edges], rotation=0)

axn[1].set_ylabel("p(singing|courtship)")

for ax in axn:
    ax.set_ylim([0, 1])
    
sns.move_legend(axn[1], bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
sns.despine(offset=4)

if bout_type == 'subboutnum':
    fig.text(0.1, 0.8, 'Behaviors split by consecutive courtship bouts ({})'.format(bout_type))
else:
    fig.text(0.1, 0.8, 'Averaging {} by bin across pairs'.format(bout_type))
fig.text(0.1, 0.85, 'Dmel (n={}), Dyak (n={})'.format(\
                    plotd[plotd['species']=='Dmel']['acquisition'].nunique(),
                    plotd[plotd['species']=='Dyak']['acquisition'].nunique()))
pl.subplots_adjust(wspace=0.6, right=0.9)

putil.label_figure(fig, figid)

figname = 'dist_to_other_frac-{}_p(sing_vs_chase_if_courting)_hist__{}'.format(bout_type, dataid)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir, figname)

#%%

# Plot EACH pair
#plotd = meanbouts_courting[meanbouts_courting['species']=='Dyak'].copy()
for sp, plotd in meanbouts_courting.groupby('species'):
    if sp=='Dyak':
        nr=2; nc=5;
    else:
        nr=3; nc=5;
    fig, axn = pl.subplots(nr, nc, sharex=True, sharey=True, 
                        figsize=(nc*2, nr*2))
    for ai, (acq, d_) in enumerate(plotd.groupby('acquisition')):
        ax=axn.flat[ai]
        sns.barplot(data=d_,
                    x='binned_dist_to_other', 
                    y='singing_binary', ax=ax, 
                    errorbar=error_type, errcolor=bg_color,
                    hue='species', palette=species_palette, 
                    edgecolor='none')
        ax.legend_.remove() 
        ax.set_title(acq, fontsize=4, loc='left')
        ax.set_box_aspect(1)
        ax.set_xlabel('')
    pl.subplots_adjust(bottom=0.2, top=0.9)
    fig.text(0.5, 0.1, 'binned distance to other (mm)', ha='center')
    fig.text(0.1, 0.93, 'Dyak: p(singing|courtship) vs. distance to other (mm)')
    # format xticks to single digit numbers:
    bin_edges = [str(int(x)) for x in dist_bins[:-1]]
    bin_edges[0] = '<{}'.format(bin_size)
    bin_edges[-1] = '>{}'.format(int(dist_bins[-2]))
    ax.set_xticks(range(len(bin_edges)))
    ax.set_xticklabels([str(x) for x in bin_edges], rotation=0)
    #ax.set_ylabel("p(singing|courtship)")

    putil.label_figure(fig, figid)
    figname = 'p(singing)_binned_dist_to_other_{}'.format(sp)
    pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

    #%%


# %%

# Split into SUBBOUTS
bout_type = 'boutnum'
ftjaaba.loc[ftjaaba['chasing_binary']==1, 'behavior'] = 'chasing'
ftjaaba.loc[ftjaaba['singing_binary']==1, 'behavior'] = 'singing'

meanbouts = ftjaaba.groupby(['species', 'acquisition',
                'subboutnum'])[['courtship', 'orienting_binary', 'chasing_binary', 'singing_binary']].mean().reset_index()
meanbouts.head()
#%%
# Overall proportion of time spent singing
#%
err_type='se'
fig, axn = pl.subplots(1, 3, sharex=True, sharey=True)
ax=axn[0]

#courting_or_not = ftjaaba.groupby(['species', 'acquisition'])[['courtship', 'orienting_binary', 'chasing_binary', 'singing_binary']].mean().reset_index()
courting_or_not = meanbouts.groupby(['species', \
                        'acquisition'])[['courtship', 'orienting_binary', 'chasing_binary', 'singing_binary']].mean().reset_index()   
sns.barplot(data=courting_or_not, x='species', 
            y='courtship', ax=ax, errorbar=err_type,
            palette=species_palette, edgecolor=bg_color, 
            errcolor=bg_color)

means_by_pair = meanbouts[meanbouts['courtship']==1]\
                .groupby(['species', 'acquisition'])[['chasing_binary', \
                        'singing_binary']].mean().reset_index()
ax=axn[1]
sns.barplot(data=means_by_pair, x='species', 
            y='chasing_binary', ax=ax, errorbar=err_type,
            palette=species_palette, edgecolor=bg_color, errcolor=bg_color)
ax.set_ylabel('p(chasing|courting)')

ax=axn[2]
sns.barplot(data=means_by_pair, x='species', 
            y='singing_binary', ax=ax, errorbar=err_type,
            palette=species_palette, edgecolor=bg_color, errcolor=bg_color)
ax.set_ylabel('p(singing|courting)')

pl.ylim([0, 1])
pl.subplots_adjust(wspace=0.5)
sns.despine(offset=4, bottom=True)
pl.xlabel('')

n_per_species = means_by_pair.groupby('species')['acquisition'].count()
fig.text(0.1, 0.9, 'Frac. of time spent in each behavior (subbout={}s, mean+/-sem)\n Dyak (n={}), Dmel (n={})'.format(subbout_dur, n_per_species['Dyak'], n_per_species['Dmel']))

putil.label_figure(fig, figid)
figname = 'frac_each_behavior_subbout-{}s__{}'.format(subbout_dur, dataid)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir)

#%%
# 
# Average ALL FRAMES together?
means_by_pair0 = ftjaaba[ftjaaba['courtship']>0].groupby(['species', 
                    'acquisition'])[['courtship', 'orienting_binary', 'chasing_binary', 'singing_binary']].mean().reset_index()

fig, axn = pl.subplots(1, 3, sharex=True, sharey=True)
ax=axn[0]
frac_courtship = ftjaaba.groupby(['species', 'acquisition'])[['courtship']].mean().reset_index()
sns.barplot(data=frac_courtship, x='species', y='courtship', ax=ax,
            palette=species_palette, edgecolor=bg_color, errorbar='se')
ax.set_ylabel('p(courting)')
ax=axn[1]
sns.barplot(data=means_by_pair0, x='species', 
            y='chasing_binary', ax=ax, errorbar='se',
            palette=species_palette, edgecolor=bg_color)
ax.set_ylabel('p(chasing|courting)')
ax=axn[2]
sns.barplot(data=means_by_pair0, x='species', 
            y='singing_binary', ax=ax, errorbar='se',
            palette=species_palette, edgecolor=bg_color)
ax.set_ylabel('p(singing|courting)')

pl.subplots_adjust(wspace=0.5)
sns.despine(offset=4, bottom=True)
for ax in fig.axes:
    ax.set_xlabel('')

n_per_species = frac_courtship.groupby('species')['acquisition'].count()
fig.text(0.1, 0.9, 'Fraction of time spent in each behavior (Dyak n={}, Dmel n={})'.format(n_per_species['Dyak'], n_per_species['Dmel']))

putil.label_figure(fig, figid)
figname = 'frac_each_behavior_averge-frames__{}'.format(dataid)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir)

#%%
# Compare FEMALE VELOCITY
# NOTE: >40 mm/s is likely an error or fly flying around the chamber

# %%
n_bins=8
max_vel = ftjaaba['female_velocity'].max()
vel_bins = np.linspace(0, max_vel, n_bins)

filt_ftjaaba = ftjaaba[ftjaaba['female_velocity']<=40].copy()

filt_ftjaaba['binned_female_velocity'] =pd.cut(filt_ftjaaba['female_velocity'], n_bins)
                                       #bins=n_bins, labels=vel_bins[:-1])   

filt_ftjaaba['binned_female_velocity_labels'] = filt_ftjaaba['binned_female_velocity'].apply(lambda x: x.left)

#%
#plotd = ftjaaba[(ftjaaba['chasing_binary']==1)
#                         | (ftjaaba['singing_binary']==1)].copy()
courting = filt_ftjaaba[filt_ftjaaba['courtship']==1].copy()
courting.loc[courting['chasing_binary']==1, 'behavior'] = 'chasing'
courting.loc[courting['singing_binary']==1, 'behavior'] = 'singing'

plot_vars = ['courtship', 'chasing_binary', 'singing_binary', 'female_velocity']
means_ = filt_ftjaaba.groupby(['species', 'acquisition', 
                           'subboutnum'
                           ])[plot_vars].mean().reset_index()

#%%
fig, axn = pl.subplots(1, 3)

ax=axn[0]
sns.histplot(data=means_[means_['courtship']==0], ax=ax,
             x='female_velocity',
             hue='species', palette=species_palette,
             stat='probability', common_norm=False,
             cumulative=False, element='step',fill=False)
ax=axn[1]
sns.histplot(data=means_[means_['chasing_binary']>0], ax=ax,
             x='female_velocity',
             hue='species', palette=species_palette,
             stat='probability', common_norm=False,
             cumulative=False, element='step',fill=False)
ax=axn[2]
sns.histplot(data=means_[means_['singing_binary']>0], ax=ax,
             x='female_velocity',
             hue='species', palette=species_palette,
             stat='probability', common_norm=False,
             cumulative=False, element='step',fill=False)
for ax in axn:
    ax.set_xlim([0, 50])
    ax.legend_.remove()
    ax.set_box_aspect(1)
pl.subplots_adjust(wspace=0.5)

#%%
#%
# Bin female_velocity during chasing and singing
xvar = 'female_velocity'
means_ = courting.groupby(['species', 'acquisition', 
                           'binned_{}'.format(xvar),
                           ])[plot_vars].mean().reset_index()
means_.head()

#%%
species_palette = {'Dmel': 'lavender', 
                   'Dyak': 'mediumorchid'}
error_type = 'se'

xvar = 'female_velocity'
max_dist = np.ceil(meanbouts_courting['dist_to_other'].max())
#bin_size=3
#bins = np.arange(0, max_dist+bin_size, bin_size)
#meanbouts_courting['binned_{}'.format(xvar)] = pd.cut(meanbouts_courting[xvar], 
#                                       bins=bins, labels=bins[:-1])  
fig, axn = pl.subplots(1, 2, sharex=True, sharey=True)
sns.pointplot(data=means_,
             x='binned_{}'.format(xvar), 
             y='chasing_binary', ax=axn[0], 
             errorbar=error_type, #errcolor=bg_color,
             hue='species', palette=species_palette) #, edgecolor=bg_color)
axn[0].legend_.remove()
axn[0].set_ylabel("p(chasing|courtship)")

sns.pointplot(data=means_, #meanbouts_courting,
             x='binned_{}'.format(xvar), 
             y='singing_binary', ax=axn[1], 
             errorbar=error_type, #errcolor=bg_color,
             hue='species', palette=species_palette) #, edgecolor=bg_color)
axn[1].set_ylabel("p(singing|courtship)")

for ax in axn:
    ax.set_box_aspect(1)
    ax.set_xlabel('female velocity (mm/s)')
# format xticks to single digit numbers:
#bin_edges = [str(int(x)) for x in vel_bins[:-1]]
bin_edges = sorted(filt_ftjaaba['binned_female_velocity_labels'].unique())
bin_labels = [str(round(x)) if i%2==0 or i==len(bin_edges)-1 else '' for i, x in enumerate(bin_edges)]

axn[0].set_xticklabels(bin_labels, rotation=0)

sns.move_legend(axn[1], bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
sns.despine(offset=4)

fig.text(0.1, 0.8, 'Behaviors split by consecutive courtship bouts ({})'.format(bout_type))
fig.text(0.1, 0.85, 'Dmel (n={}), Dyak (n={})'.format(\
                    plotd[plotd['species']=='Dmel']['acquisition'].nunique(),
                    plotd[plotd['species']=='Dyak']['acquisition'].nunique()))

putil.label_figure(fig, figid)

figname = '{}_frac-{}_sing_vs_chase_hist__{}'.format(xvar, bout_type, dataid)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir, figname)


# %%
