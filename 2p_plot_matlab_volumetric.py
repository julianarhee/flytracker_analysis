#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 04 14:41:00 2024

Author: Juliana Rhee
Email:  juliana.rhee@gmail.com

This script imports a .mat file with all the 2p/virmen data processed in Matlab
and performs some analysis on it.
"""
#%%
import os
import glob
import numpy as np
import pandas as pd

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

import utils as util
import plotting as putil
import utils_2p as util2p

import mat73

#%%
plot_style='white'
putil.set_sns_style(plot_style, min_fontsize=6)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%%

rootdir = '/Volumes/juliana/2p-data'
#session = '20240531'
# acq = 'example-yak-P1-1'
#acqnum = 9
example = 'Dyak'
is_volumetric = True

if example == 'Dyak':
    if is_volumetric:
        session = '20250424'
        flynum = 4
        acqnum = 29
        no_trigger = False
    else:
        session = '20250418'
        flynum = 2
        acqnum = 6
        no_trigger = False

elif example == 'Dmel':
    session = '20250218'
    flynum = 3
    acqnum = 10
    no_trigger = False
#%%
processed_dir = os.path.join(rootdir, session, 'processed')
acq = os.path.splitext(os.path.split(glob.glob(os.path.join(rootdir, session, 
                'virmen', '*{:03d}.mat'.format(acqnum)))[0])[1])[0]
print(acq)

# % Output figures dir
if plot_style == 'white':
    figdir = os.path.join( processed_dir, 'figures', acq , 'white')
else:
    figdir = os.path.join( processed_dir, 'figures', acq)

if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

#%% Load behavior data

# Load BEHAVIOR data
#fname = 'virft_{}.csv'.format(acq)
#virmen_fpath = os.path.join(processed_dir, 'matlab-files', fname)
#assert os.path.exists(virmen_fpath)

#ft = pd.read_csv(virmen_fpath)
ft = util2p.virmen_to_df(session, acqnum, rootdir=rootdir)

#%% Load ROI extracted traces

processed_mats = sorted(glob.glob(os.path.join( processed_dir,
                        'matlab-files', '*{:03d}*_nrAligned.mat'.format(acqnum))))
for p in processed_mats:
    print(os.path.split(p)[-1])

#%%
# Get matlab CAIMAN results
#mat_fpath = [c for c in processed_mats if 'slice%03d'.format(slicenum) in c][0]

mats = {}
for mat_fpath in processed_mats:
    #%
    # Load mat
    tmp_mdata = util2p.load_caimin_matlab(mat_fpath)
    mat_key = os.path.splitext(os.path.split(mat_fpath)[1])[0]
    mats.update({mat_key: tmp_mdata})
#%x
# %%
min_dff = 0.25
max_diff = 5
#k = list(mats.keys())[1]
#mdata = mats[k]

#%%
tc_list = []
roi_list = []
rel_time = []
roi_count = 0
for currslice, (k, mdata) in enumerate(mats.items()):
    print(currslice, k)
    # get index of CoMs sorted by position
    coms = pd.DataFrame({'x': mdata['CoM'][:, 0],
                'y': mdata['CoM'][:, 1]},
                index=np.arange(roi_count, 
                                roi_count+len(mdata['CoM'])))
    coms['z'] = currslice
    all_rois = np.array(coms.index.tolist())
    #%
    #%
    # Filter by responses
    timecourse = mdata['mean_across_trials_lr'].copy()
    # subtract the min value from each ROI timecourse
    timecourse = timecourse - timecourse.min(axis=1, keepdims=True)
    
    tc = pd.DataFrame(data=timecourse.T,
                columns=coms.index, 
                index=range(timecourse.shape[1]))#mdata['rel_tstamps'])
    tc['slice'] = currslice
    #%
    max_vals = np.max(timecourse, axis=1)
    min_vals = np.min(timecourse, axis=1)
    diff_vals = max_vals - min_vals
    incl_ixs = np.where((max_diff > diff_vals)
                        & (diff_vals > min_dff))[0]
    incl_rois = all_rois[incl_ixs]

    # %
    print("There are {}/{} ROIs with dF/F > {}".format(len(incl_rois), len(max_vals), min_dff))
    roi_count += len(coms)

    tc_list.append(tc[incl_rois])
    roi_list.append(coms)
    rel_time.append(mdata['rel_tstamps'])

ndata = pd.concat(tc_list, axis=1)
roi_pos = pd.concat(roi_list, axis=0)


#%%

tstamps = np.array(rel_time)
mean_tstamps =tstamps.mean(axis=0)

ndata.index = mean_tstamps

# %%
cmap = plt.get_cmap('viridis_r')

sorted_by_pos = roi_pos.sort_values(by=['z', 'y', 'x']).index
incl_rois = [roi for roi in sorted_by_pos if roi in ndata.columns]
n_colors = len(incl_rois) 

color_list = cmap(np.linspace(0, 1, n_colors)) #len(rel_tstamps))) #n_argmaxs))
cdict = dict((v, c) for v, c in zip(incl_rois, color_list))
#colors = [cdict[roi_to_argmax[i]] for i in incl_rois]

#rel_tstamps = mdata['rel_tstamps']

#%%
# Plot waterfall plot of ROI timecourses, sorted by peak response time
neural_fps = 1/np.mean(np.diff((mdata['iTime'])))
nsec_legend = 1 
nframes_legend = nsec_legend #* neural_fps

fig, ax = plt.subplots(figsize=(3, 5), dpi=300)
offset=1
gap = 0.2
ax.set_aspect(10)
for roi in incl_rois[0::5]: #sorted_by_peak[incl_ixs]:
    ax.plot(ndata.index.tolist(), ndata[roi]+offset, 
            c=cdict[roi],
            alpha=1, lw=0.5)
    offset+=gap
    #plt.xlim([1325,1375])
ax.set_aspect(0.75)
#ax.set_xlim([ax.get_xlim()[0], 5]) # ndata.shape[0]]) #timecourse.shape[1]])
#ax.set_ylim([0, 1]) #ax.get_ylim()[1]])a
ax.set_xlim([ax.get_xlim()[0], 4]) #timecourse.shape[1]])
ax.set_ylim([0, ax.get_ylim()[1]])

# %
# Add legend
dff_legend = 1
ax.set_yticks([0, dff_legend])
ax.set_xticks([ax.get_xlim()[0], nframes_legend])
sns.despine(offset=0, trim=True)

ax.set_xticklabels(['1 sec', ''], rotation=0, fontsize=6,
                   ha='left', va='top')
ax.set_yticklabels(['{} dF/F'.format(dff_legend), ''], rotation=90, 
                   ha='right', va='bottom', fontsize=6)

putil.label_figure(fig, acq)

figname = 'waterfall_timecourses_by_pos'
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)), bbox_inches='tight')
plt.savefig(os.path.join(figdir, '{}.svg'.format(figname)), bbox_inches='tight')

# %%

fig, ax = plt.subplots() #plt.figure()

slice_colors = sns.color_palette('viridis', n_colors=len(mats))

for i, (k, mdata) in enumerate(mats.items()):
    # average time course
    mean_timecourse = mdata['neural_timecourse'].mean(axis=0)
    mean_timecourse = mean_timecourse - mean_timecourse.min()
    neural_tstamps = mdata['iTime']
    #print(mean_timecourse.shape)

    ax.plot(neural_tstamps, mean_timecourse, lw=1,
            c=slice_colors[i], label=k) # c=bg_color)
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)

ax.set_xlim([60, 100])
# %%
