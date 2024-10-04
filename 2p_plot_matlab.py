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

import seaborn as sns
import pylab as pl

import utils as util
import plotting as putil
import mat73

#%%
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=18)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%%

rootdir = '/Volumes/juliana/2p-data'
session = '20240905'
# acq = 'example-yak-P1-1'
acqnum = 18
processed_mats = glob.glob(os.path.join(rootdir, session, 'processed', 
                                        'matlab-files', '*{:03d}.mat'.format(acqnum)))
processed_dir = os.path.join(rootdir, session, 'processed')

mat_fpath = processed_mats[0]
#print(mat_fpath)

acq = os.path.splitext(os.path.split(mat_fpath)[1])[0]
print(acq)

figdir = os.path.join(rootdir, session, 'processed', 'figures', acq)
if not os.path.exists(figdir):
    os.makedirs(figdir)

# %%
if plot_style == 'white':
    figdir = os.path.join( processed_dir, 'figures', acq , 'white')
else:
    figdir = os.path.join( processed_dir, 'figures', acq)

if not os.path.exists(figdir):
    os.makedirs(figdir)
print(figdir)

#%%
# Load mat
mat = mat73.loadmat(mat_fpath)
mdata = mat['plotdata']

for k, v in mdata.items():
    try:
        print(k, v.shape)
    except Exception as e:
        print("Issue with {}".format(k))

#%%

roi_to_argmax0 = dict(mdata['roi_by_argmax'])
roi_to_argmax = {int(k)-1: int(v) for k, v in roi_to_argmax0.items()}

# Filter by responses
min_dff = 0.
timecourse = mdata['mean_across_trials_lr']
max_vals = np.max(timecourse, axis=1)
incl_ixs = np.where(max_vals > min_dff)[0]

sorted_by_peak = np.array([int(i)-1 for i in mdata['roi_sorted_by_response_time']])
incl_rois = [r for r in sorted_by_peak if r in incl_ixs]

print("There are {}/{} ROIs with dF/F > {}".format(len(incl_rois), len(max_vals), min_dff))
# %%a
import matplotlib as mpl


CoM = mdata['CoM']
N = int(mdata['n_rois'])
# Create list of colors from colormap jet
cmap = pl.get_cmap('jet')

unique_argmax = np.unique([roi_to_argmax[roi] for roi in incl_rois]) #np.unique([v for k, v in roi_to_argmax.items()])
n_argmaxs = len(unique_argmax) #[v for k, v in roi_to_argmax.items()]).size

rel_tstamps = mdata['rel_tstamps']
color_list = cmap(np.linspace(0, 1, len(rel_tstamps))) #n_argmaxs))
cdict = dict((v, c) for v, c in zip(unique_argmax, color_list))
colors = [cdict[roi_to_argmax[i]] for i in incl_rois]

# Make continuous colorbar
norm = pl.Normalize(min(rel_tstamps), max(rel_tstamps)) #min(unique_argmax), max(unique_argmax))
clist = mpl.colors.LinearSegmentedColormap.from_list('jet', color_list)
sm = pl.cm.ScalarMappable(cmap=clist, norm=norm)

# PLOT
fig, ax =pl.subplots()
ax.imshow(mdata['Y_mean'], cmap='gray')
ax.set_aspect(1)

# Plot ROIs colored by peak response time
im = ax.scatter(CoM[incl_rois, 1], CoM[incl_rois, 0], s=10, c=colors)
ax.axis('off')

# Add colorbar for peak response time
cbar = pl.colorbar(sm, label='time (s)') #im, ax=ax, cmap=clist, norm=norm)

# %%


neural_fps = 1/np.mean(np.diff((mdata['iTime'])))
nsec_legend = 1 
nframes_legend = nsec_legend * neural_fps


fig, ax = pl.subplots()
offset=0
gap = 0.1
ax.set_aspect(10)
for roi in incl_rois[0::3]: #sorted_by_peak[incl_ixs]:
    ax.plot(timecourse[roi, :]+offset, c=cdict[roi_to_argmax[roi]],
            alpha=1, lw=1)
    offset+=gap
    #plt.xlim([1325,1375])
ax.set_xlim([ax.get_xlim()[0], timecourse.shape[1]])
ax.set_ylim([0, ax.get_ylim()[1]])

# Add legend
dff_legend = 0.5
ax.set_yticks([0, dff_legend])
ax.set_xticks([ax.get_xlim()[0], nframes_legend])
sns.despine(offset=0, trim=True)

ax.set_xticklabels(['1 sec', ''], rotation=0, fontsize=10,
                   ha='left', va='top')
ax.set_yticklabels(['{} dF/F'.format(dff_legend), ''], rotation=90, 
                   ha='right', va='bottom', fontsize=10)

#pl.figure()
#pl.hist(max_vals)

# %%

# Load BEHAVIOR data
fname = 'virft_{}.csv'.format(acq)
virmen_fpath = os.path.join(processed_dir, 'matlab-files', fname)
assert os.path.exists(virmen_fpath)

ft = pd.read_csv(virmen_fpath)

# %%

fig, ax = pl.subplots()
sns.scatterplot(data=ft, x='pos_x', y='pos_y', ax=ax,
                hue='animal_movement_speed_lab', palette='viridis', 
                edgecolor=None, s=1)
ax.legend_.remove()
#ax.plot(ft['pos_x'], ft['pos_y'], lw=1, c=bg_color)
ax.set_aspect(1)


# %%
