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
import pylab as pl

import utils as util
import plotting as putil
import utils_2p as util2p

import mat73

#%%
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=16)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%%

rootdir = '/Volumes/juliana/2p-data'
session = '20240531'
# acq = 'example-yak-P1-1'
acqnum = 9

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

processed_mats = glob.glob(os.path.join( processed_dir,
                        'matlab-files', '*{:03d}.mat'.format(acqnum)))
# Get matlab CAIMAN results
mat_fpath = processed_mats[0]
#%
# Load mat
mdata = util2p.load_caimin_matlab(mat_fpath)
#%x
# mat = mat73.loadmat(mat_fpath)
# mdata = mat['plotdata']

#%%
# Sort ROIs by peak response time

roi_to_argmax0 = dict(mdata['roi_by_argmax'])
roi_to_argmax = {int(k)-1: int(v) for k, v in roi_to_argmax0.items()}

# Filter by responses
min_dff = 0.2
timecourse = mdata['mean_across_trials_lr']
max_vals = np.max(timecourse, axis=1)
incl_ixs = np.where(max_vals > min_dff)[0]

sorted_by_peak = np.array([int(i)-1 for i in mdata['roi_sorted_by_response_time']])
incl_rois = [r for r in sorted_by_peak if r in incl_ixs]

print("There are {}/{} ROIs with dF/F > {}".format(len(incl_rois), len(max_vals), min_dff))
# %%a

# Overlay ROI positions and color by their peak resppnse time

CoM = mdata['CoM']
N = int(mdata['n_rois'])
# Create list of colors from colormap jet
cmap = pl.get_cmap('jet')

unique_argmax = np.unique([roi_to_argmax[roi] for roi in incl_rois]) #np.unique([v for k, v in roi_to_argmax.items()])
n_argmaxs = len(unique_argmax) #[v for k, v in roi_to_argmax.items()]).size

rel_tstamps = mdata['rel_tstamps']
color_list = cmap(np.linspace(0, 1, n_argmaxs)) #len(rel_tstamps))) #n_argmaxs))
cdict = dict((v, c) for v, c in zip(unique_argmax, color_list))
colors = [cdict[roi_to_argmax[i]] for i in incl_rois]

# Make continuous colorbar
norm = pl.Normalize(0, n_argmaxs) #min(rel_tstamps), max(rel_tstamps)) #min(unique_argmax), max(unique_argmax))
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
cbar = pl.colorbar(sm, label='time (s)', shrink=0.5) #im, ax=ax, cmap=clist, norm=norm)

putil.label_figure(fig, acq)

figname = 'overlay_ROIs_by_peak_response_time'
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)), bbox_inches='tight')
# %%

# Plot waterfall plot of ROI timecourses, sorted by peak response time
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
putil.label_figure(fig, acq)

figname = 'waterfall_timecourses_by_peak_response_time'
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)), bbox_inches='tight')

#%%
# average time course
mean_timecourse = mdata['neural_timecourse'].mean(axis=0)
sum_timecourse = mdata['neural_timecourse'].sum(axis=0)

neural_tstamps = mdata['iTime']
print(mean_timecourse.shape)

pl.figure()
pl.plot(neural_tstamps, mean_timecourse, lw=1, c=bg_color)
#pl.plot(neural_tstamps, sum_timecourse, lw=1, c=bg_color)

#%%

# 2p meta info
xml_fname = glob.glob(os.path.join(rootdir, session, 'raw', 
                        '*{:03d}'.format(acqnum),
                       '*-{:03d}.xml'.format(acqnum)))[0]
# frame trigger info
voltage_fname = glob.glob(os.path.join(rootdir, session, 
                        'raw', '*{:03d}'.format(acqnum),
                       '*{:03d}_*VoltageRecording_001.csv'.format(acqnum)))[0]

# Get absolute frame times for 2p images
absFrameTimes = util2p.get_timestamps_from_xml(xml_fname, verbose=True)

# Find pulses sent in the experiment data
pulseRCD = util2p.get_pulse_frames(voltage_fname)
pulseSent = ft[ft['pulse_sent']==1]['toc'] #expr[id_ML, 0]
mean_offset = util2p.get_pulse_offset(pulseRCD, pulseSent)

# Align behavior and imaging
iTime = np.array(absFrameTimes) + mean_offset

#%%
# 2p roi trace (CSV)
try:
    tc_acquisition  = glob.glob(os.path.join(rootdir,
                        session, 'processed', 
                        '*{:03d}*.csv'.format(acqnum)))[0]

    # Load 2p timecourse data
    tc = pd.read_csv(tc_acquisition)['Mean'].values
    #
    # Ensure the same length vectors
    len_tc = min(len(tc), len(iTime))
    tc = tc[:len_tc]
    iTime = iTime[:len_tc]

    pl.figure()
    pl.plot(iTime, tc, lw=1, c=bg_color)
except IndexError:
    pass

# %%

fig, ax = pl.subplots()
sns.scatterplot(data=ft, x='pos_x', y='pos_y', ax=ax,
                hue='animal_movement_speed_lab', palette='viridis', 
                edgecolor=None, s=1)
ax.legend_.remove()
#ax.plot(ft['pos_x'], ft['pos_y'], lw=1, c=bg_color)
ax.set_aspect(1)


#%%
#behav_time = [0]
#behav_time.extend(ft['toc'])
behav_time = np.array(ft['toc']) #behav_time)

# Distance between male and female
ft['distance'] = np.sqrt(np.sum((
                        ft[['target_x', 'target_y']].values - ft[['pos_x', 'pos_y']].values)**2, axis=1))

# Male and female velocities
ft['vel_fly'] = (ft[['pos_x', 'pos_y']].diff()**2).sum(axis=1).apply(np.sqrt) * (1.0 / ft['toc'].diff()) * 3
ft['vel_dot'] = (ft[['target_x', 'target_y']].diff()**2).sum(axis=1).apply(np.sqrt)
#ft['vel_dot'] = np.sqrt(np.sum(np.diff(ft[['target_x', 'target_y']], axis=0)**2, axis=1))

# Angular location of female with respect to male
zCoors = ft[['target_x', 'target_y']].values - ft[['pos_x', 'pos_y']].values
ft['target_abs_ang'] = np.arctan2(zCoors[:, 1], zCoors[:, 0])

# Normalized heading with respect to female
ft['norm_heading'] = util.circ_dist(ft['heading'], util.wrap_to_2pi(ft['target_abs_ang'] - np.pi / 2))

# Position of female in male's field of view
ft['male_fov_female_pos_x'] = ft['distance'] * np.cos(ft['norm_heading'])
ft['male_fov_female_pos_y'] = ft['distance'] * np.sin(ft['norm_heading'])

# Compute male's trajectory in VR
def transform(deltaXY, heading):
    return np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]).dot(deltaXY)

delta_pos = ft[['pos_x', 'pos_y']].diff(axis=0).fillna(0)
male_vr_pos = np.zeros((len(ft), 2))

for i in range(1, len(ft)):
    true_shift = transform(delta_pos.iloc[i - 1, :], ft['heading'].iloc[i])
    male_vr_pos[i, :] = male_vr_pos[i - 1, :] + true_shift.T
ft['male_vr_pos_x'] = male_vr_pos[:, 0]
ft['male_vr_pos_y'] = male_vr_pos[:, 1]

# Compute angle of object
#angleOL = np.arctan2(ft['target_y'], ft['target_x']) #expr[:, 6], expr[:, 5])
ft['target_angle'] = np.arctan2(ft['target_x'], ft['target_y']) #female['pos'][:, 0], female['pos'][:, 1])
# shoudl be arctan2(x, y) for stimulus to match in heatmap

# Determine phase length
ft['target_angle'] = np.round(ft['target_angle'], 5)
first_motion = np.where(np.diff(ft['target_angle']) != 0)[0][0]
phase_length = np.where(ft['target_angle'][first_motion:] == ft['target_angle'][first_motion])[0]

if phase_length[2] - phase_length[1] == 1:
    phase_length = phase_length[1] * 3
else:
    phase_length = phase_length[1]*3 #(phase_length[1] - 1) * 3

# Find number of full phases completed
n_pre_phases = np.floor(first_motion / phase_length).astype(int)
n_post_phases = np.floor((len(ft) - first_motion) / phase_length).astype(int)
total_phases = n_pre_phases + n_post_phases
firstID = first_motion - n_pre_phases * phase_length
lastID = first_motion + n_post_phases * phase_length - 1

# Change in heading
heading_diff = np.zeros(len(ft) - 1)
for i in range(len(ft) - 1):
    heading_diff[i] = util.circ_dist(ft['heading'][i], ft['heading'][i + 1])
#%
# Reshape phase data
mean_phase = np.reshape(heading_diff[firstID:lastID + 1], (phase_length, total_phases), order='F').T
stim_phase = np.reshape(ft['target_angle'][firstID:lastID + 1], (phase_length, total_phases), order='F').T
time_phase = np.reshape(behav_time[firstID:lastID + 1], (phase_length, total_phases), order='F').T

# Tracking index
R, vigor, fidelity = util2p.compute_ti(heading_diff, ft['target_angle'], behav_time)

#%% add vars to FT
heading_diff0 = [0]
heading_diff0.extend(heading_diff)
ft['heading_diff'] = heading_diff0
ft['ang_vel_fly'] = heading_diff0 * (1.0 / ft['toc'].diff()) * 3

ti0 = [None]
ti0.extend(R)
ft['tracking_index'] = ti0

#%%
fig, axn =pl.subplots(1, 2, sharex=True, sharey=True)
ax=axn[0]
ax.imshow(mean_phase, cmap='viridis', vmin=-0.1, vmax=0.1,
          interpolation='none')
ax.set_ylabel('trial number')
ax.set_xlabel('time (frame)')

ax=axn[1]
ax.imshow(stim_phase, cmap='viridis', vmin=-0.1, vmax=0.1,
          interpolation='none')
for ax in axn:
    ax.set_aspect(40)

putil.label_figure(fig, acq)
figname = 'mean_phase_v_stim_phase'
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)), bbox_inches='tight')

#%%
pl.figure()
pl.plot(R)

#util2p.compute_ti()

# %%

# Plot SUMMARY of timecourse data
zoom_pre = 50 #25
zoom_post = 150 #125 #100 #150

# Create figure
fig, axn = pl.subplots(4, 1, sharex=True,
                       figsize=(12, 8))
# Target angle
ax = axn[0] #ax1.twinx()
ax.plot(behav_time[1:], ft['target_angle'][1:], color=bg_color)
ax.set_ylabel('Target angle', color=bg_color)

# Delta heading
ax=axn[1]
ax.plot(behav_time[1:], heading_diff, color=bg_color)
ax.set_ylim([-0.5, 0.5])
ax.set_ylabel(r'$\Delta$ heading', color=bg_color)
ax.set_xticks(np.linspace(0, behav_time[-1], 5))

# Tracking index
ax=axn[2]
ax.plot(behav_time[1:], R, color=bg_color)
ax.set_ylabel('Tracking index')

# Neural timecourse
ax=axn[3]
ax.plot(neural_tstamps, mean_timecourse, lw=1, c=bg_color)
ax.set_ylabel('dF/F')
#
pl.subplots_adjust(hspace=0.1)

sns.despine(offset=0, trim=True)

for ax in axn[0:3]:
    sns.despine(ax=ax, bottom=True)

# Set figure size (in pixels, converted to inches)
#fig.set_size_inches(1200 / 96, 300/96) #234 / 96)  # Assuming 96 DPI (dots per inch)

#%
#%# Save the figure
putil.label_figure(fig, acq)

figname = 'summary_timecourse' #delta-heading_v_stimulus'
fig.savefig(os.path.join(figdir, f'{figname}.png')) #, bbox_inches='tight')

# Save a zoomed-in version
ax.set_xlim([zoom_pre, zoom_post])
figname_zoom = 'summary_timecourse_ZOOM' #'delta-heading_v_stimulus_ZOOM'
fig.savefig(os.path.join(figdir, f'{figname_zoom}.png')) #, bbox_inches='tight')

# Optionally show the plot (comment out if not needed)
# plt.show()
# %%

#import numpy as np
#import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
#import os

#for k in range(1): #mean_timecourse.shape[0]):
# Interpolate data
interp_func = interp1d(neural_tstamps, mean_timecourse, fill_value="extrapolate")
yq2 = interp_func(behav_time[1:])

# Compute baseline
idx = np.where((neural_tstamps < behav_time[first_motion]) & (neural_tstamps > (behav_time[first_motion] - 10)))
bl = np.nanmean(yq2[idx])
yq3 = (yq2 - bl) / bl

# Compute phase information
n_cycles = 3
ft['target_angle'] = np.round(ft['target_angle'], 2)
one_phase = phase_length / n_cycles

close_to_zero = np.argmin(abs(ft['target_angle'] - 0))
min_zero_val = ft['target_angle'][close_to_zero]
pos1 = np.where(ft['target_angle'] == min_zero_val)[0] + int(one_phase * 0.5)
trial_length = int(one_phase * n_cycles)
trial_starts = pos1[1::2]
trial_starts = trial_starts[0::n_cycles]

heading_diff2 = uniform_filter1d(heading_diff, size=10)

trial_tcourse, trial_headingdiff, trial_angle, trial_TI, trial_vel, trial_time = [], [], [], [], [], []

for start in trial_starts:
    try:
        if yq3[start:start + trial_length].shape[0] < trial_length:
            continue
        trial_tcourse.append(yq3[start:start + trial_length])
        trial_headingdiff.append(heading_diff2[start:start + trial_length])
        trial_angle.append(ft['target_angle'][start:start + trial_length])
        trial_TI.append(R[start:start + trial_length])
        trial_vel.append(ft['vel_fly'][start:start + trial_length])
        trial_time.append(behav_time[start:start + trial_length])
    except IndexError:
        pass

# Normalize trials
trial_tcourse = np.array(trial_tcourse)
trial_tcourseN = (trial_tcourse - trial_tcourse[:, 0][:, None]) / trial_tcourse[:, 0][:, None]

# Compute metrics
rs = np.mean(trial_TI, axis=1)
vs = np.mean(trial_vel, axis=1)
as_ = np.mean(np.abs(trial_headingdiff), axis=1) / 0.021

crt = np.where(rs > 0.3)[0]
moving = np.setdiff1d(np.where((vs > 3) | (as_ > 2))[0], crt)

# Average values for plotting
mnTime = np.mean(np.array(trial_time), axis=0)
mnTime -= mnTime[0]

#%%
# PLOT -----------------------
#colmap = pl.get_cmap('jet')(np.linspace(0, 1, mean_timecourse.shape[1]))
bg_color = [0.8, 0.8, 0.8]
courting_color = [1, 0, 0.25]
locomotion_color = [0.4, 0.5, 0.7]

# Start plotting
fig, ax1 = pl.subplots()
#ax1.set_prop_cycle(color=[courting_color, bg_color])

# Plot courting time course
ax1.plot(mnTime, np.mean(trial_tcourse[crt], axis=0), 
         color=courting_color, linestyle='-', linewidth=2)
ax1.set_ylabel(r'$\Delta F/F$')

# Plot locomotion time course
ax1.plot(mnTime, np.mean(trial_tcourse[moving], axis=0),
          color=locomotion_color, linestyle='-', linewidth=2)

# Plot female angle on the right y-axis
ax2 = ax1.twinx()
mnAngle = np.mean(trial_angle, axis=0)
ax2.plot(mnTime, mnAngle, color=bg_color, linewidth=1)
ax2.set_ylabel('Target angle')
#ax2.legend_.remove()

# Annotations
ax1.axhline(0, color='w', linestyle=':', linewidth=1)
zero_crossings = np.where(mnAngle == 0)[0]
for v in zero_crossings:
    ax1.axvline(mnTime[v], color='w', linestyle=':', linewidth=1)

labels = ['courting', 'moving', 'target']
colors = [courting_color, locomotion_color, bg_color]
legh = putil.custom_legend(labels, colors, use_line=True, lw=4, 
                    markersize=10)
ax1.legend(handles=legh, bbox_to_anchor=(1.2,1), loc='upper left', frameon=False)

# Save figures
figname = 'avg-trials_dFF-v-pos'
#fig.savefig(os.path.join(savedir, f'{figname}.png'), bbox_inches='tight', facecolor='current')

ax1.set_xlabel('Time (s)')

putil.label_figure(fig, acq)

figname = 'averaged_neural_timecourse_{}-cycles'.format(n_cycles) #delta-heading_v_stimulus'
fig.savefig(os.path.join(figdir, f'{figname}.png')) #, bbox_inches='tight')

# %%
# Plot MEANPHASE and DF/F as heatamps

fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(12, 6))

# Plot for meanPhase (left subplot)
cax1 = ax1.imshow(mean_phase[n_pre_phases:, :], aspect='auto', 
                  cmap='viridis', vmin=-0.15, vmax=0.15,
                  interpolation='none')
ax1.set_ylabel('Iteration', fontsize=15, fontname='Avenir Book')
ax1.set_xlabel('Time (frame)', fontsize=15, fontname='Avenir Book')
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_title('Heading phase')
pl.colorbar(cax1, ax=ax1)

# Plot for trialTC (right subplot)
cax2 = ax2.imshow(trial_tcourse, aspect='auto', cmap='viridis', 
                  vmin=np.min(trial_tcourse), vmax=np.max(trial_tcourse),
                  interpolation='none')
ax2.set_ylabel('Iteration', fontsize=15, fontname='Avenir Book')
ax2.set_xlabel('Time (frame)', fontsize=15, fontname='Avenir Book')
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.set_title('dF/F')
pl.colorbar(cax2, ax=ax2)

# Link the y-axes of both subplots
ax1.get_shared_y_axes().join(ax1, ax2)

putil.label_figure(fig, acq)
# Save the figure
figname = 'meanphase-heading_trialTC'
fig.savefig(os.path.join(figdir, f'{figname}.png'), bbox_inches='tight')

# %%


#%%
# Plot POSITION in VR

fig, axn = pl.subplots(1, 2)
ax=axn[0]
sns.scatterplot(data=ft.iloc[first_motion:], ax=ax,
                x='pos_x', y='pos_y', 
                hue='toc', palette='viridis',
                edgecolor=None, s=2)
ax.legend_.remove()
ax.set_aspect(1)
# %
# Distance between male and female
#ft['distance'] = np.sqrt(np.sum((ft[['target_x', 'target_y']].values - ft[['pos_x', 'pos_y']].values)**2, axis=1))

#fig, ax = pl.subplots()
ax=axn[1]
sns.scatterplot(data=ft.iloc[first_motion:], ax=ax,
                x='male_vr_pos_x', y='male_vr_pos_y', 
                hue='toc', palette='viridis',
                edgecolor=None, s=2)
ax.set_aspect(1)
ax.legend_.remove()
# %%

# Find index of closest match in behav_time to first neural_tstamp
start_ix = abs(behav_time - neural_tstamps[0]).argmin()
print(behav_time[start_ix], neural_tstamps[0])
end_ix = abs(behav_time - neural_tstamps[-1]).argmin()
print(behav_time[end_ix], neural_tstamps[-1])

# Get interprolated dF/F
interp_func = interp1d(neural_tstamps, mean_timecourse, fill_value="extrapolate")
yq2 = interp_func(behav_time[start_ix:end_ix])

# Compute baseline
idx = np.where((neural_tstamps < behav_time[first_motion]) & (neural_tstamps > (behav_time[first_motion] - 10)))
bl = np.nanmean(yq2[idx])
yq3 = (yq2 - bl) / bl

# Add interpolated dff to FT
ft['dff'] = None
ixs = ft.iloc[start_ix:end_ix].index
ft.loc[ixs, 'dff'] = yq3

# PLOT --------------

ft['ang_vel_fly_abs'] = np.abs(ft['ang_vel_fly'])
fig, axn =pl.subplots(3, 1, sharex=True)
ax=axn[0]
ax.plot(ft['toc'].iloc[start_ix:end_ix], ft['ang_vel_fly_abs'].iloc[start_ix:end_ix])
ax.set_ylabel('ang_vel')
ax=axn[1]
#ax.plot(neural_tstamps, mean_timecourse)
ax.plot(ft['toc'].iloc[start_ix:end_ix], yq3)
ax.set_ylabel('dF/F')

ax=axn[2]
ax.plot(ft['toc'].iloc[start_ix:end_ix], ft['male_fov_female_pos_x'].iloc[start_ix:end_ix])


#ax.plot(ft['ang_vel_fly'])
#%%

plot_vars = ['ang_vel_fly_abs', 'vel_fly', 'target_angle']
fig, axn = pl.subplots(1, len(plot_vars), sharey=True)

for ax, plot_var in zip(axn, plot_vars):
    sns.scatterplot(data=ft.iloc[first_motion:], ax=ax,
                x=plot_var, y='dff',
                hue='tracking_index', palette='viridis',
                edgecolor=None, s=2)
    ax.legend_.remove()
    ax.set_box_aspect(1)
    ax.invert_yaxis()
    if plot_var in ['ang_vel_fly', 'ang_vel_fly_abs', 'vel_fly']:
        ax.set_xlim([0, 100])

# 
putil.label_figure(fig, acq)
figname = 'scatter_hue-TI_dff_v_angvel'
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)), bbox_inches='tight')

# %%

fig, ax = pl.subplots()
sns.scatterplot(data=ft, ax=ax,
               x='target_x', y='target_y',
               hue='target_angle')
ax.set_aspect(1)
ax.legend_.remove()
ax.invert_xaxis()
# %%

# dict of roi: peak response time index
# roi_to_argmax

# corresponding time bins
# mdata['rel_tstamps']
#%

# Histogram of peak response times
time_bins = mdata['rel_tstamps'] #values

fig, ax = pl.subplots()
ax.hist([time_bins[v-1] for k, v in roi_to_argmax.items()], bins=20)
#ax.set_xticks(np.linspace(0, len(mdata['rel_tstamps']), 5))
ax.get_xticks()

#switch_left = np.where(np.diff(targ_angle) > 0)[0]
#%%
#import numpy as np

def get_switch_regions(targ_angle):
    from scipy.ndimage import label, find_objects

    # Finding regions where the difference in angle is positive (left-to-right switch)
    switch_left_regions, _ = label(np.diff(targ_angle) > 0)
    switch_left_objects = find_objects(switch_left_regions)

    switch_lr_onsets = [sl[0].start + 1 for sl in switch_left_objects]  # Adjust for diff offset

    # Finding regions where the difference in angle is negative (right-to-left switch)
    switch_right_regions, _ = label(np.diff(targ_angle) < 0)
    switch_right_objects = find_objects(switch_right_regions)

    switch_rl_onsets = [sr[0].start + 1 for sr in switch_right_objects]  # Adjust for diff offset

    return switch_lr_onsets, switch_rl_onsets

# do arctan(y, x) to get range 0-3 
targ_angle = np.arctan2(ft['target_y'], ft['target_x']) #female['pos'][:, 0], female['pos'][:, 1])

switch_lr_onsets, switch_rl_onsets = get_switch_regions(targ_angle) 
#%%

#pl.figure()
#pl.scatter(stimLROnsets, mdata['stimLROnsets'])

pl.figure(figsize=(12,4))
pl.plot(targ_angle)
pl.plot(switch_lr_onsets, np.ones_like(switch_lr_onsets)*1.5, 'ro', markersize=1)
pl.plot(switch_rl_onsets, np.ones_like(switch_rl_onsets)*1.5, 'go', markersize=1)

# %%
nsec_pre = 0.0
nsec_post = 1.5
behav_fps = 1/np.mean(np.diff(behav_time))
print(behav_fps)
nframes_post = int(round(nsec_post * behav_fps))
nframes_pre = int(round(nsec_pre * behav_fps))

print(nframes_post)

ft['sweep_lr'] = None
for i, sweep_start_ix in enumerate(switch_lr_onsets):
    #sweep_start_t = ft.iloc[sweep_start_ix]['toc']
    #sweep_end_t = sweep_start_t + nsec_post
    sweep_end_ix = sweep_start_ix + nframes_post #np.argmin(abs(ft['toc']-sweep_end_t))
    if sweep_end_ix > len(ft):
        break
    curr_sweep = ft.iloc[sweep_start_ix-nframes_pre:sweep_end_ix]
    #print(curr_sweep.shape)
    ft.loc[curr_sweep.index, 'sweep_lr'] = i
    ft.loc[curr_sweep.index, 'sweep_time'] = curr_sweep['toc'] - curr_sweep['toc'].iloc[0]
#[k for k, v in roi_to_argmax.items() if v >=1 and v <= ]
# %%

rel_times=[]
ang_vels=[]
targ_angles=[]
for i, d in ft.groupby('sweep_lr'):
    if i is None:
        continue

    rel_times.append(d['sweep_time'].values)
    ang_vels.append(d['ang_vel_fly'].values)
    targ_angles.append(d['target_angle'].values)

ang_vels = np.array(ang_vels)
rel_times = np.array(rel_times)
targ_angles = np.array(targ_angles)

#%%
fig, axn =pl.subplots(2, 1, sharex=True)
ax=axn[0]
ax.plot(rel_times.mean(axis=0), targ_angles.mean(axis=0),
        lw=2, color='r')

ax=axn[1]
for i, (t, v) in enumerate(zip(rel_times, ang_vels)):
    ax.plot(t, v, lw=0.2, color=bg_color, alpha=0.5)

ax.plot(rel_times.mean(axis=0), ang_vels.mean(axis=0),
        lw=2, color='r')
ax.set_ylim([-50, 50])

putil.label_figure(fig, acq)
figname = 'target_angle_v_angvel'
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)), bbox_inches='tight')

# %%
fig, ax =pl.subplots()
for i, (t, v) in enumerate(zip(rel_times, targ_angles)):
    ax.plot(t, v, lw=0.2, color=bg_color, alpha=0.5)
# %%
