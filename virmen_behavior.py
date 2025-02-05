#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: virmen_behavior.py
Created on Tue Jan 21 12:14:00 2025
Author: julianarhee
Email:  juliana.rhee@gmail.com
Description: This script is used to examine tethered behavior from FicTract and ViRMEn data
"""

#%%
import os
import glob

import numpy as np
import pandas as pd

import pylab as pl
import seaborn as sns

import utils as util
import virmen_fictrac_to_df as vfdf

#%%

rootdir = '/Volumes/Extreme Pro/courtship-tethered/visual'
session = '20250120'
srcdir = os.path.join(rootdir, session)

filenum = 8

found_fpaths = glob.glob(os.path.join(srcdir, \
                        '{}_*{:03d}*.mat'.format(session, filenum)))

# %%
import scipy
exper_fpath = [i for i in found_fpaths if i.endswith('exper.mat')][0]
storevars_fpath = [i for i in found_fpaths if not i.endswith('exper.mat')][0]

#mat = scipy.io.loadmat(mat_fpath, struct_as_record=False)

# if `_vr.mat`:
#vr.vel
#vr.arcAngle
#vr.angsize
#vr = mat['vr'][0, 0]
#vr.frametimes

# %% Load experiment metadata 

#exper_fpath = '/Volumes/Extreme Pro/courtship-tethered/visual/20250120/test.mat'
#exper = scipy.io.loadmat(exper_fpath, struct_as_record=False)
#exper['exper'][0,0]._fieldnames
#exper['exper'][0,0].__dict__['variables'][0,0].__dict__['sphere_radius'][0]

exper = vfdf.load_custom_mat(exper_fpath) #, struct_as_record=False, squeeze_me=True)

#%% Load stored FicTrac and ViRMEn variables

storevars = vfdf.virmen_to_df_from_fpath(storevars_fpath)

# %%

fig, axn = pl.subplots(1, 2)
sns.scatterplot(data=storevars, x='target_x', y='target_y', ax=axn[0])
sns.scatterplot(data=storevars, x='pos_x', y='pos_y', ax=axn[1])
for ax in axn:
    ax.set_aspect(1)
# %%
male_pos = storevars[['pos_x', 'pos_y']].values
male_heading = storevars['heading']
target_pos = storevars[['target_x', 'target_y']].values

toc = storevars['toc'].diff()

dist_to_target = np.sqrt(np.sum(target_pos**2, axis=1))

male_vel = np.sqrt( np.sum(np.diff(male_pos,axis=0)**2, axis=1)) / np.diff(storevars['toc']) / 3

#%%
def circ_dist2(x, y):
    '''
    % r = circ_dist(alpha, beta)
    %   All pairwise difference x_i-y_j around the circle computed efficiently.
    %
    %   Input:
    %     alpha       sample of linear random variable
    %     beta       sample of linear random variable
    %
    %   Output:
    %     r       matrix with pairwise differences
    %
    % References:
    %     Biostatistical Analysis, J. H. Zar, p. 651
    %
    % PHB 3/19/2009
    %
    % Circular Statistics Toolbox for Matlab

    % By Philipp Berens, 2009
    % berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html

    np.angle:  Calculates the phase angle of a complex number or 
               array of complex numbers, returning the angle in radians by default
               Uses the atan2 function internally to compute the angle between 
               the positive real axis and the complex number on the complex plane
               Returns in range (-pi, pi].
    ''' 

    #np.tile(A, (m, n)) like repmat(A, m, n) in MATLAB

    at = np.angle( np.tile( np.exp(1j*x), (1, np.size(y)) ) / np.tile( np.exp(1j*y), (np.size(x), 1) ) )

    if len(at) == 1:
        return float(at)
    else:
        return at #np.exp(1j*x) / np.exp(1j*y) )


def transform(deltaXY, heading):
    """
    Applies a 2D rotation transformation to a vector deltaXY based on the given heading angle.

    Parameters:
    deltaXY (array-like): A 2-element list or NumPy array representing the (x, y) coordinates.
    heading (float): The rotation angle in radians.

    Returns:
    np.ndarray: Transformed 2D coordinates as a NumPy array.
    """
    rotation_matrix = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading),  np.cos(heading)]
    ])
    return rotation_matrix @ np.array(deltaXY)

#%%

# Calculate male's VR position
delta_pos = np.diff(male_pos, axis=0)
male_pos_vr = np.empty((len(male_heading),2))
male_pos_vr[0, :] = np.array([0, 0])

for i, (prev_delta_pos, curr_head) in enumerate( zip(delta_pos[0:], male_heading[1:]) ):
    true_shift = transform(prev_delta_pos, curr_head)
    male_pos_vr[i+1, :] = male_pos_vr[i] + true_shift 

fig, ax =pl.subplots()
ax.plot(male_pos_vr[:, 0],  male_pos_vr[:, 1], lw=0.5)

#%%
# Calculate heading on each frame
hdiffs = []
for i, v in zip(male_heading.iloc[0:-1], male_heading.iloc[1:]):
    hd = circ_dist2(i, v) * -1 # match matlab
    hdiffs.append(hd)
heading_diff = np.array(hdiffs)

fig, ax = pl.subplots()
ax.hist( heading_diff , bins=100)

#%%
target_angle = np.arctan2(target_pos[:, 0], target_pos[:, 1])

# %%
fig, ax =pl.subplots(figsize=(10, 4))
ax.plot(storevars['toc'].iloc[2:5000+2], heading_diff[0:5000])
# %%
