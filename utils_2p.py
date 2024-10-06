#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   2p_utils.py
@Time    :   2024/10/05 17:26:48
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com

'''

import numpy as np

def compute_ti(heading_diff, female_angle, behavior_time):
    # Parameters
    trial_length = 180
    hemi_length = int(trial_length / 2)
    trialStarts = range(len(behavior_time) - 1)
    
    vigor = []
    fidelity = []
    start_times = []

    # Compute tracking index
    for i in trialStarts:
        if i > hemi_length and i < (len(heading_diff) - hemi_length): #(len(behavior_time) - hemi_length):
            x = heading_diff[(i - hemi_length):(i + hemi_length + 1)]
            f = female_angle[(i - hemi_length):(i + hemi_length + 1)]
            t = behavior_time[(i - hemi_length):(i + hemi_length + 1)]
            start_times.append(behavior_time[i])
            mat = np.corrcoef(x, f)  # Correlation matrix between stim and male turning
            fid = mat[0, 1]  # Store correlation coefficient
        elif i <= hemi_length:  # At the very start of trial
            x = heading_diff[:i + 1]
            f = female_angle[:i + 1]
            t = behavior_time[:i + 1]
            start_times.append(behavior_time[i])
            fid = 0
        else:  # At the very end of trial
            x = heading_diff[i:]
            f = female_angle[i:len(behavior_time) - 1]
            t = behavior_time[i:len(behavior_time) - 1]
            start_times.append(behavior_time[i])
            fid = 0

        fidelity.append(fid)  # Correlation
        vigor.append(np.sum(x * np.sign(f)))  # Net turning in direction of target

    vigor = np.array(vigor)
    fidelity = np.array(fidelity)
    R = vigor / np.max(vigor)  # Normalize vigor across all periods
    R = R * fidelity  # Tracking index

    return R, vigor, fidelity
