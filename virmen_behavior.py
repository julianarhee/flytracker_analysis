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

import matplotlib
# Set interactive backend
matplotlib.use('Qt5Agg')  # or 'TkAgg' if Qt5Agg doesn't work
import pylab as plt
import seaborn as sns

from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Enable interactive plotting
%matplotlib qt 

#import utils as util
import utils_2p as util2p
#import virmen_fictrac_to_df as vfdf
import plotting as putil

#%%
def wrap_pi(a):  # (-pi, pi]
    return (a + np.pi) % (2*np.pi) - np.pi

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

def calculate_male_vr_pos(expr):
    # Calculate male's VR position
    male_pos = expr[['integrated_xpos', 'integrated_ypos']].values 
    male_heading = expr['integrated_heading'] 
    delta_pos = np.diff(male_pos, axis=0)
    male_pos_vr = np.empty((len(male_heading),2))
    male_pos_vr[0, :] = np.array([0, 0])

    for i, (prev_delta_pos, curr_head) in enumerate( zip(delta_pos[0:], male_heading[1:]) ):
        true_shift = transform(prev_delta_pos, curr_head)
        male_pos_vr[i+1, :] = male_pos_vr[i] + true_shift 

    expr['male_vr_xpos'] = male_pos_vr[:, 0]
    expr['male_vr_ypos'] = male_pos_vr[:, 1]

    return expr


def convert_target_angle(expr):
    target_pos = expr[['sphere_xpos', 'sphere_ypos']].values
 
    # Numpy convention, np.arctan2(y, x) corresponds to (vertical, horizontal)
    # Returns the angle of the vector (x, y) measured from +x-axis,
    # where CCW is positive (pi to -pi)
    
    
    # Instead, we want angles centered at 0 (y spans [0, 0.75]), 
    # with x<0 being left and x>0 being right
    target_angle = wrap_pi(np.arctan2(target_pos[:, 0], target_pos[:, 1]))
  
    # To flip the angle, would have -1 * target_angle,
    # but we flip sphere_xpos already so that left-> right is positive
    expr['target_angle'] = target_angle

    return expr

def get_heading_diff(expr, heading_var='integrated_heading'):
    '''
    Calculate the stepwise wrapped difference in heading.
    Note, is same whether you use integrated_heading or heading_wrapped.

    Parameters:
        expr (pd.DataFrame): DataFrame containing the heading data.
        heading_var (str): Name of the heading column in the DataFrame.

    Returns:
        np.ndarray: Stepwise wrapped difference in heading.
    '''
    # Stepwise warpped difference in heading (new - old), 
    # CCW is positive by convention
    hdiffs = []
    for i, v in zip(expr[heading_var].iloc[0:-1], 
                    expr[heading_var].iloc[1:]):
        hd = circ_dist2(v, i) #i, v) #* -1 # match matlab
        hdiffs.append(hd)
    heading_diff = np.array(hdiffs)

    # Make CW-positive (custom convention, so left->right is positive)
    heading_diff_cw = -1*heading_diff

    # add 0
    heading_diffs = np.concatenate( ([0], heading_diff_cw) )
    
    return heading_diffs

#%%

#
#%%

def test_ti(use_abs=False):
    # Write a time series of a sin function
    time = np.linspace(0, 10, 1000)
    y = np.sin(2 * np.pi * time)
    # Convert the line defining y to a string
    y_str = 'y=np.sin(2 * np.pi * time)'

    # Write a sin function that is the same, but shifted by 0.2
    y2 = np.sin(2 * np.pi * time - 1.6)
    # Convert y2 to a string
    y2_str = 'y2=np.sin(2 * np.pi * time - 2)'

    # Write a sin function that is the same as y2, but amplitude is 0.25
    # Write a sin function where the peaks and valleys happen at the same time as y, but the width is narrower
    y3 = 0.1 * np.sin(2 * np.pi * time - 0.9)
    # Convert y3 to a string
    y3_str = 'y3=0.1 * np.sin(2 * np.pi * time - 0.9)'

    # Plot a grid that is 3 rows by 2 columns using Gridspec
    fig = plt.figure(figsize=(10, 4))
    grid = GridSpec(2, 3, figure=fig)

    # Plot the first row
    ax = fig.add_subplot(grid[0, :])
    ax.plot(time, y, label='{}, hdiff'.format(y_str))
    ax.plot(time, y2, label='{}, female_angle'.format(y2_str), color='m')
    ax.plot(time, y3, label='{}, female_angle_small'.format(y3_str), color='c')
    ax.legend(bbox_to_anchor=(1, 1), loc='lower right', frameon=False, fontsize=8)
    #%
    # Plot in 2 subplots
    # Compute tracking index where y is h_diff, and female_angle is y2
    r2, v2, f2 = compute_ti(y, y2, time, len(time)/4, use_abs=use_abs)
    r3, v3, f3 = compute_ti(y, y3, time, len(time)/4, use_abs=use_abs)

    # Plot the first column of row 2
    ax1 = fig.add_subplot(grid[1, 0])

    # Plot R for r2 and r3
    ax1.plot(time, r2, label='shifted', color='m')
    ax1.plot(time, r3, label='shifted and small', color='c')
    ax1.set_title('TI')

    # Plot v2 and v3
    ax2 = fig.add_subplot(grid[1, 2])
    ax2.plot(time, v2, label='Vigor', color='m')
    ax2.plot(time, v3, label='Vigor', color='c')
    ax2.set_title('Vigor')

    # Plot f2 and f3
    ax3 = fig.add_subplot(grid[1, 1])
    ax3.plot(time, f2, label='Fidelity', color='m')
    ax3.plot(time, f3, label='Fidelity', color='c')
    ax3.set_title('Fidelity')

    plt.subplots_adjust(hspace=0.5, wspace=0.5, right=0.8, left=0.1, top=0.8)

#%%
test_ti(use_abs=True)
#%%
# Demonstrate tracking index for different scenarios
demo_fig = demonstrate_tracking_scenarios()
plt.show()

#%%

def compute_ti(h_diff, female_angle, b_time, trial_length=1000, use_abs=False):
    """
    Compute tracking index (TI) based on correlation between male turning and female target angle.
    
    Parameters:
        h_diff (array-like): Male heading differences (turning behavior)
        female_angle (array-like): Female target angle positions
        b_time (array-like): Time stamps
        trial_length (int): Length of trial window for analysis
        
    Returns:
        tuple: (R, vigor, fidelity) where:
            R: tracking index (normalized vigor * fidelity)
            vigor: net turning in direction of target
            fidelity: correlation coefficient between male turning and target position
    """
    # Parameters
    hemi_length = round(trial_length / 2)  # half-length
    data_length = len(b_time)  # Use full length to match input arrays
    trial_starts = np.arange(data_length)  # frame IDs
    
    # Initialize output arrays with same length as input
    fidelity = np.zeros(data_length)
    vigor = np.zeros(data_length)
    
    # Compute tracking index
    for i in trial_starts:
        # If not at edges of trial
        if i > hemi_length and i < (data_length - hemi_length):
            x = h_diff[(i - hemi_length):(i + hemi_length)]  # male turning
            f = female_angle[(i - hemi_length):(i + hemi_length)]  # stim pos
            t = b_time[(i - hemi_length):(i + hemi_length)]  # time stamps
            mat = np.corrcoef(x, f)  # correlation matrix between stim and male turning
            fid = mat[0, 1]  # store correlation coefficient
            
        # If at the very start of trial (too few n), set to zero
        elif i <= hemi_length:
            x = h_diff[:i+1]
            f = female_angle[:i+1]
            t = b_time[:i+1]
            fid = 0
            
        # If at the very end of trial (too few n), set to zero
        else:
            x = h_diff[i:]
            f = female_angle[i:]
            t = b_time[i:]
            fid = 0
            
        fidelity[i] = fid  # correlation
        vigor[i] = np.sum(x * np.sign(f))  # net turning in direction of target
    

    if use_abs:
        # Handle case where all vigor values are zero to avoid division by zero
        max_abs_vigor = np.max(np.abs(vigor))
        if max_abs_vigor > 0:
            R = np.abs(vigor) / max_abs_vigor  # normalize vigor across all periods
        else:
            R = np.zeros_like(vigor)
        R = R * np.abs(fidelity)  # tracking index
    else:
        # Compute tracking index
        # R = 2 * (vigor - min(vigor)) / (max(vigor)-min(vigor)) - 1 
        R = (vigor/max(vigor)) 
        R = R * np.abs(fidelity)  # tracking index
        
    return R, vigor, fidelity


def compute_ti_enhanced(h_diff, female_angle, b_time, trial_length=1000, use_abs=False, 
                       include_amplitude_matching=True):
    """
    Enhanced version of compute_ti with amplitude matching.
    
    Parameters:
        h_diff (array-like): Male heading differences (turning behavior)
        female_angle (array-like): Female target angle positions
        b_time (array-like): Time stamps
        trial_length (int): Length of trial window for analysis
        use_abs (bool): Whether to use absolute values for tracking index
        include_amplitude_matching (bool): Whether to include amplitude matching in the calculation
        
    Returns:
        tuple: (R, vigor, fidelity, amplitude_ratio) where:
            R: tracking index (normalized vigor * fidelity * amplitude_matching)
            vigor: net turning in direction of target
            fidelity: correlation coefficient between male turning and target position
            amplitude_ratio: ratio of male response amplitude to stimulus amplitude
    """
    # Parameters
    hemi_length = round(trial_length / 2)  # half-length
    data_length = len(b_time)  # Use full length to match input arrays
    trial_starts = np.arange(data_length)  # frame IDs
    
    # Initialize output arrays with same length as input
    fidelity = np.zeros(data_length)
    vigor = np.zeros(data_length)
    amplitude_ratio = np.zeros(data_length)
    
    # Compute tracking index
    for i in trial_starts:
        # If not at edges of trial
        if i > hemi_length and i < (data_length - hemi_length):
            x = h_diff[(i - hemi_length):(i + hemi_length)]  # male turning
            f = female_angle[(i - hemi_length):(i + hemi_length)]  # stim pos
            t = b_time[(i - hemi_length):(i + hemi_length)]  # time stamps
            mat = np.corrcoef(x, f)  # correlation matrix between stim and male turning
            fid = mat[0, 1]  # store correlation coefficient
            
            # Calculate amplitude matching
            if include_amplitude_matching:
                # Calculate RMS (root mean square) amplitudes
                male_amplitude = np.sqrt(np.mean(x**2))
                stimulus_amplitude = np.sqrt(np.mean(f**2))
                
                # Avoid division by zero
                if stimulus_amplitude > 0:
                    amp_ratio = male_amplitude / stimulus_amplitude
                    # Penalize over-response and under-response
                    # Ideal ratio is 1.0, penalize deviations
                    amplitude_penalty = 1.0 / (1.0 + abs(amp_ratio - 1.0))
                else:
                    amp_ratio = 0
                    amplitude_penalty = 0
            else:
                amp_ratio = 1.0
                amplitude_penalty = 1.0
            
        # If at the very start of trial (too few n), set to zero
        elif i <= hemi_length:
            x = h_diff[:i+1]
            f = female_angle[:i+1]
            t = b_time[:i+1]
            fid = 0
            amp_ratio = 0
            amplitude_penalty = 0
            
        # If at the very end of trial (too few n), set to zero
        else:
            x = h_diff[i:]
            f = female_angle[i:]
            t = b_time[i:]
            fid = 0
            amp_ratio = 0
            amplitude_penalty = 0
            
        fidelity[i] = fid  # correlation
        vigor[i] = np.sum(x * np.sign(f))  # net turning in direction of target
        amplitude_ratio[i] = amp_ratio
    

    if use_abs:
        # Handle case where all vigor values are zero to avoid division by zero
        max_abs_vigor = np.max(np.abs(vigor))
        if max_abs_vigor > 0:
            R = np.abs(vigor) / max_abs_vigor  # normalize vigor across all periods
        else:
            R = np.zeros_like(vigor)
        R = R * np.abs(fidelity)  # tracking index
        
        # Include amplitude matching if requested
        if include_amplitude_matching:
            # Calculate amplitude penalty for the whole dataset
            male_amp_global = np.sqrt(np.mean(h_diff**2))
            stim_amp_global = np.sqrt(np.mean(female_angle**2))
            if stim_amp_global > 0:
                global_amp_ratio = male_amp_global / stim_amp_global
                global_amplitude_penalty = 1.0 / (1.0 + abs(global_amp_ratio - 1.0))
            else:
                global_amplitude_penalty = 0
            R = R * global_amplitude_penalty
    else:
        # Compute tracking index
        R = (vigor/max(vigor)) 
        R = R * np.abs(fidelity)  # tracking index
        
        # Include amplitude matching if requested
        if include_amplitude_matching:
            # Use local amplitude penalties calculated above
            R = R * amplitude_penalty
        
    return R, vigor, fidelity, amplitude_ratio


def compute_ti_with_delay_compensation(h_diff, female_angle, b_time, trial_length=1000, 
                                     max_delay_frames=50, use_abs=False, 
                                     include_amplitude_matching=True):
    """
    Compute tracking index with delay compensation to handle expected response delays.
    
    This version tests multiple delay values and uses the one that gives the best correlation,
    accounting for the fact that biological responses often have inherent delays.
    
    Parameters:
        h_diff (array-like): Male heading differences (turning behavior)
        female_angle (array-like): Female target angle positions
        b_time (array-like): Time stamps
        trial_length (int): Length of trial window for analysis
        max_delay_frames (int): Maximum delay to test (in frames)
        use_abs (bool): Whether to use absolute values for tracking index
        include_amplitude_matching (bool): Whether to include amplitude matching
        
    Returns:
        tuple: (R, vigor, fidelity, amplitude_ratio, optimal_delay) where:
            R: tracking index with delay compensation
            vigor: net turning in direction of target
            fidelity: best correlation coefficient found
            amplitude_ratio: ratio of male response amplitude to stimulus amplitude
            optimal_delay: delay (in frames) that gave the best correlation
    """
    # Parameters
    hemi_length = round(trial_length / 2)
    data_length = len(b_time)
    trial_starts = np.arange(data_length)
    
    # Initialize output arrays
    fidelity = np.zeros(data_length)
    vigor = np.zeros(data_length)
    amplitude_ratio = np.zeros(data_length)
    optimal_delay = np.zeros(data_length)
    
    # Test different delays
    delays_to_test = np.arange(0, max_delay_frames + 1, 5)  # Test every 5 frames
    
    for i in trial_starts:
        best_correlation = 0
        best_delay = 0
        best_amp_ratio = 1.0
        
        # If not at edges of trial
        if i > hemi_length and i < (data_length - hemi_length):
            # Get the window for male response
            male_start = i - hemi_length
            male_end = i + hemi_length
            x = h_diff[male_start:male_end]
            
            # Test different delays for the stimulus
            for delay in delays_to_test:
                stim_start = male_start - delay
                stim_end = male_end - delay
                
                # Check if stimulus window is valid
                if stim_start >= 0 and stim_end < data_length:
                    f = female_angle[stim_start:stim_end]
                    
                    # Calculate correlation for this delay
                    if len(x) == len(f) and len(x) > 1:
                        try:
                            mat = np.corrcoef(x, f)
                            corr = abs(mat[0, 1])  # Use absolute correlation for delay finding
                            
                            if corr > best_correlation:
                                best_correlation = corr
                                best_delay = delay
                                
                                # Calculate amplitude ratio for best delay
                                if include_amplitude_matching:
                                    male_amplitude = np.sqrt(np.mean(x**2))
                                    stimulus_amplitude = np.sqrt(np.mean(f**2))
                                    if stimulus_amplitude > 0:
                                        best_amp_ratio = male_amplitude / stimulus_amplitude
                                    else:
                                        best_amp_ratio = 0
                        except:
                            continue
            
            # Use the best delay found
            if best_delay > 0:
                stim_start = male_start - best_delay
                stim_end = male_end - best_delay
                f = female_angle[stim_start:stim_end]
                
                # Calculate final metrics with optimal delay
                mat = np.corrcoef(x, f)
                fid = mat[0, 1]  # Keep sign for directional information
            else:
                # No delay (original behavior)
                f = female_angle[male_start:male_end]
                mat = np.corrcoef(x, f)
                fid = mat[0, 1]
                
        # Handle edge cases
        elif i <= hemi_length:
            x = h_diff[:i+1]
            f = female_angle[:i+1]
            fid = 0
            best_amp_ratio = 0
            best_delay = 0
        else:
            x = h_diff[i:]
            f = female_angle[i:]
            fid = 0
            best_amp_ratio = 0
            best_delay = 0
        
        fidelity[i] = fid
        vigor[i] = np.sum(x * np.sign(f)) if len(x) == len(f) else 0
        amplitude_ratio[i] = best_amp_ratio
        optimal_delay[i] = best_delay
    
    # Calculate final tracking index
    if use_abs:
        max_abs_vigor = np.max(np.abs(vigor))
        if max_abs_vigor > 0:
            R = np.abs(vigor) / max_abs_vigor
        else:
            R = np.zeros_like(vigor)
        R = R * np.abs(fidelity)
        
        if include_amplitude_matching:
            # Global amplitude penalty
            male_amp_global = np.sqrt(np.mean(h_diff**2))
            stim_amp_global = np.sqrt(np.mean(female_angle**2))
            if stim_amp_global > 0:
                global_amp_ratio = male_amp_global / stim_amp_global
                global_amplitude_penalty = 1.0 / (1.0 + abs(global_amp_ratio - 1.0))
            else:
                global_amplitude_penalty = 0
            R = R * global_amplitude_penalty
    else:
        R = (vigor / np.max(np.abs(vigor))) * np.abs(fidelity)
        
        if include_amplitude_matching:
            # Use local amplitude penalties
            amplitude_penalty = np.zeros_like(amplitude_ratio)
            for i, amp_ratio in enumerate(amplitude_ratio):
                if amp_ratio > 0:
                    amplitude_penalty[i] = 1.0 / (1.0 + abs(amp_ratio - 1.0))
                else:
                    amplitude_penalty[i] = 0
            R = R * amplitude_penalty
    
    return R, vigor, fidelity, amplitude_ratio, optimal_delay


def demonstrate_tracking_scenarios():
    """
    Demonstrate tracking index for different behavioral scenarios using sine functions.
    Shows how the tracking index responds to various types of male-female interactions.
    """
    # Create time vector
    time = np.linspace(0, 10, 1000)
    trial_length = 250  # Window size for tracking index calculation
    
    # Define different scenarios
    scenarios = {
        'Perfect Tracking': {
            'h_diff': np.sin(2 * np.pi * time),
            'female_angle': np.sin(2 * np.pi * time),
            'description': 'Male perfectly follows female (same phase, same amplitude)'
        },
        'Perfect Anti-Tracking': {
            'h_diff': np.sin(2 * np.pi * time),
            'female_angle': -np.sin(2 * np.pi * time),
            'description': 'Male perfectly opposes female (opposite phase, same amplitude)'
        },
        'Over-Response': {
            'h_diff': 2 * np.sin(2 * np.pi * time),
            'female_angle': np.sin(2 * np.pi * time),
            'description': 'Male over-responds (same phase, 2x amplitude)'
        },
        'Under-Response': {
            'h_diff': 0.3 * np.sin(2 * np.pi * time),
            'female_angle': np.sin(2 * np.pi * time),
            'description': 'Male under-responds (same phase, 0.3x amplitude)'
        },
        'Phase Lag': {
            'h_diff': np.sin(2 * np.pi * time - np.pi/4),
            'female_angle': np.sin(2 * np.pi * time),
            'description': 'Male follows with 45° phase lag'
        },
        'No Response': {
            'h_diff': 0.1 * np.random.randn(len(time)),
            'female_angle': np.sin(2 * np.pi * time),
            'description': 'Male shows no tracking (random noise)'
        },
        'Delayed Response': {
            'h_diff': np.sin(2 * np.pi * time - np.pi/2),
            'female_angle': np.sin(2 * np.pi * time),
            'description': 'Male follows with 90° phase lag (quarter cycle delay)'
        },
        'Inverted Response': {
            'h_diff': -np.sin(2 * np.pi * time),
            'female_angle': np.sin(2 * np.pi * time),
            'description': 'Male responds in opposite direction'
        }
    }
    
    # Create figure with subplots
    n_scenarios = len(scenarios)
    fig = plt.figure(figsize=(16, 2 * n_scenarios))
    
    for i, (scenario_name, scenario_data) in enumerate(scenarios.items()):
        h_diff = scenario_data['h_diff']
        female_angle = scenario_data['female_angle']
        description = scenario_data['description']
        
        # Calculate tracking index with different methods
        R_abs, vigor_abs, fidelity_abs = compute_ti(
            h_diff, female_angle, time, trial_length, use_abs=True)
        
        R_enhanced, vigor_enhanced, fidelity_enhanced, amp_ratio_enhanced = compute_ti_enhanced(
            h_diff, female_angle, time, trial_length, use_abs=True, include_amplitude_matching=True)
        
        # Calculate with delay compensation
        R_delay, vigor_delay, fidelity_delay, amp_ratio_delay, optimal_delay = compute_ti_with_delay_compensation(
            h_diff, female_angle, time, trial_length, max_delay_frames=50, use_abs=True, include_amplitude_matching=True)
        
        # Create subplot for this scenario
        ax = fig.add_subplot(n_scenarios, 1, i + 1)
        
        # Plot the signals
        ax.plot(time, female_angle, 'm-', label='Female angle', lw=1.5, alpha=0.8)
        ax.plot(time, h_diff, 'b-', label='Male h_diff', lw=1.5, alpha=0.8)
        
        # Create twin axis for tracking index
        ax2 = ax.twinx()
        ax2.plot(time, R_abs, 'orange', label='TI (original)', lw=1.5, alpha=0.8)
        ax2.plot(time, R_enhanced, 'red', label='TI (enhanced)', lw=1.5, alpha=0.8)
        ax2.plot(time, R_delay, 'green', label='TI (delay-comp)', lw=1.5, alpha=0.8)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax.set_ylabel('Signal Amplitude')
        ax2.set_ylabel('Tracking Index')
        ax.set_title(f'{scenario_name}: {description}', fontsize=10, pad=10)
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8, frameon=False)
        
        # Add statistics text
        mean_ti_abs = np.mean(R_abs)
        mean_ti_enhanced = np.mean(R_enhanced)
        mean_ti_delay = np.mean(R_delay)
        mean_amp_ratio = np.mean(amp_ratio_enhanced)
        mean_fidelity = np.mean(np.abs(fidelity_abs))
        mean_optimal_delay = np.mean(optimal_delay)
        
        stats_text = f'Mean TI(orig): {mean_ti_abs:.3f}\nMean TI(enh): {mean_ti_enhanced:.3f}\nMean TI(delay): {mean_ti_delay:.3f}\nMean Delay: {mean_optimal_delay:.1f} frames\nMean Fidelity: {mean_fidelity:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set x-axis only for bottom plot
        if i == n_scenarios - 1:
            ax.set_xlabel('Time')
        else:
            ax.set_xticklabels([])
    
    # Add overall title first
    fig.suptitle('Tracking Index Demonstration: Different Behavioral Scenarios', 
                 fontsize=14, y=0.95)
    
    # Adjust layout to prevent title overlap
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, top=0.92)  # More space between subplots and below main title
    
    return fig


def find_tracking_chunk_n_cycles(tracking_index, toc, full_phase_starts, 
                                threshold=0.3, n_cycles=3, 
                                threshold_fraction=0.7, window_size=30):
    """
    Find a chunk that spans exactly N cycles where tracking_index is above threshold.
    
    Parameters:
        tracking_index (array-like): Tracking index values
        toc (array-like): Time stamps
        full_phase_starts (array-like): Indices where full cycles start
        threshold (float): Minimum tracking index threshold
        n_cycles (int): Exact number of cycles to span
        threshold_fraction (float): Fraction of time within window that must meet threshold
        window_size (int): Window size for rolling average calculation
        
    Returns:
        tuple: (start_idx, end_idx, duration_sec, n_cycles_actual) or None if no chunk found
    """
    if len(full_phase_starts) < n_cycles + 1:
        return None
    
    # Create smoothed threshold indicator
    above_threshold_raw = tracking_index > threshold
    above_threshold = np.zeros_like(above_threshold_raw, dtype=bool)
    half_window = window_size // 2
    
    for i in range(len(tracking_index)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(tracking_index), i + half_window + 1)
        window_above = np.sum(above_threshold_raw[start_idx:end_idx])
        window_size_actual = end_idx - start_idx
        above_threshold[i] = (window_above / window_size_actual) >= threshold_fraction
    
    # Try each possible N-cycle window
    best_chunk = None
    best_score = 0
    fallback_chunk = None
    fallback_score = 0
    
    for i in range(len(full_phase_starts) - n_cycles):
        # Define chunk boundaries: from start of cycle i to start of cycle i+n_cycles
        chunk_start = full_phase_starts[i]
        chunk_end = full_phase_starts[i + n_cycles] - 1  # End just before next cycle starts
        
        # Check if chunk is valid (within data bounds)
        if chunk_end >= len(tracking_index):
            continue
            
        # Calculate what fraction of this chunk is above threshold
        chunk_above_threshold = above_threshold[chunk_start:chunk_end+1]
        fraction_above = np.mean(chunk_above_threshold)
        duration_sec = toc[chunk_end] - toc[chunk_start]
        
        # Score based on fraction above threshold and duration
        score = fraction_above * duration_sec
        
        # Track the best chunk that meets threshold criteria
        if fraction_above >= threshold_fraction:
            if score > best_score:
                best_score = score
                best_chunk = (chunk_start, chunk_end, duration_sec, n_cycles)
        else:
            # Track the best chunk even if it doesn't meet threshold criteria (fallback)
            if score > fallback_score:
                fallback_score = score
                fallback_chunk = (chunk_start, chunk_end, duration_sec, n_cycles)
    
    # Return the best chunk that meets criteria, or the best fallback if none do
    if best_chunk is not None:
        return best_chunk
    elif fallback_chunk is not None:
        print(f"Warning: No chunk met {threshold_fraction*100:.0f}% threshold criteria. Using best available chunk with {np.mean(above_threshold[fallback_chunk[0]:fallback_chunk[1]+1])*100:.0f}% above threshold.")
        return fallback_chunk
    else:
        return None


def find_phase_starts(target_angle, return_half_phases=False):
    """
    Find the starts of each full cycle and half-sweep in the target angle trajectory.

    A full cycle is defined as -90 → 90 → -90 (in degrees, or -π/2 → π/2 → -π/2 in radians).
    A half-sweep is either -90 → 90 or 90 → -90.

    Parameters:
        target_angle (array-like): Target angle values in radians
        return_half_phases (bool): If True, also return indices of half-sweep starts

    Returns:
        tuple: (full_cycle_starts, half_sweep_starts) where:
            full_cycle_starts: indices where each full cycle starts (-90 crossing upward)
            half_sweep_starts: indices where each half-sweep starts (direction changes)
    """
    # Convert to degrees for easier threshold detection
    target_angle_deg = np.rad2deg(target_angle)

    # Find direction changes by looking at the derivative
    angle_diff = np.diff(target_angle_deg)
    # Find where direction changes (sign change in derivative)
    # Positive to negative: peak (90 degrees)
    # Negative to positive: trough (-90 degrees)
    sign_changes = np.where(np.diff(np.sign(angle_diff)) != 0)[0] + 1

    # Each direction change is a half-sweep start
    half_sweep_starts = sign_changes

    # To find full cycle starts: look for -90 → 90 → -90 pattern
    # We define the start of a full cycle as the start of a -90→90 sweep
    # i.e., a half-sweep that starts at -90 and goes upward

    # Find indices where target_angle_deg is near -90
    near_neg90 = np.isclose(target_angle_deg, -90, atol=5)
    # For each half-sweep start, check if it starts near -90 and the next half-sweep is upward
    full_cycle_starts = []
    for idx in half_sweep_starts:
        if idx < len(target_angle_deg) - 1:
            # Check if this is an upward sweep starting near -90
            if near_neg90[idx]:
                # Check direction: is the next value greater than current? (upward)
                if target_angle_deg[idx+1] > target_angle_deg[idx]:
                    full_cycle_starts.append(idx)
    full_cycle_starts = np.array(full_cycle_starts)

    if return_half_phases:
        return full_cycle_starts, half_sweep_starts
    else:
        return full_cycle_starts


#%%
plot_style = 'dark'
putil.set_sns_style(plot_style, min_fontsize=18)
bg_color = [0.7] * 3 if plot_style == 'dark' else 'k'

#%%
# Run this script, aggregate across flies
# %
import re
rootdir = '/Volumes/Juliana/2p-data'
genotype = 'R35D04-R22D06-RSET-GCaMP8m'

srcdir = os.path.join(rootdir, genotype)

figdir = os.path.join(srcdir, 'figures', 'virmen_behavior')
if not os.path.exists(figdir):
    os.makedirs(figdir)

fps = 60

#%%
#session = '20250725'
#session_list = ['20250812', '20250813', '20250814', '20250815',
#                '20250819', '20250821', '20250822']  # List of sessions to process, can be extended
session_list = ['20250821', '20250822']
#flynum = 1

# Load only one session for now
session = '20250821'
flyid = 'f2'
filenum = 30 #28 #32

# Find virmen file
vr_fpath = glob.glob(os.path.join(srcdir, session, flyid, 'virmen', 
                        '{}_*{}_{:03d}_vr.mat'.format(session, flyid, filenum)))[0]
is_vr_file = True

# Create figure directory using the basename of vr_fpath without the _vr.mat ending
acquisition = os.path.basename(vr_fpath).replace('_vr.mat', '')

# Create figure directory in parent folder as virmen folder
figdir = os.path.join(srcdir, session, flyid, 'figures')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print("Figure directory: {}".format(figdir))    

# Create figure ID from genotype, session, flyid, and filenum for saving figures, use -
figid = acquisition

# %%
# Load VR data
expr = util2p.load_mat_vr(vr_fpath)

# %% Load experiment metadata 
meta = util2p.load_virmen_meta(genotype, session, flyid, filenum, rootdir=rootdir)

# %%
# Invert xpos so that stimulus left to right -> -1 to 1
expr['sphere_xpos'] = expr['sphere_xpos']*-1

#%%
# Calculate male's VR position
expr = calculate_male_vr_pos(expr)
# Calculate angular position from male_vr_xpos and male_vr_ypos
expr['male_vr_angle'] = wrap_pi(np.arctan2(expr['male_vr_xpos'], expr['male_vr_ypos']))

# Get index of first motion
motion_start = expr[expr['is_moving']==1].index[0]
print(motion_start)

# Calcualte target angle from sphere position
expr = convert_target_angle(expr)

#%%
# Basic usage - get full phase starts only
full_phase_starts = find_phase_starts(expr['target_angle'])

#%%
# Wrap heading
expr['heading_wrapped'] = wrap_pi(expr['integrated_heading'])
expr['heading_diff_male'] = get_heading_diff(expr, 'integrated_heading')
expr['toc_diff'] = expr['toc'].diff()
expr['toc_diff'].fillna(0, inplace=True)

# Calculate angular velocity
expr['ang_vel_male'] = expr['heading_diff_male'] / expr['toc_diff'] #expr['toc'].diff().mean() 

#%%
#heading_diff_target = (np.diff(expr['target_angle']) + np.pi) % (2*np.pi) - np.pi
expr['target_angle_diff'] = get_heading_diff(expr, 'target_angle')
expr['ang_vel_target'] = expr['target_angle_diff'] / expr['toc'].diff().mean()

#%%
# Calculate tracking index
trial_length = (full_phase_starts[1] - full_phase_starts[0]) * 3
R, vigor, fidelity = compute_ti(expr['heading_diff_male'], 
                                expr['target_angle'], 
                                expr['toc'], trial_length)

# Add the tracking metrics to the dataframe
expr['tracking_index'] = R
expr['vigor'] = vigor
expr['fidelity'] = fidelity

#%%
# Plot R, vigor, and fidelity   
fig, axn = plt.subplots(3, 1, figsize=(10, 4), sharex=True,
                        gridspec_kw={'left':0.07, 'right':0.98, 'top':0.95, 'bottom':0.13, 'hspace':0.25})
ax = axn[0]
ax.plot(expr['toc'], expr['tracking_index'], color=bg_color, lw=0.5)
ax.set_ylabel('Tracking index')

ax = axn[1]
ax.plot(expr['toc'], expr['vigor'], color=bg_color, lw=0.5)
ax.set_ylabel('Vigor')

ax = axn[2]
ax.plot(expr['toc'], expr['fidelity'], color=bg_color, lw=0.5)
ax.set_ylabel('Fidelity')

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2)
putil.label_figure(fig, figid)


#%%
# Plot target_angle and tracking_index on the same plot
# Also plot heading_diff_male
fig, ax = plt.subplots(figsize=(15, 3))
ax.plot(expr['toc'], expr['target_angle'], label='Target angle',
        color=bg_color, lw=0.5)

# Plot on a twin axis
ax2 = ax.twinx()
ax2.plot(expr['toc'], expr['heading_diff_male'], label='Heading diff', 
         color='dodgerblue', lw=0.5)
# Plot tracking index
ax2.plot(expr['toc'], expr['vigor'], label='Vigor',
        color='b')
ax2.plot(expr['toc'], expr['fidelity'], label='Fidelity',
        color='c')
ax2.plot(expr['toc'], expr['tracking_index'], label='Tracking index',
        color='m')

# Combine legends
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='lower right',
          frameon=False, bbox_to_anchor=(1, 1), fontsize=8)
ax.set_xlabel('Time (s)')

# Set box aspect to be narrow (wide and short)
# This makes the plot visually narrow in height relative to width
ax.set_box_aspect(0.1)
ax2.set_box_aspect(0.1)

plt.subplots_adjust(right=0.9, left=0.05)

# label figure
putil.label_figure(fig, figid)
# save figure
figname = 'time_TI-target-heading_diff'
fig_fpath = os.path.join(figdir, '{}.png'.format(figname))
plt.savefig(fig_fpath)
print(fig_fpath)

#%%
# Plot tracking chunk
n_phase_plot = 3
ti_threshold = 0.2
window_size = 2 * fps

# Plot histogram of tracking index and show threshold
fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(expr['tracking_index'], bins=20)
ax.axvline(ti_threshold, color='r', linestyle='--', lw=0.5)
ax.set_xlabel('Tracking index')
ax.set_ylabel('Count')
ax.set_box_aspect(1)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2)

# Label figure and save
putil.label_figure(fig, figid)
figname = 'hist_tracking_index'
fig_fpath = os.path.join(figdir, '{}.png'.format(figname))
plt.savefig(fig_fpath)
print(fig_fpath)


# Find a chunk that spans exactly N cycles with good tracking
tracking_chunk = find_tracking_chunk_n_cycles(
    expr['tracking_index'], 
    expr['toc'], 
    full_phase_starts,
    threshold=ti_threshold,  # Tracking index threshold
    n_cycles=n_phase_plot,     # Exactly 3 cycles
    threshold_fraction=0.7,  # 70% of time must be above threshold
    window_size=window_size  # 30-frame rolling window
)

if tracking_chunk is not None:
    s_, e_, duration_sec, n_cycles_actual = tracking_chunk
    print(f"Found {n_cycles_actual}-cycle tracking chunk: frames {s_}-{e_}, duration {duration_sec:.1f}s")
else:
    print("No suitable 3-cycle tracking chunk found, using default range")
    s_ = motion_start
    e_ = full_phase_starts[n_phase_plot]
    #s_, e_ = 2500, 6500

#%%
import matplotlib as mpl
# Does sphere_xpos match ang_vel_male (when male is tracking)?
# ------------------------------------------------------------
# Check stimulus trajectory
print(s_, e_)
#%
# Check  male's trajectory
# Make a grid of subplots with the top row showing toc, and the bottom row showing sphere_xpos and sphere_ypos
# Use gridplot to create a grid of 2x2 subplots. The first plot will be in the top row, spanning 2 columns.

fig = plt.figure(figsize=(10, 10))
grid = GridSpec(3, 3, figure=fig)
fig.text(0.05, 0.9, 'Frames {}-{}, (TI>={})'.format(s_, e_, ti_threshold ))

# Plot stimulus trajectory
# Top row spanning 2 columns but one row

# Top row
ax1 = fig.add_subplot(grid[0, :])
sns.scatterplot(data=expr.iloc[s_:e_], x='toc', y='target_angle', ax=ax1,
                edgecolor=None, s=10, hue='sphere_xpos', palette='coolwarm', legend=0)
ax1.set_box_aspect(0.25)
# Create colorbar for ax1
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="2%", pad=0.05)
# Create norm and cmap
norm = mpl.colors.Normalize(vmin=expr['sphere_xpos'].min(), vmax=expr['sphere_xpos'].max())
cmap = 'coolwarm'
cbar1 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, 
                    shrink=0.5, label='sphere_xpos', cax=cax1)
cbar1.ax.tick_params(labelsize=9)
cbar1.ax.set_ylabel('sphere_xpos', fontsize=9)

# Middle row
ax2 = fig.add_subplot(grid[1, :])
sns.scatterplot(data=expr.iloc[s_:e_], x='toc', y='ang_vel_male', ax=ax2,
                edgecolor=None, s=10, hue='sphere_xpos', palette='coolwarm', legend=0)
ax2.set_box_aspect(0.25)

# Bottom row is 2 separate subplots, one showing male_vr_xpos and male_vr_ypos, and the other showing sphere_xpos and sphere_ypos
# They should share axes
for hi, hue_var in enumerate(['toc', 'sphere_xpos', 'ang_vel_male']):
    if hi==0:
        ax1 = fig.add_subplot(grid[2, hi])
    else:
        ax1 = fig.add_subplot(grid[2, hi], sharex=ax1, sharey=ax1)
    if hue_var=='toc':
        palette='viridis'
    else:
        palette='coolwarm'
    # Plot the same points with a line
    ax1.plot(expr.iloc[s_:e_]['male_vr_xpos'], expr.iloc[s_:e_]['male_vr_ypos'], 
            color=bg_color, lw=0.5)
    sns.scatterplot(data=expr.iloc[s_:e_], x='male_vr_xpos', y='male_vr_ypos', 
                    ax=ax1, edgecolor=None, s=10, 
                    hue=hue_var, palette=palette, legend=None)
    ax1.set_title('hue={}'.format(hue_var))
    # Remove x and y axis labels except for the leftmost plot
    if hi != 0:
        ax1.set_xlabel('')
        ax1.set_ylabel('')

plt.subplots_adjust(hspace=0.5)
# Label figure and save
putil.label_figure(fig, figid)
figname = 'scatter_hue-sphere_xpos_ang_vel_male'
fig_fpath = os.path.join(figdir, '{}.png'.format(figname))
plt.savefig(fig_fpath)
print(fig_fpath)

%%
# Compare target position and male angular velocity across the whole time seris
# Make the figure wider and fill out horizontally
fig, axn = plt.subplots(2, 1, figsize=(15, 4), sharex=True, 
                        gridspec_kw={'left':0.07, 'right':0.98, 'top':0.95, 'bottom':0.13, 'hspace':0.25})

# First subplot: color by sphere_xpos with colorbar, center at 0
ax = axn[0]
sphere_xpos_vals = expr.iloc[s_:e_]['sphere_xpos']
vmax1 = np.nanmax(np.abs(sphere_xpos_vals))
sc = ax.scatter(
    expr.iloc[s_:e_]['toc'],
    expr.iloc[s_:e_]['target_angle'],
    c=sphere_xpos_vals,
    cmap='coolwarm',
    s=5,
    edgecolor='none',
    vmin=-vmax1,
    vmax=vmax1
)
cbar = plt.colorbar(sc, ax=ax, pad=0.01)
cbar.set_label('sphere_xpos', fontsize=10)
cbar.ax.tick_params(labelsize=9)
ax.set_ylabel('target_angle', fontsize=12)
ax.set_xlabel('')  # No x-label for top plot

# Second subplot: color by ang_vel_male with colorbar, center at 0
ax = axn[1]
ang_vel_vals = expr.iloc[s_:e_]['ang_vel_male']
vmax2 = np.nanmax(np.abs(ang_vel_vals))
sc2 = ax.scatter(
    expr.iloc[s_:e_]['toc'],
    expr.iloc[s_:e_]['target_angle'],
    c=ang_vel_vals,
    cmap='coolwarm',
    s=5,
    edgecolor='none',
    vmin=-vmax2,
    vmax=vmax2
)
cbar2 = plt.colorbar(sc2, ax=ax, pad=0.01)
cbar2.set_label('ang_vel_male', fontsize=10)
cbar2.ax.tick_params(labelsize=9)
ax.set_ylabel('target_angle', fontsize=12)
ax.set_xlabel('toc', fontsize=12)

#%%
# Check time difference between consecutive frames
fig, ax = plt.subplots(figsize=(10,4))
expr['toc_diff'] = expr['toc'].diff()
#sns.scatterplot(data=expr.iloc[s_:e_], x='toc', y='toc_diff', ax=ax,
#                s=2, hue='sphere_xpos', palette='coolwarm', 
#                edgecolor=None)
ax.plot(expr['toc'], expr['toc_diff'], lw=0.5)
ax.set_ylabel('toc_diff (s)')
ax.set_xlabel('toc (s)')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2)

fig.text(0.1, 0.9, 'Time difference between consecutive frames (s)')

# Label figure and save
putil.label_figure(fig, figid)
figname = 'toc_diff'
fig_fpath = os.path.join(figdir, '{}.png'.format(figname))
plt.savefig(fig_fpath)
print(fig_fpath)


# %%
#male_pos = expr[['integrated_xpos', 'integrated_ypos']].values
#male_heading = expr['integrated_heading']
#toc = expr['toc'].diff()
#dist_to_target = np.sqrt(np.sum(target_pos**2, axis=1))
#male_vel = np.sqrt( np.sum(np.diff(male_pos,axis=0)**2, axis=1)) / np.diff(expr['toc']) / 3

# %%
# Plot target ang vel -- compare with male_ang_vel
fig, ax =plt.subplots(figsize=(10, 4))
#s_ = 2500
#e_ = 4000
#ax.set_title('Frame {}-{}'.format(s_, e_), loc='left', fontsize=12)
sns.scatterplot(data=expr.iloc[s_:e_], x='toc', y='target_angle', ax=ax,
                s=10, hue='ang_vel_target', palette='coolwarm') #, legend=0)
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1.2, 1.5), frameon=False)

ax1 = ax.twinx()
sns.scatterplot(data=expr.iloc[s_:e_], x='toc', y='heading_diff_male', ax=ax1,
                s=10, hue='ang_vel_male', palette='coolwarm')
sns.move_legend(ax1, loc='upper left', bbox_to_anchor=(1.2, 0.5), frameon=False)

fig.text(0.1, 0.9, 'Target angle and Heading diff, colored by respective ang vel\n Frames {}-{}'.format(s_, e_))

# Label figure and save
putil.label_figure(fig, figid)
figname = 'target_heading_ang_vels'
fig_fpath = os.path.join(figdir, '{}.png'.format(figname))
plt.savefig(fig_fpath)
print(fig_fpath)

# %%
# Shift ang vel to account for lag in neural response  
lag = 2
expr['ang_vel_male_shifted'] = expr['ang_vel_male'].shift(-lag)
#%
# Assign target direction
#expr.loc[expr['target_angle_diff']>0, 'target_dir'] = 'CW'
#expr.loc[expr['target_angle_diff']<0, 'target_dir'] = 'CCW'
expr.loc[expr['ang_vel_target']>0, 'target_dir'] = 'CW'
expr.loc[expr['ang_vel_target']<0, 'target_dir'] = 'CCW'

# Bin target position
chasedf = expr.copy()
chasedf['target_angle_deg'] = np.rad2deg(chasedf['target_angle'])

start_bin = -180
end_bin=180
bin_size=20
chasedf['binned_target_angle'] = pd.cut(chasedf['target_angle_deg'],
                                    bins=np.arange(start_bin, end_bin, bin_size),
                                    labels=np.arange(start_bin+bin_size/2,
                                            end_bin-bin_size/2, bin_size))  
# %%

# Plot
# Get average ang vel across bins
grouper = ['binned_target_angle', 'target_dir'] 
yvar = 'ang_vel_male'

#yvar='male_heading'
avg_ang_vel = chasedf.groupby(grouper)[yvar].mean().reset_index()

fig, ax = plt.subplots()
stim_palette = {'CW': 'cyan', 'CCW': 'magenta'} #'deepskyblue'}
sns.lineplot(data=avg_ang_vel, x='binned_target_angle', y=yvar, ax=ax,
                        hue='target_dir', palette=stim_palette, 
                        errorbar='se', marker='o') #errwidth=0.5)
ax.axvline(0, color=bg_color, ls=':', lw=1)
ax.axhline(0, color=bg_color, ls=':', lw=1)
ax.set_xlim([-185, 185])
ax.set_xticks(np.arange(-180, 181, 90))
#ax.set_ylim([-4, 4])
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=False)
ax.set_box_aspect(1)

# Label figure and save
putil.label_figure(fig, figid)
figname = 'gain_targetpos_angvel'
fig_fpath = os.path.join(figdir, '{}.png'.format(figname))
plt.savefig(fig_fpath)
print(fig_fpath)
# %%
