#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : sound_chambers.py
Created        : 2025/03/5 14:50:09
Project        : <<projectpath>>
Author         : jyr
Email          : juliana.rhee@gmail.com
Last Modified  : 
'''

#%%
import os
import glob
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import soundfile as sf
#import wave
import libs.utils as util
from scipy.io import wavfile

# %%

basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data/audio'
acq = '20250213-0915'
channel = 6
acqdir = os.path.join(basedir, acq)

audiodir = [i for i in glob.glob(os.path.join(basedir, acq, '2025*')) \
          if os.path.isdir(i)][0]
wave_files = sorted([i for i in glob.glob(os.path.join(audiodir, '*{}*.WAV'.format(channel))) \
              if 'empty' not in i], key=util.natsort)
print(len(wave_files))

# %%
# Number of channels: 1
# Frame rate: 5000
# Number of frames: 3000000

wf = wave_files[0]
samplerate, data = wavfile.read(wf)
print(f"Sample rate: {samplerate}, data shape: {data.shape}")

#%%
plt.figure()
plt.plot(data)
# %%
# Loop through each wave file and read the data into a dataframe with each channel as a column
df_ = pd.DataFrame()
# Open the .wave file
for wi, wf in enumerate(wave_files):
    samplerate, data = wavfile.read(wf)
    #audio_file = wave.open(wf, 'rb')  # 'rb' for read binary mode
    df_['ch{}'.format(wi)] = data

#%% Add time channel based on number of frames (size of df_) and samplerate
df_['time'] = np.arange(0, df_.shape[0]/samplerate, 1/samplerate)

# %%
# Find where any value of df_ is greater or less than 9000 or -9000 and make it NaN
#df_[(df_ > 7000) | (df_ < -7000)] = np.nan

#%%


from scipy.signal import butter, sosfilt

def low_pass_filter_with_peak_removal(data, cutoff, fs, order, threshold):
    """
    Applies a low-pass filter and removes peaks above a threshold.

    Args:
        data (array_like): The input waveform data.
        cutoff (float): The cutoff frequency for the low-pass filter (Hz).
        fs (float): The sampling frequency of the data (Hz).
        order (int): The order of the Butterworth filter.
        threshold (float): The threshold above which peaks will be removed.

    Returns:
        array_like: The filtered waveform with peaks removed.
    """

    # Design the Butterworth filter
    sos = butter(order, cutoff, btype='low', fs=fs, output='sos')

    # Apply the filter
    filtered_data = sosfilt(sos, data)

    # Remove peaks above the threshold
    filtered_data[np.abs(filtered_data) > threshold] = 0

    return filtered_data

#%%

## Example Usage
#fs = 1000  # Sampling frequency (Hz)
#t = np.arange(0, 1, 1/fs)  # Time vector
## Create a sample waveform with some peaks
#data = np.sin(2*np.pi*5*t) + 0.5*np.random.randn(len(t)) 
#
#cutoff_freq = 30  # Cutoff frequency (Hz)
#filter_order = 4  # Filter order
#peak_threshold = 2 # Threshold for peak removal
#
#filtered_signal = low_pass_filter_with_peak_removal(data, cutoff_freq, fs, filter_order, peak_threshold)
#plt.figure()
#plt.plot(data)
#plt.plot(filtered_signal)

import numpy as np
from scipy.interpolate import interp1d

import numpy as np
from scipy.interpolate import interp1d

def remove_clustered_peaks_and_interpolate(data, threshold):
    """
    Identifies and removes clustered large peaks in a signal and interpolates across those points.

    Args:
        data (array_like): The input signal.
        threshold (float): Amplitude threshold above which data points are removed.

    Returns:
        array_like: The signal with peaks removed and interpolated.
    """

    # Identify indices where the signal exceeds the threshold
    peak_indices = np.where(np.abs(data) > threshold)[0]

    if len(peak_indices) > 0:
        # Find groups of consecutive peaks
        peak_groups = []
        current_group = [peak_indices[0]]

        for i in range(1, len(peak_indices)):
            if peak_indices[i] == peak_indices[i - 1] + 1:
                current_group.append(peak_indices[i])
            else:
                peak_groups.append(current_group)
                current_group = [peak_indices[i]]

        peak_groups.append(current_group)  # Append the last group

        # Convert data to float (to allow NaN replacements)
        data = np.array(data, dtype=np.float64)

        # Replace each peak group with NaN
        for group in peak_groups:
            data[group] = np.nan

        # Interpolate over NaN values
        valid_indices = np.where(~np.isnan(data))[0]  # Indices of valid (non-NaN) points
        interpolator = interp1d(valid_indices, data[valid_indices], kind='linear', bounds_error=False, fill_value='extrapolate')

        # Replace NaNs with interpolated values
        data[np.isnan(data)] = interpolator(np.where(np.isnan(data))[0])

    return data

def remove_peaks_and_interpolate(data, threshold, baseline_factor=1.5, stability_window=3):
    """
    Identifies and removes large peaks in a signal, extending removal until the signal returns to a fluctuating baseline level.

    Args:
        data (array_like): The input signal.
        threshold (float): Amplitude threshold above which data points are removed.
        baseline_factor (float): Factor for determining the baseline range using standard deviation.
        stability_window (int): Number of consecutive points that must be within the baseline to stop extending.

    Returns:
        array_like: The signal with peaks removed and interpolated.
    """

    # Convert data to float (to allow NaN replacements)
    data = np.array(data, dtype=np.float64)

    # Identify peak indices where the signal exceeds the threshold
    peak_indices = np.where(np.abs(data) > threshold)[0]

    if len(peak_indices) == 0:
        return data  # No peaks found, return original signal

    # Compute baseline level using non-peak values
    non_peak_data = data[np.abs(data) <= threshold]
    baseline_mean = np.mean(non_peak_data)
    baseline_std = np.std(non_peak_data)

    # Define dynamic baseline range
    baseline_lower = baseline_mean - baseline_factor * baseline_std
    baseline_upper = baseline_mean + baseline_factor * baseline_std

    # Find groups of consecutive peaks
    peak_groups = []
    current_group = [peak_indices[0]]

    for i in range(1, len(peak_indices)):
        if peak_indices[i] == peak_indices[i - 1] + 1:
            current_group.append(peak_indices[i])
        else:
            peak_groups.append(current_group)
            current_group = [peak_indices[i]]

    peak_groups.append(current_group)  # Append the last group

    # Extend peak regions until signal stabilizes
    for group in peak_groups:
        start_idx = group[0]
        end_idx = group[-1]

        # Extend backward until the signal remains within baseline for 'stability_window' frames
        count = 0  # Track how many consecutive points are within the baseline
        while start_idx > 0:
            if baseline_lower <= data[start_idx] <= baseline_upper:
                count += 1
                if count >= stability_window:
                    break
            else:
                count = 0  # Reset if we find an unstable point
            start_idx -= 1

        # Extend forward until the signal remains within baseline for 'stability_window' frames
        count = 0
        while end_idx < len(data) - 1:
            if baseline_lower <= data[end_idx] <= baseline_upper:
                count += 1
                if count >= stability_window:
                    break
            else:
                count = 0
            end_idx += 1

        # Mask out the entire peak group including extended regions
        data[start_idx:end_idx + 1] = np.nan

    # Interpolate over NaN values
    valid_indices = np.where(~np.isnan(data))[0]  # Indices of valid (non-NaN) points
    interpolator = interp1d(valid_indices, data[valid_indices], kind='linear', bounds_error=False, fill_value='extrapolate')

    # Replace NaNs with interpolated values
    data[np.isnan(data)] = interpolator(np.where(np.isnan(data))[0])

    return data

import numpy as np
from scipy.signal import butter, sosfiltfilt

def high_pass_filter(data, cutoff, fs, order=4):
    """
    Applies a high-pass Butterworth filter to remove baseline noise.

    Args:
        data (array_like): The input signal.
        cutoff (float): The cutoff frequency for the high-pass filter (Hz).
        fs (float): The sampling frequency of the data (Hz).
        order (int): The order of the Butterworth filter (default: 4).

    Returns:
        array_like: The high-pass filtered signal.
    """

    # Design a Butterworth high-pass filter
    sos = butter(order, cutoff, btype='high', fs=fs, output='sos')

    # Apply the filter with zero-phase filtering (to prevent phase shift)
    filtered_data = sosfiltfilt(sos, data)

    return filtered_data

#%%
fs = samplerate
filter_order = 3
cutoff_freq = 2000
peak_threshold = 3000

channels = [c for c in df_.columns if c!='time']

#data = df_['ch3'].copy().values
summed = df_[channels].sum(axis=1).values
assert np.isnan(data).any() == False, "NaNs found in data"
assert np.isinf(data).any() == False, "Infs found in data"

#fsig = low_pass_filter_with_peak_removal(sig, cutoff_freq, fs, filter_order, peak_threshold)
#fsig = remove_clustered_peaks_and_interpolate(data, peak_threshold) #, window=5)
#fsig = remove_peaks_and_interpolate(summed, peak_threshold, baseline_factor=4) #, window=5)
data = df_['ch3'].copy().values
fsig = high_pass_filter(data, cutoff_freq, fs, order=filter_order)
plt.figure()
plt.plot(data)
plt.plot(fsig) #+5000)
#plt.plot(fsig2+10000)
#lt.xlim([557000, 558000])
#%%
n_channels = df_.shape[1]

fig, axn = plt.subplots(n_channels, 1) #plt.figure()
for ai, ch in enumerate(df_.columns):
    ax=axn[ai]
    ax.plot(df_[ch], label=ch, lw=0.25, color='k')
    ax.set_title(ch, loc='left')
#plt.plot(df_.sum(axis=1), lw=0.5, color='r', label='sum')
#plt.legend()
# %%
# %%
def high_pass_filter(data, cutoff=1.0, fs=44100, order=4):
    """
    Applies a high-pass Butterworth filter to remove DC offset and low-frequency drift.
    
    Args:
        data (array_like): The raw audio waveform.
        cutoff (float): The cutoff frequency (Hz) for the high-pass filter (default: 1 Hz).
        fs (int): The sampling frequency of the audio signal.
        order (int): The order of the Butterworth filter (default: 4).
        
    Returns:
        array_like: The filtered signal with a stable baseline.
    """
    sos = butter(order, cutoff, btype='high', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, medfilt

def flatten_baseline_audacity_style(data, fs, window_size=0.5, cutoff=1.0, noise_smoothing=0.1, order=4):
    """
    Removes baseline fluctuations and noise around the baseline, similar to Audacity.

    Args:
        data (array_like): The raw audio signal.
        fs (int): The sampling frequency of the audio signal.
        window_size (float): The window size (in seconds) for moving average baseline estimation (default: 0.5s).
        cutoff (float): The cutoff frequency (Hz) for high-pass filtering (default: 1.0 Hz).
        noise_smoothing (float): Window size (in seconds) for noise smoothing using a moving average (default: 0.1s).
        order (int): The order of the Butterworth filter for high-pass filtering (default: 4).

    Returns:
        array_like: The signal with a flattened baseline and reduced noise.
    """

    # Convert to float if needed
    data = np.array(data, dtype=np.float64)

    # Step 1: Remove DC Offset
    data -= np.mean(data)

    # Step 2: Apply High-Pass Filtering to Remove Low-Frequency Drift
    sos = butter(order, cutoff, btype='high', fs=fs, output='sos')
    data_hp = sosfiltfilt(sos, data)

    # Step 3: Estimate Baseline with Moving Average
    window_samples = int(window_size * fs)
    if window_samples % 2 == 0:
        window_samples += 1  # Ensure odd window size for symmetrical smoothing
    baseline = np.convolve(data_hp, np.ones(window_samples)/window_samples, mode='same')

    # Step 4: Apply Median Filtering to Reduce Sudden Baseline Noise
    smoothed_baseline = medfilt(baseline, kernel_size=5)

    # Step 5: Apply a Final Moving Average for Additional Smoothing
    smoothing_samples = int(noise_smoothing * fs)
    if smoothing_samples % 2 == 0:
        smoothing_samples += 1
    smoothed_baseline = np.convolve(smoothed_baseline, np.ones(smoothing_samples)/smoothing_samples, mode='same')

    # Step 6: Subtract Final Smoothed Baseline
    flattened_signal = data_hp - smoothed_baseline

    return flattened_signal
#%%
# Load raw WAV file
#fs, raw_signal = wav.read("your_audio_file.wav")

raw_signal = df_['ch3'].copy().values

# Convert to floating point for processing
if raw_signal.dtype != np.float32:
    raw_signal = raw_signal.astype(np.float32) / np.max(np.abs(raw_signal))

# Apply Audacity-style baseline flattening
flattened_signal = flatten_baseline_audacity_style(raw_signal, fs)

# Plot before and after filtering
fig, axn = plt.subplots(2, 1, figsize=(12, 5), sharey=True, sharex=True)
#plt.subplot(2, 1, 1)
ax=axn[0]
ax.plot(raw_signal[:10000], label="Raw Signal (Baseline Fluctuations)")
ax.axhline(0, color='r', linestyle='--', alpha=0.6)
#plt.legend()
ax.set_title("Raw Audio Signal with Baseline Fluctuations")

#plt.subplot(2, 1, 2)
ax=axn[1]
ax.plot(flattened_signal[:10000], label="Flattened Signal (Audacity-like)")
ax.axhline(0, color='r', linestyle='--', alpha=0.6)
#plt.legend()
ax.set_title("Processed Signal with Flattened Baseline")

#%%

import numpy as np
import numpy as np

def normalize_audio(data, new_min=-1, new_max=1):
    """
    Normalizes an audio signal so that the original min and max values map exactly to new_min and new_max,
    while preserving the shape of the waveform.

    Args:
        data (array_like): The input audio signal.
        new_min (float): The minimum value of the normalized range (default: -1).
        new_max (float): The maximum value of the normalized range (default: 1).

    Returns:
        array_like: The normalized signal within the specified range.
    """
    old_min = np.min(data)
    old_max = np.max(data)

    if old_max == old_min:
        return np.full_like(data, (new_max + new_min) / 2)  # Avoid division by zero for constant signals

    # Normalize the signal: map old_min -> new_min and old_max -> new_max
    normalized_data = ((data - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

    return normalized_data

def convert_range(oldval, newmin=-1, newmax=1, oldmax=None, oldmin=None):
    if oldmax is None: #and len(oldval)>1:
        oldmax = np.nanmax(oldval)
    if oldmin is None: # and len(oldval)>1:
        oldmin = np.nanmin(oldval)

    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval

def flatten_baseline(data, fs, cutoff=1.0, window_size=0.5, order=4):
    """
    Flattens the baseline of an audio signal by removing low-frequency drift,
    similar to how Audacity visualizes waveforms.

    Args:
        data (array_like): The input audio signal.
        fs (int): The sampling frequency of the audio signal.
        cutoff (float): The cutoff frequency for high-pass filtering (default: 1.0 Hz).
        window_size (float): The window size (in seconds) for adaptive baseline correction (default: 0.5s).
        order (int): The order of the Butterworth filter (default: 4).

    Returns:
        array_like: The processed signal with a flattened baseline.
    """

    # Step 1: High-Pass Filter to Remove Low-Frequency Drift
    sos = butter(order, cutoff, btype='high', fs=fs, output='sos')
    filtered_data = sosfiltfilt(sos, data)

    # Step 2: Compute Rolling Mean (Adaptive Baseline Correction)
    window_samples = int(window_size * fs)
    if window_samples % 2 == 0:
        window_samples += 1  # Ensure odd window size for symmetrical smoothing

    baseline = np.convolve(filtered_data, np.ones(window_samples) / window_samples, mode='same')

    # Step 3: Subtract Adaptive Baseline
    corrected_data = filtered_data - baseline

    # Step 4: Normalize to [-1, 1] for Visualization
    corrected_data = corrected_data / np.max(np.abs(corrected_data))

    return corrected_data


# Example usage
#%%
# Example usage
data = df_['ch3'].copy().values
data = data.astype(np.float32)

oldmax = np.nanmax(data)
oldmin = np.nanmin(data)
newmin=-1
newmax=1

normalized_signal = normalize_audio(data)
#normalized_signal = convert_range(data,
#                            newmin=-1, newmax=1,
#                            oldmax=np.nanmax(data), oldmin=np.nanmin(data))  # Default [-1, 1] range
# Apply baseline flattening
flattened_signal = flatten_baseline(normalized_signal, fs)

plt.figure()
#plt.plot(data)
plt.plot(normalized_signal[:1000])
plt.plot(flattened_signal[:1000])

# %%

actions_fpath = glob.glob(os.path.join(acqdir, '*-actions.mat'))[0]
actions = util.ft_actions_to_bout_df(actions_fpath)
actions

# set 0-index
actions['start'] = actions['start'] - 1
actions['end'] = actions['end'] - 1

# %%
video_dur = 600 # sec
video_fps = 60
video_nsec = video_fps * video_dur
video_time = np.linspace(0, video_dur, video_nsec)

len(video_time)

# %%

#%%
start_t = 600-60
end_t = 600

fig, axn = plt.subplots(n_channels, 1) #plt.figure()
for ai, ch in enumerate(channels): #df_.columns):
    ax=axn[ai]
    ax.plot(df_.loc[df_['time'].between(start_t, end_t)][ch], label=ch, lw=0.25, color='k')
    ax.set_title(ch, loc='left')
fig.suptitle('{}-{}'.format(round(start_t, 2), round(end_t, 2)))


# %%

outdir = os.path.join(acqdir, 'figures')
if not os.path.exists(outdir):
    os.makedirs(outdir)

#%%
channels = [c for c in df_.columns if c.startswith('ch')]

probability = 3 # definitely | 1=maybe, 2=probably, 3=definitely
we = actions[actions['action']=='unilateral extension'].copy()
evs = we[we['likelihood']>=probability]

#s_ = 35640 #35779-1
#e_ = 36000-1

for (s_, e_, bout, li) in evs[['start', 'end', 'boutnum', 'likelihood']].values:
    start_t = video_time[s_]
    end_t = video_time[e_]
    fig, axn = plt.subplots(n_channels, 1, sharex=True, sharey=True) #plt.figure()
    for ai, ch in enumerate(channels): #df_.columns):
        ax=axn[ai]
        start_t = video_time[s_]
        end_t = video_time[e_]
        ax.plot(df_.loc[df_['time'].between(start_t, end_t)][ch], 
                                label=ch, lw=0.25, color='k')
        ax.set_title(ch, loc='left')
    figtitle = 'bout {}\nframes {}-{} (sec {}={})'.format(bout, s_, e_, round(start_t, 2), round(end_t, 2))
    fig.suptitle(figtitle)

    figname = 'lk{}_bout{}'.format(li, bout)
    plt.savefig(os.path.join(outdir, '{}.png'.format(figname)))
    plt.close()



#%%

# BOUT 140 (last)
# CLACK followed by PULSE/Thud directly over *center* microphone

boutnum = 140
bout_start = 35865
bout_end = 35923
s_ = bout_start-100
e_ = min(bout_end + 100, len(video_time)-1)

start_t = video_time[s_]
end_t = video_time[e_]
fig, axn = plt.subplots(len(channels), 1, figsize=(6, 4.5),
                        sharex=True, sharey=True) #plt.figure()
for ai, ch in enumerate(channels): #df_.columns):
    ax=axn[ai]
    start_t = video_time[s_]
    end_t = video_time[e_]
    ax.plot(df_.loc[df_['time'].between(start_t, end_t)]['time'],
            df_.loc[df_['time'].between(start_t, end_t)][ch], 
                            label=ch, lw=0.25, color='k')
    ax.set_title(ch, loc='left')
    #ax.axvline(video_time[bout_start], color='r', linestyle='--')
    #ax.axvline(video_time[bout_end], color='r', linestyle='--')
    ax.axvspan(video_time[bout_start], video_time[bout_end], 
                   color='r', alpha=0.2)
plt.subplots_adjust(hspace=1)
fig.suptitle('bout {}\nframe {}-{}'.format(boutnum, s_, e_))

plt.savefig(os.path.join(outdir, 'bout140_red-UWE-over-mic.png'))

#%%
# BOUT: 139

# UNILATERAL over lower-right microphone: 
boutnum = 139

mic_pos = ['bot-r', 'mid-r', 'top-r', 'top-l', 'ctr', 'mid-r']
bout_starts = [27448, 27507, 27580, 27656, 27811, 27897]
bout_ends = [27474 ,27544, 27614, 27758, 27851, 27910]

s_ = min(bout_starts) -100
e_ = max(bout_ends) + 100 #, len(video_time)-1)

start_t = video_time[s_]
end_t = video_time[e_]
fig, axn = plt.subplots(len(channels), 1, figsize=(6,4.5),
                         sharex=True, sharey=True) #plt.figure()
for ai, ch in enumerate(channels): #df_.columns):
    ax=axn[ai]
    start_t = video_time[s_]
    end_t = video_time[e_]
    ax.plot(df_.loc[df_['time'].between(start_t, end_t)]['time'],
            df_.loc[df_['time'].between(start_t, end_t)][ch], 
                            label=ch, lw=0.25, color='k')
    ax.set_title(ch, loc='left')
    ylim = ax.get_ylim()
    for bout_start, bout_end, mpos in zip(bout_starts, bout_ends, mic_pos):
        # Plot transparent red rectange spanning the bout
        ax.axvspan(video_time[bout_start], video_time[bout_end], 
                   color='r', alpha=0.2)
        #ax.axvline(video_time[bout_start], color='r', linestyle='--')
        #ax.axvline(video_time[bout_end], color='r', linestyle='--')
        if ai==0: #len(channels)-1:
            ax.text(video_time[bout_start], ylim[-1]+1000, mpos)

    if ch=='ch1':
        ax.plot(video_time[27394], ylim[-1], 'b*')

plt.subplots_adjust(hspace=1)

fig.suptitle('bout {}\nframe {}-{}'.format(boutnum, s_, e_))

plt.savefig(os.path.join(outdir, 'bout139_red-UWE-over-mic.png'))

#%%
import transform_data.relative_metrics as rel
import cv2

#%%
#cap = rel.get_video_cap(acqdir, movie_fmt='mp4')
video_fpath = os.path.join(acqdir, '20250213-0915_fly1_Dyak-WT_3do_gh_2025-02-13-091817-0000.mp4')
cap = cv2.VideoCapture(video_fpath)
n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(n_frames, frame_width)
#cap = cv2.VideoCapture(vidpath)
# %%
ix = 50 #bout_starts[0]
cap.set(1, ix)
ret, im = cap.read()
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)
fig, ax = plt.subplots()
ax.imshow(im)

# %%
