#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
from re import I
import sys
import glob
import importlib
import subprocess
import cv2

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import libs.utils as util
import libs.plotting as putil

#%%
def infer_led_blocks(led_onset, n_LED_blocks, led_block_nframes, curr_leds):
    '''
    Infer LED block starts.
    Level 0 starts at frame 0, and level 1 starts at led_onset.

    Arguments:
        led_onset -- frame number of LED onset
        n_LED_blocks -- number of LED blocks
        led_block_nframes -- number of frames per LED block

    Returns:
        led_blocks -- DataFrame with LED block starts and ends
    '''
    led0_start = int(led_onset) - led_block_nframes
    if led0_start < 0:
        led0_start = 0
    led_starts = [led0_start]
    for i in range(1, n_LED_blocks):
        led_starts.append(int(led_onset) + (i - 1) * led_block_nframes)
    led_starts = np.array(led_starts, dtype=int)

    led_ends = led_starts + led_block_nframes - 1
    # Replace first led_end with actual led_stat
    led_ends[0] = led_onset - 1
    led_ends[-1] = total_nframes #- 1

    led_blocks = pd.DataFrame({
        'led_level': np.arange(n_LED_blocks, dtype=int),
        'led_start': led_starts,
        'led_end': led_ends,
    })

    # add intensity of each led block
    led_blocks.loc[led_blocks['led_level']==0, 'led_intensity'] = 0
    for l in range(1, n_LED_blocks):
        led_blocks.loc[led_blocks['led_level']==l, 'led_intensity'] = curr_leds[l-1]

    return led_blocks


def overlap_len(start_a, end_a, start_b, end_b):
    """
    Inclusive overlap length between intervals [start_a, end_a] and [start_b, end_b].
    """
    lo = max(start_a, start_b)
    hi = min(end_a, end_b)
    return max(0, hi - lo + 1)


def build_speed_blocks(
    led_blocks,
    speed_onset_df,
    n_speed_blocks,
    speed_block_nframes,
    total_nframes,
    curr_speeds,
    mode='annotated',
):
    """
    Build speed blocks within each LED block.

    Parameters
    ----------
    curr_speeds : list
        Speed values (e.g. Hz) corresponding to speed levels 0..n_speed_blocks-1.
    mode : str
        'annotated' -> use all speed_block_onset start frames directly;
                       ends are next level start - 1.
        'inferred'  -> infer all speed blocks from LED block start using
                       fixed speed_block_nframes spacing.
    """
    assert mode in ['annotated', 'inferred'], "mode must be 'annotated' or 'inferred'"

    speed_rows = []
    #speed_onsets = speed_onset_df[speed_onset_df['action'] == 'speed_block_onset'].copy()
    speed_onsets = speed_onset_df.sort_values('start').reset_index(drop=True)

    if max(led_blocks['led_intensity']) < 50: # Low LED intensity for Dmel
        led_type = 'low_led'
    else:
        led_type = 'full_led'

    if max(curr_speeds) < 80: # Low speed for Dyak
        speed_type = 'slow_speed'
    else:
        speed_type = 'standard_speed'

    last_ix = 0
    for li, led_row in led_blocks.iterrows():
        led_level = int(led_row['led_level'])
        led_start = int(led_row['led_start'])
        led_end = int(led_row['led_end'])
        led_intensity = int(led_row['led_intensity'])

        # Just use manual, in case of jitter:
        assert len(speed_onsets) == 30, "Expected 30 speed_block_onset events, found {}".format(len(speed_onsets))
        if li == 0:
            indices = np.arange(last_ix, last_ix+n_speed_blocks-1)
            assert indices[-1] != speed_onsets.index.tolist()[-1], "Using the wrong final index"
            curr_onsets = speed_onsets.iloc[indices]['start'].astype(int).to_numpy()
            starts = [0]
            starts.extend(curr_onsets)
            starts = np.array(starts, dtype=int)
            ends = np.concatenate([starts[1:] - 1, [led_end]])
        else:
            indices = np.arange(last_ix, last_ix+n_speed_blocks)
            assert indices[-1] != speed_onsets.index.tolist()[-1], "Using the wrong final index"
            curr_onsets = speed_onsets.iloc[indices]['start'].astype(int).to_numpy()
            starts = curr_onsets
            ends = np.concatenate([starts[1:] - 1, [led_end]])
        assert len(curr_onsets) == len(indices), "Expected {} speed_block_onset events, found {}".format(len(indices), len(curr_onsets))
        # Update last index  
        last_ix = indices[-1] + 1

#         if mode == 'annotated':
#             curr_onsets = speed_onsets[
#                 (speed_onsets['start'].astype(int) >= led_start-1)
#                 & (speed_onsets['start'].astype(int) <= led_end)
#             ]['start'].astype(int).sort_values().to_numpy()
#             # If first block, 1 less because annotations start at first movement onset 
#             if li == 0:
#                 curr_n_speed_blocks = n_speed_blocks - 1
#             else:
#                 curr_n_speed_blocks = n_speed_blocks
#             assert len(curr_onsets) >= curr_n_speed_blocks, (
#                 f"LED block {led_level}, ({led_start}, {led_end}): expected at least {curr_n_speed_blocks} speed_block_onset "
#                 f"events, found {len(curr_onsets)}: {curr_onsets}"
#             )
#             # Use exactly n_speed_blocks onsets; extras at block edges are ignored.
#             if li==0:
#                 starts = [0]
#                 starts.extend(curr_onsets)
#             else:
#                 starts = curr_onsets #[:n_speed_blocks]
#             starts = np.array(starts, dtype=int)
#             # Ends: each block ends one frame before the next starts; last ends at led_end.
#             ends = np.concatenate([starts[1:] - 1, [led_end]])
#         else:
#             starts = np.array(
#                 [led_start + s * speed_block_nframes for s in range(n_speed_blocks)],
#                 dtype=int,
#             )
#             ends = starts + speed_block_nframes - 1
#             ends[-1] = min(led_end, int(total_nframes) - 1)
# 
        for s_level in range(n_speed_blocks):
            speed_rows.append({
                'led_level': led_level,
                'led_type': led_type,
                'speed_level': s_level,
                'speed_hz': curr_speeds[s_level],
                'speed_type': speed_type,
                'start': int(starts[s_level]),
                'end': int(ends[s_level]),
                'led_intensity': led_intensity,
                'speed_block_source': mode,
            })

    return pd.DataFrame(speed_rows)


def build_speed_blocks_cw_ccw(
    speed_onset_df,
    n_speed_blocks,
    speed_block_nframes,
    total_nframes,
    curr_speeds,
    led_intensity=0,
    led_onset_frame=1200, # 20s in 
    mode='annotated',
    led_level=1, #'',
    led_type = 'standard',
    speed_type='standard',
):
    """
    Build speed blocks within each LED block.

    Parameters
    ----------
    curr_speeds : list
        Speed values (e.g. Hz) corresponding to speed levels 0..n_speed_blocks-1.
    mode : str
        'annotated' -> use all speed_block_onset start frames directly;
                       ends are next level start - 1.
        'inferred'  -> infer all speed blocks from LED block start using
                       fixed speed_block_nframes spacing.
    """
    assert mode in ['annotated', 'inferred'], "mode must be 'annotated' or 'inferred'"

    speed_rows = []
    #speed_onsets = speed_onset_df[speed_onset_df['action'] == 'speed_block_onset'].copy()
    speed_onsets = speed_onset_df.sort_values('start').reset_index(drop=True)
# 
# 
#     last_ix = 0
#     for li, led_row in led_blocks.iterrows():
#         led_level = int(led_row['led_level'])
#         led_start = int(led_row['led_start'])
#         led_end = int(led_row['led_end'])
#         led_intensity = int(led_row['led_intensity'])
# 
#         # Just use manual, in case of jitter:
#         assert len(speed_onsets) == 30, "Expected 30 speed_block_onset events, found {}".format(len(speed_onsets))
#         if li == 0:
#             indices = np.arange(last_ix, last_ix+n_speed_blocks-1)
#             assert indices[-1] != speed_onsets.index.tolist()[-1], "Using the wrong final index"
#             curr_onsets = speed_onsets.iloc[indices]['start'].astype(int).to_numpy()
#             starts = [0]
#             starts.extend(curr_onsets)
#             starts = np.array(starts, dtype=int)
#             ends = np.concatenate([starts[1:] - 1, [led_end]])
#         else:
#             indices = np.arange(last_ix, last_ix+n_speed_blocks)
#             assert indices[-1] != speed_onsets.index.tolist()[-1], "Using the wrong final index"
#             curr_onsets = speed_onsets.iloc[indices]['start'].astype(int).to_numpy()
#             starts = curr_onsets
#             ends = np.concatenate([starts[1:] - 1, [led_end]])
 
    #for s_level in range(n_speed_blocks):
    # first add stationary baseline:
    speed_rows.append({
        'led_level': led_level,
        'led_type': led_type,
        'speed_level': 0,
        'speed_hz': curr_speeds[0],
        'speed_type': speed_type,
        'start': 0,
        'end': led_onset_frame-1,
        'led_intensity': led_intensity,
        'speed_block_source': mode,
        'led_onset_frame': led_onset_frame
    })
    for si, speed_onset in speed_onsets.iterrows():
        start = speed_onset['start']
        end = start + speed_block_nframes - 1
        s_level = si+1
        if si == len(speed_onsets)-1:
            break
        speed_rows.append({
            'led_level': led_level,
            'led_type': led_type,
            'speed_level': s_level,
            'speed_hz': curr_speeds[s_level],
            'speed_type': speed_type,
            'start': int(start),
            'end': int(end),
            'led_intensity': led_intensity,
            'speed_block_source': mode,
            'led_onset_frame': led_onset_frame
        })

    return pd.DataFrame(speed_rows)



def count_courtship_frames(speed_blocks, actions_df, total_nframes):
    """
    Count courtship frames per LED/speed block using a frame-wise mask.
    Each courtship bout can span multiple blocks; frames are counted once per block.

    Parameters
    ----------
    speed_blocks : pd.DataFrame
        Output of build_speed_blocks.
    actions_df : pd.DataFrame
        Actions dataframe containing 'courtship' rows with 'start' and 'end'.
    total_nframes : int
        Total number of frames in the video.

    Returns
    -------
    pd.DataFrame
        Columns: led_level, speed_level, speed_hz, courtship_frames, courtship_frac.
    """
    courtship = actions_df[actions_df['action'] == 'courtship'].copy()
    courtship['start'] = courtship['start'].astype(int)
    courtship['end'] = courtship['end'].astype(int)

    courtship_mask = np.zeros(int(total_nframes), dtype=bool)
    for _, bout in courtship.iterrows():
        bstart = max(0, int(bout['start']))
        bend = min(int(total_nframes) - 1, int(bout['end']))
        if bend >= bstart:
            courtship_mask[bstart:bend + 1] = True

    counts = []
    for _, blk in speed_blocks.iterrows():
        bstart = int(blk['start'])
        bend = int(blk['end'])
        cframes = int(courtship_mask[bstart:bend + 1].sum())
        block_len = bend - bstart + 1
        led_onset_frame = int(blk['led_onset_frame'])
        # If block starts before LED onset, count courtship frames before LED onset separately
        # Add both blocks to counts
        cframes_before_led = 0
        block_len_before_led = 0
        if bstart < led_onset_frame:
            # There are 2 blocks
            blocks = [(bstart, led_onset_frame-1), (led_onset_frame, bend)]
            #print(blk['speed_hz'], blocks)
        else:
            blocks = [(bstart, bend)]
        for block in blocks:
            cframes = int(courtship_mask[block[0]:block[1] + 1].sum())
            block_len = block[1] - block[0] + 1
            is_led_off = block[0] <= led_onset_frame             
            counts.append({
                'led_level': int(blk['led_level']),
                'led_intensity': int(blk['led_intensity']),
                'led_type': blk['led_type'],
                'speed_type': blk['speed_type'],
                'speed_level': int(blk['speed_level']),
                'speed_hz': blk['speed_hz'],
                'courtship_frames': cframes,
                'courtship_frac': cframes / block_len if block_len > 0 else np.nan,
                'led_is_on': not(is_led_off)
            })

    return pd.DataFrame(counts)


#%%
plot_style='dark'
min_fontsize=12
putil.set_sns_style(plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'w'

#%%
rootdir = '/Volumes/Juliana/Caitlin_RA_data/Caitlin_projector'

meta_fpath = glob.glob(os.path.join(rootdir, '*.csv'))[0]
meta0 = pd.read_csv(meta_fpath)
meta0.head()

#%
# Get calibration data only
meta = meta0[
      (meta0['tracked in matlab and checked for swaps']==1)
    #& (meta0['calibration']==1)
    # Check if '40s_10_120_prj5ms' in traj_in column values
    & meta0['traj_in'].str.contains('40s_10_120_prj5ms')
    & (meta0['speed_blocks_marked']==1)
    #& ~(meta0['led_onset_frame'].isna())
    ].copy()
meta.shape

#%%
conds = ['species_male', 'age_male', 'days_on_retinol', 'speeds', 'intensity_light']
meta_counts = meta.groupby(conds)['file_name'].nunique()
meta_counts = meta_counts.reset_index()
#meta_counts.columns = conds + ['file_count']
print(meta_counts)

#%%
n_LED_blocks = 2
n_speed_blocks = 8

block_dur_sec = 40
dur_min =  n_speed_blocks * block_dur_sec #*n_LED_blocks * n_speed_blocks * block_dur_sec
print(dur_min)
fps = 60

total_nframes = fps*dur_min
print(total_nframes)

# block sizes (frames)
speed_block_nframes = int(round(block_dur_sec * fps))
led_block_nframes = int(round(n_speed_blocks * speed_block_nframes))
    #%

#%%
fn = meta['file_name'].unique()[0]
led_onset = 1200

# Get actions file
found_actions_paths = glob.glob(os.path.join(rootdir, fn, '*', '*-actions.mat'))
assert len(found_actions_paths)==1, 'Expected 1 actions file, found {}'.format(len(found_actions_paths))
# Load actions file
actions_df = util.load_ft_actions(found_actions_paths, split_end=False)
speed_onset_df = actions_df[actions_df['action']=='IR LED on'].copy()
speed_onset_df.loc[speed_onset_df['action']=='IR LED on', 'action'] = 'speed_block_onset'
speed_onset_df = speed_onset_df.sort_values(by='start')

led_intensity = meta[meta['file_name']==fn]['intensity_light'].values[0]
print(led_intensity)
curr_speeds = [0, 10, 20, 40, 60, 80, 100, 120]
speed_block_mode = 'annotated'

# %%
max_std_frames = 8 #10
courtship_counts_all = []
errors = []
missing_files = []
for fn in meta['file_name'].unique():
    #%
    if not os.path.exists(os.path.join(rootdir, fn)):
        print('ERR: file not found: {}'.format(fn))
        missing_files.append(fn)
        #continue
    print('Found file: {}'.format(fn))

    led_intensity = meta[meta['file_name']==fn]['intensity_light'].values[0]
    stim_dir = meta[meta['file_name']==fn]['stim_direction'].values[0]
    print(led_intensity, stim_dir)

    # Get actions file
    found_actions_paths = glob.glob(os.path.join(rootdir, fn, '*', '*-actions.mat'))
    assert len(found_actions_paths)==1, 'Expected 1 actions file, found {}'.format(len(found_actions_paths))
    # Load actions file
    actions_df = util.load_ft_actions(found_actions_paths, split_end=False)
    speed_onset_df = actions_df[actions_df['action']=='IR LED on'].copy()
    speed_onset_df.loc[speed_onset_df['action']=='IR LED on', 'action'] = 'speed_block_onset'
    speed_onset_df = speed_onset_df.sort_values(by='start')

    try:
        assert np.diff(speed_onset_df['start']).std() < max_std_frames, "Speed onset jitter is too high"
    except Exception as e:
        errors.append((fn, e))
    #%
    # --------------------------------------------------------------------------
    # 1) Infer LED block starts.
    #    Level 0 starts at frame 0, and level 1 starts at led_onset.
    # --------------------------------------------------------------------------
    # Wrap this in a simple function
#    curr_leds = [int(l) for l in meta[meta['file_name']==fn]['intensity_light'].values[0].split(',')]
#    led_blocks = infer_led_blocks(led_onset, 
#                    n_LED_blocks, led_block_nframes,
#                    curr_leds)
    # print(led_blocks)
    #%
    # --------------------------------------------------------------------------
    # 2) Build speed blocks.
    #    mode = 'annotated': use speed_block_onset timing from actions_
    #    mode = 'inferred' : use LED block timing only
    # --------------------------------------------------------------------------
    #curr_speeds = list(range(n_speed_blocks))  # replace with actual speed values if known
    #curr_speeds = [int(s) for s in meta[meta['file_name']==fn]['speeds'].values[0].split(',')]
    curr_speeds = [0, 10, 20, 40, 60, 80, 100, 120]
    speed_block_mode = 'annotated'
    try:
        speed_blocks = build_speed_blocks_cw_ccw(
            #led_blocks=led_blocks,
            speed_onset_df=speed_onset_df,
            n_speed_blocks=n_speed_blocks,
            speed_block_nframes=speed_block_nframes,
            total_nframes=total_nframes,
            curr_speeds=curr_speeds,
            mode=speed_block_mode,
            led_intensity=led_intensity,
            led_onset_frame=led_onset,
            #led_level=led_level,
            #speed_type=speed_type,
        )
    #print(speed_blocks.head())
    except Exception as e:
        print(f"Error building speed blocks for {fn}: {e}")
        errors.append((fn, e))
        continue
    #%
    # --------------------------------------------------------------------------
    # 3) Count courtship frames by overlap with each LED/speed block.
    # --------------------------------------------------------------------------
    courtship_counts = count_courtship_frames(speed_blocks, actions_df, total_nframes)
    #print(courtship_counts)
    courtship_counts['file_name'] = fn
    courtship_counts['species'] = 'Dyak' if 'yak' in fn else 'Dmel'
    age = meta[meta['file_name']==fn]['age_male'].values[0]
    ATR = meta[meta['file_name']==fn]['days_on_retinol'].values[0]
    courtship_counts['age'] = age
    courtship_counts['ATR'] = ATR
    courtship_counts['age-ATR'] = '-'.join([str(age), str(ATR)])
    courtship_counts['led_intensity'] = led_intensity
    courtship_counts['stim_dir'] = stim_dir
   
    courtship_counts_all.append(courtship_counts)

courtship_counts_all = pd.concat(courtship_counts_all)

#%%
print("Errors:")
for e in errors:
    print(e)

print("Missing files:")
for m in missing_files:
    print(m)

courtship_counts_all.reset_index(drop=True, inplace=True)
#%%
# Add some meta columns
courtship_counts_all['date'] = [int(a.split('-')[0]) for a in courtship_counts_all['file_name']]
# Find 'fly##' in file_name:
courtship_counts_all['fly_num'] = [int(a.split('fly')[1].split('_')[0]) for a in courtship_counts_all['file_name']]
courtship_counts_all['fly_id'] = [f'f{fnum}' for fnum in courtship_counts_all['fly_num']]

courtship_counts_all['acquisition'] = ['_'.join([str(a), b, c]) for a, b, c in courtship_counts_all[['date', 'fly_id', 'species']].values]

#courtship_counts_all[['file_name', 'acquisition']]
courtship_counts_all.groupby('species')['acquisition'].nunique()

# %%
# Get number of files for each age-ATR and include in legend
conds = ['species', 'age-ATR', 'led_type', 'speed_type']
age_counts = courtship_counts_all.groupby(conds)['file_name'].nunique()
age_counts = age_counts.reset_index()
age_counts.columns = conds + ['file_count']
print(age_counts)


# %%
# Overall courtship vs. LED speed (ignore conditions)
species_palette = {'Dmel': 'plum', 
                   'Dyak': 'mediumseagreen'} 
#for sp, df_ in courtship_counts_all.groupby('species'):
# Put counts of each species in the legenda
n_species = courtship_counts_all['species'].nunique()
species_counts = courtship_counts_all.groupby(['species', 'stim_dir'])['file_name'].nunique()
species_counts = species_counts.reset_index()
species_counts.columns = ['species', 'stim_dir', 'file_count']
print(species_counts)
plotd = courtship_counts_all.copy()
fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
for i, (    cdir, cdf_) in enumerate(plotd.groupby('stim_dir')):
    ax=axn[i]
    sns.lineplot(data=cdf_, ax=ax,
                x='speed_hz', y='courtship_frac', 
                hue='species', legend=0, 
                palette=species_palette)
    ax.set_title(f'Courtship by speed, stim_dir: {cdir}')
    ax.set_xlabel('Speed (Hz)')
    ax.set_ylabel('Courtship frames')
    # Custom legend 
    legend_handles = [mpl.lines.Line2D([0], [0], color=species_palette[sp], lw=4) for sp in sorted(species_counts['species'])]
    legend_labels = [f"{sp} ({species_counts[species_counts['species']==sp]['file_count'].values[0]})" for sp in sorted(species_counts['species'])]
    plt.legend(legend_handles, legend_labels, loc='lower left', bbox_to_anchor=(1, 1), 
        frameon=False, title='', fontsize=10)  
#plt.show()

#%%
# Check 0 speed:
plotd = courtship_counts_all[courtship_counts_all['speed_hz']==0].copy()
fig, ax = plt.subplots(figsize=(10, 5))
sns.stripplot(data=plotd, ax=ax,
    x='acquisition', y='courtship_frac', 
    hue='led_is_on',# palette=species_palette, dodge=False,
    legend=1)
# Rotate x-ticks to be vertical
ax.tick_params(axis='x', labelrotation=90)
ax.set_title('0 speed')
ax.set_ylabel('Fraction courtship')
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), 
    frameon=False, title='')  

#%
plotd[plotd['courtship_frac']>0.5]

#%%

# Look at distribution of fraction courtship for each acquisition
plotd = courtship_counts_all.copy()
fig, axn = plt.subplots(1, 2, figsize=(10, 5),
                        sharex=False, sharey=True)
for i, (sp, df_) in enumerate(plotd.groupby('species')):
    ax=axn[i]
    sns.stripplot(data=df_, ax=ax,
        x='acquisition', y='courtship_frac', 
        hue='speed_hz', palette='magma', dodge=False,
        legend=i==1)
    ax.set_title(f'{sp}')
    # Rotate x-ticks to be vertical
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Fraction courtship')
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), 
    frameon=False, title='speed (mm/s)')  
ax.set_ylabel('Fraction courtship')
plt.show()

plotd = courtship_counts_all.copy()
fig, axn = plt.subplots(1, 2, figsize=(10, 5),
                        sharex=False, sharey=True)
for i, (sp, df_) in enumerate(plotd.groupby('species')):
    ax=axn[i]
    sns.boxplot(data=df_, ax=ax,
        x='acquisition', y='courtship_frac')
        #hue='speed_hz', palette='magma', dodge=False,
        #legend=i==1)
    ax.set_title(f'{sp}')
    # Rotate x-ticks to be vertical
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('Fraction courtship')
plt.show()

#%%
acquisition_parentdir = rootdir
acqs = meta['file_name'].unique()
processedmat_dir = os.path.join(rootdir, 'processedmat')
create_new = True
processedmat_dir = '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_calibrated/FlyTracker/processed_mats'

from analyses.steering.src import gain_funcs as gf

#%%
# Transform data 
df0_all, errors = gf.transform_projector_data(acquisition_parentdir, acqs,
                                    processedmat_dir, movie_fmt='.avi',
                                    subdir=None, flyid1=0, flyid2=1,
                                    create_new=create_new, 
                                    reassign_acquisition_name=True)


#%%
# Assign stimulus direction from meta
for fn, df_ in df0_all.groupby('file_name'):
    currm = meta[meta['file_name']==fn]
    assert len(currm)>0, 'No meta data for {}'.format(fn)
    assert len(currm)==1, 'Multiple meta data for {}'.format(fn)
    stim_dir = currm['stim_direction'].unique()[0]
    df0_all.loc[df0_all['file_name']==fn, 'stim_direction'] = stim_dir

#%%
f1 = df0_all[df0_all['id']==0].copy()
#%%
# Assign "pr_direction" progressive or regressive:
# stim_direction is CCW, and target_position < 0: progressive
# stim_direciton is CCW, and target_position > 0: regressive
# is CW and > 0: progressive
# is CW and < 0: regressive
f1['pr_direction'] = None
f1.loc[(f1['stim_direction']=='ccw') & (f1['targ_pos_theta']<0), 'pr_direction'] = 'progressive'
f1.loc[(f1['stim_direction']=='ccw') & (f1['targ_pos_theta']>0), 'pr_direction'] = 'regressive'
f1.loc[(f1['stim_direction']=='cw') & (f1['targ_pos_theta']>0), 'pr_direction'] = 'progressive'
f1.loc[(f1['stim_direction']=='cw') & (f1['targ_pos_theta']<0), 'pr_direction'] = 'regressive'

#%%
f1['ang_vel_abs'] = np.abs(f1['ang_vel'])
f1 = util.shift_variables_by_lag(f1, file_grouper='file_name', lag=12)
f1['ang_vel_fly_shifted_abs'] = np.abs(f1['ang_vel_fly_shifted'])
f1['ang_vel_fly_shifted_deg'] = np.rad2deg(f1['ang_vel_fly_shifted'])
f1['ang_vel_fly_deg'] = np.rad2deg(f1['ang_vel_fly'])

#%%
# Add bins
f1['theta_error_deg'] = np.rad2deg(f1['theta_error'])
f1 = gf.bin_by_object_position(f1, start_bin=-180, end_bin=180, bin_size=20)
f1['binned_theta_error_num'] = pd.to_numeric(f1['binned_theta_error'], errors='coerce')

f1.reset_index(drop=True, inplace=True)

#%%
# 
# # Get average ang vel across bins
grouper = ['species', 'acquisition', 'binned_theta_error',
          'binned_theta_error_num', 'pr_direction']
yvar = 'ang_vel_fly_shifted_deg'
mean_f1 = f1.groupby(grouper)[yvar].mean().reset_index()

#%%
pr_palette = {'progressive': 'darkgreen', 'regressive': 'purple'}
lw=2
err = 'se'

# Plot
fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
for i, (sp, df_) in enumerate(f1.groupby('species')):
    ax=axn[i]
    #sns.lineplot(data=df_, ax=ax,
    #    x='binned_theta_error', y='ang_vel_fly_shifted_deg',
    #    hue='pr_direction', palette=pr_palette)
    sns.lineplot(data=df_[df_['binned_theta_error_num']<0], 
                x='binned_theta_error', y=yvar, ax=ax,
                hue='pr_direction', palette=pr_palette, 
                errorbar=err, marker='o', 
                markersize=0, markeredgewidth=0,
                err_style='bars', legend=0, lw=lw, err_kws={'linewidth': lw})
    sns.lineplot(data=df_[df_['binned_theta_error_num']>0], 
                x='binned_theta_error', y=yvar, ax=ax,
                hue='pr_direction', palette=pr_palette, 
                errorbar=err, marker='o', 
                markersize=0, markeredgewidth=0,
                err_style='bars', lw=lw, err_kws={'linewidth': lw},
                legend=i==1)
    ax.set_title(f'{sp}')
    ax.set_xlabel('Theta error (deg)')
    ax.set_ylabel('Ang vel (deg/s)')
    ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
    ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1),
                    frameon=False, title='', 
                    fontsize=min_fontsize)
ax.set_xlim([-180, 180])
ax.set_xticks(np.linspace(-180, 180, 9))

#%%

# %%
# Split by speed and age-ATR
g = sns.FacetGrid(courtship_counts_all, 
        col='speed_hz', row='age-ATR')
g.map_dataframe(sns.lineplot, x='led_intensity', y='courtship_frac',
                hue='species', palette=species_palette)
g.add_legend()
#%%

sp = 'Dyak'
yak_ages = ['3-3.0', '4-4.0', '5-5.0']
all_speeds = courtship_counts_all['speed_hz'].unique()
print(all_speeds)
speed_palette = dict(zip(sorted(all_speeds), sns.color_palette('viridis', n_colors=len(all_speeds))))

yakd = courtship_counts_all[
    (courtship_counts_all['species']=='Dyak')].copy()

#%%
sp = 'Dmel'
mel_ages = ['3-0.5', '3-1.0']
led_type = 'low_led'
mel = courtship_counts_all[
        (courtship_counts_all['species']=='Dmel')
      & (courtship_counts_all['led_type']==led_type)].copy()
tmp_mel = courtship_counts_all[
    (courtship_counts_all['species']=='Dmel') &
    (courtship_counts_all['age-ATR']=='2-0.5')].copy()
# Combine
mel = pd.concat([mel, tmp_mel])
mel_ages.append('2-0.5')
mel_ages = sorted(mel_ages)

#%%
# Average speed hz
# ------
sp = 'Dyak'
yakd = courtship_counts_all[
    (courtship_counts_all['species']==sp)].copy()
fig, axn = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 5))
for i, age_atr in enumerate(yak_ages):
    plotd = yakd[yakd['age-ATR']==age_atr].copy()
    # plot
    #for speed_, df_ in plotd.groupby('speed_type'):
    #ri = 0 if speed_ == 'slow_speed' else 1
    ax=axn[i]
    sns.lineplot(data=plotd, ax=ax,
        x='led_intensity', y='courtship_frac',
        hue='speed_hz', palette=speed_palette, 
        legend=0)
    ax.set_title(f'{sp} {age_atr}', fontsize=12)
    ax.set_box_aspect(1)
    # plot all the xtick labels
    xtick_labels = yakd['led_intensity'].unique()
    ax.set_xticks(xtick_labels)
    ax.set_xticklabels(xtick_labels, fontsize=10)
#Custom legend for speed_hz
legend_handles = [mpl.lines.Line2D([0], [0], color=speed_palette[s], lw=4) for s in sorted(all_speeds)]
legend_labels = [f'{s} Hz' for s in sorted(all_speeds)]
plt.legend(legend_handles, legend_labels, loc='lower left', bbox_to_anchor=(1, 1), 
    frameon=False, title='')  

#%%
# Dmel
sp = 'Dmel'
#ages = ['3-0.5', '3-1.0']
led_type = 'low_led'
mel_ages = sorted(mel_ages)
# plot
fig, axn = plt.subplots(1, 3, sharex=False, sharey=True, figsize=(10, 5))
for i, age_atr in enumerate(mel_ages):
    plotd = mel[mel['age-ATR']==age_atr].copy()
    # plot
    #ri = 0 if speed_ == 'slow_speed' else 1
    ax=axn[i]
    print(plotd['led_intensity'].unique())
    sns.lineplot(data=plotd, ax=ax,
        x='led_intensity', y='courtship_frac',
        hue='speed_hz', palette=speed_palette, 
        legend=0)#legend=(i==2) & (ri==1))
    ax.set_title(f'{sp} {age_atr}', fontsize=12)
    ax.set_box_aspect(1)
    # plot all the xtick labels
    xtick_labels = plotd['led_intensity'].unique()
    ax.set_xticks(xtick_labels)
    ax.set_xticklabels(xtick_labels, fontsize=10)

#Custom legend for speed_hz
legend_handles = [mpl.lines.Line2D([0], [0], color=speed_palette[s], lw=4) for s in sorted(all_speeds)]
legend_labels = [f'{s} Hz' for s in sorted(all_speeds)]
plt.legend(legend_handles, legend_labels, loc='lower left', bbox_to_anchor=(1, 1), 
    frameon=False, title='')  
plt.subplots_adjust(hspace=0.5)
sns.move_legend(ax, loc='lower left', bbox_to_anchor=(1, 1), 
    frameon=False, title='')  

#%%
# DONT split by speed:
# ------
sp = 'Dmel'
mel_ages = ['3-0.5', '3-1.0']
led_type = 'low_led'
mel = courtship_counts_all[
        (courtship_counts_all['species']=='Dmel')
      & (courtship_counts_all['led_type']==led_type)].copy()
mel_ages = sorted(mel['age-ATR'].unique())
#sp = 'Dmel'
#ages = ['3-0.5', '3-1.0']
#led_type = 'low_led'
#mel_ages = sorted(mel_ages)
# plot
fig, axn = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(10, 5))
for i, age_atr in enumerate(mel_ages):
    plotd = mel[mel['age-ATR']==age_atr].copy()
    # plot
    #ri = 0 if speed_ == 'slow_speed' else 1
    ax=axn[i]
    print(plotd['led_intensity'].unique())
    sns.lineplot(data=plotd, ax=ax,
        x='led_intensity', y='courtship_frac',
        #hue='speed_hz', palette=speed_palette, 
        legend=0)#legend=(i==2) & (ri==1))
    ax.set_title(f'{sp} {age_atr}', fontsize=12)
    ax.set_box_aspect(1)
    # plot all the xtick labels
    xtick_labels = plotd['led_intensity'].unique()
    ax.set_xticks(xtick_labels)
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.axvline(x=3, color=bg_color, linestyle='--')
# Do yak now
sp = 'Dyak'
fig, axn = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 5))
for i, age_atr in enumerate(yak_ages):
    plotd = yakd[yakd['age-ATR']==age_atr].copy()
    # plot
    #for speed_, df_ in plotd.groupby('speed_type'):
    #ri = 0 if speed_ == 'slow_speed' else 1
    ax=axn[i]
    sns.lineplot(data=plotd, ax=ax,
        x='led_intensity', y='courtship_frac',
        #hue='speed_hz', palette=speed_palette, 
        legend=0)#legend=(i==2) & (ri==1))
    ax.set_title(f'{sp} {age_atr}', fontsize=12)
    ax.set_box_aspect(1)
    # plot all the xtick labels
    xtick_labels = plotd['led_intensity'].unique()
    ax.set_xticks(xtick_labels)
    ax.set_xticklabels(xtick_labels, fontsize=10)
    #ax.axvline(x=3, color=bg_color, linestyle='--')
    # Plot line
    ax.axvline(x=20, color=bg_color, linestyle='--')
#%%

plotd = courtship_counts_all[
    (courtship_counts_all['species']=='Dyak')
    & (courtship_counts_all['age-ATR']=='5-5.0')].copy()

#fig, axn = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 5))
n_speeds = plotd['speed_hz'].nunique()
#speed_palette = dict(zip(sorted(plotd['speed_hz'].unique()), sns.color_palette('viridis', n_colors=n_speeds)))
file_palette = dict(zip(sorted(plotd['file_name'].unique()), sns.color_palette('colorblind', n_colors=len(plotd['file_name'].unique()))))
fig, axn = plt.subplots(n_speeds, 1, 
            sharex=True, sharey=True, figsize=(10, 10))
for i, (speed_, df_) in enumerate(plotd.groupby('speed_hz')):
    ax=axn[i]
    sns.lineplot(data=df_, ax=ax,
        x='led_intensity', y='courtship_frac',
        hue='file_name', palette=file_palette, #palette=speed_palette, 
        legend=i==0)
    ax.set_title(f'{speed_}')
    ax.set_box_aspect(1)

#%%
fn = '20260415-1606_fly10_Dyak-p1_5do_gh_5dR'
plotd = courtship_counts_all[
    (courtship_counts_all['species']=='Dyak')
    & (courtship_counts_all['file_name']==fn)].copy()
print(plotd.shape)

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=plotd, ax=ax,
    x='led_intensity', y='courtship_frac',
    hue='speed_hz', palette='viridis', legend=1)
ax.set_title(f'{fn}')
ax.set_box_aspect(1)

#%%
for age_atr in ages:
    plotd = mel[mel['age-ATR']==age_atr].copy()
    # plot
    ax=axn[i]
    sns.lineplot(data=plotd, ax=ax,
        x='led_intensity', y='courtship_frac',
        hue='speed_hz', palette='viridis', legend=i==2)
# %%

#%%
#conds = ['led_type', 'speed_type']
f = sns.FacetGrid(courtship_counts_all, 
row='led_type', col='speed_hz')
f.map_dataframe(sns.lineplot, 
            x='led_intensity', y='courtship_frac', 
            style='age-ATR', hue='species', 
            palette=species_palette)
f.add_legend()


#%%
xvar = 'led_level'
mel_led_type = 'low_led'
yak_led_type = 'full_led'

mel = courtship_counts_all[
    (courtship_counts_all['species']=='Dmel') &
    (courtship_counts_all['led_type']==mel_led_type)].copy()

yak = courtship_counts_all[
    (courtship_counts_all['species']=='Dyak') &
    (courtship_counts_all['led_type']==yak_led_type)].copy()

#for age_str, age_df in courtship_counts_all.groupby('age-ATR'):
#fig, ax = plt.subplots(figsize=(10, 5))
# Combine mel unique speed_hz and yak unique speed_hz
unique_speed_hz = sorted(list(set(mel['speed_hz'].unique()) | set(yak['speed_hz'].unique())))


#g = sns.FacetGrid(courtship_counts_all, col='speed_hz')
fig, axn = plt.subplots(len(unique_speed_hz), 1, 
            sharex=True, sharey=True, figsize=(4, 12))
fig.text(0.02, 0.95, f'mel: {mel_led_type}, yak: {yak_led_type}', fontsize=12)
for i, speed_hz in enumerate(unique_speed_hz):
    for sp, df_ in [('Dmel', mel), ('Dyak', yak)]:
        ax=axn[i]
        sns.lineplot(data=df_[df_['speed_hz']==speed_hz], 
                    ax=ax,
                    x=xvar, y='courtship_frac',
                    hue='species', style='age-ATR',
                    palette=species_palette,
                    legend=i==len(unique_speed_hz)-1)
        ax.set_title(f'speed: {speed_hz} Hz', loc='left')
#ax.set_box_aspect(1)
#ax.set_ylim([0, 1])
plt.subplots_adjust(hspace=0.5)
sns.move_legend(ax, loc='lower left', bbox_to_anchor=(1, 1), 
    frameon=False, title='')  
# %
print(mel.groupby('age-ATR')['file_name'].nunique()) # )
print(yak.groupby('age-ATR')['file_name'].nunique()) # 4
# Dmel:
# age-ATR
# 3-1.0    4

# Dyak:
# age-ATR
# 3-3.0    4
# 4-4.0    3
#%
#%% 
# Combine mel and yak df
comb = pd.concat([mel, yak])
comb.shape

# Get speeds where courtship level is min. > 0.4
# for a given LED level
min_frac = 0.5
mean_frac = comb.groupby(['species', 'speed_hz', 'led_intensity', 'age-ATR'],
                        as_index=False)['courtship_frac'].mean()
means_above_thr = mean_frac[mean_frac['courtship_frac']>=min_frac]

# %%
#comb[(comb['species']=='Dyak') & (comb['speed_hz']==20)].\
#    groupby('led_level')['courtship_frac'].mean()
# Plot best match?
incl_speeds = means_above_thr['speed_hz'].unique().tolist() #[40, 60 80]
plotd = comb[comb['speed_hz'].isin(incl_speeds)]

fig, ax = plt.subplots()
sns.lineplot(data=plotd, ax=ax,
            x='led_intensity', y='courtship_frac',
            hue='species', palette=species_palette)
ax.set_title('Courtship by LED and speed')
ax.set_xlabel('LED intensity')
ax.set_ylabel('Courtship frames')
plt.show()
# %%
