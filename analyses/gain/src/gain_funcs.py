#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#%%
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import libs.utils as util
import libs.plotting as putil
import transform_data.relative_metrics as rel

# %%

def transform_projector_data(acquisition_parentdir, acqs, processedmat_dir, 
                            movie_fmt='.avi',subdir=None, flyid1=0, flyid2=1,
                            create_new=False, reassign_acquisition_name=False):
    """
    Load transformed projector data for specified acquisitions.
    Assumes that acquisitions are in the acquisition_parentdir.
    Assumes that flytracker output is in the processedmat_dir.

    Arguments:
        acquisition_parentdir (str): Parent directory of acquisitions.
        acqs (list): List of acquisitions.
        processedmat_dir (str): Directory to save processed mats.
        movie_fmt (str): Movie format.
        subdir (str): Subdirectory of acquisitions.
        flyid1 (int): Flytracker male index.
        flyid2 (int): Flytracker female index.
        create_new (bool): Create new processed mats.
        reassign_acquisition_name (bool): Reassign acquisition name, TRUE if one data-fly has multiple files (like projector CW/CCW)

    Returns:
        df0 (pd.DataFrame): Processed data.
        errors (list): List of errors.
    """
    d_list = []
    errors = []
    for i, acq in enumerate(acqs):
        if i % 10 == 0:
            print('Processing {} of {}: {}'.format(i, len(acqs), acq))
        acq_dir = os.path.join(acquisition_parentdir, acq)
        try:
            # Load flytracker output
            calib, trk, feat = util.load_flytracker_data(acq_dir, filter_ori=True)
            # Transform data to relative coordinates
            df_ = rel.get_metrics_relative_to_focal_fly(acq_dir,
                                                    savedir=processedmat_dir,
                                                    movie_fmt='.avi',
                                                    mov_is_upstream=None,
                                                    flyid1=0, flyid2=1,
                                                    plot_checks=False,
                                                    create_new=create_new,
                                                    get_relative_sizes=False)
            if 'targ_centered_x' not in df_.columns:
                print(f"{acq} targ_centered_x missing, rerun relative_metrics.py ")
                # Rerun: get_metrics_relative_to_focal_fly
                df_ = rel.get_metrics_relative_to_focal_fly(acq_dir,
                                                    savedir=processedmat_dir,
                                                    movie_fmt='.avi',
                                                    mov_is_upstream=None,
                                                    flyid1=0, flyid2=1,
                                                    plot_checks=False,
                                                    create_new=True,
                                                    get_relative_sizes=False)
                if 'targ_centered_x' not in df_.columns:
                    raise ValueError(f"{acq} targ_centered_x STILL not in df_")
            #assert len(df_['targ_centered_x'].unique()) > 1, "Bad targ calculation"
        except Exception as e:
            errors.append((acq, e))
            print("ERROR: {}".format(e))
            continue
        df_['file_name'] = os.path.split(acq)[-1]
        # Get species from acquisition

        if 'species' not in df_.columns: 
            df_['species'] = util.get_species_from_acquisition_name(acq)
        if reassign_acquisition_name:
            df_['date_fly'] = ['_'.join([f.split('-')[0], f.split('_')[1]]) for f in df_['file_name']]
            df_['acquisition'] = ['_'.join([a, b]) for a, b in df_[['date_fly', 'species']].values]
        else:
            df_['acquisition'] = acq #os.path.split(acq)[-1] 
        d_list.append(df_)

    if len(d_list) == 0:
        error_summary = '\n'.join(f'  {acq}: {e}' for acq, e in errors)
        raise ValueError(
            f"No acquisitions loaded successfully ({len(errors)} failed):\n{error_summary}"
        )
    df0 = pd.concat(d_list)

    return df0, errors


def assign_paint_conditions(df0, meta):
    # Add all paint conditions
    for fn, df_ in df0.groupby('file_name'):
        currm = meta[meta['file_name']==fn]
        assert len(currm)>0, 'No meta data for {}'.format(fn)
        assert len(currm)==1, 'Multiple meta data for {}'.format(fn)
        #df0.loc[df0['file_name']==fn, 'stim_direction'] = currm['stim_direction'].values[0]
        stim_dir = currm['stim_direction'].unique()[0] #fn.split('_')[-1]
        df0.loc[df0['file_name']==fn, 'stim_direction'] = stim_dir
        df0.loc[df0['file_name']==fn, 'paint_coverage'] = currm['painted'].values[0]
        manipulation_ = currm['manipulation_male'].values[0]
        if manipulation_.startswith('no '):
            paint_side = 'none'
        elif manipulation_.startswith('left '):
            paint_side = 'left'
        elif manipulation_.startswith('right '):
            paint_side = 'right'
        elif manipulation_.startswith('both '):
            paint_side = 'both'
        df0.loc[df0['file_name']==fn, 'paint_side'] = paint_side 
    df0['date'] = [int(a.split('_')[0]) for a in df0['acquisition']]

    return df0

def load_transformed_data(parquet_path):
    """Load DataFrame from parquet. If not found, load from pkl and save as parquet for next time."""
    pkl_path = parquet_path.replace('.parquet', '.pkl') if parquet_path.endswith('.parquet') else parquet_path + '.pkl'
    if os.path.exists(parquet_path):
        print("Loading parquet from: {}".format(parquet_path))
        return pd.read_parquet(parquet_path)
    if os.path.exists(pkl_path):
        print("Loading pkl from: {}".format(pkl_path))
        df = pd.read_pickle(pkl_path)
        df.to_parquet(parquet_path, index=False)
        print("Saved parquet to: {}".format(parquet_path))
        return df
    raise FileNotFoundError("Neither {} nor {} found".format(parquet_path, pkl_path))

def get_heading_diff(expr, heading_var='integrated_heading', invert_heading=True):
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
        hd = util.circ_dist(v, i) #i, v) #* -1 # match matlab
        hdiffs.append(hd)
    heading_diff = np.array(hdiffs)

    # Make CW-positive (custom convention, so left->right is positive)
    if invert_heading:
        heading_diff_cw = -1*heading_diff
    else:
        heading_diff_cw = heading_diff

    # add 0
    heading_diffs = np.concatenate( ([0], heading_diff_cw) )
    
    return heading_diffs

def bin_x(chase_, xvar, start_bin = -180, end_bin=180, bin_size=20):
    '''
    Bin xvar by object position. Convert xvar to degrees.
    '''
    chase_['binned_{}'.format(xvar)] = pd.cut(chase_[xvar],
                                    bins=np.arange(start_bin, end_bin, bin_size),
                                    labels=np.arange(start_bin+bin_size/2, 
                                                     end_bin-bin_size/2, bin_size))
    return chase_


# Group by binned theta errors
def bin_by_object_position(chase_, start_bin = -180, end_bin=180, bin_size=20):
    '''
    Bin theta error by object position. Convert theta error to degrees.
    '''
    chase_['binned_theta_error'] = pd.cut(chase_['theta_error_deg'],
                                    bins=np.arange(start_bin, end_bin, bin_size),
                                    labels=np.arange(start_bin+bin_size/2,
                                            end_bin-bin_size/2, bin_size))    
    return chase_


# ----------------------------------------------------------------------
# Diagnostic plotting functions
# ----------------------------------------------------------------------
def find_chase_snippet(flydf, fps=60, snippet_dur_sec=3):
    """Find the longest contiguous chasing bout and return a snippet.

    Args:
        flydf: DataFrame for a single file with 'chasing' and 'frame' columns.
        fps: frames per second.
        snippet_dur_sec: max duration of snippet in seconds.

    Returns:
        stretch: DataFrame slice of the snippet.
        f_start, f_end: start and end frame numbers.
    """
    flydf = flydf.sort_values('frame').copy()
    flydf['sec'] = flydf['frame'] / fps
    flydf['chase_block'] = (flydf['chasing'] != flydf['chasing'].shift()).cumsum()
    chase_blocks = (flydf[flydf['chasing'] == 1]
                    .groupby('chase_block')['frame']
                    .agg(['min', 'max', 'count'])
                    .sort_values('count', ascending=False))
    best = chase_blocks.iloc[0]
    f_start = int(best['min'])
    f_end = min(f_start + int(snippet_dur_sec * fps), int(best['max']))
    stretch = flydf[(flydf['frame'] >= f_start) & (flydf['frame'] <= f_end)].copy()
    return stretch, f_start, f_end


def diagnostics_plot_timecourses(stretch, f_start, f_end, example_lag=6, fps=60,
                                bg_color='gray', title=None):
    """Plot time-course diagnostic: target pos, theta error, ori, ang_vel comparison.

    Args:
        stretch: DataFrame snippet with required columns.
        f_start, f_end: frame range.
        example_lag: lag in frames for shifted column name.
        fps: frames per second.
        bg_color: color for reference lines.
        title: optional plot title override.

    Returns:
        fig, axes
    """
    sec_vals = stretch['sec'].values
    t_norm = (stretch['frame'].values - f_start) / max(f_end - f_start, 1)

    stretch = stretch.copy()
    if 'theta_error_deg' not in stretch.columns:
        stretch['theta_error_deg'] = np.rad2deg(stretch['theta_error'])
    if 'targ_pos_theta_deg' not in stretch.columns:
        stretch['targ_pos_theta_deg'] = np.rad2deg(stretch['targ_pos_theta'])
    if 'ang_vel_fly_deg' not in stretch.columns:
        stretch['ang_vel_fly_deg'] = np.rad2deg(stretch['ang_vel_fly'])
    if 'ang_vel_deg' not in stretch.columns:
        stretch['ang_vel_deg'] = -1 * np.rad2deg(stretch['ang_vel'])
    if 'ori_deg' not in stretch.columns:
        stretch['ori_deg'] = np.rad2deg(stretch['ori'])

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    putil.colored_line(axes[0], sec_vals, stretch['targ_pos_theta_deg'].values,
                       t_norm, 'autumn')
    axes[0].set_ylabel('Target position (deg)')
    if title:
        axes[0].set_title(title)

    putil.colored_line(axes[1], sec_vals, stretch['theta_error_deg'].values,
                       t_norm, 'cool')
    axes[1].set_ylabel('Theta error (deg)')
    axes[1].axhline(0, color=bg_color, ls='--', lw=0.5)

    putil.colored_line(axes[2], sec_vals, stretch['ori_deg'].values,
                       t_norm, 'cool', lw=1)
    axes[2].set_ylabel('Orientation (deg)')

    axes[3].plot(stretch['sec'], stretch['ang_vel_deg'],
                 color='lime', lw=1, alpha=0.8, label='ang_vel (FlyTracker)')
    axes[3].plot(stretch['sec'], stretch['ang_vel_fly_deg'],
                 color='deepskyblue', lw=1, alpha=0.8, label='ang_vel_fly')
    axes[3].set_ylabel('Ang vel (deg/s)')
    axes[3].set_xlabel('Time (s)')
    axes[3].axhline(0, color=bg_color, ls='--', lw=0.5)
    axes[3].legend(frameon=False)
    plt.tight_layout()
    return fig, axes


def diagnostics_plot_timecourses_zoom(stretch, zoom_offset=1.0, zoom_dur=1.0,
                                      bg_color='gray'):
    """Plot a zoomed time-course showing individual frames.

    Args:
        stretch: DataFrame snippet with 'sec', 'ori_deg', 'ang_vel_fly_deg',
                 'ang_vel_deg' columns.
        zoom_offset: seconds from snippet start to begin zoom window.
        zoom_dur: duration of zoom window in seconds.
        bg_color: color for reference lines.

    Returns:
        fig, (ax_ori, ax_vel)
    """
    zoom_start = stretch['sec'].iloc[0] + zoom_offset
    zoom_end = zoom_start + zoom_dur
    zoom = stretch[(stretch['sec'] >= zoom_start) & (stretch['sec'] <= zoom_end)]
    zoom_sec = zoom['sec'].values

    fig, (ax_ori, ax_vel) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    ax_ori.plot(zoom_sec, zoom['ori_deg'], '-o', color='cyan', lw=1, ms=3)
    ax_ori.set_ylabel('Orientation (deg)')
    ax_ori.set_title(f'Zoomed: {zoom_start:.1f}–{zoom_end:.1f}s  (dots = individual frames)')

    ax_vel.plot(zoom_sec, zoom['ang_vel_fly_deg'], '-o', color='deepskyblue',
                lw=1, ms=3, label='ang_vel_fly')
    ax_vel.plot(zoom_sec, zoom['ang_vel_deg'], '-o', color='lime',
                lw=1, ms=3, label='ang_vel (FT, sign-flipped)')
    ax_vel.axhline(0, color=bg_color, ls='--', lw=0.5)
    ax_vel.set_ylabel('Ang vel (deg/s)')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.legend(frameon=False)
    plt.tight_layout()
    return fig, (ax_ori, ax_vel)


def diagnostics_plot_2d_traj_and_rel(stretch, stretch_targ, f_start, f_end,
                                 arrow_len=20, arrow_every=None,
                                 fly_cmap='cool', targ_cmap='autumn',
                                 title=None):
    """Plot 2D trajectory with orientation arrows and relative target position.

    Args:
        stretch: fly DataFrame with 'frame', 'pos_x', 'pos_y', 'ori',
                 'targ_rel_pos_x', 'targ_rel_pos_y'.
        stretch_targ: target DataFrame with 'frame', 'pos_x', 'pos_y'.
        f_start, f_end: frame range for time normalization.
        arrow_len: quiver arrow length.
        arrow_every: subsample rate (None = auto).
        fly_cmap, targ_cmap: colormaps.
        title: optional suptitle.

    Returns:
        fig, (ax_traj, ax_rel)
    """
    f_range = max(f_end - f_start, 1)
    t_norm = (stretch['frame'].values - f_start) / f_range
    t_norm_targ = (stretch_targ['frame'].values - f_start) / f_range

    if arrow_every is None:
        arrow_every = max(1, len(stretch) // 40)
    s_sub = stretch.iloc[::arrow_every]
    t_sub = t_norm[::arrow_every]

    fig, (ax_traj, ax_rel) = plt.subplots(1, 2, figsize=(12, 5))

    putil.plot_orientation_arrows(ax_traj, s_sub['pos_x'].values,
                                 s_sub['pos_y'].values, s_sub['ori'].values,
                                 t_sub, arrow_len=arrow_len, cmap=fly_cmap,
                                 width=0.006, zorder=2)
    ax_traj.scatter(stretch_targ['pos_x'], stretch_targ['pos_y'],
                    c=t_norm_targ, cmap=targ_cmap, s=4, zorder=2, label='target')
    ax_traj.plot(stretch['pos_x'].iloc[0], stretch['pos_y'].iloc[0],
                 'o', color='lime', ms=8, zorder=3, label='start')
    ax_traj.set_xlabel('x (px)')
    ax_traj.set_ylabel('y (px)')
    ax_traj.invert_yaxis()
    ax_traj.set_aspect('equal')
    ax_traj.legend(frameon=False, markerscale=2)
    ax_traj.set_title('2D trajectories')

    ax_rel.scatter(stretch['targ_rel_pos_x'], stretch['targ_rel_pos_y'],
                   c=t_norm, cmap=targ_cmap, s=6, zorder=2)
    ax_rel.plot(0, 0, 'o', color='cyan', ms=8, zorder=3, label='fly (origin)')
    ax_rel.arrow(0, 0, 0, stretch['targ_rel_pos_y'].std() * 0.5,
                 head_width=0.5, head_length=0.3, fc='cyan', ec='cyan', zorder=3)
    ax_rel.set_xlabel('Relative x')
    ax_rel.set_ylabel('Relative y (heading direction)')
    ax_rel.set_aspect('equal')
    ax_rel.legend(frameon=False)
    ax_rel.set_title('Target relative to fly')

    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig, (ax_traj, ax_rel)

