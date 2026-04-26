"""Turn-bout detection, cross-correlation, and pursuit-specific plotting utilities.

Migrated from libs/theta_error.py (Step 2 of reorganization).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import libs.utils as util
import libs.plotting as putil


# ---------------------------------------------------------------------------
# Turn-bout detection
# ---------------------------------------------------------------------------

def get_window_centered_bouts(turn_start_frames, flydf, nframes_win):
    """Extract time windows centered on each turn-start frame.

    Args:
        turn_start_frames: frame indices to center windows around.
        flydf: DataFrame with focal fly data (must have 'sec' column).
        nframes_win: number of frames before and after each turn start.

    Returns:
        turnbouts: DataFrame with added columns 'turn_bout_num', 'rel_sec',
        and 'turn_start_frame'.
    """
    d_list = []
    for i, ix in enumerate(turn_start_frames):
        start_ = ix - nframes_win
        stop_ = ix + nframes_win
        t_onset = flydf.loc[ix]['sec']
        d_ = flydf.loc[start_:stop_].copy()
        d_['turn_bout_num'] = i
        d_['rel_sec'] = (
            d_['sec'].interpolate().ffill().bfill().astype(float) - t_onset
        ).round(2)
        d_['turn_start_frame'] = ix
        if len(d_['rel_sec'].values) < 13:
            print("bad len")
            break
        d_list.append(d_)

    turnbouts = pd.concat(d_list)
    return turnbouts


def get_turn_bouts(flydf, min_dist_to_other=15, min_ang_acc=100,
                   min_facing_angle=np.deg2rad(90),
                   min_vel=15, nframes_win=0.1 * 60, filter_dur=False):
    """Detect turn bouts from a focal-fly DataFrame.

    Filters for frames meeting angular acceleration, chasing, proximity,
    facing-angle, and velocity thresholds, then extracts centered windows.

    Returns:
        turnbouts DataFrame, or None if no bouts found.
    """
    passdf = flydf[
        (flydf['stim_hz'] > 0)
        & (flydf['ang_acc'] >= min_ang_acc)
        & (flydf['chasing_binary'] > 0)
        & (flydf['good_frames'] == 1)
        & (flydf['dist_to_other'] <= min_dist_to_other)
        & (flydf['facing_angle'] < min_facing_angle)
        & (flydf['vel'] > min_vel)
    ]
    high_ang_vel_bouts = util.get_indices_of_consecutive_rows(passdf)
    if len(high_ang_vel_bouts) == 0:
        print("No turn bouts! - {}".format(flydf['acquisition'].unique()))
        return None

    if filter_dur:
        min_bout_len = 2 / 60
        fps = 60.0
        incl_bouts = util.filter_bouts_by_frame_duration(
            high_ang_vel_bouts, min_bout_len, fps
        )
        print("{} of {} bouts pass min dur {}sec".format(
            len(incl_bouts), len(high_ang_vel_bouts), min_bout_len))
        turn_start_frames = [c[0] for c in incl_bouts]
    else:
        turn_start_frames = [c[0] for c in high_ang_vel_bouts]

    turnbouts = get_window_centered_bouts(turn_start_frames, flydf, nframes_win)
    return turnbouts


def select_turn_bouts_for_plotting(flydf, min_ang_acc=100, min_dist_to_other=25,
                                   min_facing_angle=np.deg2rad(90),
                                   min_vel=15, fps=60.0):
    """Select longer turn bouts (multiple turns) suitable for plotting.

    Unlike ``get_turn_bouts`` which returns individual turn-centered windows,
    this returns the start frames of duration-filtered bouts *and* all
    high-acceleration start frames for overlay.

    Returns:
        turn_bout_starts: list of start frames for filtered bouts.
        high_ang_start_frames: list of all high-acceleration start frames.
    """
    passdf = flydf[
        (flydf['stim_hz'] > 0)
        & (flydf['ang_acc'] > min_ang_acc)
        & (flydf['dist_to_other'] <= min_dist_to_other)
        & (flydf['facing_angle'] < min_facing_angle)
        & (flydf['vel'] > min_vel)
    ]
    high_ang_vel_bouts = util.get_indices_of_consecutive_rows(passdf)
    print(len(high_ang_vel_bouts))

    min_bout_len = 2 / fps
    incl_bouts = util.filter_bouts_by_frame_duration(
        high_ang_vel_bouts, min_bout_len, fps
    )
    print("{} of {} bouts pass min dur {}sec".format(
        len(incl_bouts), len(high_ang_vel_bouts), min_bout_len))

    turn_bout_starts = [c[0] for c in incl_bouts]
    high_ang_start_frames = [c[0] for c in high_ang_vel_bouts]
    print(len(high_ang_start_frames))
    return turn_bout_starts, high_ang_start_frames


def count_n_turns_in_window(flydf, turn_bout_starts, high_ang_start_frames,
                            fps=60.0):
    """Count the number of turns within a 2-second window around each bout start.

    Args:
        flydf: focal-fly DataFrame.
        turn_bout_starts: start frames of the longer bouts.
        high_ang_start_frames: all high-acceleration start frames.
        fps: frames per second.

    Returns:
        turn_counts DataFrame with columns: turn_ix, frame_num, n_turns,
        start_ix, stop_ix.
    """
    nframes_win = 2 * fps
    t_list = []
    for turn_ix, frame_num in enumerate(turn_bout_starts):
        start_ix = frame_num - nframes_win
        stop_ix = frame_num + nframes_win
        plotdf = flydf.loc[start_ix:stop_ix]
        n_turns = len(plotdf[plotdf['frame'].isin(high_ang_start_frames)])
        t_list.append(pd.DataFrame(
            {'turn_ix': turn_ix, 'frame_num': frame_num,
             'n_turns': n_turns, 'start_ix': start_ix, 'stop_ix': stop_ix},
            index=[turn_ix],
        ))
    turn_counts = pd.concat(t_list)
    return turn_counts


# ---------------------------------------------------------------------------
# Cross-correlation helpers
# ---------------------------------------------------------------------------

def cross_corr_each_bout(turnbouts, v1='facing_angle', v2='ang_vel', fps=60):
    """Compute cross-correlation for each turn bout.

    Args:
        turnbouts: DataFrame with 'turn_bout_num' column.
        v1, v2: column names for the two signals.
        fps: frames per second.

    Returns:
        xcorr: array (n_bouts x corr_length).
        all_lags: lag index arrays.
        t_lags: per-bout time lags in seconds.
    """
    xcorr = []
    t_lags = []
    all_lags = []
    for _i, d_ in turnbouts.groupby('turn_bout_num'):
        correlation, lags, _lag_frames, t_lag = util.cross_correlation_lag(
            d_[v2].interpolate().ffill().bfill(),
            d_[v1].interpolate().ffill().bfill(), fps=fps,
        )
        xcorr.append(correlation)
        all_lags.append(lags)
        t_lags.append(t_lag)

    return np.array(xcorr), np.array(all_lags), np.array(t_lags)


def get_turn_psth_values(plotdf, high_ang_start_frames, interval=10,
                         yvar1='facing_angle', yvar2='ang_vel',
                         nframes_win=0.1 * 60, fps=60):
    """Compute peri-stimulus time histogram values around each turn.

    For each turn, extracts a window, aligns time to turn onset, and
    computes cross-correlation.

    Returns:
        turns_: DataFrame with yvar1, yvar2, rel_sec, turn_ix, turn_frame.
        t_lags: array of per-turn time lags (seconds).
    """
    d_list = []
    t_lags = []
    for i, ix in enumerate(high_ang_start_frames[0::interval]):
        start_ = ix - nframes_win
        stop_ = ix + nframes_win
        t_onset = plotdf.loc[ix]['sec']
        d_ = plotdf.loc[start_:stop_].copy()
        if d_.shape[0] < (nframes_win * 2 + 1):
            continue
        d_['rel_sec'] = d_['sec'] - t_onset
        correlation, lags, lag_frames, t_lag = util.cross_correlation_lag(
            d_[yvar2].interpolate(),
            d_[yvar1].interpolate(), fps=fps,
        )
        d_[[yvar1, yvar2, 'rel_sec']] = (
            d_[[yvar1, yvar2, 'rel_sec']].interpolate().ffill().bfill()
        )
        d_['rel_sec'] = d_['rel_sec'].round(3)
        d_['turn_ix'] = i
        d_['turn_frame'] = ix
        d_list.append(d_[[yvar1, yvar2, 'rel_sec', 'turn_ix', 'turn_frame']])
        t_lags.append(t_lag)
    t_lags = np.array(t_lags)
    turns_ = pd.concat(d_list).reset_index(drop=True)
    return turns_, t_lags


def shift_vars_by_lag(flydf, high_ang_start_frames, med_lag, fps=60):
    """Shift stimulus variables by the median cross-correlation lag.

    At each turn frame, stimulus variables (theta_error, facing_angle, etc.)
    are read from ``med_lag`` seconds earlier while fly response variables
    are read at the original frame.

    Returns:
        shifted: DataFrame aligned by lag.
        unshifted: DataFrame at original frames.
    """
    lag_frames = np.round(med_lag * fps)

    unshifted_vars = [
        'ang_vel', 'ang_acc', 'turn_size', 'ang_vel_fly', 'ang_acc_fly',
        'ang_vel_fly_smoothed', 'ang_acc_fly_smoothed',
    ]
    shifted_vars = [
        'theta_error', 'theta_error_dt', 'facing_angle', 'facing_angle_vel',
        'facing_angle_vel_abs', 'facing_angle_acc',
        'theta_error_smoothed', 'theta_error_dt_smoothed',
    ]
    all_vars = unshifted_vars + shifted_vars

    orig_list = []
    d_list = []
    for f in high_ang_start_frames:
        fly_ = flydf.loc[f][unshifted_vars]
        targ_ = flydf.loc[f - lag_frames][shifted_vars].interpolate().ffill().bfill()
        d_ = pd.concat([fly_, targ_], axis=0)
        d_['turn_start_frame'] = f

        d2_ = flydf.loc[f][all_vars]
        d2_['turn_start_frame'] = f
        orig_list.append(d2_)
        d_list.append(d_)

    shifted = pd.concat(d_list, axis=1).T.astype(float)
    unshifted = pd.concat(orig_list, axis=1).T.astype(float)
    return shifted, unshifted


def aggregate_turns_across_flies(ftjaaba, v1='theta_error', v2='ang_vel_fly',
                                 min_n_turns=5, min_ang_acc=100, min_vel=15,
                                 min_dist_to_other=25,
                                 min_facing_angle=np.deg2rad(90),
                                 fps=60, nframes_win=0.1 * 60):
    """Detect turn bouts per acquisition, run cross-correlation, and aggregate.

    Returns:
        aggr_turns: concatenated DataFrame of all per-acquisition turn bouts
        with an added 'delta_t_lag' column.
    """
    no_turns = []
    few_turns = []
    t_list = []

    for acq, flydf in ftjaaba.groupby('acquisition'):
        turns_ = get_turn_bouts(
            flydf, min_ang_acc=min_ang_acc, min_vel=min_vel,
            min_dist_to_other=min_dist_to_other,
            min_facing_angle=min_facing_angle,
            nframes_win=nframes_win,
        )
        if turns_ is None:
            no_turns.append(acq)
            continue
        if turns_['turn_bout_num'].nunique() < min_n_turns:
            few_turns.append(acq)
            continue

        turns_ = turns_.reset_index(drop=True)
        xcorr, lags, t_lags = cross_corr_each_bout(turns_, v1=v1, v2=v2)

        for (turn_ix, t), l in zip(turns_.groupby('turn_bout_num'), t_lags):
            turns_.loc[t.index, 'delta_t_lag'] = l

        t_list.append(turns_)

    aggr_turns = pd.concat(t_list)
    return aggr_turns


def get_theta_errors_before_turns(aggr_turns, fps=60):
    """Get the theta error preceding each turn onset, shifted by mean lag.

    Returns:
        turn_starts: DataFrame with one row per turn onset and a
        'previous_theta_error' column.
    """
    t_list = []
    for _acq, d_ in aggr_turns.groupby('acquisition'):
        for _ti, t_ in d_.groupby('turn_bout_num'):
            mean_delta_t = d_['delta_t_lag'].mean()
            t_['previous_theta_error'] = t_['theta_error'].shift(
                int(mean_delta_t * fps)
            )
            t_list.append(
                t_[t_['frame'] == int(t_['turn_start_frame'].unique())]
            )
    turn_starts = pd.concat(t_list).reset_index(drop=True)
    return turn_starts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_individual_turns(turnbouts, v1='facing_angle', v2='ang_vel'):
    """Plot individual turn-bout traces for two variables side by side."""
    fig, axn = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
    ax = axn[0]
    sns.lineplot(data=turnbouts, x='rel_sec', y=v1, ax=ax,
                 hue='turn_bout_num', palette='Reds', legend=0,
                 lw=0.5, alpha=0.5)
    ax.set_ylabel(r'$\theta_{E}$')
    ax = axn[1]
    sns.lineplot(data=turnbouts, x='rel_sec', y=v2, ax=ax,
                 hue='turn_bout_num', palette='Blues', legend=0,
                 lw=0.5, alpha=0.5)
    ax.set_ylabel(r'$\omega_{f}$')
    plt.subplots_adjust(wspace=0.5, top=0.8)
    return fig


def plot_cross_corr_results(turnbouts, xcorr, lags, t_lags,
                            v1='facing_angle', v2='ang_vel',
                            v1_label=None, v2_label=None,
                            col1='r', col2='dodgerblue',
                            bg_color=[0.7] * 3,
                            fig_w=10, fig_h=5):
    """Plot mean aligned traces, cross-correlation, and lag histogram."""
    if v1_label is None:
        v1_label = r'$\theta_{E}$' + '\n{}'.format(v1)
    if v2_label is None:
        v2_label = r'$\omega_{f}$' + '\n{}'.format(v2)

    fig, axn = plt.subplots(1, 3, figsize=(fig_w, fig_h))
    ax1 = axn[0]
    ax2 = ax1.twinx()
    sns.lineplot(data=turnbouts, x='rel_sec', y=v1, ax=ax1, lw=0.5, color=col1)
    sns.lineplot(data=turnbouts, x='rel_sec', y=v2, ax=ax2, lw=0.5, color=col2)
    for ax, sp, lb, col in zip(
        [ax1, ax2], ['left', 'right'], [v1_label, v2_label], [col1, col2]
    ):
        ax.set_ylabel(lb)
        putil.change_spine_color(ax, col, sp)
    ax1.axvline(x=0, color=bg_color, linestyle='--')

    ax = axn[1]
    xcorr_mean = np.array(xcorr).mean(axis=0)
    xcorr_sem = np.array(xcorr).std(axis=0) / np.sqrt(len(xcorr))
    lags_mean = np.array(lags).mean(axis=0)
    ax.plot(lags_mean, xcorr_mean, color=bg_color)
    ax.fill_between(lags_mean, xcorr_mean - xcorr_sem,
                     xcorr_mean + xcorr_sem, color=bg_color, alpha=0.5)
    ax.set_ylabel('cross corr.')
    ax.set_xlabel('lag of {} relative to {}'.format(v2, v1))

    ax = axn[2]
    ax.hist(t_lags, bins=20, color=bg_color)
    med_lag = np.median(np.array(t_lags))
    ax.axvline(x=med_lag, color='w', linestyle='--')
    ax.set_title('Median lag: {:.2f}ms'.format(med_lag * 1000))
    ax.set_xlabel("lags (sec)")

    for ax in fig.axes:
        ax.set_box_aspect(1)
    plt.subplots_adjust(wspace=1)
    return fig


def compare_regr_pre_post_shift(flydf, shifted, x='facing_angle',
                                y='ang_vel', y1='turn_size',
                                x1='facing_angle_vel', lag_frames=2,
                                col=[0.7] * 3, markersize=5):
    """Regression plots of response on stimulus, before and after lag shift."""
    turn_start_frames = shifted['turn_start_frame'].values
    unshifted = flydf.loc[turn_start_frames].copy()

    scatter_kws = {'s': markersize}
    fig, axn = plt.subplots(2, 3, figsize=(15, 10))

    plotdf_shifted = shifted[
        (shifted['turn_size'] <= np.pi) & (shifted['turn_size'] >= -np.pi)
    ]
    plotdf_unshifted = unshifted[
        (unshifted['turn_size'] <= np.pi) & (unshifted['turn_size'] >= -np.pi)
    ]

    for row_idx, (plotd_, title_) in enumerate([
        (plotdf_unshifted.copy(), 'unshifted'),
        (plotdf_shifted.copy(), 'lag {}'.format(lag_frames)),
    ]):
        for ai, (x_, y_) in enumerate(zip([x, x, x1], [y, y1, y])):
            ax = axn[row_idx, ai]
            sns.regplot(data=plotd_, x=x_, y=y_, ax=ax, color=col,
                        scatter_kws=scatter_kws)
            ax.set_title(title_)
            putil.annotate_regr(plotd_, ax, x=x_, y=y_, fontsize=12)
            model_ = sm.OLS(plotd_[y_], sm.add_constant(plotd_[x_]))
            res_ = model_.fit()
            fit_str = 'OLS: y = {:.2f}x + {:.2f}, R2={:.2f}'.format(
                res_.params[1], res_.params[0], res_.rsquared)
            ax.text(0.1, 0.9, fit_str, transform=ax.transAxes)
            if 'dt' in x_ or 'vel' in x_:
                ax.set_xlabel(r'$\omega_{\theta_{E}}$' + '\n{}'.format(x_))
            else:
                ax.set_xlabel(r'$\theta_{E}$' + '\n{}'.format(x_))
            if 'turn_size' in y_:
                ax.set_ylabel(y_)
            else:
                ax.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(y_))

    for ax in axn.flat:
        ax.set_box_aspect(1)
    return fig


def plot_timecourses_for_turn_bouts(plotdf, high_ang_start_frames,
                                    xvar='sec',
                                    varset='varset2_smoothed',
                                    targ_color='r',
                                    fly_color='cornflowerblue',
                                    accel_color=[0.7] * 3):
    """Plot theta-error, angular velocity, and acceleration time courses.

    Three vertically stacked panels with twin y-axes.

    Args:
        varset: 'varset1' for FlyTracker outputs, 'varset2' for re-computed
            theta_error / ang_vel_fly, 'varset2_smoothed' for smoothed.
    """
    if 'varset2' in varset:
        smooth_sfx = '_smoothed' if 'smoothed' in varset else ''
        var1 = 'theta_error{}'.format(smooth_sfx)
        vel_var1 = 'theta_error_dt{}'.format(smooth_sfx)
        var2 = 'ang_vel_fly{}'.format(smooth_sfx)
        acc_var2 = 'ang_acc_fly{}'.format(smooth_sfx)
        center_yaxis = True
    else:
        var1 = 'facing_angle'
        vel_var1 = 'facing_angle_vel'
        var2 = 'ang_vel'
        acc_var2 = 'ang_acc'
        center_yaxis = False

    fig, axn = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    ax = axn[0]
    ax.plot(plotdf[xvar], plotdf[var1], targ_color)
    ax.set_ylabel(r'$\theta_{E}$' + '\n{}'.format(var1))
    putil.change_spine_color(ax, targ_color, 'left')
    ax.axhline(y=0, color=targ_color, linestyle='--', lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(plotdf[xvar], plotdf[var2], fly_color)
    ax2.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(var2))
    putil.change_spine_color(ax2, fly_color, 'right')
    if center_yaxis:
        curr_ylim = np.round(plotdf[var2].abs().max(), 0)
        ax2.set_ylim(-curr_ylim, curr_ylim)
    mean_ang_vel = np.mean(
        plotdf[plotdf['frame'].isin(high_ang_start_frames)][var2]
    )
    ax2.axhline(y=mean_ang_vel, color='w', linestyle='--', lw=0.5)
    ax2.axhline(y=-mean_ang_vel, color='w', linestyle='--', lw=0.5)

    ax = axn[1]
    ax.plot(plotdf[xvar], plotdf[vel_var1], targ_color)
    ax.set_ylabel(r'$\omega_{\theta_{E}}$' + '\n{}'.format(vel_var1))
    putil.change_spine_color(ax, targ_color, 'left')
    ax2 = ax.twinx()
    ax2.plot(plotdf[xvar], plotdf[var2], fly_color)
    ax2.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(var2))
    putil.change_spine_color(ax2, fly_color, 'right')

    ax = axn[2]
    ax.plot(plotdf[xvar], plotdf[acc_var2], fly_color)
    ax.set_ylabel(acc_var2)

    for f in high_ang_start_frames:
        ax.plot(plotdf.loc[plotdf['frame'] == f][xvar],
                plotdf.loc[plotdf['frame'] == f][acc_var2],
                color='w', marker='o', markersize=5)

    return fig


def plot_psth_all_turns(turns_, yvar1='theta_error', yvar2='ang_vel_fly',
                        col1='r', col2='cornflowerblue',
                        bg_color=[0.7] * 3,
                        ax1=None, ax2=None, lw_all=0.5, lw_mean=3):
    """Plot all individual turn traces and their mean for two variables."""
    if ax1 is None or ax2 is None:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

    for _i, d_ in turns_.groupby('turn_ix'):
        sns.lineplot(data=d_, x='rel_sec', y=yvar1, ax=ax1,
                     color=col1, lw=lw_all, alpha=0.5)
        sns.lineplot(data=d_, x='rel_sec', y=yvar2, ax=ax2,
                     color=col2, lw=lw_all, alpha=0.5)

    if turns_[yvar2].min() < -1 * turns_[yvar2].max():
        ylim2 = turns_[yvar2].abs().max()
        ax2.set_ylim([-ylim2, ylim2])
    if turns_[yvar1].min() < -1 * turns_[yvar1].max():
        ylim1 = turns_[yvar1].abs().max()
        ax1.set_ylim([-ylim1, ylim1])

    mean_turns_ = turns_.groupby('rel_sec').mean().reset_index()
    for xv, col, ax, var_ in zip(
        [yvar1, yvar2], [col1, col2], [ax1, ax2], [yvar1, yvar2]
    ):
        ax.plot(mean_turns_['rel_sec'], mean_turns_[var_], color=col, lw=lw_mean)

    ax1.set_ylabel(r'$\theta_{E}$' + '\n{}'.format(yvar1))
    ax2.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(yvar2))
    putil.change_spine_color(ax1, col1, 'left')
    putil.change_spine_color(ax2, col2, 'right')
    ax1.axvline(x=0, color=bg_color, linestyle='--')
    fig = ax1.get_figure()
    for ax in fig.axes:
        ax.set_box_aspect(1)
    return fig


def plot_mean_cross_corr_results(mean_turns_, correlation, lags, t_lags,
                                 t_lag=2,
                                 yvar1='theta_error', yvar2='ang_vel_fly',
                                 col1='r', col2='cornflowerblue',
                                 bg_color=[0.7] * 3):
    """Plot mean turn traces, cross-correlation, and lag distribution."""
    fig, axn = plt.subplots(1, 3, figsize=(16, 4))
    ax1 = axn[0]
    ax2 = ax1.twinx()
    for xv, col, ax in zip([yvar1, yvar2], [col1, col2], [ax1, ax2]):
        sns.lineplot(data=mean_turns_, x='rel_sec', y=xv, ax=ax,
                     color=col, lw=3)
    ax1.set_ylabel(r'$\theta_{E}$' + '\n{}'.format(yvar1))
    ax2.set_ylabel(r'$\omega_{f}$' + '\n{}'.format(yvar2))
    putil.change_spine_color(ax1, col1, 'left')
    putil.change_spine_color(ax2, col2, 'right')
    ax1.set_xlabel('time (sec)')
    ax1.set_title('mean bout vals')

    ax = axn[1]
    ax.plot(lags, correlation, c=bg_color)
    max1 = np.argmax(mean_turns_[yvar1])
    max2 = np.argmax(mean_turns_[yvar2])
    peak_diff_sec = (
        mean_turns_['rel_sec'].iloc[max2] - mean_turns_['rel_sec'].iloc[max1]
    )
    ax.set_ylabel('cross correlation')
    ax.set_xlabel('lag of {} rel. to {} (frames)'.format(yvar2, yvar1))
    ax.set_title('Peak diff: {:.2f}sec\nx-corr peak: {:.2f}msec'.format(
        peak_diff_sec, t_lag * 1E3))

    ax = axn[2]
    ax.hist(t_lags, color=bg_color)
    ax.set_title('Median lag: {:.2f}msec'.format(np.median(t_lags) * 1E3))
    ax.set_xlabel("lags (sec)")

    for ax in fig.axes:
        ax.set_box_aspect(1)
    plt.subplots_adjust(wspace=1)
    return fig
