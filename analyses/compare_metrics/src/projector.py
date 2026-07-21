#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
projector.py

Compare basic male kinematics across species (Dmel vs. Dyak) in the 2D-projector
CW/CCW calibration dataset, split by whether the male is courting/chasing.

DATASET (identical to analyses/calibration/src/test_cw_ccw.py):
    rootdir          = /Volumes/Juliana/Caitlin_RA_data/Caitlin_projector
    processedmat_dir = /Volumes/Juliana/2d_projector_analysis
                       /circle_diffspeeds_calibrated/FlyTracker/processed_mats
    protocol         = '40s_10_120_prj5ms'   (filtered from the metadata CSV)

The processed_mats directory holds one `<file_name>_df.parquet` per recording.
Each male is recorded TWICE - once with a clockwise (cw) and once with a
counter-clockwise (ccw) moving target - so there are two files per fly. The
parquet `acquisition` column already collapses the two directions into a single
per-fly identifier (e.g. '20260417_fly6_Dmel'), so grouping by `acquisition`
yields ONE value per fly for metrics that are not split by target direction.

Each frame is labeled courting vs non-courting using the manual `-actions.mat`
annotations (`courtship` column), then summarized per fly:

    - forward velocity of the male   (FlyTracker `vel`, mm/s)
    - angular velocity of the male   (FlyTracker `ang_vel`, plotted as |rad/s|)

Figures:
    Figure 1 - courting / chasing frames only
        1a. mean forward velocity per fly, Dmel vs. Dyak
        1b. mean |angular velocity| per fly, Dmel vs. Dyak
    Figure 2 - NON-courting / non-chasing frames (same two panels)

Written #%% cell-style (prototype interactively in the VSCode interactive
window, per the repo workflow); also runnable as a CLI:

    python projector.py [--rootdir PATH] [--processedmat-dir PATH]
                        [--protocol NAME] [--create-new] [--min-frames N]
"""
#%%
import os
import glob
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as spstats

import libs.utils as util
import libs.plotting as putil
import transform_data.relative_metrics as rel
from analyses.p1_levels.src import load_calibration_data as lcd


# ---------------------------------------------------------------------------
# Dataset — same as analyses/calibration/src/test_cw_ccw.py
# ---------------------------------------------------------------------------
# Raw video + manual -actions.mat annotations (Caitlin projector experiments)
DEFAULT_ROOTDIR = '/Volumes/Juliana/Caitlin_RA_data/Caitlin_projector'
# Per-recording processed parquets (FlyTracker -> relative metrics)
DEFAULT_PROCESSEDMAT_DIR = (
    '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_calibrated'
    '/FlyTracker/processed_mats'
)
# Output figures land here
DEFAULT_FIGDIR = (
    '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds_calibrated'
    '/FlyTracker/compare_metrics'
)
# Metadata protocol filter selecting the CW/CCW calibration recordings.
DEFAULT_PROTOCOL = '40s_10_120_prj5ms'
# Cached aggregate (all recordings, both flies, with courtship annotations).
CACHE_FNAME = '_calibrated_cw_ccw_df_all.parquet'

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Male is the focal fly (id=0) in the 2-fly projector assay.
MALE_ID = 0
# Per-fly identifier: 'acquisition' collapses the cw & ccw recordings of one
# male into a single value (e.g. '20260417_fly6_Dmel').
FLY_ID = 'acquisition'
# Column names in the processed projector dataframe.
FWD_VEL_VAR = 'vel'          # forward/translational speed (mm/s)
# FlyTracker `ang_vel` is already an UNSIGNED angular-speed magnitude (rad/s);
# the signed body-axis angular velocity is a separate column (`ang_vel_fly`).
ANG_VEL_VAR = 'ang_vel'

SPECIES_ORDER = ['Dmel', 'Dyak']
SPECIES_PALETTE = lcd.SPECIES_PALETTE   # {'Dmel': 'plum', 'Dyak': 'mediumseagreen'}

# Figure layout
FIGSIZE = (5, 3)       # per-panel size in inches (width, height); panels are square
MIN_FONTSIZE = 7


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _find_actions_file(rootdir, fn):
    """Locate the manual -actions.mat for recording ``fn`` (nested or flat)."""
    found = glob.glob(os.path.join(rootdir, fn, '*', '*-actions.mat'))
    if len(found) == 0:
        found = glob.glob(os.path.join(rootdir, fn, fn, '{}-actions.mat'.format(fn)))
    if len(found) == 0:
        found = glob.glob(os.path.join(rootdir, fn, '{}-actions.mat'.format(fn)))
    return found


def load_projector_data(rootdir, processedmat_dir, protocol=DEFAULT_PROTOCOL,
                        create_new=False):
    """
    Load the CW/CCW projector dataset, one per-recording parquet per file.

    Mirrors analyses/calibration/src/test_cw_ccw.py: filters the metadata to
    the ``protocol`` recordings, loads each `<file_name>_df.parquet` from
    ``processedmat_dir`` (which already carries `species`, `acquisition`,
    `stim_direction`, `vel`, `ang_vel`, ...), then attaches the manual
    `courtship` label from each recording's `-actions.mat`.

    Parameters
    ----------
    rootdir : str
        Root directory with raw video/actions data and the metadata CSV.
    processedmat_dir : str
        Directory holding the `<file_name>_df.parquet` files.
    protocol : str
        Substring matched against the metadata `traj_in` column.
    create_new : bool
        Passed through to rel.load_processed_df (True forces a miss -> None).

    Returns
    -------
    df_all : pd.DataFrame
        Per-frame data for all recordings (both flies), with a `courtship`
        column. Each fly appears as two recordings (cw + ccw) that share one
        `acquisition` value.
    meta : pd.DataFrame
        Metadata filtered to the protocol recordings.
    errors : list
        (file_name, Exception) tuples for files that failed to load.

    Notes
    -----
    The concatenated, courtship-annotated dataframe is cached to
    ``<processedmat_dir>/<CACHE_FNAME>`` so subsequent runs skip the per-file
    actions parsing. Pass ``create_new=True`` to rebuild it.
    """
    meta_fpath = glob.glob(os.path.join(rootdir, '*.csv'))[0]
    meta0 = pd.read_csv(meta_fpath)
    meta = meta0[
        (meta0['tracked in matlab and checked for swaps'] == 1)
        & meta0['traj_in'].str.contains(protocol, na=False)
        & (meta0['speed_blocks_marked'] == 1)
    ].copy()
    acqs = meta['file_name'].unique()
    print("Protocol '{}': {} recordings".format(protocol, len(acqs)))

    # --- Use cached aggregate if present ---
    cache_path = os.path.join(processedmat_dir, CACHE_FNAME)
    if not create_new and os.path.exists(cache_path):
        print("Loading cached aggregate: {}".format(cache_path))
        df_all = pd.read_parquet(cache_path)
        print("  {} rows, {} unique flies".format(
            len(df_all), df_all[FLY_ID].nunique()))
        return df_all, meta, []

    df_list, errors = [], []
    for i, fn in enumerate(acqs):
        if i % 10 == 0:
            print('  Loading {}/{}: {}'.format(i, len(acqs), fn))
        try:
            df_ = rel.load_processed_df(processedmat_dir, acq=fn,
                                        create_new=False)
            if df_ is None:
                raise FileNotFoundError(
                    'No processed parquet for {}'.format(fn))

            # Attach manual courtship annotations (not stored in the parquet).
            found_actions = _find_actions_file(rootdir, fn)
            if len(found_actions) >= 1:
                actions = util.load_ft_actions(found_actions[:1], split_end=False)
                df_ = util.assign_action_frames_to_df(df_, actions)
            if 'courtship' not in df_.columns:
                # Recording with no annotated courtship bouts.
                df_['courtship'] = 0

            df_list.append(df_)
        except Exception as e:
            print('  Error {}: {}'.format(fn, e))
            errors.append((fn, e))

    if len(df_list) == 0:
        raise ValueError('No recordings loaded successfully')

    df_all = pd.concat(df_list, ignore_index=True)
    df_all['courtship'] = df_all['courtship'].fillna(0)
    n_flies = df_all[FLY_ID].nunique()
    print("Loaded {} recordings -> {} unique flies (cw+ccw collapsed)".format(
        len(df_list), n_flies))

    # --- Cache the annotated aggregate (atomic write) ---
    tmp_path = cache_path + '.tmp'
    df_all.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, cache_path)
    print("Saved aggregate to: {}".format(cache_path))

    return df_all, meta, errors


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize_per_acquisition(df_all, fwd_var=FWD_VEL_VAR, ang_var=ANG_VEL_VAR):
    """
    Per-fly mean male kinematics, split by courting vs non-courting.

    Grouping is by `acquisition` (= FLY_ID), which pools both the cw and ccw
    recordings of a male into a single value. This is what we want for metrics
    that are not split by target direction: ONE value per fly.

    Parameters
    ----------
    df_all : pd.DataFrame
        Per-frame data (both flies). Must contain 'id', 'courtship',
        'species', 'acquisition', and the fwd/ang velocity columns.
    fwd_var, ang_var : str
        Column names for forward and angular velocity.

    Returns
    -------
    summary : pd.DataFrame
        One row per (species, acquisition, court_state) with columns:
        fwd_vel, ang_vel_abs, n_frames.
    """
    male = df_all[df_all['id'] == MALE_ID].copy()

    # Manual -actions courtship label: 1 = courting/chasing frame.
    male['court_state'] = np.where(
        male['courtship'] == 1, 'courting', 'non-courting')

    # `ang_vel` is already an unsigned turning magnitude; .abs() is a defensive
    # no-op (the signed angular velocity lives in `ang_vel_fly`).
    male['ang_vel_abs'] = male[ang_var].abs()

    # Group by FLY_ID so cw + ccw frames of the same male are pooled.
    summary = (
        male.groupby(['species', FLY_ID, 'court_state'])
        .agg(fwd_vel=(fwd_var, 'mean'),
             ang_vel_abs=('ang_vel_abs', 'mean'),
             n_frames=(fwd_var, 'size'))
        .reset_index()
    )
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _annotate_species_test(ax, data, metric, species_order):
    """Mann-Whitney U test between the two species; annotate p on the axis."""
    groups = [data.loc[data['species'] == sp, metric].dropna().values
              for sp in species_order]
    if len(groups) == 2 and all(len(g) > 0 for g in groups):
        try:
            _, p = spstats.mannwhitneyu(groups[0], groups[1],
                                        alternative='two-sided')
            ax.set_title('{}\nMWU p = {:.3g}'.format(ax.get_title(), p),
                         fontsize=7)
        except ValueError:
            pass


def species_stats_string(data, metric, label, species_order=SPECIES_ORDER):
    """
    Concise, publication-ready statistics for a two-species comparison of a
    per-fly metric (one value per fly), for parenthetical reporting.

    Because the metric is already averaged within each fly, each fly is a single
    independent observation. We describe the sample of flies with mean +/- SEM
    (the standard for reporting the precision of a group mean across independent
    units; SD would describe raw spread but is not what a group comparison
    conveys), and compare the two species with a Mann-Whitney U test.

    Parameters
    ----------
    data : pd.DataFrame
        Per-fly summary (already filtered to one courtship state).
    metric : str
        Column to summarize (e.g. 'fwd_vel').
    label : str
        Human-readable metric name with units (e.g. 'Forward velocity (mm/s)').
    species_order : list of str

    Returns
    -------
    str
        e.g. "Forward velocity (mm/s): Dmel 12.3 +/- 0.8 (n=10), "
             "Dyak 9.4 +/- 0.6 (n=12); MWU U=38, p=0.012".
    """
    groups = [data.loc[data['species'] == sp, metric].dropna().values
              for sp in species_order]
    parts = []
    for sp, g in zip(species_order, groups):
        if len(g) == 0:
            parts.append('{} n=0'.format(sp))
            continue
        m = np.mean(g)
        sem = spstats.sem(g) if len(g) > 1 else 0.0
        parts.append('{} {:.3g} +/- {:.2g} (n={})'.format(sp, m, sem, len(g)))
    desc = ', '.join(parts)

    if len(groups) == 2 and all(len(g) > 0 for g in groups):
        try:
            u, p = spstats.mannwhitneyu(groups[0], groups[1],
                                        alternative='two-sided')
            test = 'MWU U={:.0f}, p={:.3g}'.format(u, p)
        except ValueError:
            test = 'MWU n/a'
    else:
        test = 'MWU n/a'

    return '{}: {}; {}'.format(label, desc, test)


def plot_species_comparison(summary, court_state, species_order=SPECIES_ORDER,
                            palette=SPECIES_PALETTE, min_frames=0,
                            dot_color='k', panel_size=FIGSIZE):
    """
    Two-panel figure: forward velocity (left) and |angular velocity| (right),
    Dmel vs. Dyak, one point per acquisition, for a given courtship state.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of summarize_per_acquisition.
    court_state : str
        'courting' or 'non-courting'.
    species_order : list of str
    palette : dict
    min_frames : int
        Drop acquisitions with fewer than this many frames in the state.
    dot_color : color
        Color for the per-acquisition strip dots.
    panel_size : tuple (w, h)
        Size of each individual panel in inches.

    Returns
    -------
    fig, axn
    """
    sub = summary[(summary['court_state'] == court_state)
                  & (summary['n_frames'] >= min_frames)].copy()

    metrics = [('fwd_vel', 'Mean forward velocity (mm/s)'),
               ('ang_vel_abs', 'Mean |angular velocity| (rad/s)')]

    n_panels = len(metrics)
    panel_w, panel_h = panel_size
    fig, axn = plt.subplots(1, n_panels,
                            figsize=(panel_w * n_panels, panel_h))

    for ax, (metric, ylabel) in zip(axn, metrics):
        sns.boxplot(data=sub, x='species', y=metric, order=species_order,
                    hue='species', hue_order=species_order, palette=palette,
                    dodge=False, showfliers=False, width=0.5, ax=ax)
        if ax.legend_ is not None:
            ax.legend_.remove()
        sns.stripplot(data=sub, x='species', y=metric, order=species_order,
                      color=dot_color, size=4, alpha=0.7, jitter=True, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel)
        ax.set_box_aspect(1)
        _annotate_species_test(ax, sub, metric, species_order)

    n_by_sp = (sub.groupby('species')[FLY_ID].nunique()
               .reindex(species_order).fillna(0).astype(int))
    n_str = ', '.join('{}: n={}'.format(sp, n_by_sp[sp]) for sp in species_order)
    fig.suptitle('{} frames  ({})'.format(court_state, n_str), fontsize=9)
    fig.tight_layout()

    # Parenthetical stats (mean +/- SEM per species + MWU) below the panels.
    stats_lines = [species_stats_string(sub, metric, label, species_order)
                   for metric, label in metrics]
    fig.text(0.5, -0.02, '\n'.join(stats_lines), ha='center', va='top',
             fontsize=6, family='monospace')

    return fig, axn


# ---------------------------------------------------------------------------
# Config (defaults; overridden by CLI args when run as a script)
# ---------------------------------------------------------------------------
# parse_known_args lets the script also run in interactive / #%% mode where
# sys.argv contains ipykernel or VSCode runner arguments.
_p = argparse.ArgumentParser(
    description='Compare male kinematics across species (projector dataset).')
_p.add_argument('--rootdir', default=None,
                help='Raw data root (default: DEFAULT_ROOTDIR).')
_p.add_argument('--processedmat-dir', dest='processedmat_dir', default=None,
                help='Processed parquet dir (default: DEFAULT_PROCESSEDMAT_DIR).')
_p.add_argument('--protocol', default=DEFAULT_PROTOCOL,
                help='Metadata traj_in protocol filter (default: {}).'.format(
                    DEFAULT_PROTOCOL))
_p.add_argument('--create-new', action='store_true',
                help='Force rel.load_processed_df to miss the cache.')
_p.add_argument('--min-frames', type=int, default=0,
                help='Drop flies with fewer than N frames in a state.')
_args, _ = _p.parse_known_args()

rootdir = _args.rootdir or DEFAULT_ROOTDIR
processedmat_dir = _args.processedmat_dir or DEFAULT_PROCESSEDMAT_DIR
protocol = _args.protocol
create_new = _args.create_new
min_frames = _args.min_frames

plot_style = 'white'
putil.set_sns_style(plot_style, min_fontsize=MIN_FONTSIZE)
bg_color = [0.7] * 3 if plot_style == 'dark' else 'k'

figdir = DEFAULT_FIGDIR
os.makedirs(figdir, exist_ok=True)
_script = __file__ if '__file__' in globals() else 'projector.py'
figid = '{}\n{}'.format(_script, processedmat_dir)
print('Saving figures to: {}'.format(figdir))

#%%
# ---------------------------------------------------------------------------
# Load + summarize
# ---------------------------------------------------------------------------
df_all, meta, errors = load_projector_data(
    rootdir, processedmat_dir, protocol=protocol, create_new=create_new)

summary = summarize_per_acquisition(df_all)
print(summary.groupby(['species', 'court_state'])[FLY_ID].nunique())

#%%
# ---------------------------------------------------------------------------
# Figure 1 - courting / chasing frames
# ---------------------------------------------------------------------------
fig1, _ = plot_species_comparison(summary, 'courting', min_frames=min_frames,
                                   dot_color=bg_color)
putil.label_figure(fig1, figid)
fig1.savefig(os.path.join(figdir, 'fig1_courting_fwd-ang-vel_mel-v-yak.png'),
             bbox_inches='tight')

#%%
# ---------------------------------------------------------------------------
# Figure 2 - NON-courting / non-chasing frames
# ---------------------------------------------------------------------------
fig2, _ = plot_species_comparison(summary, 'non-courting', min_frames=min_frames,
                                   dot_color=bg_color)
putil.label_figure(fig2, figid)
fig2.savefig(os.path.join(figdir, 'fig2_non-courting_fwd-ang-vel_mel-v-yak.png'),
             bbox_inches='tight')

print('Done. Saved 2 figures to:\n  {}'.format(figdir))

# %%
