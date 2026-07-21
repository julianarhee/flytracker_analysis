#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
free_behavior.py

Compare basic male AND female kinematics across species (Dmel vs. Dyak) in the
free-behavior 38mm dyad (male-female) dataset, split by whether the male is
courting/chasing. This is the free-behavior analog of ``projector.py`` (which
runs the same comparison on the 2D-projector CW/CCW calibration dataset).

DATASET (the '38mm_dyad_GG' assay in analyses/gain/src/steering_gain.py):
    Giacomo's free-behavior mel/yak recordings, transformed to relative metrics
    and aggregated into a single parquet:
        /Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker
            /transformed_data_GG.parquet

    Each recording is one male-female pair. FlyTracker `id` 0 = male, 1 = female;
    `acquisition` is the per-recording identifier (e.g.
    '20240112-1030-fly2-melWT_3do_sh_melWT_3do_gh'). Grouping by `acquisition`
    yields ONE value per pair.

Unlike the projector dataset, the parquet carries no manual `-actions.mat` or
JAABA courtship annotations, so courtship is derived kinematically from the
MALE's behavior (chasing OR unilateral wing extension / singing; see
COURTSHIP_* thresholds below, matching the kinematic gates in
analyses/strain_variation/src/strain_funcs.derive_courtship_labels). The male's
per-frame courtship label is then broadcast to the female of the same frame, so
"courting frames" mean the same set of frames for both sexes.

Per fly we summarize:
    - forward velocity            (FlyTracker `vel`, mm/s)
    - angular velocity magnitude  (FlyTracker `ang_vel`, plotted as |rad/s|)

Figures (one point per acquisition, Dmel vs. Dyak):
    Figure 1 - MALE,   courting / chasing frames        (fwd vel | |ang vel|)
    Figure 2 - MALE,   NON-courting frames              (fwd vel | |ang vel|)
    Figure 3 - FEMALE, courting frames (male courting)  (fwd vel | |ang vel|)
    Figure 4 - FEMALE, NON-courting frames              (fwd vel | |ang vel|)

Written #%% cell-style (prototype interactively in the VSCode interactive
window, per the repo workflow); also runnable as a CLI:

    python free_behavior.py [--parquet PATH] [--figdir PATH] [--min-frames N]
"""
#%%
import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as spstats

import libs.plotting as putil
from analyses.p1_levels.src import load_calibration_data as lcd


# ---------------------------------------------------------------------------
# Dataset — the '38mm_dyad_GG' assay (analyses/gain/src/steering_gain.py)
# ---------------------------------------------------------------------------
# Aggregated relative-metrics parquet (FlyTracker -> relative metrics).
DEFAULT_PARQUET = (
    '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker'
    '/transformed_data_GG.parquet'
)
# Output figures land here.
DEFAULT_FIGDIR = (
    '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker'
    '/compare_metrics'
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# FlyTracker ids in the male-female dyad.
MALE_ID = 0
FEMALE_ID = 1
# Per-pair identifier: one free-behavior recording per male-female pair.
FLY_ID = 'acquisition'
# Column names in the transformed dataframe.
FWD_VEL_VAR = 'vel'          # forward/translational speed (mm/s)
# FlyTracker `ang_vel` is already an UNSIGNED angular-speed magnitude (rad/s);
# the signed body-axis angular velocity is a separate column (`ang_vel_fly`).
ANG_VEL_VAR = 'ang_vel'

# Kinematic courtship gates applied to the MALE (radians for angles, mm for
# distance, mm/s for speed). Matches the kinematic gates used in
# analyses/strain_variation/src/strain_funcs.derive_courtship_labels.
COURTSHIP_CHASING = dict(min_vel=10.0, max_facing_angle=np.deg2rad(60),
                         max_dist_to_other=20.0)
COURTSHIP_SINGING = dict(min_wing_ang=np.deg2rad(30), max_facing_angle=np.deg2rad(90),
                         max_dist_to_other=35.0)

# Columns needed from the (large) parquet; keep this minimal to save memory.
LOAD_COLUMNS = ['id', 'species', FLY_ID, 'frame', FWD_VEL_VAR, ANG_VEL_VAR,
                'max_wing_ang', 'facing_angle', 'dist_to_other']

SPECIES_ORDER = ['Dmel', 'Dyak']
SPECIES_PALETTE = lcd.SPECIES_PALETTE   # {'Dmel': 'plum', 'Dyak': 'mediumseagreen'}

# Figure layout
FIGSIZE = (5, 3)       # per-panel size in inches (width, height); panels are square
MIN_FONTSIZE = 7


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _male_courtship_mask(male, chasing=COURTSHIP_CHASING, singing=COURTSHIP_SINGING):
    """Boolean per-frame courtship mask for the male (chasing OR singing)."""
    is_chasing = (
        (male[FWD_VEL_VAR] >= chasing['min_vel'])
        & (male['facing_angle'] <= chasing['max_facing_angle'])
        & (male['dist_to_other'] <= chasing['max_dist_to_other'])
    )
    is_singing = (
        (male['max_wing_ang'] >= singing['min_wing_ang'])
        & (male['facing_angle'] <= singing['max_facing_angle'])
        & (male['dist_to_other'] <= singing['max_dist_to_other'])
    )
    return (is_chasing | is_singing).fillna(False)


def load_free_behavior_data(parquet_path=DEFAULT_PARQUET, columns=LOAD_COLUMNS):
    """
    Load the free-behavior GG dataset and attach a per-frame courtship label.

    Courtship is derived from the MALE's kinematics (chasing OR singing) and
    broadcast to the female of the same (acquisition, frame), so both sexes
    share the same courting/non-courting frame partition.

    Parameters
    ----------
    parquet_path : str
        Path to the aggregated relative-metrics parquet.
    columns : list of str
        Columns to read (kept minimal because the parquet is large).

    Returns
    -------
    df_all : pd.DataFrame
        Per-frame data for both flies, with a `courtship` (0/1) column and a
        `sex` ('male'/'female') column.
    """
    print('Loading: {}'.format(parquet_path))
    df_all = pd.read_parquet(parquet_path, columns=columns)
    print('  {} rows, {} acquisitions ({})'.format(
        len(df_all), df_all[FLY_ID].nunique(),
        dict(df_all[[FLY_ID, 'species']].drop_duplicates()
             .groupby('species')[FLY_ID].count())))

    df_all['sex'] = np.where(df_all['id'] == MALE_ID, 'male', 'female')

    # Courtship label from the male, then broadcast to the female by frame.
    male = df_all[df_all['id'] == MALE_ID].copy()
    male_court = pd.DataFrame({
        FLY_ID: male[FLY_ID].values,
        'frame': male['frame'].values,
        'courtship': _male_courtship_mask(male).astype(int).values,
    })
    df_all = df_all.merge(male_court, on=[FLY_ID, 'frame'], how='left')
    df_all['courtship'] = df_all['courtship'].fillna(0).astype(int)

    return df_all


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize_per_acquisition(df_all, fly_id, fwd_var=FWD_VEL_VAR,
                              ang_var=ANG_VEL_VAR):
    """
    Per-acquisition mean kinematics for one sex, split by courting vs non-courting.

    Grouping is by `acquisition` (= FLY_ID): ONE value per recording (pair).

    Parameters
    ----------
    df_all : pd.DataFrame
        Per-frame data (both flies). Must contain 'id', 'courtship', 'species',
        'acquisition', and the fwd/ang velocity columns.
    fly_id : int
        MALE_ID (0) or FEMALE_ID (1) — which fly to summarize.
    fwd_var, ang_var : str
        Column names for forward and angular velocity.

    Returns
    -------
    summary : pd.DataFrame
        One row per (species, acquisition, court_state) with columns:
        fwd_vel, ang_vel_abs, n_frames.
    """
    fly = df_all[df_all['id'] == fly_id].copy()

    # Courtship is defined by the male; a frame is 'courting' for both sexes.
    fly['court_state'] = np.where(
        fly['courtship'] == 1, 'courting', 'non-courting')

    # `ang_vel` is already an unsigned turning magnitude; .abs() is a defensive
    # no-op (the signed angular velocity lives in `ang_vel_fly`).
    fly['ang_vel_abs'] = fly[ang_var].abs()

    summary = (
        fly.groupby(['species', FLY_ID, 'court_state'])
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


def plot_species_comparison(summary, court_state, sex, species_order=SPECIES_ORDER,
                            palette=SPECIES_PALETTE, min_frames=0,
                            dot_color='k', panel_size=FIGSIZE):
    """
    Two-panel figure: forward velocity (left) and |angular velocity| (right),
    Dmel vs. Dyak, one point per acquisition, for a given sex + courtship state.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of summarize_per_acquisition.
    court_state : str
        'courting' or 'non-courting'.
    sex : str
        'male' or 'female' (used only for titles/labels).
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
    fig.suptitle('{} {} frames  ({})'.format(sex, court_state, n_str), fontsize=9)
    fig.tight_layout()
    return fig, axn


# ---------------------------------------------------------------------------
# Config (defaults; overridden by CLI args when run as a script)
# ---------------------------------------------------------------------------
# parse_known_args lets the script also run in interactive / #%% mode where
# sys.argv contains ipykernel or VSCode runner arguments.
_p = argparse.ArgumentParser(
    description='Compare male & female kinematics across species '
                '(free-behavior 38mm dyad GG dataset).')
_p.add_argument('--parquet', default=None,
                help='Aggregated parquet path (default: DEFAULT_PARQUET).')
_p.add_argument('--figdir', default=None,
                help='Output figure dir (default: DEFAULT_FIGDIR).')
_p.add_argument('--min-frames', type=int, default=0,
                help='Drop flies with fewer than N frames in a state.')
_args, _ = _p.parse_known_args()

parquet_path = _args.parquet or DEFAULT_PARQUET
min_frames = _args.min_frames

plot_style = 'white'
putil.set_sns_style(plot_style, min_fontsize=MIN_FONTSIZE)
bg_color = [0.7] * 3 if plot_style == 'dark' else 'k'

figdir = _args.figdir or DEFAULT_FIGDIR
os.makedirs(figdir, exist_ok=True)
_script = __file__ if '__file__' in globals() else 'free_behavior.py'
figid = '{}\n{}'.format(_script, parquet_path)
print('Saving figures to: {}'.format(figdir))

#%%
# ---------------------------------------------------------------------------
# Load + summarize
# ---------------------------------------------------------------------------
df_all = load_free_behavior_data(parquet_path)

summary_male = summarize_per_acquisition(df_all, MALE_ID)
summary_female = summarize_per_acquisition(df_all, FEMALE_ID)
print('MALE frames per species/state:')
print(summary_male.groupby(['species', 'court_state'])[FLY_ID].nunique())
print('FEMALE frames per species/state:')
print(summary_female.groupby(['species', 'court_state'])[FLY_ID].nunique())

#%%
# ---------------------------------------------------------------------------
# Figure 1 - MALE, courting / chasing frames
# ---------------------------------------------------------------------------
fig1, _ = plot_species_comparison(summary_male, 'courting', 'male',
                                  min_frames=min_frames, dot_color=bg_color)
putil.label_figure(fig1, figid)
fig1.savefig(os.path.join(figdir, 'fig1_male_courting_fwd-ang-vel_mel-v-yak.png'),
             bbox_inches='tight')

#%%
# ---------------------------------------------------------------------------
# Figure 2 - MALE, NON-courting frames
# ---------------------------------------------------------------------------
fig2, _ = plot_species_comparison(summary_male, 'non-courting', 'male',
                                  min_frames=min_frames, dot_color=bg_color)
putil.label_figure(fig2, figid)
fig2.savefig(os.path.join(figdir, 'fig2_male_non-courting_fwd-ang-vel_mel-v-yak.png'),
             bbox_inches='tight')

#%%
# ---------------------------------------------------------------------------
# Figure 3 - FEMALE, courting frames (male courting)
# ---------------------------------------------------------------------------
fig3, _ = plot_species_comparison(summary_female, 'courting', 'female',
                                  min_frames=min_frames, dot_color=bg_color)
putil.label_figure(fig3, figid)
fig3.savefig(os.path.join(figdir, 'fig3_female_courting_fwd-ang-vel_mel-v-yak.png'),
             bbox_inches='tight')

#%%
# ---------------------------------------------------------------------------
# Figure 4 - FEMALE, NON-courting frames
# ---------------------------------------------------------------------------
fig4, _ = plot_species_comparison(summary_female, 'non-courting', 'female',
                                  min_frames=min_frames, dot_color=bg_color)
putil.label_figure(fig4, figid)
fig4.savefig(os.path.join(figdir, 'fig4_female_non-courting_fwd-ang-vel_mel-v-yak.png'),
             bbox_inches='tight')

print('Done. Saved 4 figures to:\n  {}'.format(figdir))

# %%
