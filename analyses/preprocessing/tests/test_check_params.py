#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for preprocessing / check_params transformations.

Covers:
  - recompute_ang_vel: unwrapping, smoothing, head-tail flip handling
  - find_action_snippet: contiguous bout detection, snippet duration
  - diagnostics_plot_timecourses: figure creation and axis structure
  - diagnostics_plot_timecourses_zoom: zoomed view
  - diagnostics_plot_2d_traj_and_rel: trajectory + relative position
  - colored_line: LineCollection helper
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

import libs.utils as util
import libs.plotting as putil


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------
@pytest.fixture
def constant_ori_series():
    """Constant orientation — angular velocity should be ~0."""
    n = 120
    return pd.Series(np.full(n, 1.0))


@pytest.fixture
def linear_ori_series():
    """Linearly increasing orientation at 1 rad/frame → ang_vel ≈ fps."""
    n = 120
    return pd.Series(np.linspace(0, 2 * np.pi, n))


@pytest.fixture
def ori_with_flip():
    """Orientation series containing a single head-tail flip (~pi jump)."""
    n = 120
    ori = np.full(n, 1.0)
    ori[60:] += np.pi  # abrupt pi jump at frame 60
    return pd.Series(ori)


@pytest.fixture
def chasing_df():
    """DataFrame with two chasing bouts: a short one and a long one."""
    n = 300
    frames = np.arange(n)
    chasing = np.zeros(n, dtype=int)
    chasing[10:30] = 1    # 20-frame bout
    chasing[100:250] = 1  # 150-frame bout (longest)
    return pd.DataFrame({
        'frame': frames,
        'chasing': chasing,
        'sec': frames / 60.0,
        'ori': np.random.uniform(-np.pi, np.pi, n),
        'pos_x': np.cumsum(np.random.randn(n)) + 200,
        'pos_y': np.cumsum(np.random.randn(n)) + 200,
        'theta_error': np.random.uniform(-0.5, 0.5, n),
        'targ_pos_theta': np.random.uniform(-np.pi, np.pi, n),
        'ang_vel': np.random.randn(n) * 0.1,
        'ang_vel_fly': np.random.randn(n) * 0.1,
    })


@pytest.fixture
def stretch_and_targ(chasing_df):
    """Return a stretch snippet + matching target df for plotting tests."""
    stretch, f_start, f_end = util.find_action_snippet(chasing_df, action='chasing',
                                                        fps=60, snippet_dur_sec=2)
    n = len(stretch)
    targ = pd.DataFrame({
        'frame': stretch['frame'].values,
        'pos_x': stretch['pos_x'].values + 30,
        'pos_y': stretch['pos_y'].values + 30,
    })
    stretch = stretch.copy()
    stretch['targ_rel_pos_x'] = np.random.randn(n) * 5
    stretch['targ_rel_pos_y'] = np.random.randn(n) * 5 + 10
    return stretch, targ, f_start, f_end


# -------------------------------------------------------------------
# Tests: recompute_ang_vel
# -------------------------------------------------------------------
class TestRecomputeAngVel:

    def test_constant_ori_gives_zero_velocity(self, constant_ori_series):
        ang_vel = util.recompute_ang_vel(constant_ori_series, fps=60)
        assert len(ang_vel) == len(constant_ori_series)
        assert np.allclose(ang_vel, 0, atol=1e-10)

    def test_linear_ori_gives_constant_velocity(self, linear_ori_series):
        ang_vel = util.recompute_ang_vel(linear_ori_series, fps=60)
        # Middle portion avoids convolution edge effects
        mid = ang_vel.values[10:-10]
        assert np.std(mid) < 0.5, "angular velocity should be roughly constant"
        expected_rate = (2 * np.pi / (len(linear_ori_series) - 1)) * 60
        assert np.abs(np.mean(mid) - expected_rate) < 1.0

    def test_flip_is_unwrapped(self, ori_with_flip):
        ang_vel = util.recompute_ang_vel(ori_with_flip, fps=60)
        # After unwrapping with discont=pi/2, the pi-jump should be corrected
        # so ang_vel shouldn't have a huge spike
        assert np.all(np.abs(ang_vel) < 200), \
            "pi-jump should be unwrapped, no spike > 200 deg/s expected"

    def test_nan_handling(self):
        ori = pd.Series([0.5, np.nan, np.nan, 0.6, 0.7, 0.8, 0.9, 1.0])
        ang_vel = util.recompute_ang_vel(ori, fps=60)
        assert not np.any(np.isnan(ang_vel)), "NaNs should be interpolated away"

    def test_output_length_matches_input(self):
        ori = pd.Series(np.random.uniform(-np.pi, np.pi, 50))
        ang_vel = util.recompute_ang_vel(ori, fps=60)
        assert len(ang_vel) == 50

    def test_preserves_index(self):
        idx = pd.RangeIndex(start=100, stop=120, step=1)
        ori = pd.Series(np.linspace(0, 1, 20), index=idx)
        ang_vel = util.recompute_ang_vel(ori, fps=60)
        assert (ang_vel.index == idx).all()


# -------------------------------------------------------------------
# Tests: find_action_snippet
# -------------------------------------------------------------------
class TestFindActionSnippet:

    def test_finds_longest_bout(self, chasing_df):
        stretch, f_start, f_end = util.find_action_snippet(
            chasing_df, action='chasing', fps=60, snippet_dur_sec=10)
        # The longest bout is frames 100-249 (150 frames)
        assert f_start == 100

    def test_snippet_respects_duration_limit(self, chasing_df):
        dur = 1.0  # 1 second = 60 frames
        stretch, f_start, f_end = util.find_action_snippet(
            chasing_df, action='chasing', fps=60, snippet_dur_sec=dur)
        assert (f_end - f_start) <= int(dur * 60)

    def test_stretch_only_contains_requested_frames(self, chasing_df):
        stretch, f_start, f_end = util.find_action_snippet(
            chasing_df, action='chasing', fps=60, snippet_dur_sec=2)
        assert stretch['frame'].min() >= f_start
        assert stretch['frame'].max() <= f_end

    def test_sec_column_added(self, chasing_df):
        stretch, _, _ = util.find_action_snippet(
            chasing_df, action='chasing', fps=60, snippet_dur_sec=2)
        assert 'sec' in stretch.columns

    def test_raises_on_no_action_frames(self):
        df = pd.DataFrame({
            'frame': np.arange(100),
            'chasing': np.zeros(100, dtype=int),
        })
        with pytest.raises((IndexError, ValueError)):
            util.find_action_snippet(df, action='chasing', fps=60,
                                      snippet_dur_sec=2)

    def test_works_with_custom_action_column(self):
        n = 200
        df = pd.DataFrame({
            'frame': np.arange(n),
            'tracking': np.zeros(n, dtype=int),
        })
        df.loc[50:120, 'tracking'] = 1
        stretch, f_start, f_end = util.find_action_snippet(
            df, action='tracking', fps=60, snippet_dur_sec=10)
        assert f_start == 50


# -------------------------------------------------------------------
# Tests: diagnostic plot functions
# -------------------------------------------------------------------
class TestDiagnosticsPlotTimecourses:

    def test_creates_4_panel_figure(self, stretch_and_targ):
        stretch, _, f_start, f_end = stretch_and_targ
        fig, axes = putil.diagnostics_plot_timecourses(stretch, f_start, f_end)
        assert len(axes) == 4
        plt.close(fig)

    def test_returns_figure_and_axes(self, stretch_and_targ):
        stretch, _, f_start, f_end = stretch_and_targ
        fig, axes = putil.diagnostics_plot_timecourses(stretch, f_start, f_end)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_title_is_set(self, stretch_and_targ):
        stretch, _, f_start, f_end = stretch_and_targ
        fig, axes = putil.diagnostics_plot_timecourses(stretch, f_start, f_end,
                                                        title='Test Title')
        assert axes[0].get_title() == 'Test Title'
        plt.close(fig)


class TestDiagnosticsPlotTimecourseZoom:

    def test_creates_2_panel_figure(self, stretch_and_targ):
        stretch, _, f_start, f_end = stretch_and_targ
        stretch = stretch.copy()
        stretch['ori_deg'] = np.rad2deg(stretch['ori'])
        stretch['ang_vel_fly_deg'] = np.rad2deg(stretch['ang_vel_fly'])
        stretch['ang_vel_deg'] = -np.rad2deg(stretch['ang_vel'])
        fig, (ax1, ax2) = putil.diagnostics_plot_timecourses_zoom(stretch)
        assert ax1 is not None and ax2 is not None
        plt.close(fig)


class TestDiagnosticsPlot2dTrajAndRel:

    def test_creates_two_subplots(self, stretch_and_targ):
        stretch, targ, f_start, f_end = stretch_and_targ
        fig, (ax_traj, ax_rel) = putil.diagnostics_plot_2d_traj_and_rel(
            stretch, targ, f_start, f_end)
        assert ax_traj is not None
        assert ax_rel is not None
        plt.close(fig)

    def test_y_axis_inverted_on_trajectory(self, stretch_and_targ):
        stretch, targ, f_start, f_end = stretch_and_targ
        fig, (ax_traj, _) = putil.diagnostics_plot_2d_traj_and_rel(
            stretch, targ, f_start, f_end)
        assert ax_traj.yaxis_inverted()
        plt.close(fig)

    def test_equal_aspect_ratio(self, stretch_and_targ):
        stretch, targ, f_start, f_end = stretch_and_targ
        fig, (ax_traj, ax_rel) = putil.diagnostics_plot_2d_traj_and_rel(
            stretch, targ, f_start, f_end)
        assert ax_traj.get_aspect() in ('equal', 1.0)
        assert ax_rel.get_aspect() in ('equal', 1.0)
        plt.close(fig)


# -------------------------------------------------------------------
# Tests: colored_line (in libs/plotting.py)
# -------------------------------------------------------------------
class TestColoredLine:

    def test_adds_collection_to_axis(self):
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        c = np.linspace(0, 1, 50)
        lc = putil.colored_line(ax, x, y, c, 'viridis')
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_returns_line_collection(self):
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 20)
        y = np.cos(x)
        c = np.linspace(0, 1, 20)
        lc = putil.colored_line(ax, x, y, c, 'cool')
        from matplotlib.collections import LineCollection
        assert isinstance(lc, LineCollection)
        plt.close(fig)
