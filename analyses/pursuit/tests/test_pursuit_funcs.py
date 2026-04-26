"""Tests for analyses/pursuit/src/pursuit_funcs.py."""

import numpy as np
import pandas as pd
import pytest

import analyses.pursuit.src.pursuit_funcs as pf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flydf():
    """Minimal focal-fly DataFrame with required columns for turn detection."""
    n = 200
    rng = np.random.default_rng(0)
    frames = np.arange(n)
    df = pd.DataFrame({
        'frame': frames,
        'sec': frames / 60.0,
        'stim_hz': np.ones(n) * 10,
        'ang_acc': rng.uniform(0, 200, n),
        'chasing_binary': np.ones(n),
        'good_frames': np.ones(n),
        'dist_to_other': rng.uniform(5, 30, n),
        'facing_angle': rng.uniform(0, np.pi, n),
        'vel': rng.uniform(10, 30, n),
        'ang_vel': rng.standard_normal(n),
        'ang_vel_fly': rng.standard_normal(n),
        'vel_fly': rng.uniform(0, 20, n),
        'acquisition': 'test_acq',
    })
    return df


@pytest.fixture
def turnbouts_simple():
    """Synthetic turnbouts DataFrame with 3 bouts, each 13 frames."""
    d_list = []
    for bout in range(3):
        frames = np.arange(13)
        d = pd.DataFrame({
            'rel_sec': (frames - 6) / 60.0,
            'turn_bout_num': bout,
            'facing_angle': np.sin(frames * 0.5) * 0.3,
            'ang_vel': np.cos(frames * 0.5) * 2.0,
            'frame': frames + bout * 20,
            'sec': (frames + bout * 20) / 60.0,
        })
        d_list.append(d)
    return pd.concat(d_list, ignore_index=True)


# ---------------------------------------------------------------------------
# get_window_centered_bouts
# ---------------------------------------------------------------------------

class TestGetWindowCenteredBouts:

    def test_returns_correct_columns(self, flydf):
        turn_starts = [50, 100]
        result = pf.get_window_centered_bouts(turn_starts, flydf, nframes_win=6)
        assert 'turn_bout_num' in result.columns
        assert 'rel_sec' in result.columns
        assert 'turn_start_frame' in result.columns

    def test_window_size(self, flydf):
        nframes_win = 6
        turn_starts = [50]
        result = pf.get_window_centered_bouts(turn_starts, flydf, nframes_win)
        assert len(result) == 2 * nframes_win + 1

    def test_multiple_bouts_numbered(self, flydf):
        turn_starts = [40, 80, 120]
        result = pf.get_window_centered_bouts(turn_starts, flydf, nframes_win=6)
        assert result['turn_bout_num'].nunique() == 3


# ---------------------------------------------------------------------------
# cross_corr_each_bout
# ---------------------------------------------------------------------------

class TestCrossCorrEachBout:

    def test_returns_three_arrays(self, turnbouts_simple):
        xcorr, lags, t_lags = pf.cross_corr_each_bout(
            turnbouts_simple, v1='facing_angle', v2='ang_vel'
        )
        assert isinstance(xcorr, np.ndarray)
        assert isinstance(lags, np.ndarray)
        assert isinstance(t_lags, np.ndarray)

    def test_one_lag_per_bout(self, turnbouts_simple):
        _, _, t_lags = pf.cross_corr_each_bout(
            turnbouts_simple, v1='facing_angle', v2='ang_vel'
        )
        n_bouts = turnbouts_simple['turn_bout_num'].nunique()
        assert len(t_lags) == n_bouts


# ---------------------------------------------------------------------------
# Plotting smoke tests
# ---------------------------------------------------------------------------

class TestPlotIndividualTurns:

    def test_returns_figure(self, turnbouts_simple):
        fig = pf.plot_individual_turns(
            turnbouts_simple, v1='facing_angle', v2='ang_vel'
        )
        import matplotlib.pyplot as plt
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)


class TestPlotPsthAllTurns:

    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        n = 50
        turns_ = pd.DataFrame({
            'rel_sec': np.tile(np.linspace(-0.1, 0.1, 10), 5),
            'turn_ix': np.repeat(range(5), 10),
            'theta_error': np.random.randn(n),
            'ang_vel_fly': np.random.randn(n),
        })
        fig = pf.plot_psth_all_turns(turns_)
        assert fig is not None
        plt.close(fig)
