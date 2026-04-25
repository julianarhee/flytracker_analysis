"""Tests for cross_correlation_lag and shift_variables_by_lag in libs/utils.py."""

import numpy as np
import pandas as pd
import pytest

import libs.utils as util


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_signals():
    """Two sine waves with a known lag (10 frames at 60 fps)."""
    fps = 60
    t = np.arange(0, 2, 1.0 / fps)  # 2 seconds
    lag_frames = 10
    x = np.sin(2 * np.pi * 2 * t)
    y = np.sin(2 * np.pi * 2 * (t - lag_frames / fps))
    return x, y, fps, lag_frames


@pytest.fixture
def shift_df():
    """DataFrame with two groups and monotonic angular-velocity values."""
    n = 30
    df = pd.DataFrame({
        'file_name': ['A'] * n + ['B'] * n,
        'frame': list(range(n)) * 2,
        'ang_vel_fly': np.arange(2 * n, dtype=float),
        'vel_fly': np.arange(2 * n, dtype=float) * 0.5,
        'vel': np.arange(2 * n, dtype=float) * 0.1,
        'ang_vel': np.arange(2 * n, dtype=float) * 2.0,
    })
    return df


# ---------------------------------------------------------------------------
# cross_correlation_lag
# ---------------------------------------------------------------------------

class TestCrossCorrelationLag:

    def test_returns_four_values(self, sine_signals):
        x, y, fps, _ = sine_signals
        result = util.cross_correlation_lag(x, y, fps=fps)
        assert len(result) == 4

    def test_peak_lag_magnitude_matches_known_shift(self, sine_signals):
        x, y, fps, expected_lag = sine_signals
        _, _, lag_frames, t_lag = util.cross_correlation_lag(x, y, fps=fps)
        assert abs(abs(lag_frames) - expected_lag) <= 1, (
            f"Expected |lag| ~{expected_lag}, got {lag_frames}"
        )
        assert abs(abs(t_lag) - expected_lag / fps) < 2 / fps

    def test_zero_lag_for_identical_signals(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        _, _, lag_frames, _ = util.cross_correlation_lag(x, x, fps=60)
        assert lag_frames == 0

    def test_correlation_length(self, sine_signals):
        x, y, fps, _ = sine_signals
        ccorr, lags, _, _ = util.cross_correlation_lag(x, y, fps=fps)
        assert len(ccorr) == len(lags)
        assert len(lags) == 2 * len(x) - 1


# ---------------------------------------------------------------------------
# shift_variables_by_lag
# ---------------------------------------------------------------------------

class TestShiftVariablesByLag:

    def test_new_columns_created(self, shift_df):
        result = util.shift_variables_by_lag(shift_df.copy(),
                                             file_grouper='file_name', lag=2)
        expected_cols = [
            'ang_vel_fly_shifted', 'vel_fly_shifted',
            'ang_vel_abs_shifted', 'vel_shifted',
            'vel_shifted_abs', 'ang_vel_shifted',
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_shift_values_correct(self, shift_df):
        lag = 3
        result = util.shift_variables_by_lag(shift_df.copy(),
                                             file_grouper='file_name', lag=lag)
        grp_a = result[result['file_name'] == 'A']
        original = shift_df[shift_df['file_name'] == 'A']['ang_vel_fly'].values
        shifted = grp_a['ang_vel_fly_shifted'].values
        n = len(original)
        np.testing.assert_array_equal(shifted[:n - lag], original[lag:])
        assert np.all(np.isnan(shifted[n - lag:]))

    def test_groups_independent(self, shift_df):
        """Shift in group A should not bleed into group B."""
        lag = 5
        result = util.shift_variables_by_lag(shift_df.copy(),
                                             file_grouper='file_name', lag=lag)
        grp_b = result[result['file_name'] == 'B']
        original_b = shift_df[shift_df['file_name'] == 'B']['ang_vel_fly'].values
        shifted_b = grp_b['ang_vel_fly_shifted'].values
        np.testing.assert_array_equal(shifted_b[:len(original_b) - lag],
                                      original_b[lag:])

    def test_lag_zero_is_identity(self, shift_df):
        result = util.shift_variables_by_lag(shift_df.copy(),
                                             file_grouper='file_name', lag=0)
        np.testing.assert_array_equal(
            result['ang_vel_fly_shifted'].values,
            result['ang_vel_fly'].values,
        )

    def test_abs_columns_are_absolute(self, shift_df):
        result = util.shift_variables_by_lag(shift_df.copy(),
                                             file_grouper='file_name', lag=2)
        assert (result['ang_vel_abs_shifted'].dropna() >= 0).all()
        assert (result['vel_shifted_abs'].dropna() >= 0).all()
