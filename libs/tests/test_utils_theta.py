"""Tests for split_theta_error and count_chasing_frames in libs/utils.py."""

import numpy as np
import pandas as pd
import pytest

import libs.utils as util


# ---------------------------------------------------------------------------
# split_theta_error
# ---------------------------------------------------------------------------

class TestSplitThetaError:

    def test_labels_small_errors(self):
        df = pd.DataFrame({'theta_error': [0.0, 0.05, -0.05]})
        result = util.split_theta_error(df.copy(), theta_error_small=np.deg2rad(10))
        assert (result['error_size'] == 'small').all()

    def test_labels_large_errors(self):
        df = pd.DataFrame({'theta_error': [1.0, -1.0, 0.8]})
        result = util.split_theta_error(df.copy(), theta_error_large=np.deg2rad(25))
        assert (result['error_size'] == 'large').all()

    def test_medium_errors_are_none(self):
        df = pd.DataFrame({'theta_error': [0.3]})
        result = util.split_theta_error(
            df.copy(),
            theta_error_small=np.deg2rad(10),
            theta_error_large=np.deg2rad(25),
        )
        assert result['error_size'].iloc[0] is None

    def test_adds_error_size_column(self):
        df = pd.DataFrame({'theta_error': [0.0, 0.5, 1.0]})
        result = util.split_theta_error(df.copy())
        assert 'error_size' in result.columns


# ---------------------------------------------------------------------------
# count_chasing_frames
# ---------------------------------------------------------------------------

class TestCountChasingFrames:

    def test_basic_counts(self):
        df = pd.DataFrame({
            'species': ['A'] * 10,
            'acquisition': ['acq1'] * 10,
            'frame': range(10),
            'chasing_binary': [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
        })
        result = util.count_chasing_frames(df)
        assert result['n_frames_chasing'].iloc[0] == 4
        assert result['n_frames_total'].iloc[0] == 10
        assert abs(result['frac_frames_chasing'].iloc[0] - 0.4) < 1e-10

    def test_multiple_groups(self):
        df = pd.DataFrame({
            'species': ['A'] * 5 + ['B'] * 5,
            'acquisition': ['a1'] * 5 + ['b1'] * 5,
            'frame': list(range(5)) * 2,
            'chasing_binary': [1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        })
        result = util.count_chasing_frames(df)
        assert len(result) == 2

    def test_custom_chase_var(self):
        df = pd.DataFrame({
            'species': ['A'] * 5,
            'acquisition': ['a1'] * 5,
            'frame': range(5),
            'my_chase': [1, 0, 1, 0, 1],
        })
        result = util.count_chasing_frames(df, chase_var='my_chase')
        assert result['n_frames_chasing'].iloc[0] == 3
