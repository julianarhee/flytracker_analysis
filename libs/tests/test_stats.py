"""Tests for libs/stats.py."""

import numpy as np
import pandas as pd
import pytest

import libs.stats as lstats


# ---------------------------------------------------------------------------
# get_R2_ols
# ---------------------------------------------------------------------------

class TestGetR2Ols:

    def test_perfect_linear_fit(self):
        df = pd.DataFrame({
            'x': np.arange(20, dtype=float),
            'y': np.arange(20, dtype=float) * 3.0 + 1.0,
        })
        lr, r2 = lstats.get_R2_ols(df, 'x', 'y')
        assert r2 > 0.999
        assert abs(lr.coef_[0] - 3.0) < 0.01

    def test_returns_lr_and_r2(self):
        df = pd.DataFrame({'x': [1.0, 2, 3, 4], 'y': [2.0, 4, 5, 8]})
        lr, r2 = lstats.get_R2_ols(df, 'x', 'y')
        assert hasattr(lr, 'coef_')
        assert 0 <= r2 <= 1

    def test_handles_nans(self):
        df = pd.DataFrame({
            'x': [1.0, np.nan, 3, 4, 5],
            'y': [2.0, 4, np.nan, 8, 10],
        })
        lr, r2 = lstats.get_R2_ols(df, 'x', 'y')
        assert np.isfinite(r2)


# ---------------------------------------------------------------------------
# add_multiple_comparisons
# ---------------------------------------------------------------------------

class TestAddMultipleComparisons:

    def test_adds_correction_columns(self):
        df = pd.DataFrame({
            'frequency': [1, 5, 10],
            'p_value': [0.01, 0.03, 0.5],
        })
        result = lstats.add_multiple_comparisons(df.copy())
        for col in ['p_fdr', 'sig_fdr', 'p_bonf', 'sig_bonf']:
            assert col in result.columns

    def test_bonferroni_more_conservative(self):
        df = pd.DataFrame({
            'frequency': [1, 2, 3],
            'p_value': [0.02, 0.03, 0.04],
        })
        result = lstats.add_multiple_comparisons(df.copy())
        assert (result['p_bonf'] >= result['p_fdr']).all()
