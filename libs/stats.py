"""Statistical helper functions.

Migrated from libs/theta_error.py (Step 3-6 of reorganization).
"""

import numpy as np
import pandas as pd
import scipy.stats as spstats


def get_R2_ols(plotd, xvar, yvar):
    """Fit a simple OLS linear regression and return the model and R-squared.

    Returns:
        lr: fitted sklearn LinearRegression object.
        r2: coefficient of determination.
    """
    from sklearn import linear_model
    X = plotd[xvar].interpolate().ffill().bfill().values
    y = plotd[yvar].interpolate().ffill().bfill().values
    lr = linear_model.LinearRegression()
    lr.fit(X.reshape(len(X), 1), y)
    r2 = lr.score(X.reshape(len(X), 1), y)
    return lr, r2


def mixed_anova_stats(stimhz_means, yvar, within='stim_hz', between='species',
                      subject='acquisition',
                      between1='Dmel', between2='Dyak'):
    """Mixed ANOVA with between and within factors plus Mann-Whitney U tests.

    Uses pingouin for the ANOVA and scipy for pairwise Mann-Whitney U tests
    at each level of the within factor.

    Returns:
        results_df: per-level Mann-Whitney U results.
        aov: pingouin ANOVA table.
    """
    import pingouin as pg
    aov = pg.mixed_anova(data=stimhz_means, dv=yvar,
                         within=within, between=between,
                         subject=subject)
    results = []
    for freq, subset in stimhz_means.groupby(within):
        group1 = subset[subset['species'] == between1][yvar]
        group2 = subset[subset['species'] == between2][yvar]
        stat, pval = spstats.mannwhitneyu(group1, group2,
                                          alternative='two-sided')
        results.append({
            'frequency': freq,
            'U_statistic': stat,
            'p_value': pval,
        })
    results_df = pd.DataFrame(results).sort_values('frequency')
    return results_df, aov


def add_multiple_comparisons(results_df):
    """Add FDR and Bonferroni corrected p-values to a results DataFrame.

    Expects a 'p_value' column.  Adds p_fdr, sig_fdr, p_bonf, sig_bonf.
    """
    from statsmodels.stats.multitest import multipletests
    pvals = results_df['p_value'].values
    reject_fdr, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05,
                                                 method='fdr_bh')
    reject_bonf, pvals_bonf, _, _ = multipletests(pvals, alpha=0.05,
                                                   method='bonferroni')
    results_df['p_fdr'] = pvals_fdr
    results_df['sig_fdr'] = reject_fdr
    results_df['p_bonf'] = pvals_bonf
    results_df['sig_bonf'] = reject_bonf
    return results_df


def pval_to_stars(pval):
    """Convert a p-value to a significance string."""
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    return 'ns'
