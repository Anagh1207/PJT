"""
=============================================================================
Statistical Validation Module
=============================================================================
Phase 7: Statistical significance tests for model comparisons.

Implements:
  - Paired t-test
  - Wilcoxon signed-rank test
  - Bootstrap confidence intervals
  - Cross-validation variance analysis

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def paired_t_test(scores_a, scores_b, alternative='two-sided'):
    """
    Perform paired t-test between two sets of scores.

    Parameters
    ----------
    scores_a, scores_b : list or np.ndarray
        Per-fold or per-seed metric values.
    alternative : str

    Returns
    -------
    dict with statistic, p_value, significant (at 0.05)
    """
    t_stat, p_value = sp_stats.ttest_rel(scores_a, scores_b,
                                         alternative=alternative)
    return {
        'test': 'Paired t-test',
        'statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }


def wilcoxon_test(scores_a, scores_b, alternative='two-sided'):
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative).

    Parameters
    ----------
    scores_a, scores_b : array-like

    Returns
    -------
    dict
    """
    try:
        stat, p_value = sp_stats.wilcoxon(scores_a, scores_b,
                                          alternative=alternative)
    except ValueError:
        # All differences are zero
        stat, p_value = 0.0, 1.0

    return {
        'test': 'Wilcoxon signed-rank',
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }


def bootstrap_confidence_interval(scores, confidence=0.95, n_bootstrap=1000,
                                  random_state=42):
    """
    Compute bootstrap confidence interval for the mean.

    Parameters
    ----------
    scores : array-like
    confidence : float
    n_bootstrap : int

    Returns
    -------
    dict with mean, ci_lower, ci_upper
    """
    np.random.seed(random_state)
    scores = np.array(scores)
    boot_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        'mean': scores.mean(),
        'std': scores.std(),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence': confidence,
    }


def run_all_significance_tests(proposed_scores, baseline_scores_dict,
                               metric_name='F1'):
    """
    Run all statistical tests comparing the proposed model to each baseline.

    Parameters
    ----------
    proposed_scores : list
        Per-seed metric values for the proposed DPSN model.
    baseline_scores_dict : dict
        model_name → list of per-seed metric values.
    metric_name : str

    Returns
    -------
    results_df : pd.DataFrame
    """
    rows = []
    for model_name, baseline_scores in baseline_scores_dict.items():
        # Ensure same length
        min_len = min(len(proposed_scores), len(baseline_scores))
        ps = proposed_scores[:min_len]
        bs = baseline_scores[:min_len]

        t_result = paired_t_test(ps, bs)
        w_result = wilcoxon_test(ps, bs)
        ci_proposed = bootstrap_confidence_interval(ps)
        ci_baseline = bootstrap_confidence_interval(bs)

        rows.append({
            'Metric': metric_name,
            'Proposed (DPSN)': f"{ci_proposed['mean']:.4f} ± {ci_proposed['std']:.4f}",
            'Baseline': model_name,
            'Baseline Score': f"{ci_baseline['mean']:.4f} ± {ci_baseline['std']:.4f}",
            'Improvement': f"{(ci_proposed['mean'] - ci_baseline['mean']):.4f}",
            'Paired t-test p': f"{t_result['p_value']:.4f}",
            't-test Sig.': '✓' if t_result['significant'] else '✗',
            'Wilcoxon p': f"{w_result['p_value']:.4f}",
            'Wilcoxon Sig.': '✓' if w_result['significant'] else '✗',
            'DPSN 95% CI': f"[{ci_proposed['ci_lower']:.4f}, {ci_proposed['ci_upper']:.4f}]",
            'Baseline 95% CI': f"[{ci_baseline['ci_lower']:.4f}, {ci_baseline['ci_upper']:.4f}]",
        })

    return pd.DataFrame(rows)
