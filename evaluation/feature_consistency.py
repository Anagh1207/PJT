"""
=============================================================================
Feature Weight Consistency Check — SHAP vs DPSN
=============================================================================
Phase 8c: Compare DPSN's learned feature weights against SHAP importance
rankings from XGBoost and LightGBM.

Quantifies agreement via Spearman rank correlation and Kendall's tau.

Author: AI Research Assistant
Date: August 2026
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kendalltau

from config import PLOT_DPI, PLOT_STYLE, FIGURES_DIR, RESULTS_DIR

plt.rcParams.update(PLOT_STYLE)


def load_feature_rankings():
    """
    Load DPSN feature weights and SHAP importance rankings.

    Returns
    -------
    merged_df : pd.DataFrame with columns:
        Feature, DPSN_Weight, SHAP_XGBoost, SHAP_LightGBM
    """
    # DPSN weights
    dpsn_path = os.path.join(RESULTS_DIR, 'dpsn_feature_weights.csv')
    if os.path.exists(dpsn_path):
        dpsn_df = pd.read_csv(dpsn_path)
        dpsn_df = dpsn_df.rename(columns={'Learned Weight': 'DPSN_Weight'})
    else:
        print(f"  [WARN] DPSN weights not found: {dpsn_path}")
        return None

    # SHAP XGBoost
    xgb_path = os.path.join(RESULTS_DIR, 'shap_importance_xgboost.csv')
    if os.path.exists(xgb_path):
        xgb_df = pd.read_csv(xgb_path)
        xgb_df = xgb_df.rename(columns={'Mean |SHAP|': 'SHAP_XGBoost'})
    else:
        xgb_df = pd.DataFrame({'Feature': dpsn_df['Feature'],
                                'SHAP_XGBoost': 0})

    # SHAP LightGBM
    lgb_path = os.path.join(RESULTS_DIR, 'shap_importance_lightgbm.csv')
    if os.path.exists(lgb_path):
        lgb_df = pd.read_csv(lgb_path)
        lgb_df = lgb_df.rename(columns={'Mean |SHAP|': 'SHAP_LightGBM'})
    else:
        lgb_df = pd.DataFrame({'Feature': dpsn_df['Feature'],
                                'SHAP_LightGBM': 0})

    # Merge on Feature
    merged = dpsn_df.merge(xgb_df, on='Feature', how='left')
    merged = merged.merge(lgb_df, on='Feature', how='left')
    merged = merged.fillna(0)

    return merged


def check_weight_differentiation(weights, threshold=0.001):
    """
    Check if feature weights are meaningfully differentiated.

    Parameters
    ----------
    weights : array-like
    threshold : float, minimum range to consider differentiated

    Returns
    -------
    is_differentiated : bool
    weight_range : float
    weight_std : float
    """
    w = np.array(weights)
    weight_range = w.max() - w.min()
    weight_std = np.std(w)
    is_differentiated = weight_range > threshold

    return is_differentiated, weight_range, weight_std


def compute_rank_correlations(merged_df):
    """
    Compute Spearman and Kendall rank correlations between
    DPSN weights and SHAP importance.

    Returns
    -------
    results : dict with correlation values and p-values
    """
    results = {}

    dpsn_weights = merged_df['DPSN_Weight'].values

    # Check if DPSN weights are differentiated
    is_diff, w_range, w_std = check_weight_differentiation(dpsn_weights)
    results['dpsn_differentiated'] = is_diff
    results['dpsn_weight_range'] = w_range
    results['dpsn_weight_std'] = w_std

    if not is_diff:
        results['warning'] = (
            'DPSN feature weights are NOT differentiated '
            f'(range={w_range:.6f}, std={w_std:.6f}). '
            'All features received equal weight (1/N). '
            'Rank correlation is undefined/meaningless. '
            'The feature-weighting component of DPSN did not learn '
            'meaningful distinctions during training.'
        )
        results['spearman_xgb'] = float('nan')
        results['spearman_xgb_p'] = float('nan')
        results['spearman_lgb'] = float('nan')
        results['spearman_lgb_p'] = float('nan')
        results['kendall_xgb'] = float('nan')
        results['kendall_xgb_p'] = float('nan')
        results['kendall_lgb'] = float('nan')
        results['kendall_lgb_p'] = float('nan')
        return results

    # Spearman with XGBoost
    rho_xgb, p_xgb = spearmanr(dpsn_weights, merged_df['SHAP_XGBoost'].values)
    results['spearman_xgb'] = rho_xgb
    results['spearman_xgb_p'] = p_xgb

    # Spearman with LightGBM
    rho_lgb, p_lgb = spearmanr(dpsn_weights, merged_df['SHAP_LightGBM'].values)
    results['spearman_lgb'] = rho_lgb
    results['spearman_lgb_p'] = p_lgb

    # Kendall with XGBoost
    tau_xgb, p_tau_xgb = kendalltau(dpsn_weights,
                                     merged_df['SHAP_XGBoost'].values)
    results['kendall_xgb'] = tau_xgb
    results['kendall_xgb_p'] = p_tau_xgb

    # Kendall with LightGBM
    tau_lgb, p_tau_lgb = kendalltau(dpsn_weights,
                                     merged_df['SHAP_LightGBM'].values)
    results['kendall_lgb'] = tau_lgb
    results['kendall_lgb_p'] = p_tau_lgb

    # SHAP cross-check: XGBoost vs LightGBM correlation
    rho_shap, p_shap = spearmanr(merged_df['SHAP_XGBoost'].values,
                                  merged_df['SHAP_LightGBM'].values)
    results['spearman_xgb_lgb'] = rho_shap
    results['spearman_xgb_lgb_p'] = p_shap

    return results


def plot_feature_weight_comparison(merged_df, corr_results, save_dir=None):
    """
    Plot side-by-side comparison of DPSN weights vs SHAP importance.
    """
    save_dir = save_dir or FIGURES_DIR

    # Normalize each column to [0, 1] for visual comparison
    df = merged_df.copy()
    for col in ['DPSN_Weight', 'SHAP_XGBoost', 'SHAP_LightGBM']:
        max_val = df[col].max()
        if max_val > 0:
            df[col + '_norm'] = df[col] / max_val
        else:
            df[col + '_norm'] = df[col]

    # Sort by LightGBM SHAP (most informative ranking)
    df = df.sort_values('SHAP_LightGBM', ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    # Panel 1: SHAP LightGBM
    axes[0].barh(df['Feature'], df['SHAP_LightGBM_norm'],
                 color='#C73E1D', edgecolor='white', alpha=0.85)
    axes[0].set_xlabel('Normalized Importance', fontweight='bold')
    axes[0].set_title('SHAP — LightGBM', fontweight='bold', fontsize=12)
    axes[0].grid(axis='x', alpha=0.3, linestyle='--')

    # Panel 2: SHAP XGBoost
    axes[1].barh(df['Feature'], df['SHAP_XGBoost_norm'],
                 color='#F18F01', edgecolor='white', alpha=0.85)
    axes[1].set_xlabel('Normalized Importance', fontweight='bold')
    axes[1].set_title('SHAP — XGBoost', fontweight='bold', fontsize=12)
    axes[1].grid(axis='x', alpha=0.3, linestyle='--')

    # Panel 3: DPSN Learned Weights
    is_diff = corr_results.get('dpsn_differentiated', False)
    bar_color = '#D81B60' if is_diff else '#999999'
    axes[2].barh(df['Feature'], df['DPSN_Weight_norm'],
                 color=bar_color, edgecolor='white', alpha=0.85)
    axes[2].set_xlabel('Normalized Weight', fontweight='bold')

    if is_diff:
        rho = corr_results.get('spearman_lgb', float('nan'))
        axes[2].set_title(f'DPSN Learned Weights\n(ρ={rho:.3f} vs LightGBM)',
                          fontweight='bold', fontsize=12)
    else:
        axes[2].set_title('DPSN Learned Weights\n⚠ NOT DIFFERENTIATED',
                          fontweight='bold', fontsize=12, color='#E53935')

    axes[2].grid(axis='x', alpha=0.3, linestyle='--')

    # Add cross-check annotation
    rho_shap = corr_results.get('spearman_xgb_lgb', float('nan'))
    if not np.isnan(rho_shap):
        fig.text(0.5, 0.01,
                 f'SHAP Cross-Check: XGBoost vs LightGBM ρ = {rho_shap:.3f}',
                 ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = os.path.join(save_dir, 'feature_weight_consistency.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Feature consistency plot: {path}")


def run_feature_consistency_check(save_dir=None):
    """
    Run the full SHAP vs DPSN feature weight consistency analysis.

    Returns
    -------
    consistency_df : pd.DataFrame
    """
    save_dir = save_dir or FIGURES_DIR

    print("\n  ── SHAP vs DPSN Feature Weight Consistency ───────────────")

    # Load rankings
    merged_df = load_feature_rankings()
    if merged_df is None:
        print("  [WARN] Cannot load feature rankings.")
        return pd.DataFrame()

    print(f"\n  Feature Rankings Loaded ({len(merged_df)} features):")
    print(merged_df.to_string(index=False))

    # Compute rank correlations
    corr_results = compute_rank_correlations(merged_df)

    # Report findings
    if not corr_results.get('dpsn_differentiated', True):
        print(f"\n  ⚠️  WARNING: {corr_results['warning']}")
        print(f"\n  This is a key finding: the DPSN feature-weighting module")
        print(f"  initialized all weights to 1/N and they remained at 1/N")
        print(f"  throughout training. This component did not contribute to")
        print(f"  the model's predictive capacity.")
    else:
        print(f"\n  Spearman ρ (DPSN vs XGBoost SHAP):  "
              f"{corr_results['spearman_xgb']:.4f} "
              f"(p={corr_results['spearman_xgb_p']:.4f})")
        print(f"  Spearman ρ (DPSN vs LightGBM SHAP): "
              f"{corr_results['spearman_lgb']:.4f} "
              f"(p={corr_results['spearman_lgb_p']:.4f})")
        print(f"  Kendall τ  (DPSN vs XGBoost SHAP):  "
              f"{corr_results['kendall_xgb']:.4f} "
              f"(p={corr_results['kendall_xgb_p']:.4f})")
        print(f"  Kendall τ  (DPSN vs LightGBM SHAP): "
              f"{corr_results['kendall_lgb']:.4f} "
              f"(p={corr_results['kendall_lgb_p']:.4f})")

    rho_shap = corr_results.get('spearman_xgb_lgb', float('nan'))
    if not np.isnan(rho_shap):
        print(f"\n  SHAP Cross-Check (XGBoost vs LightGBM): ρ={rho_shap:.4f} "
              f"(p={corr_results.get('spearman_xgb_lgb_p', 0):.4f})")

    # Plot
    plot_feature_weight_comparison(merged_df, corr_results, save_dir)

    # Save results
    # Build summary rows
    summary_rows = [
        {'Metric': 'DPSN Weight Range',
         'Value': corr_results['dpsn_weight_range']},
        {'Metric': 'DPSN Weight Std',
         'Value': corr_results['dpsn_weight_std']},
        {'Metric': 'Weights Differentiated',
         'Value': corr_results['dpsn_differentiated']},
    ]

    for key in ['spearman_xgb', 'spearman_xgb_p', 'spearman_lgb',
                'spearman_lgb_p', 'kendall_xgb', 'kendall_xgb_p',
                'kendall_lgb', 'kendall_lgb_p', 'spearman_xgb_lgb',
                'spearman_xgb_lgb_p']:
        if key in corr_results:
            summary_rows.append({'Metric': key, 'Value': corr_results[key]})

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_DIR, 'feature_consistency_analysis.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  [OK] Consistency analysis saved: {csv_path}")

    # Save merged rankings
    merged_path = os.path.join(RESULTS_DIR, 'feature_rankings_merged.csv')
    merged_df.to_csv(merged_path, index=False, float_format='%.6f')

    return summary_df
