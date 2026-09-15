"""
=============================================================================
Synthetic Data Validation Module — Statistical & ML Consistency Validation
=============================================================================
Evaluates synthetic dataset fidelity relative to the real benchmark dataset:
  1. Kolmogorov-Smirnov distribution tests for continuous features
  2. Chi-squared tests for binary categorical features
  3. Correlation matrix difference (Frobenius norm)
  4. Train-on-Synthetic, Test-on-Benchmark model generalization
  5. Publication-quality diagnostic visual plots

Author: AI Research Assistant
Date: September 2026
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ALL_FEATURES, CONTINUOUS_FEATURES, BINARY_FEATURES, TARGET_COLUMN,
    VISUALIZATIONS_DIR, FIGURES_DIR, PLOT_DPI, PLOT_STYLE
)

plt.style.use('default')

def validate_synthetic_dataset(
    benchmark_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    save_plots: bool = True
) -> dict:
    """
    Runs full statistical and machine learning validation comparing synthetic vs benchmark dataset.
    """
    print("\n" + "="*70)
    print("  SYNTHETIC DATASET VALIDATION (Benchmark vs Synthetic)")
    print("="*70)
    
    # 1. Feature Distribution Tests
    ks_results = {}
    for col in CONTINUOUS_FEATURES:
        stat, p_val = stats.ks_2samp(benchmark_df[col], synthetic_df[col])
        ks_results[col] = {'ks_stat': stat, 'p_value': p_val, 'passed': p_val > 0.05}
        
    chi2_results = {}
    for col in BINARY_FEATURES:
        obs = np.array([
            [ (benchmark_df[col] == 0).sum(), (benchmark_df[col] == 1).sum() ],
            [ (synthetic_df[col] == 0).sum(), (synthetic_df[col] == 1).sum() ]
        ])
        chi2_stat, p_val, _, _ = stats.chi2_contingency(obs)
        chi2_results[col] = {'chi2_stat': chi2_stat, 'p_value': p_val, 'passed': p_val > 0.05}
        
    # 2. Correlation Similarity
    bench_corr = benchmark_df[ALL_FEATURES].corr().fillna(0).values
    synth_corr = synthetic_df[ALL_FEATURES].corr().fillna(0).values
    corr_frobenius_diff = float(np.linalg.norm(bench_corr - synth_corr))
    mean_abs_corr_diff = float(np.mean(np.abs(bench_corr - synth_corr)))
    
    # 3. Target Class Balance
    bench_target_dist = benchmark_df[TARGET_COLUMN].value_counts(normalize=True).to_dict()
    synth_target_dist = synthetic_df[TARGET_COLUMN].value_counts(normalize=True).to_dict()
    
    # 4. Train-on-Synthetic, Test-on-Benchmark ML Generalization
    X_syn = synthetic_df[ALL_FEATURES].values
    y_syn = synthetic_df[TARGET_COLUMN].values
    X_bench = benchmark_df[ALL_FEATURES].values
    y_bench = benchmark_df[TARGET_COLUMN].values
    
    clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    clf.fit(X_syn, y_syn)
    
    bench_preds = clf.predict(X_bench)
    bench_probs = clf.predict_proba(X_bench)[:, 1]
    
    ml_utility = {
        'accuracy': float(accuracy_score(y_bench, bench_preds)),
        'f1_score': float(f1_score(y_bench, bench_preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_bench, bench_probs))
    }
    
    validation_summary = {
        'ks_tests': ks_results,
        'chi2_tests': chi2_results,
        'corr_frobenius_diff': corr_frobenius_diff,
        'mean_abs_corr_diff': mean_abs_corr_diff,
        'benchmark_target_dist': bench_target_dist,
        'synthetic_target_dist': synth_target_dist,
        'ml_utility_train_syn_test_bench': ml_utility
    }
    
    # Print summary report
    print(f"\n[Statistical Similarity]")
    ks_passed_count = sum(1 for r in ks_results.values() if r['passed'])
    print(f"  Continuous KS-tests passed (p > 0.05): {ks_passed_count}/{len(CONTINUOUS_FEATURES)}")
    chi2_passed_count = sum(1 for r in chi2_results.values() if r['passed'])
    print(f"  Binary Chi2-tests passed (p > 0.05): {chi2_passed_count}/{len(BINARY_FEATURES)}")
    print(f"  Correlation matrix Frobenius distance: {corr_frobenius_diff:.4f}")
    print(f"  Mean absolute correlation difference: {mean_abs_corr_diff:.4f}")
    
    print(f"\n[ML Utility — Train on Synthetic -> Test on Real Benchmark]")
    print(f"  Accuracy: {ml_utility['accuracy']:.4f}")
    print(f"  F1 Score: {ml_utility['f1_score']:.4f}")
    print(f"  ROC AUC : {ml_utility['roc_auc']:.4f}")
    
    if save_plots:
        plot_synthetic_validation_diagnostics(benchmark_df, synthetic_df)
        
    return validation_summary


def plot_synthetic_validation_diagnostics(
    benchmark_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    save_dir: str = VISUALIZATIONS_DIR
):
    """Generates publication-quality figures comparing benchmark vs synthetic distributions."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 1. Distribution Overlays (Top continuous features)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(ALL_FEATURES[:12]):
        ax = axes[idx]
        sns.kdeplot(benchmark_df[col], label='Benchmark Real', color='#1E88E5', fill=True, alpha=0.4, ax=ax)
        sns.kdeplot(synthetic_df[col], label='Synthetic CTGAN', color='#E53935', fill=True, alpha=0.4, ax=ax)
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "synthetic_distribution_overlays.png"), dpi=PLOT_DPI)
    fig.savefig(os.path.join(FIGURES_DIR, "Fig_Synthetic_Distributions.pdf"))
    plt.close()
    
    # 2. Correlation Matrix Comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    bench_corr = benchmark_df[ALL_FEATURES].corr()
    synth_corr = synthetic_df[ALL_FEATURES].corr()
    diff_corr = (bench_corr - synth_corr).abs()
    
    sns.heatmap(bench_corr, ax=ax1, cmap='coolwarm', vmin=-1, vmax=1, cbar=False, annot=False)
    ax1.set_title('Benchmark Real Correlation', fontweight='bold')
    
    sns.heatmap(synth_corr, ax=ax2, cmap='coolwarm', vmin=-1, vmax=1, cbar=False, annot=False)
    ax2.set_title('Synthetic CTGAN Correlation', fontweight='bold')
    
    sns.heatmap(diff_corr, ax=ax3, cmap='Reds', vmin=0, vmax=0.5, cbar=True, annot=False)
    ax3.set_title('Absolute Difference', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "synthetic_correlation_comparison.png"), dpi=PLOT_DPI)
    plt.close()
    
    # 3. PCA Feature Space Overlap
    pca = PCA(n_components=2, random_state=42)
    pca.fit(benchmark_df[ALL_FEATURES])
    
    bench_pca = pca.transform(benchmark_df[ALL_FEATURES])
    synth_pca = pca.transform(synthetic_df[ALL_FEATURES])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(bench_pca[:, 0], bench_pca[:, 1], alpha=0.4, label='Benchmark Real', color='#1E88E5', s=20)
    ax.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.4, label='Synthetic CTGAN', color='#E53935', s=20)
    ax.set_title('PCA Manifold Overlap: Benchmark vs Synthetic', fontsize=12, fontweight='bold')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "synthetic_pca_manifold_overlap.png"), dpi=PLOT_DPI)
    plt.close()
    
    print(f"[SyntheticValidation] Diagnostic plots saved to {save_dir}")


if __name__ == "__main__":
    from data.benchmark_data import prepare_benchmark_dataset
    from data.synthetic_data import generate_synthetic_dataset
    
    bench_df = prepare_benchmark_dataset()
    syn_df = generate_synthetic_dataset(bench_df, n_samples=1000, epochs=20)
    summary = validate_synthetic_dataset(bench_df, syn_df)
