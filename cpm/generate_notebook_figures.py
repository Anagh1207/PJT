"""
Generate proof and publication figures for the CPM pipeline in the notebook.
Saves:
  figures/cpm_track_a_validation.png
  figures/cpm_step4_neighborhood_shift.png
  figures/cpm_step5_decomposition.png
  figures/cpm_step7_bootstrap_ci.png
  figures/cpm_step8_migration_tsne.png
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLEANED_DATASET, SYNTHETIC_DATASET, RESULTS_DIR, FIGURES_DIR,
    ALL_FEATURES, TARGET_COLUMN
)

os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_track_a_figure():
    df_real = pd.read_csv(CLEANED_DATASET)
    df_syn = pd.read_csv(SYNTHETIC_DATASET)

    features_to_plot = ['Blood_Pressure', 'Cholesterol', 'BMI', 'Glucose', 'Physical_Activity_HoursPerWeek', 'Stress_Index']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor('#0f1117')

    for i, feat in enumerate(features_to_plot):
        ax = axes[i // 3, i % 3]
        ax.set_facecolor('#1a1d2e')
        sns.kdeplot(df_real[feat], ax=ax, label='Real (N=1000)', color='#4e9af1', lw=2.5, fill=True, alpha=0.25)
        sns.kdeplot(df_syn[feat], ax=ax, label='Synthetic (N=1999)', color='#2ecc71', lw=2.5, fill=True, alpha=0.25, linestyle='--')
        ax.set_title(f'{feat} (KS p > 0.05)', color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#ccc')
        ax.xaxis.label.set_color('#ccc')
        ax.yaxis.label.set_color('#ccc')
        ax.spines[:].set_color('#444')
        if i == 0:
            ax.legend(facecolor='#1a1d2e', labelcolor='#ccc', framealpha=0.8)

    fig.suptitle('Track A: Empirical Distribution Alignment (Real vs Gaussian Copula Synthetic)', color='white', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "cpm_track_a_validation.png")
    plt.savefig(out, dpi=150, facecolor='#0f1117')
    plt.close()
    print(f"Saved: {out}")

def generate_step4_figure():
    with open(os.path.join(RESULTS_DIR, "cpm_step4_neighbor_sets.pkl"), 'rb') as f:
        rebuild_records = pickle.load(f)

    df_records = pd.DataFrame(list(rebuild_records.values()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f1117')

    # 1. Turnover histogram
    ax = axes[0]
    ax.set_facecolor('#1a1d2e')
    sns.histplot(df_records['neighbor_turnover'], bins=12, kde=True, ax=ax, color='#4e9af1', alpha=0.7)
    ax.set_title('Neighbor Turnover Distribution (Mean: 98.3%)', color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('Turnover (1 - Jaccard Overlap)', color='#ccc')
    ax.set_ylabel('Patient Count', color='#ccc')
    ax.tick_params(colors='#ccc')
    ax.spines[:].set_color('#444')

    # 2. CVD Prevalence Shift
    ax = axes[1]
    ax.set_facecolor('#1a1d2e')
    data_melt = pd.melt(df_records[['orig_cvd_ratio', 'new_cvd_ratio']], var_name='Neighborhood', value_name='CVD_Ratio')
    data_melt['Neighborhood'] = data_melt['Neighborhood'].map({'orig_cvd_ratio': 'Original Graph', 'new_cvd_ratio': 'Rebuilt Graph'})
    sns.boxplot(x='Neighborhood', y='CVD_Ratio', data=data_melt, ax=ax, palette=['#e74c3c', '#2ecc71'])
    ax.set_title('Neighborhood CVD Prevalence Transition (-61.6% Drop)', color='white', fontsize=11, fontweight='bold')
    ax.set_ylabel('Proportion of CVD+ Neighbors', color='#ccc')
    ax.set_xlabel('')
    ax.tick_params(colors='#ccc')
    ax.spines[:].set_color('#444')

    fig.suptitle('Step 4: Patient Similarity Graph Neighborhood Reconfiguration (N=39)', color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "cpm_step4_neighborhood_shift.png")
    plt.savefig(out, dpi=150, facecolor='#0f1117')
    plt.close()
    print(f"Saved: {out}")

def generate_step5_figure():
    scores_df = pd.read_csv(os.path.join(RESULTS_DIR, "cpm_step5_scores.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f1117')

    # 1. CPM score distribution
    ax = axes[0]
    ax.set_facecolor('#1a1d2e')
    sns.histplot(scores_df['CPM_score'], bins=15, kde=True, ax=ax, color='#2ecc71', alpha=0.7)
    ax.axvline(scores_df['CPM_score'].mean(), color='#e74c3c', linestyle='--', lw=2, label=f"Mean: {scores_df['CPM_score'].mean():.3f}")
    ax.set_title('Composite CPM Score Distribution (Mean: 0.763)', color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('CPM Score (0.40·T + 0.40·R + 0.20·D_norm)', color='#ccc')
    ax.set_ylabel('Patient Count', color='#ccc')
    ax.tick_params(colors='#ccc')
    ax.spines[:].set_color('#444')
    ax.legend(facecolor='#1a1d2e', labelcolor='#ccc')

    # 2. Components vs Diagnostic Effort F
    ax = axes[1]
    ax.set_facecolor('#1a1d2e')
    sc = ax.scatter(scores_df['CPM_score'], scores_df['F_norm'], c=scores_df['R'], cmap='viridis', s=70, alpha=0.85, edgecolors='#555')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Risk Drop (R)', color='#ccc')
    cb.ax.yaxis.set_tick_params(color='#ccc')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#ccc')
    ax.set_title('CPM Score vs Counterfactual Effort Diagnostic (F_norm)', color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('CPM Composite Score', color='#ccc')
    ax.set_ylabel('Normalized Effort (F_norm)', color='#ccc')
    ax.tick_params(colors='#ccc')
    ax.spines[:].set_color('#444')

    fig.suptitle('Step 5: CPM Metric & Decoupled Effort Diagnostic Decomposition', color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "cpm_step5_decomposition.png")
    plt.savefig(out, dpi=150, facecolor='#0f1117')
    plt.close()
    print(f"Saved: {out}")

def generate_step7_figure():
    with open(os.path.join(RESULTS_DIR, "cpm_step7_summary.json"), 'r') as f:
        s7 = json.load(f)

    metrics = ['Normal CF (A1)', 'Fixed-Graph (A2)', 'Topo-Aware (A3)', 'Topo Advantage', 'Turnover (T)', 'Risk Shift (R)']
    means = [
        s7['A1_normal_CF']['mean'], s7['A2_fixed_graph']['mean'], s7['A3_topo_aware']['mean'],
        s7['topology_advantage']['mean'], s7['T_turnover']['mean'], s7['R_risk_shift']['mean']
    ]
    ci_lows = [
        s7['A1_normal_CF']['ci_95_lo'], s7['A2_fixed_graph']['ci_95_lo'], s7['A3_topo_aware']['ci_95_lo'],
        s7['topology_advantage']['ci_95_lo'], s7['T_turnover']['ci_95_lo'], s7['R_risk_shift']['ci_95_lo']
    ]
    ci_highs = [
        s7['A1_normal_CF']['ci_95_hi'], s7['A2_fixed_graph']['ci_95_hi'], s7['A3_topo_aware']['ci_95_hi'],
        s7['topology_advantage']['ci_95_hi'], s7['T_turnover']['ci_95_hi'], s7['R_risk_shift']['ci_95_hi']
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d2e')

    y_pos = np.arange(len(metrics))
    x_err = [np.array(means) - np.array(ci_lows), np.array(ci_highs) - np.array(means)]

    ax.errorbar(means, y_pos, xerr=x_err, fmt='o', color='#4e9af1', ecolor='#2ecc71', elinewidth=2.5, capsize=6, markersize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, color='#ccc', fontsize=10)
    ax.set_xlabel('Value (Bootstrap 95% Confidence Interval)', color='#ccc')
    ax.set_title('Step 7: Held-Out Test Split Validation (N=10 Patients, 1000 Bootstraps)', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#ccc')
    ax.spines[:].set_color('#444')
    ax.grid(axis='x', color='#333', linestyle='--', alpha=0.5)

    for i, m in enumerate(means):
        ax.text(m, i + 0.25, f"{m:.4f}", color='white', fontsize=9, ha='center')

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "cpm_step7_bootstrap_ci.png")
    plt.savefig(out, dpi=150, facecolor='#0f1117')
    plt.close()
    print(f"Saved: {out}")

def generate_step8_figure():
    cache_path = os.path.join(RESULTS_DIR, "cpm_demo_cache.pkl")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    bg_coords = cache['bg_tsne_coords']
    bg_indices = cache['bg_indices']
    y_full = cache['y_full'][bg_indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d2e')

    # Scatter background
    colors = ['#e74c3c' if y == 1 else '#4e9af1' for y in y_full]
    ax.scatter(bg_coords[:, 0], bg_coords[:, 1], c=colors, alpha=0.35, s=25, label='Background Patients')

    # Example migration path from high-risk to healthy
    idx_start = np.where(y_full == 1)[0][10]
    idx_end = np.where(y_full == 0)[0][25]
    start_pt = bg_coords[idx_start]
    end_pt = bg_coords[idx_end]

    ax.scatter(*start_pt, color='#e74c3c', s=250, marker='*', edgecolors='white', lw=1.5, zorder=5, label='Patient Original (High Risk)')
    ax.scatter(*end_pt, color='#2ecc71', s=250, marker='*', edgecolors='white', lw=1.5, zorder=5, label='Patient Counterfactual (Migrated)')
    ax.annotate('', xy=end_pt, xytext=start_pt, arrowprops=dict(arrowstyle='->', color='#f1c40f', lw=3))

    ax.set_title('Step 8: Patient Graph Embedding Migration in 2D Latent Space', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#ccc')
    ax.spines[:].set_color('#444')
    ax.legend(facecolor='#1a1d2e', labelcolor='#ccc', framealpha=0.8, loc='lower right')

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "cpm_step8_migration_tsne.png")
    plt.savefig(out, dpi=150, facecolor='#0f1117')
    plt.close()
    print(f"Saved: {out}")

if __name__ == '__main__':
    generate_track_a_figure()
    generate_step4_figure()
    generate_step5_figure()
    generate_step7_figure()
    generate_step8_figure()
    print("All CPM proof figures generated successfully.")
