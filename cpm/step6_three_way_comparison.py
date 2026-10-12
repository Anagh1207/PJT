"""
=============================================================================
Step 6 - Three-Way Comparison Experiment (CPM Pipeline)
=============================================================================
Compares three counterfactual conditions on the same high-risk dev cohort:

  Approach 1: Normal CF       - tabular risk reduction only (LightGBM)
  Approach 2: Fixed-Graph CF  - GNN risk, original neighbor topology frozen
  Approach 3: Topology-Aware  - GNN risk + graph migration (CPM pipeline)

Reports effect sizes (Cohen's d) and statistical tests (paired Wilcoxon) for
all pairwise comparisons. This is the primary evidence for the paper's novelty
claim.

Artifacts:
  results/cpm_step6_three_way.csv     - per-patient rows for all 3 approaches
  results/cpm_step6_summary.json      - aggregate stats + statistical tests
  figures/cpm_step6_comparison.png    - publication-ready comparison figure
=============================================================================
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLEANED_DATASET, SYNTHETIC_DATASET, RESULTS_DIR, FIGURES_DIR,
    ALL_FEATURES, TARGET_COLUMN, RANDOM_STATE,
    MODEL_DIR, GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT, CPM_CONFIG,
    COUNTERFACTUAL_CONFIG
)
from models.gnn_models import GraphSAGE
from graph_construction import (
    build_patient_similarity_graph, rebuild_graph_for_patient,
    compute_neighborhood_stats
)
from sklearn.model_selection import train_test_split

STEP3_CSV   = os.path.join(RESULTS_DIR, "cpm_step3_counterfactuals.csv")
STEP4_PKL   = os.path.join(RESULTS_DIR, "cpm_step4_neighbor_sets.pkl")
STEP5_CSV   = os.path.join(RESULTS_DIR, "cpm_step5_scores.csv")
STEP6_CSV   = os.path.join(RESULTS_DIR, "cpm_step6_three_way.csv")
STEP6_JSON  = os.path.join(RESULTS_DIR, "cpm_step6_summary.json")
STEP6_FIG   = os.path.join(FIGURES_DIR, "cpm_step6_comparison.png")
CHECKPOINT  = os.path.join(MODEL_DIR, "graphsage_optimal.pt")


def _cohens_d(a, b):
    """Cohen's d effect size (paired)."""
    diff = np.array(a) - np.array(b)
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


def _wilcoxon_test(a, b):
    """Wilcoxon signed-rank test for paired samples."""
    try:
        stat, pval = wilcoxon(np.array(a) - np.array(b))
        return float(stat), float(pval)
    except Exception:
        return float('nan'), 1.0


def run_step6():
    print("\n" + "="*70)
    print("  STEP 6 - THREE-WAY COUNTERFACTUAL COMPARISON")
    print("="*70)

    # ── Load data ──────────────────────────────────────────────────────────
    df_real = pd.read_csv(CLEANED_DATASET)
    df_syn  = pd.read_csv(SYNTHETIC_DATASET)
    df_full = pd.concat([df_real, df_syn], ignore_index=True)
    X_real  = df_real[ALL_FEATURES].values.astype(float)
    y_real  = df_real[TARGET_COLUMN].values.astype(int)

    # Split same as training
    idx_all = np.arange(len(df_real))
    idx_tv, idx_test, y_tv, _ = train_test_split(
        idx_all, y_real, test_size=0.20, stratify=y_real, random_state=RANDOM_STATE)
    idx_train, idx_val, _, _ = train_test_split(
        idx_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RANDOM_STATE)
    idx_dev = np.concatenate([idx_train, idx_val])
    X_dev = X_real[idx_dev]
    y_dev = y_real[idx_dev]

    print(f"  [Step6] Dev population: {len(X_dev)} patients | Test held out.")

    # ── Load GraphSAGE ─────────────────────────────────────────────────────
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    optimal_k = ck.get('optimal_k', CPM_CONFIG['graph_k'])
    model = GraphSAGE(len(ALL_FEATURES), GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    print(f"  [Step6] Loaded GraphSAGE checkpoint (K*={optimal_k}).")

    # Dev graph (for Approach 2 - fixed topology)
    edge_index_dev, edge_weight_dev, _ = build_patient_similarity_graph(
        X_dev, k=optimal_k, feature_names=ALL_FEATURES, metric='clinical')

    # Full population graph (for Approach 3 - topology-aware)
    X_full = df_full[ALL_FEATURES].values.astype(float)
    y_full = df_full[TARGET_COLUMN].values.astype(int)
    edge_index_full, edge_weight_full, _ = build_patient_similarity_graph(
        X_full, k=optimal_k, feature_names=ALL_FEATURES, metric='clinical')
    print(f"  [Step6] Built dev graph ({len(X_dev)} nodes) and full graph ({len(X_full)} nodes).")

    # ── Train LightGBM for Approach 1 (tabular baseline) ──────────────────
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
    lgb_model.fit(X_dev, y_dev)
    print(f"  [Step6] Trained LightGBM on dev population.")

    # ── Load Step 3 & 4 artifacts ─────────────────────────────────────────
    cf_df = pd.read_csv(STEP3_CSV)
    cf_df = cf_df[cf_df['success']].reset_index(drop=True)
    cpm_df = pd.read_csv(STEP5_CSV)
    with open(STEP4_PKL, 'rb') as fh:
        rebuild_records = pickle.load(fh)

    pop_min = X_dev.min(axis=0)
    pop_max = X_dev.max(axis=0)
    pop_range = pop_max - pop_min
    pop_range[pop_range == 0] = 1.0

    lower_factors = ['Blood_Pressure', 'Cholesterol', 'BMI', 'Glucose',
                     'Triglycerides', 'Stress_Index', 'Heart_Rate', 'Smoking', 'Diabetes']
    increase_factors = ['Physical_Activity_HoursPerWeek']
    lower_indices   = [ALL_FEATURES.index(f) for f in lower_factors
                       if f in COUNTERFACTUAL_CONFIG['modifiable_features']]
    increase_indices = [ALL_FEATURES.index(f) for f in increase_factors
                        if f in COUNTERFACTUAL_CONFIG['modifiable_features']]

    print(f"\n  [Step6] Computing all three conditions for {len(cf_df)} patients...\n")

    records = []
    for _, row in cf_df.iterrows():
        global_idx = int(row['global_idx'])
        local_idx  = int(row['local_dev_idx'])
        x_orig = X_dev[local_idx].copy()
        x_cf   = np.array([row[f'cf_{feat}'] for feat in ALL_FEATURES], dtype=float)

        # ── Approach 1: Normal CF (LightGBM) ─────────────────────────────
        p1_orig = float(lgb_model.predict_proba(x_orig.reshape(1, -1))[0, 1])
        p1_cf   = float(lgb_model.predict_proba(x_cf.reshape(1, -1))[0, 1])
        risk_red_1 = p1_orig - p1_cf

        # ── Approach 2: Fixed-Graph CF (GNN, frozen topology) ─────────────
        X_dev_t = torch.tensor(X_dev, dtype=torch.float)
        with torch.no_grad():
            p2_orig = torch.softmax(model(X_dev_t, edge_index_dev)[local_idx], dim=0)[1].item()

        X_dev_mod = X_dev.copy()
        X_dev_mod[local_idx] = x_cf
        with torch.no_grad():
            p2_cf = torch.softmax(
                model(torch.tensor(X_dev_mod, dtype=torch.float), edge_index_dev)[local_idx],
                dim=0)[1].item()
        risk_red_2 = p2_orig - p2_cf

        # ── Approach 3: Topology-Aware CF (GNN + graph rebuild) ───────────
        # Patient's global index in full population
        X_full_mod = X_full.copy()
        X_full_mod[global_idx] = x_cf
        edge_new, weight_new, _ = rebuild_graph_for_patient(
            X_full, global_idx, x_cf, k=optimal_k,
            feature_names=ALL_FEATURES, metric='clinical')
        with torch.no_grad():
            p3_orig = torch.softmax(
                model(torch.tensor(X_full, dtype=torch.float), edge_index_full)[global_idx],
                dim=0)[1].item()
            p3_cf = torch.softmax(
                model(torch.tensor(X_full_mod, dtype=torch.float), edge_new)[global_idx],
                dim=0)[1].item()
        risk_red_3 = p3_orig - p3_cf

        # ── CPM score for this patient ────────────────────────────────────
        cpm_row = cpm_df[cpm_df['global_idx'] == global_idx]
        cpm_score = float(cpm_row['CPM_score'].iloc[0]) if not cpm_row.empty else None
        f_norm    = float(cpm_row['F_norm'].iloc[0])    if not cpm_row.empty else None
        turnover  = float(cpm_row['T'].iloc[0])         if not cpm_row.empty else None
        risk_shift= float(cpm_row['R'].iloc[0])         if not cpm_row.empty else None
        emb_disp  = float(cpm_row['D_norm'].iloc[0])    if not cpm_row.empty else None

        records.append({
            'global_idx':      global_idx,
            'local_dev_idx':   local_idx,
            # Approach 1
            'A1_orig_risk':    round(p1_orig, 4),
            'A1_cf_risk':      round(p1_cf, 4),
            'A1_risk_red':     round(risk_red_1, 4),
            # Approach 2
            'A2_orig_risk':    round(p2_orig, 4),
            'A2_cf_risk':      round(p2_cf, 4),
            'A2_risk_red':     round(risk_red_2, 4),
            # Approach 3
            'A3_orig_risk':    round(p3_orig, 4),
            'A3_cf_risk':      round(p3_cf, 4),
            'A3_risk_red':     round(risk_red_3, 4),
            # CPM topology components
            'CPM_score':       cpm_score,
            'T_turnover':      turnover,
            'R_risk_shift':    risk_shift,
            'D_norm_emb':      emb_disp,
            'F_norm_effort':   f_norm,
            # Key comparison
            'topo_advantage':  round(risk_red_3 - risk_red_2, 4),
        })

    result_df = pd.DataFrame(records)
    n = len(result_df)

    # ── Statistical tests ─────────────────────────────────────────────────
    a1 = result_df['A1_risk_red'].values
    a2 = result_df['A2_risk_red'].values
    a3 = result_df['A3_risk_red'].values

    stat_1v2, p_1v2 = _wilcoxon_test(a1, a2)
    stat_1v3, p_1v3 = _wilcoxon_test(a1, a3)
    stat_2v3, p_2v3 = _wilcoxon_test(a2, a3)
    d_1v2 = _cohens_d(a1, a2)
    d_1v3 = _cohens_d(a1, a3)
    d_2v3 = _cohens_d(a2, a3)

    print("\n" + "-"*70)
    print("  STEP 6 - THREE-WAY COMPARISON RESULTS")
    print("-"*70)
    print(f"\n  N = {n} patients\n")
    print(f"  {'Metric':<35} {'A1 (Normal)':>12} {'A2 (Fixed-G)':>13} {'A3 (Topo-Aware)':>15}")
    print(f"  {'-'*35} {'-'*12} {'-'*13} {'-'*15}")
    for label, arr in [("Mean Risk Reduction", [a1, a2, a3]),
                       ("Std Risk Reduction",  [a1.std(), a2.std(), a3.std()]),
                       ("Min Risk Reduction",  [a1.min(), a2.min(), a3.min()]),
                       ("Max Risk Reduction",  [a1.max(), a2.max(), a3.max()])]:
        if label == "Mean Risk Reduction":
            print(f"  {label:<35} {arr[0].mean():>12.4f} {arr[1].mean():>13.4f} {arr[2].mean():>15.4f}")
        else:
            print(f"  {label:<35} {arr[0]:>12.4f} {arr[1]:>13.4f} {arr[2]:>15.4f}")

    print(f"\n  Pairwise Statistical Tests (Wilcoxon signed-rank, paired):")
    print(f"    A1 vs A2:  stat={stat_1v2:.2f}, p={p_1v2:.4e}, Cohen's d={d_1v2:.4f}")
    print(f"    A1 vs A3:  stat={stat_1v3:.2f}, p={p_1v3:.4e}, Cohen's d={d_1v3:.4f}")
    print(f"    A2 vs A3:  stat={stat_2v3:.2f}, p={p_2v3:.4e}, Cohen's d={d_2v3:.4f}")

    # Key finding
    a3_gt_a2 = (a3 > a2).sum()
    print(f"\n  Key finding: Topology-Aware > Fixed-Graph in {a3_gt_a2}/{n} patients ({a3_gt_a2/n:.1%})")
    print(f"  Mean topology advantage (A3 - A2): {result_df['topo_advantage'].mean():.4f}")

    # ── Sensitivity check (qualitative direction) ─────────────────────────
    print("\n  Sensitivity check on CPM weighting (direction test):")
    from scipy.stats import spearmanr
    r_cpm_topo, _ = spearmanr(result_df['CPM_score'], result_df['topo_advantage'])
    print(f"    Spearman r(CPM_score, topology_advantage) = {r_cpm_topo:.4f}")
    print(f"    -> Higher CPM score patients show {'more' if r_cpm_topo > 0 else 'less'} "
          f"topology-aware benefit.")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        'n_patients':     n,
        'A1_normal_CF':   {'mean_risk_red': round(float(a1.mean()), 4),
                           'std_risk_red':  round(float(a1.std()), 4)},
        'A2_fixed_graph': {'mean_risk_red': round(float(a2.mean()), 4),
                           'std_risk_red':  round(float(a2.std()), 4)},
        'A3_topo_aware':  {'mean_risk_red': round(float(a3.mean()), 4),
                           'std_risk_red':  round(float(a3.std()), 4)},
        'statistical_tests': {
            'A1_vs_A2': {'wilcoxon_stat': stat_1v2, 'p_value': p_1v2, 'cohens_d': d_1v2},
            'A1_vs_A3': {'wilcoxon_stat': stat_1v3, 'p_value': p_1v3, 'cohens_d': d_1v3},
            'A2_vs_A3': {'wilcoxon_stat': stat_2v3, 'p_value': p_2v3, 'cohens_d': d_2v3},
        },
        'topo_advantage': {
            'mean':       round(float(result_df['topo_advantage'].mean()), 4),
            'pct_positive': round(float(a3_gt_a2 / n), 4),
        },
        'sensitivity_spearman_r': round(float(r_cpm_topo), 4),
    }

    # ── Figure ────────────────────────────────────────────────────────────
    _plot_three_way(result_df, summary, STEP6_FIG)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    result_df.to_csv(STEP6_CSV, index=False)
    with open(STEP6_JSON, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  [Step6] Results saved to: {STEP6_CSV}")
    print(f"  [Step6] Summary saved to:  {STEP6_JSON}")
    print(f"  [Step6] Figure saved to:   {STEP6_FIG}")

    return result_df, summary


def _plot_three_way(df, summary, save_path):
    """Publication-ready three-way comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.patch.set_facecolor('#0f1117')

    colors = {'A1': '#4e9af1', 'A2': '#f4a261', 'A3': '#2ecc71'}
    labels = {
        'A1': 'Normal CF\n(Tabular LGB)',
        'A2': 'Fixed-Graph CF\n(GNN, frozen topology)',
        'A3': 'Topology-Aware CF\n(GNN + graph migration)',
    }

    a1 = df['A1_risk_red'].values
    a2 = df['A2_risk_red'].values
    a3 = df['A3_risk_red'].values

    # (1) Box plot comparison
    ax = axes[0]
    ax.set_facecolor('#1a1d27')
    bp = ax.boxplot([a1, a2, a3], patch_artist=True, widths=0.5,
                    medianprops=dict(color='white', linewidth=2))
    for patch, col in zip(bp['boxes'], [colors['A1'], colors['A2'], colors['A3']]):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    for whisker in bp['whiskers'] + bp['caps']:
        whisker.set_color('#888')
    for flier in bp['fliers']:
        flier.set(marker='o', color='#aaa', alpha=0.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Normal\nCF', 'Fixed\nGraph', 'Topo\nAware'], color='#ccc', fontsize=9)
    ax.set_ylabel('Risk Reduction', color='#ccc')
    ax.set_title('Risk Reduction by Approach', color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#ccc')
    ax.spines[:].set_color('#444')
    ax.yaxis.label.set_color('#ccc')

    # (2) Scatter: CPM score vs topology advantage
    ax = axes[1]
    ax.set_facecolor('#1a1d27')
    sc = ax.scatter(df['CPM_score'], df['topo_advantage'],
                    c=df['T_turnover'], cmap='plasma', s=60, alpha=0.85, edgecolors='#555', linewidths=0.5)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Neighbor Turnover', color='#ccc', fontsize=9)
    cb.ax.yaxis.set_tick_params(color='#ccc')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#ccc')
    ax.axhline(0, color='#f66', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('CPM Score', color='#ccc')
    ax.set_ylabel('Topology Advantage (A3 - A2 Risk Reduction)', color='#ccc')
    ax.set_title('CPM Score vs Topology Advantage', color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#ccc')
    ax.spines[:].set_color('#444')

    # (3) Bar chart: mean risk reduction per approach with error bars
    ax = axes[2]
    ax.set_facecolor('#1a1d27')
    means = [a1.mean(), a2.mean(), a3.mean()]
    stds  = [a1.std(), a2.std(), a3.std()]
    x_pos = [0, 1, 2]
    bars = ax.bar(x_pos, means, color=[colors['A1'], colors['A2'], colors['A3']],
                  alpha=0.85, width=0.5, zorder=3)
    ax.errorbar(x_pos, means, yerr=stds, fmt='none', ecolor='white', capsize=6, linewidth=2, zorder=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Normal\nCF', 'Fixed\nGraph', 'Topo\nAware'], color='#ccc', fontsize=9)
    ax.set_ylabel('Mean Risk Reduction (+/- SD)', color='#ccc')
    ax.set_title('Mean Risk Reduction by Approach', color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#ccc')
    ax.spines[:].set_color('#444')
    ax.grid(axis='y', color='#333', linestyle='--', alpha=0.5)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, mean + 0.01, f'{mean:.3f}',
                ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

    # Significance annotation on bar chart
    pval = summary['statistical_tests']['A2_vs_A3']['p_value']
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
    y_max = max(means) + max(stds) + 0.05
    ax.annotate('', xy=(2, y_max), xytext=(1, y_max),
                arrowprops=dict(arrowstyle='-', color='white', lw=1.5))
    ax.text(1.5, y_max + 0.01, f'A2 vs A3: {sig}', ha='center', color='white', fontsize=9)

    fig.suptitle('Three-Way Counterfactual Comparison\nCPM Pipeline - CVD Risk Prediction',
                 color='white', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  [Step6] Figure saved: {save_path}")


if __name__ == "__main__":
    result_df, summary = run_step6()
    print("\nStep 6 complete.")
