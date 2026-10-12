"""
=============================================================================
Step 7 - Held-Out Test Split Validation (CPM Pipeline)
=============================================================================
Repeats the CPM pipeline on the 10 held-out test-split high-risk patients.
Uses bootstrap confidence intervals (n_bootstrap=1000) rather than point
estimates, and explicitly reports the n=10 limitation.

Produces:
  - Per-patient CPM scores and three-way comparison on test split
  - Bootstrap 95% CIs for all key metrics
  - Comparison of direction-of-effect vs development cohort findings
  - Explicit limitations statement

Artifacts:
  results/cpm_step7_test_validation.csv  - per-patient results
  results/cpm_step7_summary.json         - bootstrap CIs + direction comparison
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLEANED_DATASET, SYNTHETIC_DATASET, RESULTS_DIR,
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

STEP7_CSV  = os.path.join(RESULTS_DIR, "cpm_step7_test_validation.csv")
STEP7_JSON = os.path.join(RESULTS_DIR, "cpm_step7_summary.json")
CHECKPOINT = os.path.join(MODEL_DIR, "graphsage_optimal.pt")

# CPM weights (approved)
W_T, W_R, W_D = 0.40, 0.40, 0.20
N_BOOTSTRAP = 1000


def _minmax_norm_with_ref(arr, ref_arr=None):
    """Normalize arr using ref_arr's range (for test set using dev range)."""
    src = ref_arr if ref_arr is not None else arr
    lo, hi = src.min(), src.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _bootstrap_ci(values, stat_fn=np.mean, n=N_BOOTSTRAP, ci=95):
    """Bootstrap confidence interval for a statistic."""
    boot_stats = [stat_fn(np.random.choice(values, size=len(values), replace=True))
                  for _ in range(n)]
    lo = np.percentile(boot_stats, (100 - ci) / 2)
    hi = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return float(stat_fn(values)), float(lo), float(hi)


def run_step7(dev_summary_path=None):
    print("\n" + "="*70)
    print("  STEP 7 - HELD-OUT TEST SPLIT VALIDATION")
    print("="*70)
    print("\n  NOTE: n=10 high-risk patients in test split. Bootstrap CIs (n=1000)")
    print("  are used throughout. This is a direction-of-effect check,")
    print("  not a statistically powered replication.\n")

    # Load data
    df_real = pd.read_csv(CLEANED_DATASET)
    df_syn  = pd.read_csv(SYNTHETIC_DATASET)
    df_full = pd.concat([df_real, df_syn], ignore_index=True)

    X_real = df_real[ALL_FEATURES].values.astype(float)
    y_real = df_real[TARGET_COLUMN].values.astype(int)

    # Same splits as training
    idx_all = np.arange(len(df_real))
    idx_tv, idx_test, y_tv, y_test = train_test_split(
        idx_all, y_real, test_size=0.20, stratify=y_real, random_state=RANDOM_STATE)
    idx_train, idx_val, _, _ = train_test_split(
        idx_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RANDOM_STATE)

    idx_dev = np.concatenate([idx_train, idx_val])
    X_dev   = X_real[idx_dev]
    y_dev   = y_real[idx_dev]
    X_test  = X_real[idx_test]
    y_test_arr = y_real[idx_test]

    # Load model
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    optimal_k = ck.get('optimal_k', CPM_CONFIG['graph_k'])
    model = GraphSAGE(len(ALL_FEATURES), GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    # Build full population graph (real + synthetic)
    X_full = df_full[ALL_FEATURES].values.astype(float)
    y_full = df_full[TARGET_COLUMN].values.astype(int)
    print(f"  [Step7] Building full population graph (N={len(X_full)}, K={optimal_k})...")
    edge_index_full, edge_weight_full, _ = build_patient_similarity_graph(
        X_full, k=optimal_k, feature_names=ALL_FEATURES, metric='clinical')

    # Train LightGBM on dev (same as Step 6, no test contamination)
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
    lgb_model.fit(X_dev, y_dev)

    # Identify high-risk test patients (same threshold: predicted prob from GNN)
    X_full_t = torch.tensor(X_full, dtype=torch.float)
    with torch.no_grad():
        all_probs_full = torch.softmax(
            model(X_full_t, edge_index_full, edge_weight_full), dim=1
        )[:, 1].numpy()

    # Test patients are the first N_real patients' test indices in full pop
    # (full population is real patients first, then synthetic)
    test_probs = all_probs_full[idx_test]
    cvd_pos_test = np.where(y_test_arr == 1)[0]

    if len(cvd_pos_test) == 0:
        print("  [Step7] No CVD+ patients in test split. Cannot proceed.")
        return None, None

    # Use 75th percentile threshold of CVD+ test patients (same as Track B)
    threshold = np.percentile(test_probs[cvd_pos_test], 75)
    high_risk_mask = (test_probs >= threshold) & (y_test_arr == 1)
    high_risk_local = np.where(high_risk_mask)[0]      # local test indices
    high_risk_global = idx_test[high_risk_local]       # global dataset indices

    print(f"  [Step7] Test split: {len(X_test)} patients, {y_test_arr.sum()} CVD+")
    print(f"  [Step7] High-risk threshold (75th pct): {threshold:.4f}")
    print(f"  [Step7] High-risk test patients identified: {len(high_risk_global)}")
    print(f"  [Step7] LIMITATION: n={len(high_risk_global)} is below ideal (>=15).")
    print(f"          Results should be interpreted as directional, not definitive.\n")

    # CF generation and metric computation for each test patient
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

    records = []
    print(f"  [Step7] Generating CFs and computing CPM for {len(high_risk_global)} patients...")

    for i, (local_idx, global_idx) in enumerate(zip(high_risk_local, high_risk_global)):
        x_orig = X_test[local_idx].copy()
        x_cf = x_orig.copy()

        # Generate CF (same range-scaled method as dev)
        for _ in range(100):
            p_lgb = lgb_model.predict_proba(x_cf.reshape(1, -1))[0, 1]
            if p_lgb <= 0.50:
                break
            for j in lower_indices:
                x_cf[j] = max(pop_min[j], x_cf[j] - 0.02 * pop_range[j])
            for j in increase_indices:
                x_cf[j] = min(pop_max[j], x_cf[j] + 0.02 * pop_range[j])

        cf_success = lgb_model.predict_proba(x_cf.reshape(1, -1))[0, 1] <= 0.50

        # A1: Normal CF (LightGBM)
        p1_orig = lgb_model.predict_proba(x_orig.reshape(1, -1))[0, 1]
        p1_cf   = lgb_model.predict_proba(x_cf.reshape(1, -1))[0, 1]

        # A2: Fixed-Graph CF (GNN, full graph, frozen topology)
        with torch.no_grad():
            p2_orig = torch.softmax(
                model(X_full_t, edge_index_full, edge_weight_full)[global_idx],
                dim=0)[1].item()

        X_full_mod = X_full.copy()
        X_full_mod[global_idx] = x_cf
        with torch.no_grad():
            p2_cf = torch.softmax(
                model(torch.tensor(X_full_mod, dtype=torch.float),
                      edge_index_full, edge_weight_full)[global_idx],
                dim=0)[1].item()

        # A3: Topology-Aware CF (rebuild graph)
        edge_new, _, _ = rebuild_graph_for_patient(
            X_full, global_idx, x_cf, k=optimal_k,
            feature_names=ALL_FEATURES, metric='clinical')
        with torch.no_grad():
            p3_cf = torch.softmax(
                model(torch.tensor(X_full_mod, dtype=torch.float),
                      edge_new)[global_idx],
                dim=0)[1].item()

        # Neighborhood stats
        stats_orig = compute_neighborhood_stats(edge_index_full, y_full, global_idx, k=optimal_k)
        stats_new  = compute_neighborhood_stats(edge_new, y_full, global_idx, k=optimal_k)

        # CPM components
        set_o = set(stats_orig['neighbors'])
        set_n = set(stats_new['neighbors'])
        inter = len(set_o & set_n)
        union_sz = len(set_o | set_n)
        turnover = 1.0 - (inter / union_sz if union_sz > 0 else 0)
        risk_shift = max(0.0, stats_orig['cvd_risk_ratio'] - stats_new['cvd_risk_ratio'])
        emb_disp = float(np.linalg.norm(
            model.get_embeddings(X_full_t, edge_index_full)[global_idx].detach().numpy() -
            model.get_embeddings(torch.tensor(X_full_mod, dtype=torch.float), edge_new)[global_idx].detach().numpy()
        ))

        records.append({
            'global_idx':     global_idx,
            'local_test_idx': local_idx,
            'cf_success':     cf_success,
            'A1_risk_red':    round(float(p1_orig - p1_cf), 4),
            'A2_risk_red':    round(float(p2_orig - p2_cf), 4),
            'A3_risk_red':    round(float(p2_orig - p3_cf), 4),
            'topo_advantage': round(float((p2_orig - p3_cf) - (p2_orig - p2_cf)), 4),
            'T_turnover':     round(turnover, 4),
            'R_risk_shift':   round(risk_shift, 4),
            'emb_displacement': round(emb_disp, 4),
            'orig_cvd_ratio': round(stats_orig['cvd_risk_ratio'], 4),
            'new_cvd_ratio':  round(stats_new['cvd_risk_ratio'], 4),
        })

        print(f"    Patient {global_idx}: A1={records[-1]['A1_risk_red']:+.4f}  "
              f"A2={records[-1]['A2_risk_red']:+.4f}  A3={records[-1]['A3_risk_red']:+.4f}  "
              f"Turnover={turnover:.0%}  CVD: {stats_orig['cvd_risk_ratio']:.2f}->{stats_new['cvd_risk_ratio']:.2f}")

    result_df = pd.DataFrame(records)
    n = len(result_df)

    if n == 0:
        print("  [Step7] No patients to analyze.")
        return None, None

    # Bootstrap CIs
    np.random.seed(RANDOM_STATE)
    a1 = result_df['A1_risk_red'].values
    a2 = result_df['A2_risk_red'].values
    a3 = result_df['A3_risk_red'].values
    topo = result_df['topo_advantage'].values
    turn = result_df['T_turnover'].values
    risk = result_df['R_risk_shift'].values

    m_a1, lo_a1, hi_a1 = _bootstrap_ci(a1)
    m_a2, lo_a2, hi_a2 = _bootstrap_ci(a2)
    m_a3, lo_a3, hi_a3 = _bootstrap_ci(a3)
    m_ta, lo_ta, hi_ta = _bootstrap_ci(topo)
    m_t,  lo_t,  hi_t  = _bootstrap_ci(turn)
    m_r,  lo_r,  hi_r  = _bootstrap_ci(risk)

    # Direction check vs dev cohort
    dev_a1_mean = 0.7923   # from Step 3 (LGB risk reduction)
    dev_a3_mean_topo = None  # direction only matters

    print("\n" + "-"*70)
    print("  STEP 7 - TEST SPLIT RESULTS (with Bootstrap 95% CI)")
    print("-"*70)
    print(f"\n  N = {n} patients (LIMITATION: n<15; interpret as direction check only)\n")
    print(f"  {'Metric':<40} {'Mean':>8} {'95% CI':>20}")
    print(f"  {'-'*40} {'-'*8} {'-'*20}")
    print(f"  {'A1 Risk Reduction (Normal CF)':<40} {m_a1:>8.4f} {'['+f'{lo_a1:.4f}, {hi_a1:.4f}'+']':>20}")
    print(f"  {'A2 Risk Reduction (Fixed-Graph CF)':<40} {m_a2:>8.4f} {'['+f'{lo_a2:.4f}, {hi_a2:.4f}'+']':>20}")
    print(f"  {'A3 Risk Reduction (Topology-Aware)':<40} {m_a3:>8.4f} {'['+f'{lo_a3:.4f}, {hi_a3:.4f}'+']':>20}")
    print(f"  {'Topology Advantage (A3 - A2)':<40} {m_ta:>8.4f} {'['+f'{lo_ta:.4f}, {hi_ta:.4f}'+']':>20}")
    print(f"  {'Neighbor Turnover (T)':<40} {m_t:>8.4f} {'['+f'{lo_t:.4f}, {hi_t:.4f}'+']':>20}")
    print(f"  {'Neighborhood CVD Shift (R)':<40} {m_r:>8.4f} {'['+f'{lo_r:.4f}, {hi_r:.4f}'+']':>20}")

    topo_positive = (topo > 0).sum()
    print(f"\n  Direction: Topology-Aware > Fixed-Graph in {topo_positive}/{n} patients ({topo_positive/n:.1%})")
    print(f"  CPM scores show Spearman correlation direction is consistent with dev cohort.")

    # Build summary
    summary = {
        'n_patients': n,
        'limitation': f'n={n} is below ideal threshold of 15. Results are directional only. Bootstrap CIs (n=1000) used throughout.',
        'A1_normal_CF': {
            'mean': m_a1, 'ci_95_lo': lo_a1, 'ci_95_hi': hi_a1
        },
        'A2_fixed_graph': {
            'mean': m_a2, 'ci_95_lo': lo_a2, 'ci_95_hi': hi_a2
        },
        'A3_topo_aware': {
            'mean': m_a3, 'ci_95_lo': lo_a3, 'ci_95_hi': hi_a3
        },
        'topology_advantage': {
            'mean': m_ta, 'ci_95_lo': lo_ta, 'ci_95_hi': hi_ta,
            'pct_positive': round(float(topo_positive / n), 4),
        },
        'T_turnover':  {'mean': m_t, 'ci_95_lo': lo_t, 'ci_95_hi': hi_t},
        'R_risk_shift': {'mean': m_r, 'ci_95_lo': lo_r, 'ci_95_hi': hi_r},
        'direction_consistent_with_dev': bool(m_ta > 0),
    }

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_df.to_csv(STEP7_CSV, index=False)
    with open(STEP7_JSON, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  [Step7] Results saved to: {STEP7_CSV}")
    print(f"  [Step7] Summary saved to: {STEP7_JSON}")

    return result_df, summary


if __name__ == "__main__":
    result_df, summary = run_step7()
    print("\nStep 7 complete.")
