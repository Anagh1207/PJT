"""
=============================================================================
Step 3 — Constrained Counterfactual Generation (CPM Pipeline)
=============================================================================
For every patient in the Track B high-risk cohort, generates a constrained
counterfactual: a modified feature vector that reduces predicted CVD risk
below 0.5, by changing ONLY mutable features within physiologically
plausible ranges, minimising the total feature change.

Uses the greedy directional approach from evaluation/topology_counterfactual.py
with a batch wrapper that tracks success/failure and change magnitudes.

Artifacts:
  results/cpm_step3_counterfactuals.csv — per-patient CF vectors + flags
  results/cpm_step3_summary.json         — success rate, per-feature change stats
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLEANED_DATASET, RESULTS_DIR, ALL_FEATURES, CONTINUOUS_FEATURES,
    BINARY_FEATURES, TARGET_COLUMN, RANDOM_STATE, CPM_CONFIG,
    COUNTERFACTUAL_CONFIG, MODEL_DIR, GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT
)
from evaluation.topology_counterfactual import generate_counterfactual_vector
from models.gnn_models import GraphSAGE
from graph_construction import build_patient_similarity_graph

# ── Paths ─────────────────────────────────────────────────────────────────
CF_CSV      = os.path.join(RESULTS_DIR, "cpm_step3_counterfactuals.csv")
CF_SUMMARY  = os.path.join(RESULTS_DIR, "cpm_step3_summary.json")
CHECKPOINT  = os.path.join(MODEL_DIR, "graphsage_optimal.pt")
COHORT_CSV  = os.path.join(RESULTS_DIR, "cpm_track_b_cohort.csv")


def _get_splits(df):
    idx = np.arange(len(df))
    y   = df[TARGET_COLUMN].values
    idx_tv, idx_test, y_tv, _ = train_test_split(
        idx, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    idx_train, idx_val, _, _ = train_test_split(
        idx_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RANDOM_STATE
    )
    return np.concatenate([idx_train, idx_val]), idx_test


def run_step3(
    model=None,
    X_dev: np.ndarray = None,
    y_dev: np.ndarray = None,
    edge_index=None,
    edge_weight=None,
    cohort_df: pd.DataFrame = None,
    target_risk: float = 0.50,
    max_iter: int = 500,
    step_size: float = 0.05
) -> pd.DataFrame:
    """
    Batch counterfactual generation for the high-risk cohort.

    Returns DataFrame with original + CF vectors, success flag, and change metrics.
    Can be called with pre-loaded objects (from Track B output) or will reload from disk.
    """
    print("\n" + "="*70)
    print("  STEP 3 — CONSTRAINED COUNTERFACTUAL GENERATION")
    print("="*70)

    # ── Load from disk if not passed in ───────────────────────────────────
    if cohort_df is None:
        cohort_df = pd.read_csv(COHORT_CSV)
        print(f"  [Step3] Loaded cohort from {COHORT_CSV}: {len(cohort_df)} patients")

    df = pd.read_csv(CLEANED_DATASET)
    X_all = df[ALL_FEATURES].values.astype(float)
    y_all = df[TARGET_COLUMN].values
    idx_dev, idx_test = _get_splits(df)
    X_dev_local = X_all[idx_dev]
    y_dev_local = y_all[idx_dev]

    if model is None:
        ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        model = GraphSAGE(
            in_channels=X_dev_local.shape[1],
            hidden_channels=GNN_HIDDEN_DIM,
            embed_channels=GNN_EMBED_DIM,
            dropout=GNN_DROPOUT
        )
        model.load_state_dict(ck['model_state_dict'])
        model.eval()
        optimal_k = ck['optimal_k']
        feature_names = ck['feature_names']
        edge_index, edge_weight, _ = build_patient_similarity_graph(
            X_dev_local, k=optimal_k, feature_names=feature_names, metric='clinical'
        )
        print(f"  [Step3] Loaded checkpoint and rebuilt graph (K={optimal_k})")

    # ── Feature change direction table (mutable features) ────────────────
    print("\n  [Step3] Mutable vs Immutable feature list:")
    print(f"  IMMUTABLE: {COUNTERFACTUAL_CONFIG['immutable_features']}")
    print(f"  MUTABLE:   {COUNTERFACTUAL_CONFIG['modifiable_features']}")
    print("\n  [Step3] Clinical modification directions:")
    print("  Lower (risk-reducing): Blood_Pressure, Cholesterol, BMI, Glucose,")
    print("           Triglycerides, Stress_Index, Heart_Rate, Smoking, Diabetes")
    print("  Increase: Physical_Activity_HoursPerWeek")
    print("  ECG_Score: directional — lower toward population mean")

    # Train LightGBM risk model for counterfactual search (tabular CVD risk)
    import lightgbm as lgb
    import warnings
    warnings.filterwarnings('ignore')
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
    lgb_model.fit(X_dev_local, y_dev_local)

    pop_min = X_dev_local.min(axis=0)
    pop_max = X_dev_local.max(axis=0)
    pop_range = pop_max - pop_min
    pop_range[pop_range == 0] = 1.0

    lower_factors = ['Blood_Pressure', 'Cholesterol', 'BMI', 'Glucose', 'Triglycerides', 'Stress_Index', 'Heart_Rate', 'Smoking', 'Diabetes']
    increase_factors = ['Physical_Activity_HoursPerWeek']
    lower_indices = [ALL_FEATURES.index(f) for f in lower_factors if f in COUNTERFACTUAL_CONFIG['modifiable_features']]
    increase_indices = [ALL_FEATURES.index(f) for f in increase_factors if f in COUNTERFACTUAL_CONFIG['modifiable_features']]

    # ── Batch generation ──────────────────────────────────────────────────
    records = []
    n_success = 0
    n_fail    = 0

    local_indices = cohort_df['local_dev_idx'].tolist()
    global_indices = cohort_df['global_idx'].tolist()

    print(f"\n  [Step3] Generating CFs for {len(local_indices)} patients...")

    for i, (local_idx, global_idx) in enumerate(zip(local_indices, global_indices)):
        local_idx = int(local_idx)
        x_orig = X_dev_local[local_idx].copy()
        x_cf = x_orig.copy()

        # Get original risks
        X_tensor = torch.tensor(X_dev_local, dtype=torch.float)
        with torch.no_grad():
            orig_gnn_prob = torch.softmax(
                model(X_tensor, edge_index, edge_weight)[local_idx], dim=0
            )[1].item()

        orig_lgb_prob = float(lgb_model.predict_proba(x_orig.reshape(1, -1))[0, 1])

        # Generate range-scaled constrained counterfactual
        for step in range(max_iter):
            p = float(lgb_model.predict_proba(x_cf.reshape(1, -1))[0, 1])
            if p <= target_risk:
                break
            for j in lower_indices:
                x_cf[j] = max(pop_min[j], x_cf[j] - step_size * pop_range[j])
            for j in increase_indices:
                x_cf[j] = min(pop_max[j], x_cf[j] + step_size * pop_range[j])

        cf_lgb_prob = float(lgb_model.predict_proba(x_cf.reshape(1, -1))[0, 1])
        success = bool(cf_lgb_prob <= target_risk)

        # Compute GNN fixed-graph risk for comparison
        X_mod = X_dev_local.copy()
        X_mod[local_idx] = x_cf
        X_mod_tensor = torch.tensor(X_mod, dtype=torch.float)
        with torch.no_grad():
            cf_gnn_prob = torch.softmax(
                model(X_mod_tensor, edge_index, edge_weight)[local_idx], dim=0
            )[1].item()

        # Compute change metrics
        diff = x_cf - x_orig
        n_features_changed = int((np.abs(diff) > 1e-4).sum())
        l2_dist = float(np.linalg.norm(diff))
        normalized_l2 = float(l2_dist / np.sqrt(len(ALL_FEATURES)))

        if success:
            n_success += 1
        else:
            n_fail += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(local_indices):
            print(f"  ... {i+1}/{len(local_indices)} | "
                  f"Success so far: {n_success} | "
                  f"Current: {'SUCCESS' if success else 'FAIL'} "
                  f"(LGB: {orig_lgb_prob:.3f} -> {cf_lgb_prob:.3f} | GNN: {orig_gnn_prob:.3f} -> {cf_gnn_prob:.3f})")

        row = {
            'global_idx':          global_idx,
            'local_dev_idx':       local_idx,
            'orig_risk_prob':      round(orig_lgb_prob, 4),
            'cf_risk_prob':        round(cf_lgb_prob, 4),
            'orig_gnn_prob':       round(orig_gnn_prob, 4),
            'cf_gnn_fixed_prob':   round(cf_gnn_prob, 4),
            'risk_reduction':      round(orig_lgb_prob - cf_lgb_prob, 4),
            'success':             success,
            'n_features_changed':  n_features_changed,
            'l2_feature_dist':     round(l2_dist, 4),
            'normalized_l2':       round(normalized_l2, 4),
        }
        # Append original and CF feature values
        for j, feat in enumerate(ALL_FEATURES):
            row[f'orig_{feat}'] = round(float(x_orig[j]), 4)
            row[f'cf_{feat}']   = round(float(x_cf[j]), 4)
            row[f'delta_{feat}'] = round(float(x_cf[j] - x_orig[j]), 4)

        records.append(row)

    result_df = pd.DataFrame(records)
    success_rate = n_success / len(records) if records else 0.0

    # ── Summary ──────────────────────────────────────────────────────────
    success_df = result_df[result_df['success']]
    summary = {
        'total_patients': len(records),
        'n_success': n_success,
        'n_fail': n_fail,
        'success_rate': round(success_rate, 4),
        'target_risk_threshold': target_risk,
        'mean_risk_reduction_successful': round(
            float(success_df['risk_reduction'].mean()), 4) if len(success_df) > 0 else 0.0,
        'mean_features_changed': round(float(result_df['n_features_changed'].mean()), 2),
        'mean_l2_dist': round(float(result_df['l2_feature_dist'].mean()), 4),
    }

    # Per-feature change summary (successful CFs only)
    if len(success_df) > 0:
        per_feature = {}
        for feat in COUNTERFACTUAL_CONFIG['modifiable_features']:
            col = f'delta_{feat}'
            if col in success_df.columns:
                per_feature[feat] = {
                    'mean_delta':   round(float(success_df[col].mean()), 4),
                    'pct_changed':  round(float((success_df[col].abs() > 1e-4).mean()), 4),
                }
        summary['per_feature_change'] = per_feature

    print("\n" + "-"*70)
    print("  STEP 3 — RESULTS SUMMARY")
    print("-"*70)
    print(f"  Total patients: {summary['total_patients']}")
    print(f"  Success rate:   {n_success}/{len(records)} ({success_rate:.1%})")
    print(f"  Mean risk reduction (successes): {summary['mean_risk_reduction_successful']:.4f}")
    print(f"  Mean features changed:           {summary['mean_features_changed']:.1f}")
    print(f"  Mean L2 feature distance:        {summary['mean_l2_dist']:.4f}")

    if n_fail > 0:
        print(f"\n  NOTE: {n_fail} patients did not reach target risk ({target_risk}) within "
              f"{max_iter} iterations. These are flagged success=False and will be "
              "excluded from Steps 4-6 (reported as a finding, not a bug).")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_df.to_csv(CF_CSV, index=False)
    with open(CF_SUMMARY, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  [Step3] Results saved to: {CF_CSV}")
    print(f"  [Step3] Summary saved to:  {CF_SUMMARY}")

    return result_df, summary


if __name__ == "__main__":
    df_cf, summ = run_step3()
    print(f"\nStep 3 complete. {summ['n_success']} successful CFs.")
