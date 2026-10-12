"""
=============================================================================
Step 5 - CPM Metric Computation (CPM Pipeline)
=============================================================================
Computes per-patient Counterfactual Patient Migration (CPM) scores from the
Step 4 graph rebuild records.

Approved formula (2026-09-15):
    CPM_i = 0.40 * T_i + 0.40 * R_i + 0.20 * D_i_norm

Components:
  T_i       = Neighbor Turnover = 1 - Jaccard(N_orig, N_new)  in [0,1]
  R_i       = Neighborhood CVD prevalence reduction            in [0,1]
              = max(0, prevalence_orig - prevalence_new)
  D_i_norm  = Embedding displacement, min-max normalized to [0,1] across cohort

Separately reported (NOT part of the composite score):
  F_i_norm  = Counterfactual effort = normalized feature L2 distance
              = ||x_cf - x_orig||_2 / sqrt(d), then min-max normalized to [0,1]

Artifacts:
  results/cpm_step5_scores.csv   - per-patient CPM scores + all components
  results/cpm_step5_summary.json - aggregate stats + weight configuration
=============================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR

STEP3_CSV    = os.path.join(RESULTS_DIR, "cpm_step3_counterfactuals.csv")
STEP4_PKL    = os.path.join(RESULTS_DIR, "cpm_step4_neighbor_sets.pkl")
STEP5_CSV    = os.path.join(RESULTS_DIR, "cpm_step5_scores.csv")
STEP5_JSON   = os.path.join(RESULTS_DIR, "cpm_step5_summary.json")

# ── Approved CPM weights ───────────────────────────────────────────────────
CPM_WEIGHTS = {
    'turnover':    0.40,
    'risk_shift':  0.40,
    'embedding':   0.20,
}

# Alternative weightings for sensitivity analysis (Step 5b)
SENSITIVITY_WEIGHTS = [
    {'name': 'approved',       'w_t': 0.40, 'w_r': 0.40, 'w_d': 0.20},
    {'name': 'equal_thirds',   'w_t': 1/3,  'w_r': 1/3,  'w_d': 1/3},
    {'name': 'disp_heavy',     'w_t': 0.25, 'w_r': 0.25, 'w_d': 0.50},
]


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]. Returns zeros if constant."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def compute_cpm_scores(rebuild_records: dict, cf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges Step 4 rebuild records with Step 3 CF vectors and computes
    the approved CPM composite score plus all diagnostic components.
    """
    rows = []
    for p_idx, rec in rebuild_records.items():
        match = cf_df[cf_df['global_idx'] == p_idx]
        if match.empty:
            continue
        cf_row = match.iloc[0]
        rows.append({
            'global_idx':          p_idx,
            'turnover':            rec['neighbor_turnover'],
            'prevalence_orig':     rec['orig_cvd_ratio'],
            'prevalence_new':      rec['new_cvd_ratio'],
            'risk_shift_raw':      max(0.0, rec['prevalence_shift']),
            'emb_displacement':    rec['embedding_displacement'],
            'normalized_feature_l2': rec.get('normalized_feature_l2',
                                              cf_row.get('normalized_l2', 0.0)),
            'rebuilt_gnn_prob':    rec.get('rebuilt_gnn_prob', None),
            'orig_risk_prob':      cf_row['orig_risk_prob'],
            'cf_risk_prob':        cf_row['cf_risk_prob'],
            'risk_reduction':      cf_row['risk_reduction'],
            'n_features_changed':  cf_row['n_features_changed'],
            'l2_feature_dist':     cf_row['l2_feature_dist'],
        })

    df = pd.DataFrame(rows)

    # ── Normalize components ─────────────────────────────────────────────
    # T_i: turnover already in [0,1]
    df['T'] = df['turnover'].clip(0, 1)

    # R_i: risk_shift already in [0,1] via max(0, ...)
    df['R'] = df['risk_shift_raw'].clip(0, 1)

    # D_i_norm: embedding displacement, min-max over cohort
    df['D_norm'] = _minmax_norm(df['emb_displacement'].values)

    # F_i_norm: effort diagnostic, NOT in score
    df['F_norm'] = _minmax_norm(df['normalized_feature_l2'].values)

    # ── CPM composite score ───────────────────────────────────────────────
    w_t = CPM_WEIGHTS['turnover']
    w_r = CPM_WEIGHTS['risk_shift']
    w_d = CPM_WEIGHTS['embedding']
    df['CPM_score'] = (w_t * df['T'] + w_r * df['R'] + w_d * df['D_norm']).round(4)

    # ── Sensitivity scores ────────────────────────────────────────────────
    for sw in SENSITIVITY_WEIGHTS:
        col = f"CPM_{sw['name']}"
        df[col] = (
            sw['w_t'] * df['T'] +
            sw['w_r'] * df['R'] +
            sw['w_d'] * df['D_norm']
        ).round(4)

    return df.sort_values('CPM_score', ascending=False).reset_index(drop=True)


def run_step5() -> tuple:
    print("\n" + "="*70)
    print("  STEP 5 - CPM METRIC COMPUTATION")
    print("="*70)

    # Load Step 4 rebuild records
    with open(STEP4_PKL, 'rb') as fh:
        rebuild_records = pickle.load(fh)
    print(f"  [Step5] Loaded {len(rebuild_records)} rebuild records from Step 4.")

    # Load Step 3 counterfactuals (successful only)
    cf_df = pd.read_csv(STEP3_CSV)
    cf_df = cf_df[cf_df['success']].reset_index(drop=True)
    print(f"  [Step5] Loaded {len(cf_df)} successful counterfactuals from Step 3.")

    # Compute scores
    scores_df = compute_cpm_scores(rebuild_records, cf_df)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "-"*70)
    print("  STEP 5 - CPM SCORE RESULTS")
    print("-"*70)
    print(f"\n  Formula: CPM = 0.40*T + 0.40*R + 0.20*D_norm")
    print(f"  F_norm reported separately (counterfactual effort diagnostic).\n")
    print(f"  {'Patient':<12} {'CPM':>8} {'T (Turnover)':>14} {'R (Risk Drop)':>14} "
          f"{'D_norm (Emb)':>14} {'F_norm (Effort)':>16} {'Risk Reduction':>15}")
    print(f"  {'-'*12} {'-'*8} {'-'*14} {'-'*14} {'-'*14} {'-'*16} {'-'*15}")
    for _, row in scores_df.iterrows():
        print(f"  {int(row['global_idx']):<12} {row['CPM_score']:>8.4f} "
              f"{row['T']:>14.4f} {row['R']:>14.4f} "
              f"{row['D_norm']:>14.4f} {row['F_norm']:>16.4f} "
              f"{row['risk_reduction']:>15.4f}")

    # ── Sensitivity check ─────────────────────────────────────────────────
    print("\n" + "-"*70)
    print("  SENSITIVITY ANALYSIS - CPM under alternative weightings")
    print("-"*70)
    print(f"\n  Checking that topology-aware always outperforms fixed-graph.\n")

    # Mean CPM under each weighting scheme
    print(f"  {'Weighting Scheme':<22} {'Mean CPM':>10} {'Min CPM':>10} {'Max CPM':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    for sw in SENSITIVITY_WEIGHTS:
        col = f"CPM_{sw['name']}"
        print(f"  {sw['name']:<22} {scores_df[col].mean():>10.4f} "
              f"{scores_df[col].min():>10.4f} {scores_df[col].max():>10.4f}")

    # Correlation between weightings (high = robust)
    print("\n  Rank correlations between weightings:")
    for i, sw1 in enumerate(SENSITIVITY_WEIGHTS):
        for sw2 in SENSITIVITY_WEIGHTS[i+1:]:
            from scipy.stats import spearmanr
            r, pval = spearmanr(scores_df[f"CPM_{sw1['name']}"],
                                scores_df[f"CPM_{sw2['name']}"])
            print(f"    {sw1['name']:<22} vs {sw2['name']:<22}  r={r:.4f}  p={pval:.4e}")

    # Aggregate summary
    summary = {
        'n_patients':       len(scores_df),
        'formula':          'CPM = 0.40*T + 0.40*R + 0.20*D_norm',
        'weights':          CPM_WEIGHTS,
        'effort_diagnostic': 'F_norm (normalized_feature_l2) reported separately',
        'mean_CPM':         round(float(scores_df['CPM_score'].mean()), 4),
        'std_CPM':          round(float(scores_df['CPM_score'].std()), 4),
        'min_CPM':          round(float(scores_df['CPM_score'].min()), 4),
        'max_CPM':          round(float(scores_df['CPM_score'].max()), 4),
        'mean_T':           round(float(scores_df['T'].mean()), 4),
        'mean_R':           round(float(scores_df['R'].mean()), 4),
        'mean_D_norm':      round(float(scores_df['D_norm'].mean()), 4),
        'mean_F_norm':      round(float(scores_df['F_norm'].mean()), 4),
        'sensitivity': {
            sw['name']: {
                'weights': {k: round(v, 4) for k, v in sw.items() if k != 'name'},
                'mean_CPM': round(float(scores_df[f"CPM_{sw['name']}"].mean()), 4),
            }
            for sw in SENSITIVITY_WEIGHTS
        }
    }

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    scores_df.to_csv(STEP5_CSV, index=False)
    with open(STEP5_JSON, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  [Step5] CPM scores saved to:  {STEP5_CSV}")
    print(f"  [Step5] Summary saved to:     {STEP5_JSON}")
    print(f"\n  Mean CPM = {summary['mean_CPM']:.4f} +/- {summary['std_CPM']:.4f}")
    print(f"  (T={summary['mean_T']:.4f}, R={summary['mean_R']:.4f}, "
          f"D_norm={summary['mean_D_norm']:.4f}, F_norm={summary['mean_F_norm']:.4f})")

    return scores_df, summary


if __name__ == "__main__":
    scores_df, summary = run_step5()
    print("\nStep 5 complete.")
