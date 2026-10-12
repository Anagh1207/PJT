"""
=============================================================================
Track B — High-Risk Patient Cohort Identification (CPM Pipeline)
=============================================================================
Loads the existing GraphSAGE checkpoint and patient similarity graph,
computes predicted CVD probabilities for the development population
(training + validation split — excludes test split), and identifies the
high-risk cohort for counterfactual analysis.

Threshold strategy:
  'percentile_75_cvd' — 75th-percentile of predicted probs among confirmed
                         CVD patients (label=1) in the dev set. Data-adaptive;
                         ensures a meaningful cohort size.
  'fixed_0.5'         — prob > 0.5 AND true label=1 (fallback).

The test split is NEVER touched here — it is held for Step 7.

Artifacts:
  results/cpm_track_b_cohort.csv  — cohort indices + predicted probs + true labels
  results/cpm_track_b_summary.txt — cohort size, threshold, justification
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
    CLEANED_DATASET, MODEL_DIR, RESULTS_DIR, ALL_FEATURES, TARGET_COLUMN,
    RANDOM_STATE, GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT, CPM_CONFIG
)
from models.gnn_models import GraphSAGE
from graph_construction import build_patient_similarity_graph

# ── Paths ─────────────────────────────────────────────────────────────────
COHORT_CSV     = os.path.join(RESULTS_DIR, "cpm_track_b_cohort.csv")
SUMMARY_PATH   = os.path.join(RESULTS_DIR, "cpm_track_b_summary.json")
CHECKPOINT     = os.path.join(MODEL_DIR, "graphsage_optimal.pt")


def _get_splits(df: pd.DataFrame):
    """Returns (idx_train, idx_val, idx_test) using the fixed GNN split."""
    idx = np.arange(len(df))
    y   = df[TARGET_COLUMN].values
    idx_tv, idx_test, y_tv, _ = train_test_split(
        idx, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    idx_train, idx_val, _, _ = train_test_split(
        idx_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RANDOM_STATE
    )
    return idx_train, idx_val, idx_test


def load_graphsage_model(checkpoint_path: str, n_features: int) -> tuple:
    """Loads GraphSAGE from checkpoint. Returns (model, optimal_k, feature_names)."""
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = GraphSAGE(
        in_channels=n_features,
        hidden_channels=GNN_HIDDEN_DIM,
        embed_channels=GNN_EMBED_DIM,
        dropout=GNN_DROPOUT
    )
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model, ck['optimal_k'], ck['feature_names']


def run_track_b() -> pd.DataFrame:
    """
    Main entry point for Track B.

    Returns the high-risk cohort DataFrame with columns:
      [global_idx, predicted_prob, true_label, split]
    """
    print("\n" + "="*70)
    print("  TRACK B — HIGH-RISK PATIENT COHORT IDENTIFICATION")
    print("="*70)

    df = pd.read_csv(CLEANED_DATASET)
    X  = df[ALL_FEATURES].values.astype(float)
    y  = df[TARGET_COLUMN].values

    print(f"  [TrackB] Full dataset: {len(df)} patients, "
          f"{y.sum()} CVD+ ({y.mean():.1%})")

    # Split — test split is withheld
    idx_train, idx_val, idx_test = _get_splits(df)
    idx_dev = np.concatenate([idx_train, idx_val])  # development population
    print(f"  [TrackB] Dev population (train+val): {len(idx_dev)} patients | "
          f"Test held out: {len(idx_test)} patients (NEVER TOUCHED HERE)")

    # Load checkpoint
    print(f"  [TrackB] Loading GraphSAGE checkpoint from: {CHECKPOINT}")
    model, optimal_k, feature_names = load_graphsage_model(CHECKPOINT, n_features=X.shape[1])
    print(f"  [TrackB] Checkpoint loaded. K*={optimal_k}, metric=clinical")

    # Build clinical similarity graph for the dev population
    print(f"  [TrackB] Building clinical similarity graph for dev population "
          f"(N={len(idx_dev)}, K={optimal_k})...")
    X_dev = X[idx_dev]
    y_dev = y[idx_dev]
    edge_index, edge_weight, graph_stats = build_patient_similarity_graph(
        X_dev, k=optimal_k, feature_names=feature_names, metric='clinical'
    )

    # Run inference on dev population
    X_tensor = torch.tensor(X_dev, dtype=torch.float)
    with torch.no_grad():
        logits = model(X_tensor, edge_index, edge_weight)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # ── Threshold strategy ────────────────────────────────────────────────
    strategy = CPM_CONFIG['high_risk_threshold_strategy']
    cvd_mask_dev = (y_dev == 1)
    probs_cvd = probs[cvd_mask_dev]

    if strategy == 'percentile_75_cvd':
        threshold = float(np.percentile(probs_cvd, 75))
        threshold_label = f"75th-percentile of CVD+ predicted probs ({threshold:.4f})"
        high_risk_mask = (probs >= threshold) & (y_dev == 1)
    else:  # fixed_0.5 fallback
        threshold = 0.5
        threshold_label = "Fixed threshold 0.5 with true label = 1"
        high_risk_mask = (probs > threshold) & (y_dev == 1)

    high_risk_local_indices = np.where(high_risk_mask)[0]
    high_risk_global_indices = idx_dev[high_risk_local_indices]

    # Build cohort DataFrame
    cohort_df = pd.DataFrame({
        'global_idx':     high_risk_global_indices,
        'local_dev_idx':  high_risk_local_indices,
        'predicted_prob': probs[high_risk_local_indices],
        'true_label':     y_dev[high_risk_local_indices],
        'split':          'dev',
    }).sort_values('predicted_prob', ascending=False).reset_index(drop=True)

    # ── Also compute test-split preview (for Step 7 gate, shown here early) ──
    print("\n  [TrackB] Previewing test-split high-risk count (for Step 7 gate)...")
    X_test = X[idx_test]
    y_test = y[idx_test]
    edge_index_test, edge_weight_test, _ = build_patient_similarity_graph(
        X_test, k=optimal_k, feature_names=feature_names, metric='clinical'
    )
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    with torch.no_grad():
        logits_test  = model(X_test_tensor, edge_index_test, edge_weight_test)
        probs_test   = torch.softmax(logits_test, dim=1)[:, 1].cpu().numpy()

    if strategy == 'percentile_75_cvd':
        probs_cvd_test = probs_test[y_test == 1]
        threshold_test = float(np.percentile(probs_cvd_test, 75)) if len(probs_cvd_test) > 0 else 0.5
        test_high_risk_mask = (probs_test >= threshold_test) & (y_test == 1)
    else:
        test_high_risk_mask = (probs_test > 0.5) & (y_test == 1)

    n_test_high_risk = int(test_high_risk_mask.sum())

    # ── Summary report ────────────────────────────────────────────────────
    summary = {
        'dataset_size': int(len(df)),
        'dev_population_size': int(len(idx_dev)),
        'test_population_size': int(len(idx_test)),
        'test_cvd_positive': int((y_test == 1).sum()),
        'threshold_strategy': strategy,
        'threshold_value': round(threshold, 4),
        'threshold_label': threshold_label,
        'dev_high_risk_cohort_size': int(len(cohort_df)),
        'dev_cohort_mean_prob': round(float(cohort_df['predicted_prob'].mean()), 4),
        'dev_cohort_min_prob':  round(float(cohort_df['predicted_prob'].min()), 4),
        'test_high_risk_preview_count': n_test_high_risk,
        'step7_meaningful': n_test_high_risk >= CPM_CONFIG['min_test_cohort_for_step7'],
        'step7_flag': (
            "OK — test cohort is large enough for Step 7."
            if n_test_high_risk >= CPM_CONFIG['min_test_cohort_for_step7']
            else f"WARNING — only {n_test_high_risk} high-risk patients in test split "
                 f"(minimum {CPM_CONFIG['min_test_cohort_for_step7']}). "
                 "Step 7 will use bootstrap CIs and report this limitation explicitly."
        ),
    }

    # Print summary
    print("\n" + "-"*70)
    print("  TRACK B — RESULTS SUMMARY")
    print("-"*70)
    print(f"  Threshold:         {threshold_label}")
    print(f"  Dev cohort size:   {summary['dev_high_risk_cohort_size']} patients")
    print(f"  Mean predicted prob (cohort): {summary['dev_cohort_mean_prob']:.4f}")
    print(f"  Min  predicted prob (cohort): {summary['dev_cohort_min_prob']:.4f}")
    print(f"\n  [Step 7 pre-check]")
    print(f"  Test split: {summary['test_population_size']} patients, "
          f"{summary['test_cvd_positive']} CVD+")
    print(f"  Test high-risk count (preview): {n_test_high_risk}")
    print(f"  {summary['step7_flag']}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cohort_df.to_csv(COHORT_CSV, index=False)
    with open(SUMMARY_PATH, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  [TrackB] Cohort saved to: {COHORT_CSV}")
    print(f"  [TrackB] Summary saved to: {SUMMARY_PATH}")

    # ── Also return graph data for downstream reuse ───────────────────────
    return cohort_df, {
        'model': model,
        'optimal_k': optimal_k,
        'feature_names': feature_names,
        'X_dev': X_dev,
        'y_dev': y_dev,
        'idx_dev': idx_dev,
        'idx_test': idx_test,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'probs_dev': probs,
        'summary': summary,
    }


if __name__ == "__main__":
    cohort_df, context = run_track_b()
    print(f"\nTrack B complete. {len(cohort_df)} high-risk patients identified.")
