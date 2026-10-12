"""
=============================================================================
Step 4 — Graph Position Rebuild (CPM Pipeline)
=============================================================================
For every patient in the Track B high-risk cohort with a generated counterfactual:
1. Recomputes that patient's similarity to the entire population (real + synthetic,
   N ≈ 3,000) using the clinical similarity metric and optimal K (15).
2. Captures:
   - Before & after neighbor sets
   - Before & after neighborhood CVD composition (CVD prevalence)
   - Before & after GraphSAGE node embeddings (64-dim representation)
   - GraphSAGE predicted probability on the rebuilt graph

Artifacts:
  results/cpm_step4_neighbor_sets.pkl — comprehensive per-patient rebuild records
  results/cpm_step4_summary.json       — aggregate summary stats
=============================================================================
"""

import os
import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLEANED_DATASET, SYNTHETIC_DATASET, RESULTS_DIR, ALL_FEATURES,
    TARGET_COLUMN, MODEL_DIR, GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT,
    CPM_CONFIG
)
from models.gnn_models import GraphSAGE
from graph_construction import (
    build_patient_similarity_graph, rebuild_graph_for_patient,
    compute_neighborhood_stats
)

STEP3_CSV       = os.path.join(RESULTS_DIR, "cpm_step3_counterfactuals.csv")
STEP4_PKL       = os.path.join(RESULTS_DIR, "cpm_step4_neighbor_sets.pkl")
STEP4_SUMMARY   = os.path.join(RESULTS_DIR, "cpm_step4_summary.json")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "graphsage_optimal.pt")


def run_step4():
    print("\n" + "="*70)
    print("  STEP 4 — GRAPH POSITION REBUILD")
    print("="*70)

    # 1. Load data
    df_real = pd.read_csv(CLEANED_DATASET)
    df_syn  = pd.read_csv(SYNTHETIC_DATASET)
    df_full = pd.concat([df_real, df_syn], ignore_index=True)
    N_real = len(df_real)
    N_full = len(df_full)
    print(f"  [Step4] Real patients: {N_real}, Synthetic patients: {len(df_syn)}, Total population: {N_full}")

    X_full = df_full[ALL_FEATURES].values.astype(float)
    y_full = df_full[TARGET_COLUMN].values.astype(int)

    # 2. Load Step 3 counterfactuals
    if not os.path.exists(STEP3_CSV):
        raise FileNotFoundError(f"Step 3 counterfactuals not found at {STEP3_CSV}")
    cf_df = pd.read_csv(STEP3_CSV)
    successful_cf = cf_df[cf_df['success']].reset_index(drop=True)
    print(f"  [Step4] Loaded {len(successful_cf)} successful counterfactual patients from Step 3.")

    # 3. Load GraphSAGE model
    ck = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    optimal_k = ck.get('optimal_k', CPM_CONFIG['graph_k'])
    metric = CPM_CONFIG['graph_metric']

    model = GraphSAGE(
        in_channels=len(ALL_FEATURES),
        hidden_channels=GNN_HIDDEN_DIM,
        embed_channels=GNN_EMBED_DIM,
        dropout=GNN_DROPOUT
    )
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    print(f"  [Step4] Loaded GraphSAGE checkpoint (K*={optimal_k}, metric={metric})")

    # 4. Build baseline graph for full population
    print(f"  [Step4] Building baseline graph for full population (N={N_full}, K={optimal_k})...")
    t0 = time.time()
    edge_index_orig, edge_weight_orig, _ = build_patient_similarity_graph(
        X_full, k=optimal_k, feature_names=ALL_FEATURES, metric=metric
    )
    print(f"  [Step4] Baseline graph built in {time.time()-t0:.2f}s ({edge_index_orig.shape[1]} edges).")

    # Compute baseline embeddings
    X_tensor = torch.tensor(X_full, dtype=torch.float)
    with torch.no_grad():
        emb_orig_all = model.get_embeddings(X_tensor, edge_index_orig, edge_weight_orig).numpy()

    # 5. Rebuild graph for each counterfactual patient
    rebuild_records = {}
    turnovers = []
    prev_shifts = []
    emb_disps = []

    print(f"\n  [Step4] Rebuilding graph positions for {len(successful_cf)} counterfactual patients...")

    for i, row in successful_cf.iterrows():
        p_idx = int(row['global_idx'])
        cf_vec = np.array([row[f'cf_{feat}'] for feat in ALL_FEATURES], dtype=float)
        orig_vec = X_full[p_idx].copy()

        # Original neighborhood stats & embedding
        stats_orig = compute_neighborhood_stats(edge_index_orig, y_full, p_idx, k=optimal_k)
        orig_neighbors = stats_orig['neighbors']
        orig_emb = emb_orig_all[p_idx]

        # Rebuild graph for this patient
        edge_index_new, edge_weight_new, _ = rebuild_graph_for_patient(
            X_full, p_idx, cf_vec, k=optimal_k, feature_names=ALL_FEATURES, metric=metric
        )

        # New neighborhood stats & embedding
        stats_new = compute_neighborhood_stats(edge_index_new, y_full, p_idx, k=optimal_k)
        new_neighbors = stats_new['neighbors']

        # Forward pass on rebuilt graph
        X_mod = X_full.copy()
        X_mod[p_idx] = cf_vec
        X_mod_t = torch.tensor(X_mod, dtype=torch.float)
        with torch.no_grad():
            rebuilt_emb = model.get_embeddings(X_mod_t, edge_index_new, edge_weight_new)[p_idx].numpy()
            rebuilt_logits = model(X_mod_t, edge_index_new, edge_weight_new)[p_idx]
            rebuilt_prob = torch.softmax(rebuilt_logits, dim=0)[1].item()

        # Metrics for this patient
        set_orig = set(orig_neighbors)
        set_new  = set(new_neighbors)
        inter = len(set_orig.intersection(set_new))
        union = len(set_orig.union(set_new))
        jaccard = float(inter / union) if union > 0 else 0.0
        turnover = 1.0 - jaccard

        orig_prev = stats_orig['cvd_risk_ratio']
        new_prev  = stats_new['cvd_risk_ratio']
        prev_shift = orig_prev - new_prev

        emb_disp = float(np.linalg.norm(orig_emb - rebuilt_emb))

        turnovers.append(turnover)
        prev_shifts.append(prev_shift)
        emb_disps.append(emb_disp)

        rebuild_records[p_idx] = {
            'global_idx':          p_idx,
            'orig_neighbors':      orig_neighbors,
            'new_neighbors':       new_neighbors,
            'orig_cvd_ratio':      orig_prev,
            'new_cvd_ratio':       new_prev,
            'prevalence_shift':    prev_shift,
            'jaccard_similarity':  jaccard,
            'neighbor_turnover':   turnover,
            'orig_emb':            orig_emb.tolist(),
            'rebuilt_emb':         rebuilt_emb.tolist(),
            'embedding_displacement': emb_disp,
            'rebuilt_gnn_prob':    round(rebuilt_prob, 4),
            'feature_l2_dist':     float(np.linalg.norm(cf_vec - orig_vec)),
            'normalized_feature_l2': float(np.linalg.norm(cf_vec - orig_vec) / np.sqrt(len(ALL_FEATURES)))
        }

        if (i + 1) % 10 == 0 or (i + 1) == len(successful_cf):
            print(f"  ... Rebuilt {i+1}/{len(successful_cf)} | "
                  f"Turnover: {turnover:.1%} | "
                  f"Prevalence Shift: {orig_prev:.2f} -> {new_prev:.2f} (Delta={prev_shift:.2f}) | "
                  f"Emb Disp: {emb_disp:.3f}")

    # Summary
    summary = {
        'n_rebuilt_patients': len(rebuild_records),
        'mean_neighbor_turnover': round(float(np.mean(turnovers)), 4),
        'mean_prevalence_shift':  round(float(np.mean(prev_shifts)), 4),
        'mean_embedding_displacement': round(float(np.mean(emb_disps)), 4),
        'min_turnover': round(float(np.min(turnovers)), 4),
        'max_turnover': round(float(np.max(turnovers)), 4),
        'min_prevalence_shift': round(float(np.min(prev_shifts)), 4),
        'max_prevalence_shift': round(float(np.max(prev_shifts)), 4),
    }

    print("\n" + "-"*70)
    print("  STEP 4 — REBUILD RESULTS SUMMARY")
    print("-"*70)
    print(f"  Rebuilt patients:              {summary['n_rebuilt_patients']}")
    print(f"  Mean neighbor turnover:        {summary['mean_neighbor_turnover']:.1%}")
    print(f"  Mean neighborhood CVD shift:   {summary['mean_prevalence_shift']:.4f}")
    print(f"  Mean embedding displacement:   {summary['mean_embedding_displacement']:.4f}")

    # Save
    with open(STEP4_PKL, 'wb') as fh:
        pickle.dump(rebuild_records, fh)
    with open(STEP4_SUMMARY, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  [Step4] Rebuild records saved to: {STEP4_PKL}")
    print(f"  [Step4] Summary saved to:          {STEP4_SUMMARY}")
    return rebuild_records, summary


if __name__ == "__main__":
    records, summ = run_step4()
    print("\nStep 4 complete.")
