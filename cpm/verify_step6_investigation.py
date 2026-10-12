"""
Diagnostic script to investigate Step 6 risk values, logits, and neighborhood aggregation
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLEANED_DATASET, SYNTHETIC_DATASET, RESULTS_DIR,
    ALL_FEATURES, TARGET_COLUMN, RANDOM_STATE,
    MODEL_DIR, GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT, CPM_CONFIG
)
from models.gnn_models import GraphSAGE
from graph_construction import build_patient_similarity_graph, rebuild_graph_for_patient, compute_neighborhood_stats

STEP3_CSV   = os.path.join(RESULTS_DIR, "cpm_step3_counterfactuals.csv")
STEP4_PKL   = os.path.join(RESULTS_DIR, "cpm_step4_neighbor_sets.pkl")
STEP5_CSV   = os.path.join(RESULTS_DIR, "cpm_step5_scores.csv")
CHECKPOINT  = os.path.join(MODEL_DIR, "graphsage_optimal.pt")

def investigate():
    # 1. Load data
    df_real = pd.read_csv(CLEANED_DATASET)
    df_syn  = pd.read_csv(SYNTHETIC_DATASET)
    df_full = pd.concat([df_real, df_syn], ignore_index=True)
    X_real  = df_real[ALL_FEATURES].values.astype(float)
    y_real  = df_real[TARGET_COLUMN].values.astype(int)

    idx_all = np.arange(len(df_real))
    idx_tv, idx_test, y_tv, _ = train_test_split(
        idx_all, y_real, test_size=0.20, stratify=y_real, random_state=RANDOM_STATE)
    idx_train, idx_val, _, _ = train_test_split(
        idx_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RANDOM_STATE)
    idx_dev = np.concatenate([idx_train, idx_val])
    X_dev = X_real[idx_dev]
    y_dev = y_real[idx_dev]

    X_full = df_full[ALL_FEATURES].values.astype(float)
    y_full = df_full[TARGET_COLUMN].values.astype(int)

    # Load model
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    optimal_k = ck.get('optimal_k', 15)
    model = GraphSAGE(len(ALL_FEATURES), GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    # Graphs
    edge_index_dev, edge_weight_dev, _ = build_patient_similarity_graph(
        X_dev, k=optimal_k, feature_names=ALL_FEATURES, metric='clinical')
    edge_index_full, edge_weight_full, _ = build_patient_similarity_graph(
        X_full, k=optimal_k, feature_names=ALL_FEATURES, metric='clinical')

    # Fit LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
    lgb_model.fit(X_dev, y_dev)

    # Load Step 3 & 4 & 5
    cf_df = pd.read_csv(STEP3_CSV)
    cpm_df = pd.read_csv(STEP5_CSV)
    with open(STEP4_PKL, 'rb') as f:
        rebuild_records = pickle.load(f)

    # Pick 5 patients across high, mid, and low CPM scores
    # cpm_df is sorted by CPM_score descending
    n_pts = len(cpm_df)
    chosen_indices = [
        0,                  # highest CPM
        n_pts // 4,         # upper-mid
        n_pts // 2,         # median
        (3 * n_pts) // 4,   # lower-mid
        n_pts - 1           # lowest CPM
    ]
    sample_pids = [int(cpm_df.iloc[idx]['global_idx']) for idx in chosen_indices]

    print("="*80)
    print("INVESTIGATION OF 5 PATIENTS ACROSS CPM SPECTRUM")
    print(f"Chosen Patient IDs: {sample_pids}")
    print("="*80)

    rows_task1 = []
    rows_task3 = []

    X_full_t = torch.tensor(X_full, dtype=torch.float)
    with torch.no_grad():
        full_logits_orig = model(X_full_t, edge_index_full)
        full_probs_orig = torch.softmax(full_logits_orig, dim=1)[:, 1].numpy()

    X_dev_t = torch.tensor(X_dev, dtype=torch.float)
    with torch.no_grad():
        dev_logits_orig = model(X_dev_t, edge_index_dev)
        dev_probs_orig = torch.softmax(dev_logits_orig, dim=1)[:, 1].numpy()

    for pid in sample_pids:
        cf_row = cf_df[cf_df['global_idx'] == pid].iloc[0]
        cpm_row = cpm_df[cpm_df['global_idx'] == pid].iloc[0]
        local_dev_idx = int(cf_row['local_dev_idx'])

        x_orig = X_dev[local_dev_idx].copy()
        x_cf = np.array([cf_row[f'cf_{feat}'] for feat in ALL_FEATURES], dtype=float)

        # A1: Normal CF (LightGBM)
        p1_orig = float(lgb_model.predict_proba(x_orig.reshape(1, -1))[0, 1])
        p1_cf   = float(lgb_model.predict_proba(x_cf.reshape(1, -1))[0, 1])

        # A2: Fixed-Graph CF on X_full (same population as A3!) AND on X_dev
        # Let's compute both to see if population size caused a difference!
        # Fixed-Graph on X_full:
        X_full_mod = X_full.copy()
        X_full_mod[pid] = x_cf
        with torch.no_grad():
            logits2_full = model(torch.tensor(X_full_mod, dtype=torch.float), edge_index_full)[pid]
            p2_full = torch.softmax(logits2_full, dim=0)[1].item()

        # Fixed-Graph on X_dev:
        X_dev_mod = X_dev.copy()
        X_dev_mod[local_dev_idx] = x_cf
        with torch.no_grad():
            logits2_dev = model(torch.tensor(X_dev_mod, dtype=torch.float), edge_index_dev)[local_dev_idx]
            p2_dev = torch.softmax(logits2_dev, dim=0)[1].item()

        # A3: Topology-Aware CF (rebuild on X_full)
        edge_new, weight_new, _ = rebuild_graph_for_patient(
            X_full, pid, x_cf, k=optimal_k, feature_names=ALL_FEATURES, metric='clinical'
        )
        with torch.no_grad():
            logits3 = model(torch.tensor(X_full_mod, dtype=torch.float), edge_new)[pid]
            p3_cf = torch.softmax(logits3, dim=0)[1].item()

        # Orig risk on X_full vs X_dev
        orig_logit_full = full_logits_orig[pid]
        orig_prob_full = full_probs_orig[pid]

        orig_logit_dev = dev_logits_orig[local_dev_idx]
        orig_prob_dev = dev_probs_orig[local_dev_idx]

        rows_task1.append({
            'Patient_ID': pid,
            'CPM_Score': cpm_row['CPM_score'],
            'Orig_Risk_GNN_full': orig_prob_full,
            'Orig_Risk_GNN_dev': orig_prob_dev,
            'Normal_CF_Risk_LGB': p1_cf,
            'Fixed_Graph_Risk_full': p2_full,
            'Fixed_Graph_Risk_dev': p2_dev,
            'Topo_Aware_Risk': p3_cf,
        })

        rows_task3.append({
            'Patient_ID': pid,
            'Orig_Logits_full [c0, c1]': f"[{orig_logit_full[0].item():.4f}, {orig_logit_full[1].item():.4f}]",
            'Orig_Logit_Diff (c1 - c0)': orig_logit_full[1].item() - orig_logit_full[0].item(),
            'Fixed_Logits_full [c0, c1]': f"[{logits2_full[0].item():.4f}, {logits2_full[1].item():.4f}]",
            'Fixed_Logit_Diff (c1 - c0)': logits2_full[1].item() - logits2_full[0].item(),
            'Topo_Logits [c0, c1]': f"[{logits3[0].item():.4f}, {logits3[1].item():.4f}]",
            'Topo_Logit_Diff (c1 - c0)': logits3[1].item() - logits3[0].item(),
            'Logit_Shift_Topo_vs_Fixed': (logits2_full[1].item() - logits2_full[0].item()) - (logits3[1].item() - logits3[0].item())
        })

    df1 = pd.DataFrame(rows_task1)
    df3 = pd.DataFrame(rows_task3)

    print("\n--- TASK 1: RAW RISK VALUES SIDE BY SIDE ---")
    print(df1.to_string(index=False))

    print("\n--- TASK 3: LOGIT VALUES & LOGIT SHIFTS (PRE-SOFTMAX) ---")
    print(df3.to_string(index=False))

    # TASK 4: Hand-check of one patient
    pid_check = sample_pids[0]
    print(f"\n--- TASK 4: DEEP HAND-CHECK FOR PATIENT #{pid_check} ---")
    rec = rebuild_records[pid_check]
    print(f"Old neighbors of #{pid_check}: {rec['orig_neighbors']}")
    print(f"New neighbors of #{pid_check}: {rec['new_neighbors']}")
    print(f"Old neighborhood CVD ratio: {rec['orig_cvd_ratio']:.4f}")
    print(f"New neighborhood CVD ratio: {rec['new_cvd_ratio']:.4f}")

    # Inspect what GraphSAGE conv is doing on these neighbors
    # Let's inspect neighbor feature embeddings and convolution aggregation
    new_neighs = rec['new_neighbors']
    orig_neighs = rec['orig_neighbors']
    print(f"Labels of old neighbors: {y_full[orig_neighs].tolist()}")
    print(f"Labels of new neighbors: {y_full[new_neighs].tolist()}")

    # Check predictions of old vs new neighbors
    with torch.no_grad():
        orig_neigh_probs = full_probs_orig[orig_neighs]
        new_neigh_probs = full_probs_orig[new_neighs]
    print(f"Mean GNN prob of old neighbors: {orig_neigh_probs.mean():.4f}")
    print(f"Mean GNN prob of new neighbors: {new_neigh_probs.mean():.4f}")

if __name__ == "__main__":
    investigate()
