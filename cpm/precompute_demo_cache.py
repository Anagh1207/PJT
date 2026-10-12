"""
=============================================================================
CPM Demo Precomputation Cache Generator
=============================================================================
Precomputes embeddings, 2D t-SNE coordinates, and patient metadata so the
Streamlit CPM What-If page loads and runs smoothly without lag.

Saves:
  results/cpm_demo_cache.pkl
=============================================================================
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLEANED_DATASET, SYNTHETIC_DATASET, RESULTS_DIR,
    ALL_FEATURES, TARGET_COLUMN, RANDOM_STATE,
    MODEL_DIR, GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT, CPM_CONFIG
)
from models.gnn_models import GraphSAGE
from graph_construction import build_patient_similarity_graph

CACHE_PKL = os.path.join(RESULTS_DIR, "cpm_demo_cache.pkl")
CHECKPOINT = os.path.join(MODEL_DIR, "graphsage_optimal.pt")
STEP5_CSV = os.path.join(RESULTS_DIR, "cpm_step5_scores.csv")
STEP6_CSV = os.path.join(RESULTS_DIR, "cpm_step6_three_way.csv")


def precompute_cache():
    print("\n" + "="*70)
    print("  PRECOMPUTING CPM DEMO CACHE")
    print("="*70)

    # 1. Load Data
    df_real = pd.read_csv(CLEANED_DATASET)
    df_syn  = pd.read_csv(SYNTHETIC_DATASET)
    df_full = pd.concat([df_real, df_syn], ignore_index=True)

    X_real = df_real[ALL_FEATURES].values.astype(float)
    y_real = df_real[TARGET_COLUMN].values.astype(int)

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

    # 2. Model & Graph
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    optimal_k = ck.get('optimal_k', CPM_CONFIG['graph_k'])
    model = GraphSAGE(len(ALL_FEATURES), GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    edge_index_full, edge_weight_full, _ = build_patient_similarity_graph(
        X_full, k=optimal_k, feature_names=ALL_FEATURES, metric='clinical')

    # 3. Embeddings
    X_full_t = torch.tensor(X_full, dtype=torch.float)
    with torch.no_grad():
        embeddings = model.get_embeddings(X_full_t, edge_index_full).numpy()
    print(f"  [Cache] Computed full embeddings: {embeddings.shape}")

    # 4. Background t-SNE coordinates for fast visualization
    # Subsample 500 representative nodes for background plot
    np.random.seed(42)
    sample_size = min(500, len(embeddings))
    bg_indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    bg_tsne_coords = tsne.fit_transform(embeddings[bg_indices])
    print(f"  [Cache] Precomputed background 2D t-SNE coordinates for {sample_size} nodes.")

    # 5. Fit LightGBM surrogate
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
    lgb_model.fit(X_dev, y_dev)
    print("  [Cache] LightGBM surrogate fitted.")

    # 6. Feature bounds & stats
    pop_min = X_dev.min(axis=0)
    pop_max = X_dev.max(axis=0)
    pop_range = pop_max - pop_min
    pop_range[pop_range == 0] = 1.0

    # 7. Pre-computed step 5 & 6 scores
    step5_df = pd.read_csv(STEP5_CSV) if os.path.exists(STEP5_CSV) else None
    step6_df = pd.read_csv(STEP6_CSV) if os.path.exists(STEP6_CSV) else None

    cache_data = {
        'X_full': X_full,
        'y_full': y_full,
        'X_dev': X_dev,
        'y_dev': y_dev,
        'idx_dev': idx_dev,
        'idx_test': idx_test,
        'optimal_k': optimal_k,
        'pop_min': pop_min,
        'pop_max': pop_max,
        'pop_range': pop_range,
        'embeddings': embeddings,
        'bg_indices': bg_indices,
        'bg_tsne_coords': bg_tsne_coords,
        'lgb_model': lgb_model,
        'step5_df': step5_df,
        'step6_df': step6_df,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CACHE_PKL, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"  [Cache] Successfully saved precomputed cache to: {CACHE_PKL}")
    print("="*70 + "\n")


if __name__ == "__main__":
    precompute_cache()
