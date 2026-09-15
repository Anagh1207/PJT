"""
=============================================================================
Topology Metrics Module — Graph Shift & Neighborhood Dynamics
=============================================================================
Quantifies graph topology changes when patient clinical features are modified
for counterfactual explanations:
  1. Jaccard Neighbor Overlap (Neighborhood stability)
  2. Neighborhood CVD Risk Composition Change (1-hop risk shift)
  3. Graph Embedding Space Movement (Representation trajectory)
  4. Composite Topology Shift Index

Author: AI Research Assistant
Date: September 2026
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VISUALIZATIONS_DIR, FIGURES_DIR, PLOT_DPI

def jaccard_neighbor_overlap(neighbors_orig: list, neighbors_new: list) -> float:
    """
    Computes Jaccard Similarity between original neighbors and new neighbors:
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    set_orig = set(neighbors_orig)
    set_new = set(neighbors_new)
    
    if not set_orig and not set_new:
        return 1.0
    
    intersection = len(set_orig.intersection(set_new))
    union = len(set_orig.union(set_new))
    
    return float(intersection / union) if union > 0 else 0.0


def neighborhood_risk_change(stats_orig: dict, stats_new: dict) -> dict:
    """
    Measures change in CVD-risk composition of patient's neighborhood.
    """
    ratio_orig = stats_orig.get('cvd_risk_ratio', 0.0)
    ratio_new = stats_new.get('cvd_risk_ratio', 0.0)
    
    return {
        'orig_cvd_ratio': ratio_orig,
        'new_cvd_ratio': ratio_new,
        'risk_reduction_ratio': float(ratio_orig - ratio_new)
    }


def embedding_displacement(emb_orig: np.ndarray, emb_new: np.ndarray) -> float:
    """
    Computes Euclidean L2 distance in GNN embedding space:
    d = ||h_orig - h_new||_2
    """
    return float(np.linalg.norm(emb_orig - emb_new))


def compute_composite_topology_metrics(
    patient_idx: int,
    neighbors_orig: list,
    neighbors_new: list,
    stats_orig: dict,
    stats_new: dict,
    emb_orig: np.ndarray = None,
    emb_new: np.ndarray = None
) -> dict:
    """
    Calculates comprehensive topology shift metrics for a patient counterfactual.
    """
    jaccard = jaccard_neighbor_overlap(neighbors_orig, neighbors_new)
    risk_stats = neighborhood_risk_change(stats_orig, stats_new)
    
    emb_dist = 0.0
    if emb_orig is not None and emb_new is not None:
        emb_dist = embedding_displacement(emb_orig, emb_new)
        
    return {
        'patient_idx': patient_idx,
        'jaccard_overlap': jaccard,
        'orig_neighbors_count': len(neighbors_orig),
        'new_neighbors_count': len(neighbors_new),
        'orig_cvd_ratio': risk_stats['orig_cvd_ratio'],
        'new_cvd_ratio': risk_stats['new_cvd_ratio'],
        'risk_ratio_change': risk_stats['risk_reduction_ratio'],
        'embedding_distance': emb_dist
    }


def plot_topology_shift_distributions(results_df: pd.DataFrame, save_dir: str = VISUALIZATIONS_DIR):
    """Generates visualizations summarizing topology shift metrics across test patients."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 1. Jaccard Overlap Distribution
    sns.histplot(results_df['jaccard_overlap'], ax=axes[0], color='#1E88E5', kde=True, bins=15)
    axes[0].set_title('Jaccard Neighbor Overlap', fontweight='bold')
    axes[0].set_xlabel('Overlap Score (1.0 = Unchanged, 0.0 = New Neighbors)')
    
    # 2. Neighborhood Risk Ratio Shift
    sns.histplot(results_df['risk_ratio_change'], ax=axes[1], color='#43A047', kde=True, bins=15)
    axes[1].set_title('Neighborhood CVD Risk Reduction', fontweight='bold')
    axes[1].set_xlabel('Δ CVD Neighbor Ratio (Original - New)')
    
    # 3. Embedding Distance
    sns.histplot(results_df['embedding_distance'], ax=axes[2], color='#8E24AA', kde=True, bins=15)
    axes[2].set_title('Graph Embedding Displacement', fontweight='bold')
    axes[2].set_xlabel('L2 Distance in Embedding Space')
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "topology_shift_distributions.png"), dpi=PLOT_DPI)
    fig.savefig(os.path.join(FIGURES_DIR, "Fig_Topology_Shift.pdf"))
    plt.close()
    
    print(f"[TopologyMetrics] Saved topology shift diagnostic plot to {save_dir}")
