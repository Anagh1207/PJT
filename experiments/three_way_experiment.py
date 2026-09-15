"""
=============================================================================
Three-Way Counterfactual Experiment — Comparative Evaluation Framework
=============================================================================
Compares three counterfactual paradigms across high-risk patients:
  Approach 1: Normal CF (Tabular LightGBM baseline, no graph)
  Approach 2: Fixed-Graph CF (GraphSAGE, frozen topology)
  Approach 3: Topology-Aware CF (GraphSAGE, rebuilt graph topology)

Produces comparative summary tables, LaTeX tables, and publication figures.

Author: AI Research Assistant
Date: September 2026
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EXPERIMENT_CONFIG, ALL_FEATURES, TARGET_COLUMN,
    RESULTS_DIR, VISUALIZATIONS_DIR, FIGURES_DIR, PLOT_DPI
)
from evaluation.topology_counterfactual import (
    generate_counterfactual_vector, run_topology_aware_counterfactual_analysis
)

def run_three_way_experiment(
    graphsage_model,
    X: np.ndarray,
    y: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    feature_names: list = ALL_FEATURES,
    k: int = 10,
    n_patients: int = 50
) -> pd.DataFrame:
    """
    Runs the three-way counterfactual comparison across n_patients high-risk CVD patients.
    """
    print("\n" + "="*70)
    print(f"  RUNNING THREE-WAY COUNTERFACTUAL EXPERIMENT (N = {n_patients} Patients)")
    print("="*70)
    
    # Train tabular LightGBM baseline for Approach 1 (Normal CF)
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    lgb_model.fit(X, y)
    
    # Identify high-risk patients (predicted risk > 0.6 by GraphSAGE)
    graphsage_model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float)
    with torch.no_grad():
        probs = torch.softmax(graphsage_model(X_tensor, edge_index, edge_weight), dim=1)[:, 1].numpy()
        
    high_risk_indices = np.where((probs > 0.6) & (y == 1))[0]
    if len(high_risk_indices) < n_patients:
        high_risk_indices = np.where(probs > 0.5)[0]
        
    target_patient_indices = high_risk_indices[:n_patients]
    print(f"[ThreeWayExp] Evaluating {len(target_patient_indices)} high-risk patients...")
    
    results = []
    
    for idx_num, p_idx in enumerate(target_patient_indices):
        # 1. Approach 1: Normal CF (LightGBM)
        x_cf_lgb = generate_counterfactual_vector(
            lgb_model, X, p_idx, model_type='lightgbm'
        )
        lgb_orig_risk = float(lgb_model.predict_proba(X[p_idx].reshape(1, -1))[0, 1])
        lgb_cf_risk = float(lgb_model.predict_proba(x_cf_lgb.reshape(1, -1))[0, 1])
        lgb_risk_red = lgb_orig_risk - lgb_cf_risk
        lgb_n_changed = int((np.abs(x_cf_lgb - X[p_idx]) > 1e-4).sum())
        lgb_l2_dist = float(np.linalg.norm(x_cf_lgb - X[p_idx]))
        
        # 2 & 3. Approach 2 (Fixed-Graph CF) & Approach 3 (Topology-Aware CF)
        topo_res = run_topology_aware_counterfactual_analysis(
            graphsage_model, X, y, edge_index, edge_weight, p_idx, feature_names=feature_names, k=k
        )
        
        metrics = topo_res['topology_metrics']
        
        results.append({
            'patient_idx': p_idx,
            # Approach 1: Normal CF
            'normal_risk_orig': lgb_orig_risk,
            'normal_risk_cf': lgb_cf_risk,
            'normal_risk_red': lgb_risk_red,
            'normal_n_changed': lgb_n_changed,
            'normal_l2_dist': lgb_l2_dist,
            
            # Approach 2: Fixed-Graph CF
            'fixed_risk_orig': topo_res['orig_risk_prob'],
            'fixed_risk_cf': topo_res['fixed_graph_risk_prob'],
            'fixed_risk_red': topo_res['risk_reduction_fixed'],
            'fixed_n_changed': topo_res['n_features_changed'],
            'fixed_l2_dist': topo_res['l2_feature_distance'],
            
            # Approach 3: Topology-Aware CF
            'topo_risk_orig': topo_res['orig_risk_prob'],
            'topo_risk_cf': topo_res['topology_aware_risk_prob'],
            'topo_risk_red': topo_res['risk_reduction_topo'],
            'topo_n_changed': topo_res['n_features_changed'],
            'topo_l2_dist': topo_res['l2_feature_distance'],
            'jaccard_overlap': metrics['jaccard_overlap'],
            'orig_cvd_ratio': metrics['orig_cvd_ratio'],
            'new_cvd_ratio': metrics['new_cvd_ratio'],
            'risk_ratio_change': metrics['risk_ratio_change'],
            'embedding_distance': metrics['embedding_distance']
        })
        
    df_results = pd.DataFrame(results)
    
    # Compute summary comparative table
    summary_data = [
        {
            'Approach': 'Normal CF (LightGBM)',
            'Graph Topology': 'None (Tabular)',
            'Mean Risk Reduction': f"{df_results['normal_risk_red'].mean():.4f}",
            'Mean Features Changed': f"{df_results['normal_n_changed'].mean():.2f}",
            'Mean L2 Distance': f"{df_results['normal_l2_dist'].mean():.4f}",
            'Jaccard Overlap': 'N/A',
            'Neighborhood CVD Change': 'N/A',
            'Embedding Shift': 'N/A'
        },
        {
            'Approach': 'Fixed-Graph CF (GraphSAGE)',
            'Graph Topology': 'Frozen Original',
            'Mean Risk Reduction': f"{df_results['fixed_risk_red'].mean():.4f}",
            'Mean Features Changed': f"{df_results['fixed_n_changed'].mean():.2f}",
            'Mean L2 Distance': f"{df_results['fixed_l2_dist'].mean():.4f}",
            'Jaccard Overlap': '1.0000',
            'Neighborhood CVD Change': '0.0000',
            'Embedding Shift': '0.0000'
        },
        {
            'Approach': 'Topology-Aware CF (GraphSAGE)',
            'Graph Topology': 'Dynamic Rebuilt',
            'Mean Risk Reduction': f"{df_results['topo_risk_red'].mean():.4f}",
            'Mean Features Changed': f"{df_results['topo_n_changed'].mean():.2f}",
            'Mean L2 Distance': f"{df_results['topo_l2_dist'].mean():.4f}",
            'Jaccard Overlap': f"{df_results['jaccard_overlap'].mean():.4f}",
            'Neighborhood CVD Change': f"{df_results['risk_ratio_change'].mean():.4f}",
            'Embedding Shift': f"{df_results['embedding_distance'].mean():.4f}"
        }
    ]
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*70)
    print("  THREE-WAY EXPERIMENT SUMMARY COMPARISON TABLE")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    # Export results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "three_way_cf_comparison.csv")
    summary_csv_path = os.path.join(RESULTS_DIR, "three_way_cf_summary.csv")
    df_results.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Save LaTeX table
    latex_path = os.path.join(RESULTS_DIR, "three_way_cf_table.tex")
    summary_df.to_latex(latex_path, index=False, caption="Comparative Performance of Three Counterfactual Paradigms")
    
    plot_three_way_comparison(df_results)
    
    return summary_df


def plot_three_way_comparison(df_results: pd.DataFrame, save_dir: str = VISUALIZATIONS_DIR):
    """Generates comparative plot showing risk reduction and topology shift across paradigms."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Risk Reduction Comparison
    risk_df = pd.DataFrame({
        'Normal CF (LightGBM)': df_results['normal_risk_red'],
        'Fixed-Graph CF': df_results['fixed_risk_red'],
        'Topology-Aware CF': df_results['topo_risk_red']
    })
    sns.boxplot(data=risk_df, ax=axes[0], palette=['#2E86AB', '#A23B72', '#D81B60'])
    axes[0].set_title('Predicted Risk Score Reduction Across Paradigms', fontweight='bold')
    axes[0].set_ylabel('Risk Reduction (Original Risk - CF Risk)')
    
    # 2. Neighborhood Risk Shift (Topology-Aware)
    sns.scatterplot(
        data=df_results, x='orig_cvd_ratio', y='new_cvd_ratio',
        size='embedding_distance', hue='jaccard_overlap',
        palette='viridis', ax=axes[1], sizes=(40, 200)
    )
    axes[1].plot([0, 1], [0, 1], 'r--', label='No Topology Change')
    axes[1].set_title('1-Hop Neighborhood CVD Proportion Transition', fontweight='bold')
    axes[1].set_xlabel('Original Neighborhood CVD Ratio')
    axes[1].set_ylabel('Rebuilt Counterfactual CVD Ratio')
    axes[1].legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "three_way_counterfactual_comparison.png"), dpi=PLOT_DPI)
    fig.savefig(os.path.join(FIGURES_DIR, "Fig_Three_Way_Comparison.pdf"))
    plt.close()
    
    print(f"[ThreeWayExp] Saved comparative figures to {save_dir}")


if __name__ == "__main__":
    print("Three-way experiment module initialized.")
