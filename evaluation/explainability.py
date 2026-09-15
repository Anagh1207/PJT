"""
=============================================================================
Explainability Module — SHAP + GNNExplainer
=============================================================================
Phase 8: Interpretability for classical ML and GNN models.

Implements:
  - SHAP for classical ML (TreeExplainer / KernelExplainer)
  - GNNExplainer for GNN models
  - Feature importance rankings
  - Patient-level explanations

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import PLOT_DPI, PLOT_STYLE, FIGURES_DIR

plt.rcParams.update(PLOT_STYLE)


def compute_shap_values(model, X_train, X_test, feature_names,
                        model_name='Model', save_dir=None):
    """
    Compute SHAP values for a classical ML model.

    Uses TreeExplainer for tree-based models, KernelExplainer otherwise.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_train : array-like (for background)
    X_test : array-like (to explain)
    feature_names : list of str
    model_name : str
    save_dir : str

    Returns
    -------
    shap_values : np.ndarray
    feature_importance : pd.DataFrame
    """
    import shap

    save_dir = save_dir or FIGURES_DIR

    # Choose explainer
    tree_models = ['RandomForestClassifier', 'XGBClassifier',
                   'LGBMClassifier', 'DecisionTreeClassifier',
                   'GradientBoostingClassifier']

    model_type = type(model).__name__

    if model_type in tree_models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        # KernelExplainer for LR, SVM etc.
        background = shap.sample(X_train, min(100, len(X_train)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test[:100])

    # Handle multi-output (take positive class)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Feature importance from mean |SHAP|
    mean_abs_shap = np.abs(sv).mean(axis=0)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap
    }).sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)

    # Summary bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    fi = feature_importance.sort_values('Mean |SHAP|', ascending=True)
    ax.barh(fi['Feature'], fi['Mean |SHAP|'], color='#D81B60',
            edgecolor='white', alpha=0.85)
    ax.set_xlabel('Mean |SHAP value|', fontweight='bold')
    ax.set_title(f'SHAP Feature Importance — {model_name}',
                 fontweight='bold', fontsize=13)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()

    path = os.path.join(save_dir, f'shap_importance_{model_name.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] SHAP importance plot: {path}")

    # Beeswarm plot
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        if hasattr(X_test, 'values'):
            X_for_plot = X_test
        else:
            X_for_plot = pd.DataFrame(X_test, columns=feature_names)

        shap.summary_plot(sv, X_for_plot[:100],
                         feature_names=feature_names,
                         show=False, max_display=14)
        path2 = os.path.join(save_dir, f'shap_summary_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(path2, dpi=PLOT_DPI, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close('all')
        print(f"  [OK] SHAP summary plot: {path2}")
    except Exception as e:
        print(f"  [WARN] SHAP summary plot failed: {e}")

    return sv, feature_importance


def run_gnn_explainer(model, x, edge_index, node_idx, feature_names,
                      edge_weight=None, save_dir=None):
    """
    Run GNNExplainer on a specific node.

    Parameters
    ----------
    model : trained GNN model
    x : tensor (N, F)
    edge_index : tensor (2, E)
    node_idx : int
        Index of the node to explain.
    feature_names : list of str
    edge_weight : tensor, optional
    save_dir : str

    Returns
    -------
    feature_mask : tensor — importance of each feature
    edge_mask : tensor — importance of each edge
    """
    from torch_geometric.explain import Explainer, GNNExplainer

    save_dir = save_dir or FIGURES_DIR

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    explanation = explainer(x, edge_index, index=node_idx)

    feature_mask = explanation.node_mask[node_idx] if explanation.node_mask is not None else None
    edge_mask = explanation.edge_mask if explanation.edge_mask is not None else None

    # Feature importance plot
    if feature_mask is not None:
        fm = feature_mask.detach().numpy()
        fig, ax = plt.subplots(figsize=(8, 5))
        sorted_idx = np.argsort(fm)
        ax.barh([feature_names[i] for i in sorted_idx], fm[sorted_idx],
                color='#8E24AA', edgecolor='white', alpha=0.85)
        ax.set_xlabel('Feature Importance (GNNExplainer)', fontweight='bold')
        ax.set_title(f'GNN Explanation — Node {node_idx}',
                     fontweight='bold', fontsize=13)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()

        path = os.path.join(save_dir, f'gnn_explanation_node_{node_idx}.png')
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  [OK] GNN explanation: {path}")

    return feature_mask, edge_mask


def compute_gnn_feature_importance(model, x, edge_index, test_mask,
                                   feature_names, edge_weight=None,
                                   num_nodes=50, save_dir=None):
    """
    Aggregate GNNExplainer results across multiple test nodes
    to get global feature importance.

    Parameters
    ----------
    model : trained GNN
    x, edge_index : tensors
    test_mask : tensor
    feature_names : list
    num_nodes : int — how many test nodes to explain
    save_dir : str

    Returns
    -------
    importance_df : pd.DataFrame
    """
    from torch_geometric.explain import Explainer, GNNExplainer

    save_dir = save_dir or FIGURES_DIR

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    test_indices = test_mask.nonzero(as_tuple=True)[0].numpy()
    sample_indices = test_indices[:min(num_nodes, len(test_indices))]

    all_masks = []
    for idx in sample_indices:
        try:
            explanation = explainer(x, edge_index, index=int(idx))
            if explanation.node_mask is not None:
                fm = explanation.node_mask[int(idx)].detach().numpy()
                all_masks.append(fm)
        except Exception:
            continue

    if not all_masks:
        print("  [WARN] No valid GNN explanations obtained.")
        return pd.DataFrame()

    avg_mask = np.mean(all_masks, axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'GNN Importance': avg_mask
    }).sort_values('GNN Importance', ascending=False).reset_index(drop=True)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    fi = importance_df.sort_values('GNN Importance', ascending=True)
    ax.barh(fi['Feature'], fi['GNN Importance'], color='#8E24AA',
            edgecolor='white', alpha=0.85)
    ax.set_xlabel('Mean Feature Importance (GNNExplainer)', fontweight='bold')
    ax.set_title('GNN Feature Importance (Aggregated)',
                 fontweight='bold', fontsize=13)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()

    path = os.path.join(save_dir, 'gnn_feature_importance_aggregated.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Aggregated GNN feature importance: {path}")

    return importance_df
