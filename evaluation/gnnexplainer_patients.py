"""
=============================================================================
GNNExplainer — Per-Patient Explanations with Subgraph Visualization
=============================================================================
Phase 8 (enhanced): Run GNNExplainer on representative test patients
(TP, FP, TN, FN) and generate interpretable subgraph visualizations.

Author: AI Research Assistant
Date: August 2026
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import PLOT_DPI, PLOT_STYLE, FIGURES_DIR, RESULTS_DIR

plt.rcParams.update(PLOT_STYLE)


def select_representative_patients(y_true, y_pred, y_prob, test_mask):
    """
    Select representative patients from the test set:
    - True Positive (TP): correctly predicted CVD
    - False Positive (FP): wrongly predicted CVD
    - True Negative (TN): correctly predicted no CVD
    - False Negative (FN): missed CVD case
    - High-confidence TP: highest probability TP

    Parameters
    ----------
    y_true : array, true labels (full dataset)
    y_pred : array, predicted labels (test set only)
    y_prob : array, predicted probabilities (test set only)
    test_mask : tensor, boolean mask for test set

    Returns
    -------
    patients : dict {category: global_node_index}
    """
    test_indices = test_mask.nonzero(as_tuple=True)[0].numpy()
    patients = {}

    tp_mask = (y_pred == 1) & (y_true[test_indices] == 1)
    fp_mask = (y_pred == 1) & (y_true[test_indices] == 0)
    tn_mask = (y_pred == 0) & (y_true[test_indices] == 0)
    fn_mask = (y_pred == 0) & (y_true[test_indices] == 1)

    # TP — pick the one with highest probability
    if tp_mask.any():
        tp_local = np.where(tp_mask)[0]
        best_tp = tp_local[np.argmax(y_prob[tp_local])]
        patients['True Positive (TP)'] = test_indices[best_tp]

        # Also pick a "medium confidence" TP if available
        if len(tp_local) > 1:
            sorted_tp = tp_local[np.argsort(y_prob[tp_local])]
            mid_tp = sorted_tp[len(sorted_tp) // 2]
            patients['Mid-Confidence TP'] = test_indices[mid_tp]

    # FP — pick the one with highest probability (most confident mistake)
    if fp_mask.any():
        fp_local = np.where(fp_mask)[0]
        best_fp = fp_local[np.argmax(y_prob[fp_local])]
        patients['False Positive (FP)'] = test_indices[best_fp]

    # TN — pick one with lowest probability (most confident correct)
    if tn_mask.any():
        tn_local = np.where(tn_mask)[0]
        best_tn = tn_local[np.argmin(y_prob[tn_local])]
        patients['True Negative (TN)'] = test_indices[best_tn]

    # FN — pick one with lowest probability (most confident missed case)
    if fn_mask.any():
        fn_local = np.where(fn_mask)[0]
        best_fn = fn_local[np.argmin(y_prob[fn_local])]
        patients['False Negative (FN)'] = test_indices[best_fn]

    return patients


def run_gnnexplainer_on_patient(model, x, edge_index, node_idx,
                                 feature_names, num_epochs=200):
    """
    Run GNNExplainer on a single patient node.

    Returns
    -------
    feature_mask : np.ndarray or None
    edge_mask : np.ndarray or None
    explanation : Explanation object
    """
    from torch_geometric.explain import Explainer, GNNExplainer

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=num_epochs),
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

    feature_mask = None
    edge_mask = None

    if explanation.node_mask is not None:
        feature_mask = explanation.node_mask[node_idx].detach().cpu().numpy()

    if explanation.edge_mask is not None:
        edge_mask = explanation.edge_mask.detach().cpu().numpy()

    return feature_mask, edge_mask, explanation


def plot_patient_feature_importance(feature_mask, feature_names, node_idx,
                                     category, y_true_label, y_pred_label,
                                     y_prob_value, save_dir=None):
    """
    Plot feature importance bar chart for a single patient.
    """
    save_dir = save_dir or FIGURES_DIR

    fig, ax = plt.subplots(figsize=(9, 6))

    sorted_idx = np.argsort(feature_mask)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values = feature_mask[sorted_idx]

    # Color by importance (gradient)
    norm = plt.Normalize(vmin=sorted_values.min(), vmax=sorted_values.max())
    cmap = plt.cm.RdPu
    bar_colors = [cmap(norm(v)) for v in sorted_values]

    bars = ax.barh(sorted_features, sorted_values, color=bar_colors,
                   edgecolor='white', alpha=0.9)

    ax.set_xlabel('Feature Importance (GNNExplainer)', fontweight='bold')
    ax.set_title(f'GNNExplainer — Patient {node_idx} ({category})\n'
                 f'True: {"CVD" if y_true_label == 1 else "No CVD"} | '
                 f'Pred: {"CVD" if y_pred_label == 1 else "No CVD"} | '
                 f'P(CVD) = {y_prob_value:.4f}',
                 fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    safe_cat = category.lower().replace(' ', '_').replace('(', '').replace(')', '')
    path = os.path.join(save_dir, f'gnnexplainer_features_{safe_cat}.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Feature importance: {path}")
    return path


def plot_patient_subgraph(edge_index, edge_mask, node_idx, y_all,
                           category, top_k_edges=15, save_dir=None):
    """
    Visualize the local subgraph around a patient with edge importance.
    """
    save_dir = save_dir or FIGURES_DIR

    try:
        import networkx as nx
    except ImportError:
        print("  [WARN] networkx not available, skipping subgraph plot")
        return None

    edge_index_np = edge_index.cpu().numpy()
    y_np = y_all.numpy() if isinstance(y_all, torch.Tensor) else np.array(y_all)

    # Find edges connected to the target node
    src, dst = edge_index_np[0], edge_index_np[1]
    node_mask = (src == node_idx) | (dst == node_idx)
    local_edge_indices = np.where(node_mask)[0]

    if len(local_edge_indices) == 0:
        print(f"  [WARN] No edges found for node {node_idx}")
        return None

    # Sort by edge importance and take top-k
    if edge_mask is not None:
        local_importances = edge_mask[local_edge_indices]
        top_k = min(top_k_edges, len(local_edge_indices))
        top_local = local_edge_indices[np.argsort(local_importances)[-top_k:]]
    else:
        top_local = local_edge_indices[:top_k_edges]

    # Build subgraph
    G = nx.Graph()
    involved_nodes = set()
    edge_weights = []

    for idx in top_local:
        s, d = int(src[idx]), int(dst[idx])
        w = float(edge_mask[idx]) if edge_mask is not None else 0.5
        G.add_edge(s, d, weight=w)
        involved_nodes.add(s)
        involved_nodes.add(d)
        edge_weights.append(w)

    if len(G.nodes()) == 0:
        return None

    # Node colors by class
    node_colors = []
    for n in G.nodes():
        if n == node_idx:
            node_colors.append('#FFD600')  # Yellow for target
        elif n < len(y_np) and y_np[n] == 1:
            node_colors.append('#E53935')  # Red for CVD
        else:
            node_colors.append('#2196F3')  # Blue for no CVD

    # Node sizes
    node_sizes = [400 if n == node_idx else 150 for n in G.nodes()]

    # Edge widths based on importance
    if edge_weights:
        max_w = max(edge_weights) if max(edge_weights) > 0 else 1
        edge_widths = [1 + 4 * (w / max_w) for w in edge_weights]
    else:
        edge_widths = [1.5] * len(G.edges())

    # Edge colors
    if edge_weights:
        edge_colors = [plt.cm.Reds(w / max_w) for w in edge_weights]
    else:
        edge_colors = ['#999999'] * len(G.edges())

    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color=edge_colors, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors='white',
                           linewidths=1.5)

    # Label target node
    nx.draw_networkx_labels(G, pos, {node_idx: f'P{node_idx}'},
                            font_size=10, font_weight='bold', ax=ax)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#FFD600', edgecolor='gray',
                       label=f'Target Patient ({node_idx})'),
        mpatches.Patch(facecolor='#E53935', edgecolor='gray',
                       label='CVD Positive'),
        mpatches.Patch(facecolor='#2196F3', edgecolor='gray',
                       label='CVD Negative'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    ax.set_title(f'GNNExplainer Subgraph — Patient {node_idx} ({category})\n'
                 f'Top {len(top_local)} edges by importance',
                 fontweight='bold', fontsize=13)
    ax.axis('off')

    plt.tight_layout()
    safe_cat = category.lower().replace(' ', '_').replace('(', '').replace(')', '')
    path = os.path.join(save_dir, f'gnnexplainer_subgraph_{safe_cat}.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Subgraph visualization: {path}")
    return path


def run_gnnexplainer_patients(model, x, edge_index, y, test_mask,
                                y_pred, y_prob, feature_names,
                                edge_weight=None, save_dir=None):
    """
    Run GNNExplainer on representative patients and generate all outputs.

    Parameters
    ----------
    model : trained DPSN/GNN model
    x : tensor (N, F)
    edge_index : tensor (2, E)
    y : tensor (N,) or array
    test_mask : tensor (N,)
    y_pred : array, predictions for test nodes
    y_prob : array, predicted probabilities for test nodes
    feature_names : list of str
    edge_weight : optional tensor
    save_dir : str

    Returns
    -------
    results_df : pd.DataFrame with patient-level explanations
    """
    save_dir = save_dir or FIGURES_DIR

    print("\n  ── GNNExplainer: Per-Patient Analysis ────────────────────")

    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)
    test_indices = test_mask.nonzero(as_tuple=True)[0].numpy()

    # Select representative patients
    patients = select_representative_patients(y_np, y_pred, y_prob, test_mask)

    if not patients:
        print("  [WARN] No representative patients found.")
        return pd.DataFrame()

    print(f"  Selected {len(patients)} representative patients:")
    for cat, idx in patients.items():
        local_pos = np.where(test_indices == idx)[0][0]
        print(f"    {cat}: Node {idx} "
              f"(true={y_np[idx]}, pred={y_pred[local_pos]}, "
              f"prob={y_prob[local_pos]:.4f})")

    # Run GNNExplainer on each
    model.eval()
    results_rows = []

    for category, global_idx in patients.items():
        print(f"\n  Running GNNExplainer for: {category} (Node {global_idx})...")

        try:
            feature_mask, edge_mask, explanation = run_gnnexplainer_on_patient(
                model, x, edge_index, global_idx, feature_names)

            # Get this patient's prediction info
            local_pos = np.where(test_indices == global_idx)[0][0]
            true_label = y_np[global_idx]
            pred_label = y_pred[local_pos]
            prob_value = y_prob[local_pos]

            # Plot feature importance
            if feature_mask is not None:
                plot_patient_feature_importance(
                    feature_mask, feature_names, global_idx,
                    category, true_label, pred_label, prob_value,
                    save_dir=save_dir)

                # Record top features
                top_3_idx = np.argsort(feature_mask)[-3:][::-1]
                top_features = [feature_names[i] for i in top_3_idx]
                top_weights = [float(feature_mask[i]) for i in top_3_idx]
            else:
                top_features = ['N/A'] * 3
                top_weights = [0.0] * 3

            # Plot subgraph
            plot_patient_subgraph(
                edge_index, edge_mask, global_idx, y_np,
                category, top_k_edges=15, save_dir=save_dir)

            results_rows.append({
                'Category': category,
                'Node Index': global_idx,
                'True Label': int(true_label),
                'Predicted Label': int(pred_label),
                'P(CVD)': float(prob_value),
                'Top Feature 1': top_features[0],
                'Weight 1': top_weights[0],
                'Top Feature 2': top_features[1],
                'Weight 2': top_weights[1],
                'Top Feature 3': top_features[2],
                'Weight 3': top_weights[2],
                'Num Important Edges': int(np.sum(edge_mask > 0.1))
                    if edge_mask is not None else 0,
            })

        except Exception as e:
            print(f"  [WARN] GNNExplainer failed for {category}: {e}")
            import traceback
            traceback.print_exc()

    results_df = pd.DataFrame(results_rows)
    csv_path = os.path.join(RESULTS_DIR, 'gnnexplainer_patient_explanations.csv')
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n  [OK] Patient explanations saved: {csv_path}")

    return results_df
