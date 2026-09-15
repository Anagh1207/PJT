"""
=============================================================================
Publication Figures Module — Phase 9
=============================================================================
Generates all 15 publication-quality figures at 300 DPI for IEEE paper.

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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE

from config import (
    PLOT_DPI, PLOT_STYLE, FIGURES_DIR, ALL_MODEL_COLORS,
    COLOR_NEGATIVE, COLOR_POSITIVE, GNN_MODEL_COLORS, GRAPH_COLORS
)

plt.rcParams.update(PLOT_STYLE)


def fig01_model_comparison(all_results, save_dir=None):
    """Bar chart comparing all models across metrics."""
    save_dir = save_dir or FIGURES_DIR
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    models = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(metrics))
    n = len(models)
    width = 0.8 / n
    offsets = np.linspace(-(n-1)*width/2, (n-1)*width/2, n)

    for i, model in enumerate(models):
        r = all_results[model]
        vals = [r.get(m, 0) for m in metrics]
        color = ALL_MODEL_COLORS.get(model, '#333')
        ax.bar(x + offsets[i], vals, width, label=model, color=color,
               edgecolor='white', alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Comprehensive Model Comparison — All Methods',
                 fontweight='bold', fontsize=14)
    ax.set_ylim([0, 1.15])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=4, fontsize=8, frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    path = os.path.join(save_dir, 'fig01_model_comparison.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Fig 01: {path}")


def fig02_roc_comparison(all_results, y_test, save_dir=None):
    """Combined ROC curves for all models."""
    save_dir = save_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, r in all_results.items():
        if 'y_prob' not in r:
            continue
        # Use per-model y_true if available, else fall back to global y_test
        y_true = r.get('y_true', y_test)
        fpr, tpr, _ = roc_curve(y_true, r['y_prob'])
        color = ALL_MODEL_COLORS.get(name, '#333')
        lw = 3.0 if name == 'DPSN' else 1.8
        ax.plot(fpr, tpr, color=color, linewidth=lw,
                label=f"{name} ({r['roc_auc']:.4f})",
                alpha=0.9)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curves \u2014 All Models', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=8, frameon=True)
    ax.set_aspect('equal')
    plt.tight_layout()

    path = os.path.join(save_dir, 'fig02_roc_comparison.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Fig 02: {path}")


def fig03_pr_comparison(all_results, y_test, save_dir=None):
    """Combined PR curves."""
    save_dir = save_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(8, 7))
    baseline = sum(np.array(y_test) == 1) / len(y_test)

    for name, r in all_results.items():
        if 'y_prob' not in r:
            continue
        # Use per-model y_true if available, else fall back to global y_test
        y_true = r.get('y_true', y_test)
        prec, rec, _ = precision_recall_curve(y_true, r['y_prob'])
        ap = average_precision_score(y_true, r['y_prob'])
        color = ALL_MODEL_COLORS.get(name, '#333')
        lw = 3.0 if name == 'DPSN' else 1.8
        ax.plot(rec, prec, color=color, linewidth=lw,
                label=f"{name} (AP={ap:.4f})", alpha=0.9)

    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1, alpha=0.4)
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision-Recall Curves \u2014 All Models',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=8, frameon=True)
    plt.tight_layout()

    path = os.path.join(save_dir, 'fig03_pr_comparison.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Fig 03: {path}")


def fig04_05_06_metric_comparisons(all_results, save_dir=None):
    """Individual bar charts for Accuracy, F1, and AUC."""
    save_dir = save_dir or FIGURES_DIR

    for metric, title, fname in [
        ('accuracy', 'Accuracy', 'fig04_accuracy_comparison'),
        ('f1', 'F1 Score', 'fig05_f1_comparison'),
        ('roc_auc', 'ROC-AUC', 'fig06_auc_comparison'),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))
        models = list(all_results.keys())
        values = [all_results[m].get(metric, 0) for m in models]
        colors = [ALL_MODEL_COLORS.get(m, '#333') for m in models]

        bars = ax.bar(models, values, color=colors, edgecolor='white',
                      alpha=0.88, width=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.4f}', ha='center', va='bottom', fontweight='bold',
                    fontsize=8)

        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(f'{title} Comparison — All Models',
                     fontweight='bold', fontsize=13)
        ax.set_ylim([0, 1.12])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=30, ha='right', fontsize=9)
        plt.tight_layout()

        path = os.path.join(save_dir, f'{fname}.png')
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [OK] {fname}: {path}")


def fig07_graph_construction_workflow(save_dir=None):
    """Schematic diagram of graph construction workflow."""
    save_dir = save_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.set_axis_off()

    boxes = [
        (1, 3, 'Patient\nFeatures\n(N×14)', '#E3F2FD', '#1565C0'),
        (4, 4.5, 'Euclidean\nKNN', '#E8F5E9', '#2E7D32'),
        (4, 3, 'Cosine\nSimilarity', '#FFF3E0', '#E65100'),
        (4, 1.5, 'Clinical\nSimilarity', '#FCE4EC', '#AD1457'),
        (7.5, 3, 'Patient\nSimilarity\nGraph', '#F3E5F5', '#8E24AA'),
        (10.5, 3, 'GNN\nModel', '#FFEBEE', '#D81B60'),
        (13, 3, 'CVD\nPrediction', '#E8EAF6', '#283593'),
    ]

    for x, y, text, facecolor, edgecolor in boxes:
        fancy = FancyBboxPatch((x-0.9, y-0.6), 1.8, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor=facecolor, edgecolor=edgecolor,
                               linewidth=2)
        ax.add_patch(fancy)
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                fontweight='bold', color=edgecolor)

    # Arrows
    arrows = [(2, 3, 3, 4.5), (2, 3, 3, 3), (2, 3, 3, 1.5),
              (5, 4.5, 6.5, 3), (5, 3, 6.5, 3), (5, 1.5, 6.5, 3),
              (8.5, 3, 9.5, 3), (11.5, 3, 12, 3)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555',
                                   lw=1.5, connectionstyle='arc3,rad=0'))

    ax.set_title('Graph Construction Workflow',
                 fontweight='bold', fontsize=14, pad=20)

    path = os.path.join(save_dir, 'fig07_graph_construction_workflow.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Fig 07: {path}")


def fig08_gnn_architecture(save_dir=None):
    """GNN architecture diagram."""
    save_dir = save_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.set_axis_off()

    blocks = [
        (1, 2.5, 'Input\nFeatures\n(14)', '#E3F2FD', '#1565C0'),
        (3.5, 2.5, 'GNN Layer 1\n(Conv + BN\n+ Activation)', '#E8F5E9', '#2E7D32'),
        (6, 2.5, 'Dropout', '#FFF8E1', '#F57F17'),
        (8.5, 2.5, 'GNN Layer 2\n(Conv + BN\n+ Activation)', '#F3E5F5', '#8E24AA'),
        (11, 2.5, 'Dropout', '#FFF8E1', '#F57F17'),
        (13, 2.5, 'Classifier\n(Linear→2)', '#FFEBEE', '#D81B60'),
    ]

    for x, y, text, fc, ec in blocks:
        fancy = FancyBboxPatch((x-0.9, y-0.7), 1.8, 1.4,
                               boxstyle="round,pad=0.1",
                               facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(fancy)
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                fontweight='bold', color=ec)

    for i in range(len(blocks) - 1):
        x1 = blocks[i][0] + 0.9
        x2 = blocks[i+1][0] - 0.9
        ax.annotate('', xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2))

    ax.text(7, 4.5, 'Standard GNN Architecture (GCN / GraphSAGE / GAT)',
            ha='center', fontweight='bold', fontsize=14, color='#333')

    path = os.path.join(save_dir, 'fig08_gnn_architecture.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Fig 08: {path}")


def fig09_dpsn_architecture(save_dir=None):
    """DPSN architecture diagram."""
    save_dir = save_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.set_axis_off()

    # Main pipeline
    blocks = [
        (1.5, 3.5, 'Input\nFeatures\n(N×14)', '#E3F2FD', '#1565C0'),
        (4, 5, 'Feature\nEncoder', '#E8F5E9', '#2E7D32'),
        (4, 2, 'Learnable\nFeature\nWeights', '#FFF3E0', '#E65100'),
        (7, 5, 'Dynamic Edge\nReweighting\nMLP', '#FCE4EC', '#AD1457'),
        (7, 2, 'Attention\nEdge\nScoring', '#F3E5F5', '#8E24AA'),
        (10, 3.5, 'GAT\nBackbone\n(2 layers)', '#FFEBEE', '#D81B60'),
        (13, 5, 'Node\nEmbeddings', '#E8EAF6', '#283593'),
        (13, 2, 'Classifier\n→ CVD Risk', '#FBE9E7', '#BF360C'),
    ]

    for x, y, text, fc, ec in blocks:
        w, h = 2.0, 1.2
        fancy = FancyBboxPatch((x-w/2, y-h/2), w, h,
                               boxstyle="round,pad=0.1",
                               facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(fancy)
        ax.text(x, y, text, ha='center', va='center', fontsize=7.5,
                fontweight='bold', color=ec)

    # Arrows
    connections = [
        (1.5, 3.5, 4, 5), (1.5, 3.5, 4, 2),
        (4, 5, 7, 5), (4, 2, 7, 2),
        (7, 5, 10, 3.5), (7, 2, 10, 3.5),
        (10, 3.5, 13, 5), (10, 3.5, 13, 2),
    ]
    for x1, y1, x2, y2 in connections:
        ax.annotate('', xy=(x2-1, y2), xytext=(x1+1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555',
                                   lw=1.5, connectionstyle='arc3,rad=0.1'))

    # Dynamic feedback loop
    ax.annotate('', xy=(7, 5.8), xytext=(10, 4.3),
                arrowprops=dict(arrowstyle='->', color='#D81B60',
                               lw=2, linestyle='--',
                               connectionstyle='arc3,rad=-0.4'))
    ax.text(9, 6.2, 'Dynamic\nTopology\nUpdate', ha='center',
            fontsize=7, color='#D81B60', fontstyle='italic')

    ax.text(8, 0.3, 'Dynamic Patient Similarity Network (DPSN) — Proposed Architecture',
            ha='center', fontweight='bold', fontsize=14, color='#333')

    path = os.path.join(save_dir, 'fig09_dpsn_architecture.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Fig 09: {path}")


def fig10_ablation_chart(ablation_results, save_dir=None):
    """Ablation study comparison chart."""
    save_dir = save_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(12, 6))

    variants = list(ablation_results.keys())
    metrics = ['accuracy', 'f1', 'roc_auc', 'mcc']
    labels = ['Accuracy', 'F1', 'ROC-AUC', 'MCC']
    x = np.arange(len(variants))

    colors_m = ['#1E88E5', '#43A047', '#8E24AA', '#F57C00']

    width = 0.18
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [ablation_results[v].get(metric, 0) for v in variants]
        ax.bar(x + i * width - 1.5 * width, vals, width, label=label,
               color=colors_m[i], edgecolor='white', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('A: ', '').replace('B: ', '')
                        .replace('C: ', '').replace('D: ', '')
                        for v in variants], fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Ablation Study — Component Contribution Analysis',
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=9, frameon=True)
    ax.set_ylim([0, 1.12])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    path = os.path.join(save_dir, 'fig10_ablation_study.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Fig 10: {path}")


def fig11_statistical_significance(sig_df, save_dir=None):
    """Statistical significance chart with error bars."""
    save_dir = save_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(12, 6))

    if sig_df is not None and len(sig_df) > 0:
        baselines = sig_df['Baseline'].tolist()
        improvements = [float(x) for x in sig_df['Improvement'].tolist()]
        significant = ['✓' in str(x) for x in sig_df['t-test Sig.'].tolist()]

        colors = ['#43A047' if s else '#E53935' for s in significant]
        bars = ax.bar(baselines, improvements, color=colors,
                      edgecolor='white', alpha=0.85)
        ax.axhline(y=0, color='black', linestyle='-', lw=0.8)
        ax.set_ylabel('Improvement over Baseline', fontweight='bold')
        ax.set_title('Statistical Significance of DPSN Improvements',
                     fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=30, ha='right', fontsize=9)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#43A047', label='Significant (p<0.05)'),
                          Patch(facecolor='#E53935', label='Not Significant')]
        ax.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig11_statistical_significance.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Fig 11: {path}")


def fig14_15_embedding_visualization(embeddings, labels, save_dir=None):
    """t-SNE and UMAP node embedding visualizations."""
    save_dir = save_dir or FIGURES_DIR

    # t-SNE
    print("    Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls, color, label in [(0, COLOR_NEGATIVE, 'No CVD'),
                               (1, COLOR_POSITIVE, 'CVD')]:
        mask = labels == cls
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color, label=label,
                   alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    ax.set_xlabel('t-SNE Dimension 1', fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontweight='bold')
    ax.set_title('Node Embeddings — t-SNE Visualization',
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=10, frameon=True)
    plt.tight_layout()

    path = os.path.join(save_dir, 'fig14_tsne_embeddings.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Fig 14: {path}")

    # UMAP
    try:
        import umap
        print("    Computing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb_umap = reducer.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(8, 7))
        for cls, color, label in [(0, COLOR_NEGATIVE, 'No CVD'),
                                   (1, COLOR_POSITIVE, 'CVD')]:
            mask = labels == cls
            ax.scatter(emb_umap[mask, 0], emb_umap[mask, 1], c=color,
                       label=label, alpha=0.6, s=20, edgecolors='white',
                       linewidth=0.3)
        ax.set_xlabel('UMAP Dimension 1', fontweight='bold')
        ax.set_ylabel('UMAP Dimension 2', fontweight='bold')
        ax.set_title('Node Embeddings — UMAP Visualization',
                     fontweight='bold', fontsize=13)
        ax.legend(fontsize=10, frameon=True)
        plt.tight_layout()

        path = os.path.join(save_dir, 'fig15_umap_embeddings.png')
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [OK] Fig 15: {path}")
    except Exception as e:
        print(f"  [WARN] UMAP visualization failed: {e}")


def generate_architecture_diagrams(save_dir=None):
    """Generate all architecture diagrams (Figs 7, 8, 9)."""
    fig07_graph_construction_workflow(save_dir)
    fig08_gnn_architecture(save_dir)
    fig09_dpsn_architecture(save_dir)
