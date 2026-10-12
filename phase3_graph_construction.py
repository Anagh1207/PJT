
import os
import sys
import io
import warnings

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

from config import (
    PROCESSED_DATASET, TARGET_COLUMN, GRAPH_DIR, FIGURES_DIR,
    RESULTS_DIR, PLOT_DPI, PLOT_STYLE, RANDOM_STATE,
    GRAPH_COLORS, COLOR_NEGATIVE, COLOR_POSITIVE
)
from utils.data_utils import load_dataset
from graph_construction import build_all_graphs, save_graph

plt.rcParams.update(PLOT_STYLE)
np.random.seed(RANDOM_STATE)


def print_graph_stats(name, stats):
    """Pretty-print graph statistics."""
    print(f"\n  ── {name} ──")
    print(f"    Nodes:                {stats['num_nodes']}")
    print(f"    Edges (undirected):   {stats['num_edges']}")
    print(f"    Density:              {stats['density']:.6f}")
    print(f"    Avg Degree:           {stats['avg_degree']:.2f} ± {stats['std_degree']:.2f}")
    print(f"    Degree Range:         [{stats['min_degree']}, {stats['max_degree']}]")
    print(f"    Connected Components: {stats['num_connected_components']}")
    print(f"    Largest Component:    {stats['largest_component_size']}")
    print(f"    Edge Weight:          {stats['edge_weight_mean']:.4f} ± {stats['edge_weight_std']:.4f}")
    print(f"                          [{stats['edge_weight_min']:.4f}, {stats['edge_weight_max']:.4f}]")


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_graph_structure(graphs, y, save_dir):
    """Visualize sampled subgraphs for each graph type."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    sample_n = 100  # visualize a subgraph of 100 nodes
    np.random.seed(RANDOM_STATE)
    sample_idx = np.random.choice(len(y), size=sample_n, replace=False)
    sample_set = set(sample_idx.tolist())
    idx_map = {old: new for new, old in enumerate(sample_idx)}

    for ax_idx, (name, (edge_index, edge_weight, stats)) in enumerate(graphs.items()):
        ax = axes[ax_idx]

        # Filter edges within sample
        ei = edge_index.numpy()
        src_list, dst_list = [], []
        for i in range(ei.shape[1]):
            s, d = int(ei[0, i]), int(ei[1, i])
            if s in sample_set and d in sample_set:
                src_list.append(idx_map[s])
                dst_list.append(idx_map[d])

        G = nx.Graph()
        G.add_nodes_from(range(sample_n))
        G.add_edges_from(zip(src_list, dst_list))

        # Color by class
        node_colors = [COLOR_POSITIVE if y.iloc[sample_idx[i]] == 1
                       else COLOR_NEGATIVE for i in range(sample_n)]

        pos = nx.spring_layout(G, seed=RANDOM_STATE, k=1.5/np.sqrt(sample_n))
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.5,
                               edge_color='#888888')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30,
                               node_color=node_colors, alpha=0.8,
                               edgecolors='white', linewidths=0.3)

        color = GRAPH_COLORS.get(name, '#333')
        ax.set_title(f"{name}\n({stats['num_edges']} edges, density={stats['density']:.4f})",
                     fontweight='bold', fontsize=11, color=color)
        ax.set_axis_off()

    fig.suptitle('Graph Structure Visualization (100-node subgraph)',
                 fontsize=14, fontweight='bold', y=0.98)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLOR_NEGATIVE, label='No CVD'),
                       Patch(facecolor=COLOR_POSITIVE, label='CVD')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = os.path.join(save_dir, 'graph_structure_visualization.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Graph structure visualization: {path}")


def plot_degree_histograms(graphs, save_dir):
    """Overlay degree distributions for all 4 graph types."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, (edge_index, edge_weight, stats) in graphs.items():
        degrees = stats['degree_distribution']
        color = GRAPH_COLORS.get(name, '#333')
        ax.hist(degrees, bins=30, alpha=0.5, color=color, label=name,
                edgecolor='white', density=True)

    ax.set_xlabel('Node Degree', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Node Degree Distribution — All Graph Types',
                 fontweight='bold', fontsize=13)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    path = os.path.join(save_dir, 'node_degree_histogram.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Node degree histogram: {path}")


def plot_density_comparison(graphs, save_dir):
    """Bar chart comparing graph densities."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(graphs.keys())
    densities = [graphs[n][2]['density'] for n in names]
    colors = [GRAPH_COLORS.get(n, '#333') for n in names]

    bars = ax.bar(names, densities, color=colors, edgecolor='white',
                  alpha=0.88, width=0.55)
    for bar, d in zip(bars, densities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f'{d:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Graph Density', fontweight='bold')
    ax.set_title('Graph Density Comparison', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=15, ha='right')

    path = os.path.join(save_dir, 'graph_density_comparison.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Graph density comparison: {path}")


def plot_edge_weight_distribution(graphs, save_dir):
    """KDE plots of edge weight distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_idx, (name, (edge_index, edge_weight, stats)) in enumerate(graphs.items()):
        ax = axes[ax_idx]
        weights = edge_weight.numpy()

        # Subsample if too many edges
        if len(weights) > 50000:
            idx = np.random.choice(len(weights), 50000, replace=False)
            weights = weights[idx]

        color = GRAPH_COLORS.get(name, '#333')
        ax.hist(weights, bins=50, color=color, alpha=0.7, edgecolor='white',
                density=True)
        ax.set_xlabel('Edge Weight', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title(name, fontweight='bold', fontsize=11, color=color)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Annotate stats
        ax.axvline(weights.mean(), color='black', linestyle='--', lw=1.5, alpha=0.7)
        ax.text(0.95, 0.95, f'μ = {weights.mean():.3f}\nσ = {weights.std():.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle('Edge Weight Distributions', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(save_dir, 'edge_weight_distribution.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Edge weight distribution: {path}")


def plot_similarity_score_distribution(graphs, save_dir):
    """Combined similarity score distribution (KDE overlay)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, (edge_index, edge_weight, stats) in graphs.items():
        weights = edge_weight.numpy()
        if len(weights) > 50000:
            idx = np.random.choice(len(weights), 50000, replace=False)
            weights = weights[idx]

        color = GRAPH_COLORS.get(name, '#333')
        ax.hist(weights, bins=50, alpha=0.4, color=color, label=name,
                edgecolor='white', density=True)

    ax.set_xlabel('Similarity Score / Edge Weight', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Similarity Score Distribution — All Graph Types',
                 fontweight='bold', fontsize=13)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    path = os.path.join(save_dir, 'similarity_score_distribution.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Similarity score distribution: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  PHASE 3 — GRAPH CONSTRUCTION")
    print("  Building Patient Similarity Graphs")
    print("=" * 80)

    # ── Load Data ─────────────────────────────────────────────────────
    print("\n  Loading processed dataset...")
    X, y, feature_names = load_dataset(PROCESSED_DATASET, TARGET_COLUMN)
    X_np = X.values if hasattr(X, 'values') else np.array(X)

    print(f"    Samples:  {X_np.shape[0]}")
    print(f"    Features: {X_np.shape[1]}")

    # ── Build All Graphs ──────────────────────────────────────────────
    graphs = build_all_graphs(X_np, feature_names)

    # ── Print Statistics ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  GRAPH STATISTICS")
    print("=" * 80)

    stats_rows = []
    for name, (ei, ew, stats) in graphs.items():
        print_graph_stats(name, stats)
        stats_rows.append({
            'Graph Type': name,
            'Nodes': stats['num_nodes'],
            'Edges': stats['num_edges'],
            'Density': stats['density'],
            'Avg Degree': stats['avg_degree'],
            'Std Degree': stats['std_degree'],
            'Min Degree': stats['min_degree'],
            'Max Degree': stats['max_degree'],
            'Components': stats['num_connected_components'],
            'Largest Component': stats['largest_component_size'],
            'Edge Weight Mean': stats['edge_weight_mean'],
            'Edge Weight Std': stats['edge_weight_std'],
        })

    # Save statistics CSV
    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(RESULTS_DIR, 'graph_statistics.csv')
    stats_df.to_csv(stats_path, index=False, float_format='%.6f')
    print(f"\n  [OK] Graph statistics saved: {stats_path}")

    # ── Save Graphs ───────────────────────────────────────────────────
    print("\n  Saving graph data...")
    for name, (ei, ew, stats) in graphs.items():
        fname = name.lower().replace(' ', '_') + '.pt'
        fpath = os.path.join(GRAPH_DIR, fname)
        save_graph(ei, ew, stats, fpath)
        print(f"    [OK] {name} → {fpath}")

    # ── Generate Visualizations ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("  GENERATING PHASE 3 VISUALIZATIONS")
    print("=" * 80)

    plot_graph_structure(graphs, y, FIGURES_DIR)
    plot_degree_histograms(graphs, FIGURES_DIR)
    plot_density_comparison(graphs, FIGURES_DIR)
    plot_edge_weight_distribution(graphs, FIGURES_DIR)
    plot_similarity_score_distribution(graphs, FIGURES_DIR)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  [OK] PHASE 3 — GRAPH CONSTRUCTION COMPLETE")
    print("=" * 80)
    print(f"\n  Outputs:")
    print(f"    Graphs:         {GRAPH_DIR}")
    print(f"    Statistics:     {stats_path}")
    print(f"    Figures:        {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
