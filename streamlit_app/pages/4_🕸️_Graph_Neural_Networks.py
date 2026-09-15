"""
Page 4: Graph Neural Networks
GNN model performance, graph construction stats, learning curves, and embeddings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpers import (
    inject_custom_css, setup_sidebar, render_page_header, render_kpi_card,
    render_section_header, render_insight, render_warning, render_gradient_divider,
    load_csv, apply_plotly_theme, make_bar_chart, make_heatmap,
    get_figure_path, figure_exists, COLORS, MODEL_COLORS, GRAPH_COLORS,
)

st.set_page_config(page_title="Graph Neural Networks — CVD-GNN", page_icon="🕸️", layout="wide")
inject_custom_css()
setup_sidebar()

# ═══════════════════════════════════════════════════════════════════════════
render_page_header(
    "🕸️ Graph Neural Networks",
    "3 GNN architectures × 4 graph types — comprehensive node-level classification"
)

# Load data
gnn_df = load_csv("gnn_metrics.csv")
graph_stats_df = load_csv("graph_statistics.csv")

if gnn_df.empty:
    st.error("❌ Could not load GNN metrics. Make sure `results/gnn_metrics.csv` exists.")
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────────────────
best_gnn = gnn_df.loc[gnn_df["F1"].idxmax()]
best_auc_gnn = gnn_df.loc[gnn_df["ROC-AUC"].idxmax()]

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card("🕸️", "12", "Model×Graph Combos", "3 GNNs × 4 Graphs")
with col2:
    render_kpi_card("🏆", f"{best_gnn['F1']:.3f}", "Best GNN F1", best_gnn["Model"])
with col3:
    render_kpi_card("📊", f"{best_auc_gnn['ROC-AUC']:.3f}", "Best GNN ROC-AUC", best_auc_gnn["Model"])
with col4:
    render_kpi_card("🔗", "4", "Graph Types", "KNN, Cosine, Clinical, Dynamic")

st.markdown("<br>", unsafe_allow_html=True)

# ── Graph Construction Overview ──────────────────────────────────────────
render_section_header("🏗️ Graph Construction Statistics", "Properties of the 4 patient similarity graphs")

if not graph_stats_df.empty:
    col_table, col_chart = st.columns([2, 3])

    with col_table:
        st.dataframe(
            graph_stats_df.style.format({
                "Density": "{:.6f}",
                "Avg Degree": "{:.3f}",
            }).background_gradient(subset=["Edges", "Density"], cmap="Blues"),
            use_container_width=True, hide_index=True,
        )

    with col_chart:
        fig_edges = go.Figure()
        graph_types = graph_stats_df["Graph Type"].tolist()
        edges = graph_stats_df["Edges"].tolist()
        g_colors = [GRAPH_COLORS.get(g, COLORS["accent_blue"]) for g in graph_types]

        fig_edges.add_trace(go.Bar(
            x=graph_types,
            y=edges,
            marker=dict(color=g_colors, opacity=0.88),
            text=edges,
            textposition="outside",
            textfont=dict(size=13, color="#E8EAED"),
        ))
        fig_edges.update_layout(
            title="Number of Edges per Graph Type",
            yaxis_title="Edge Count",
            height=400,
        )
        fig_edges = apply_plotly_theme(fig_edges)
        st.plotly_chart(fig_edges, use_container_width=True)

    render_insight(
        "Graph Connectivity Insight",
        "The <strong>Dynamic PSG</strong> has the highest connectivity (9,653 edges, density=0.019), "
        "while <strong>Clinical Similarity</strong> is extremely sparse (13 edges) due to the high threshold (0.80)."
    )

render_gradient_divider()

# ── Graph Figures ────────────────────────────────────────────────────────
render_section_header("🖼️ Graph Visualizations", "Structure, edge weights, node degrees, and density")

graph_fig_tabs = st.tabs(["Graph Structure", "Edge Weights", "Node Degrees", "Density Comparison", "Similarity Scores"])

graph_figures = {
    "Graph Structure": "graph_structure_visualization.png",
    "Edge Weights": "edge_weight_distribution.png",
    "Node Degrees": "node_degree_histogram.png",
    "Density Comparison": "graph_density_comparison.png",
    "Similarity Scores": "similarity_score_distribution.png",
}

for tab, (name, fname) in zip(graph_fig_tabs, graph_figures.items()):
    with tab:
        if figure_exists(fname):
            st.image(get_figure_path(fname), caption=name, use_container_width=True)
        else:
            st.info(f"Figure `{fname}` not found.")

render_gradient_divider()

# ── GNN Metrics Table ────────────────────────────────────────────────────
render_section_header("📋 GNN Model Performance", "All 12 model × graph combinations")

st.dataframe(
    gnn_df.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
        "F1": "{:.4f}", "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}", "MCC": "{:.4f}",
    }).background_gradient(
        subset=["F1", "ROC-AUC", "MCC"], cmap="YlOrRd", vmin=0.65, vmax=0.9
    ).highlight_max(subset=["F1", "ROC-AUC", "MCC"], color="rgba(67,160,71,0.3)"),
    use_container_width=True, hide_index=True, height=500,
)

render_gradient_divider()

# ── Performance Heatmap ──────────────────────────────────────────────────
render_section_header("🔥 Performance Heatmap", "GNN Architecture × Graph Type (F1 Score)")

# Build the heatmap matrix
gnn_architectures = ["GCN", "GraphSAGE", "GAT"]
graph_types_ordered = ["Euclidean KNN", "Cosine Similarity", "Clinical Similarity", "Dynamic PSG"]

z_matrix = []
for arch in gnn_architectures:
    row = []
    for gt in graph_types_ordered:
        mask = gnn_df["Model"].str.contains(arch) & gnn_df["Model"].str.contains(gt.split()[0] if gt != "Dynamic PSG" else "Dynamic")
        if mask.any():
            row.append(gnn_df.loc[mask, "F1"].values[0])
        else:
            row.append(0)
    z_matrix.append(row)

col_metric_sel, _ = st.columns([1, 3])
with col_metric_sel:
    heatmap_metric = st.selectbox(
        "Select metric for heatmap:",
        ["F1", "ROC-AUC", "Accuracy", "Precision", "Recall", "MCC"],
        key="gnn_heatmap_metric",
    )

# Rebuild heatmap for selected metric
z_matrix_sel = []
for arch in gnn_architectures:
    row = []
    for gt in graph_types_ordered:
        mask = gnn_df["Model"].str.contains(arch) & gnn_df["Model"].str.contains(gt.split()[0] if gt != "Dynamic PSG" else "Dynamic")
        if mask.any():
            row.append(gnn_df.loc[mask, heatmap_metric].values[0])
        else:
            row.append(0)
    z_matrix_sel.append(row)

fig_heatmap = make_heatmap(
    z_matrix_sel,
    graph_types_ordered,
    gnn_architectures,
    f"GNN {heatmap_metric} — Architecture × Graph Type",
    colorscale="Viridis",
)
fig_heatmap.update_layout(height=350)
st.plotly_chart(fig_heatmap, use_container_width=True)

render_insight(
    "Best GNN Configuration",
    f"<strong>{best_gnn['Model']}</strong> achieved the highest F1 score ({best_gnn['F1']:.3f}). "
    "GraphSAGE consistently performs best across graph types due to its neighborhood sampling approach."
)

render_gradient_divider()

# ── Best Model per Graph Type ────────────────────────────────────────────
render_section_header("🏅 Best GNN per Graph Type", "Which architecture wins on each graph?")

best_per_graph = {}
for gt in graph_types_ordered:
    mask = gnn_df["Model"].str.contains(gt.split()[0] if gt != "Dynamic PSG" else "Dynamic")
    if mask.any():
        best_idx = gnn_df.loc[mask, "F1"].idxmax()
        best_per_graph[gt] = {
            "Model": gnn_df.loc[best_idx, "Model"],
            "F1": gnn_df.loc[best_idx, "F1"],
            "ROC-AUC": gnn_df.loc[best_idx, "ROC-AUC"],
        }

cols = st.columns(4)
for i, (gt, info) in enumerate(best_per_graph.items()):
    with cols[i]:
        arch_name = info["Model"].split("(")[0].strip()
        render_kpi_card(
            "🥇",
            f"{info['F1']:.3f}",
            gt,
            f"{arch_name} • AUC: {info['ROC-AUC']:.3f}"
        )

render_gradient_divider()

# ── Learning Curves ──────────────────────────────────────────────────────
render_section_header("📈 Learning Curves", "Training and validation loss/accuracy over epochs")

lc_tabs = st.tabs(["GCN", "GraphSAGE", "GAT", "DPSN"])
lc_files = {
    "GCN": "learning_curves_gcn.png",
    "GraphSAGE": "learning_curves_graphsage.png",
    "GAT": "learning_curves_gat.png",
    "DPSN": "learning_curves_dpsn.png",
}

for tab, (name, fname) in zip(lc_tabs, lc_files.items()):
    with tab:
        if figure_exists(fname):
            st.image(get_figure_path(fname), caption=f"{name} Learning Curves", use_container_width=True)
        else:
            st.info(f"Figure `{fname}` not found.")

render_gradient_divider()

# ── GNN Architecture Figures ─────────────────────────────────────────────
render_section_header("🏛️ Architecture Diagrams", "Visual representations of the GNN model architectures")

arch_tabs = st.tabs(["GNN Architecture", "Graph Construction Workflow"])

with arch_tabs[0]:
    if figure_exists("fig08_gnn_architecture.png"):
        st.image(get_figure_path("fig08_gnn_architecture.png"), caption="GNN Architecture Overview", use_container_width=True)

with arch_tabs[1]:
    if figure_exists("fig07_graph_construction_workflow.png"):
        st.image(get_figure_path("fig07_graph_construction_workflow.png"), caption="Graph Construction Workflow", use_container_width=True)

render_gradient_divider()

# ── Embedding Visualizations ─────────────────────────────────────────────
render_section_header("🌌 Embedding Visualizations", "t-SNE and UMAP projections of GNN node embeddings")

embed_tabs = st.tabs(["t-SNE Embeddings", "UMAP Embeddings"])

with embed_tabs[0]:
    if figure_exists("fig14_tsne_embeddings.png"):
        st.image(get_figure_path("fig14_tsne_embeddings.png"), caption="t-SNE Embeddings", use_container_width=True)
    else:
        st.info("t-SNE embedding figure not found.")

with embed_tabs[1]:
    if figure_exists("fig15_umap_embeddings.png"):
        st.image(get_figure_path("fig15_umap_embeddings.png"), caption="UMAP Embeddings", use_container_width=True)
    else:
        st.info("UMAP embedding figure not found.")
