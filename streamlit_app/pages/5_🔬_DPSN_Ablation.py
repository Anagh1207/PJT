"""
Page 5: DPSN Analysis & Ablation Study
DPSN architecture, ablation variants, feature weights, and component contribution analysis.
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
    load_csv, apply_plotly_theme, make_bar_chart,
    get_figure_path, figure_exists, COLORS,
)

st.set_page_config(page_title="DPSN & Ablation — CVD-GNN", page_icon="🔬", layout="wide")
inject_custom_css()
setup_sidebar()

# ═══════════════════════════════════════════════════════════════════════════
render_page_header(
    "🔬 DPSN Analysis & Ablation Study",
    "Dynamic Patient Similarity Network — architecture, ablation variants, and component contributions"
)

# Load data
ablation_df = load_csv("ablation_summary.csv")
ablation_full = load_csv("ablation_results.csv")
dpsn_weights_df = load_csv("dpsn_feature_weights.csv")

# ── DPSN Architecture ────────────────────────────────────────────────────
render_section_header("🏛️ DPSN Architecture", "The novel Dynamic Patient Similarity Network")

col_arch, col_desc = st.columns([3, 2])

with col_arch:
    if figure_exists("fig09_dpsn_architecture.png"):
        st.image(get_figure_path("fig09_dpsn_architecture.png"), caption="DPSN Architecture Diagram", use_container_width=True)
    else:
        st.info("DPSN architecture diagram not found.")

with col_desc:
    st.markdown("""
    <div class="kpi-card" style="text-align: left; padding: 24px;">
        <div style="font-weight: 700; color: #D81B60; font-size: 16px; margin-bottom: 12px;">
            🧬 Key Components
        </div>
        <div style="color: #E8EAED; font-size: 14px; line-height: 2.0;">
            <strong>1. Dynamic Edge Learning</strong><br>
            <span style="color: #9AA0A6;">Learns patient similarity edges dynamically</span><br>
            <strong>2. Clinical Similarity</strong><br>
            <span style="color: #9AA0A6;">Domain-weighted clinical risk factors</span><br>
            <strong>3. Attention Mechanism</strong><br>
            <span style="color: #9AA0A6;">Multi-head attention over neighbors</span><br>
            <strong>4. Feature Weighting</strong><br>
            <span style="color: #9AA0A6;">Learned importance per input feature</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # DPSN Hyperparameters
    st.markdown("""
    <div class="kpi-card" style="text-align: left; padding: 24px;">
        <div style="font-weight: 700; color: #1E88E5; font-size: 16px; margin-bottom: 12px;">
            ⚙️ Hyperparameters
        </div>
        <div style="color: #E8EAED; font-size: 13px; line-height: 2.0;">
            Hidden Dim: <span class="stat-badge">64</span><br>
            Embed Dim: <span class="stat-badge">32</span><br>
            Top-K Edges: <span class="stat-badge">15</span><br>
            Edge Hidden: <span class="stat-badge">32</span><br>
            Temperature: <span class="stat-badge">0.5</span><br>
            GAT Heads: <span class="stat-badge">4</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

render_gradient_divider()

# ── KPIs ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card("🎯", "0.813", "DPSN F1 Score", "Rank: 7th / 9")
with col2:
    render_kpi_card("📊", "0.988", "DPSN ROC-AUC", "Competitive with best")
with col3:
    render_kpi_card("🔬", "5", "Ablation Variants", "Component-wise analysis")
with col4:
    render_kpi_card("📉", "-0.047", "Max F1 Drop", "No Attention variant")

st.markdown("<br>", unsafe_allow_html=True)

# ── Ablation Results ─────────────────────────────────────────────────────
render_section_header("📊 Ablation Study Results", "Impact of removing each DPSN component")

if not ablation_df.empty:
    tab_table, tab_chart = st.tabs(["📋 Results Table", "📊 Visual Comparison"])

    with tab_table:
        st.dataframe(
            ablation_df.style.format({
                "F1": "{:.4f}", "ROC-AUC": "{:.4f}", "MCC": "{:.4f}", "F1 Drop": "{:+.4f}",
            }).background_gradient(
                subset=["F1", "ROC-AUC"], cmap="YlOrRd", vmin=0.8, vmax=0.9
            ),
            use_container_width=True, hide_index=True,
        )

    with tab_chart:
        # Grouped bar chart
        fig_abl = go.Figure()

        colors = [COLORS["accent_magenta"], COLORS["accent_blue"], COLORS["accent_emerald"]]
        for i, metric in enumerate(["F1", "ROC-AUC", "MCC"]):
            fig_abl.add_trace(go.Bar(
                x=ablation_df["Variant"],
                y=ablation_df[metric],
                name=metric,
                marker_color=colors[i],
                marker_opacity=0.88,
                text=ablation_df[metric].round(3),
                textposition="outside",
                textfont=dict(size=10),
            ))

        fig_abl.update_layout(
            title="Ablation Study — Metric Comparison",
            barmode="group",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(tickangle=-20),
        )
        fig_abl = apply_plotly_theme(fig_abl)
        st.plotly_chart(fig_abl, use_container_width=True)

    # F1 Drop analysis
    render_section_header("📉 Component Contribution (F1 Drop)", "How much performance drops when each component is removed")

    if "F1 Drop" in ablation_df.columns:
        drop_df = ablation_df[ablation_df["Variant"] != "Full DPSN"].copy()
        drop_df["F1 Drop Abs"] = drop_df["F1 Drop"].abs()
        drop_df = drop_df.sort_values("F1 Drop Abs", ascending=True)

        drop_colors = [
            COLORS["accent_emerald"] if v >= 0 else COLORS["accent_magenta"]
            for v in drop_df["F1 Drop"]
        ]

        fig_drop = go.Figure(go.Bar(
            y=drop_df["Variant"],
            x=drop_df["F1 Drop"],
            orientation="h",
            marker=dict(color=drop_colors, opacity=0.88),
            text=[f"{v:+.4f}" for v in drop_df["F1 Drop"]],
            textposition="outside",
            textfont=dict(size=12, color="#E8EAED"),
        ))
        fig_drop.update_layout(
            title="",
            xaxis_title="F1 Score Change (Negative = Performance Drop)",
            height=300,
            xaxis=dict(zeroline=True, zerolinecolor="#9AA0A6", zerolinewidth=1),
        )
        fig_drop = apply_plotly_theme(fig_drop)
        st.plotly_chart(fig_drop, use_container_width=True)

        render_insight(
            "Ablation Finding",
            "Removing the <strong>Attention mechanism</strong> caused the largest F1 drop (-0.047), "
            "indicating it is the most critical component. Interestingly, removing attention actually "
            "<em>improved</em> all metrics, suggesting potential over-fitting with the attention layer."
        )

render_gradient_divider()

# ── Pre-generated Ablation Figure ────────────────────────────────────────
if figure_exists("fig10_ablation_study.png"):
    render_section_header("📈 Publication Figure", "Ablation study visualization for IEEE paper")
    st.image(get_figure_path("fig10_ablation_study.png"), caption="Fig 10: Ablation Study", use_container_width=True)

render_gradient_divider()

# ── DPSN Feature Weights ─────────────────────────────────────────────────
render_section_header("⚖️ DPSN Learned Feature Weights", "Feature importance as learned by the DPSN model")

if not dpsn_weights_df.empty:
    col_table, col_chart = st.columns([1, 2])

    with col_table:
        st.dataframe(
            dpsn_weights_df.style.format({"Learned Weight": "{:.6f}"}),
            use_container_width=True, hide_index=True,
        )

    with col_chart:
        sorted_w = dpsn_weights_df.sort_values("Learned Weight", ascending=True)
        fig_w = go.Figure(go.Bar(
            y=sorted_w["Feature"],
            x=sorted_w["Learned Weight"],
            orientation="h",
            marker=dict(
                color=sorted_w["Learned Weight"],
                colorscale="Magenta",
                opacity=0.88,
            ),
            text=sorted_w["Learned Weight"].round(4),
            textposition="outside",
            textfont=dict(size=11, color="#E8EAED"),
        ))
        fig_w.update_layout(
            title="DPSN Feature Weights",
            xaxis_title="Learned Weight",
            height=450,
        )
        fig_w = apply_plotly_theme(fig_w)
        st.plotly_chart(fig_w, use_container_width=True)

    if figure_exists("dpsn_feature_weights.png"):
        with st.expander("📊 Pre-generated Feature Weights Figure"):
            st.image(get_figure_path("dpsn_feature_weights.png"), caption="DPSN Feature Weights", use_container_width=True)

    render_warning(
        "Uniform Weights Detected",
        "All DPSN feature weights converged to equal values (1/14 ≈ 0.0714), "
        "suggesting the feature weighting module did not differentiate between features. "
        "This aligns with the ablation result showing no F1 drop when removing feature weighting."
    )
