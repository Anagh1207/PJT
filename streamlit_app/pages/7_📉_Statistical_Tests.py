"""
Page 7: Statistical Validation
Paired t-tests, Wilcoxon signed-rank tests, bootstrap confidence intervals, and significance analysis.
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
    load_csv, apply_plotly_theme, make_heatmap,
    get_figure_path, figure_exists, COLORS,
)

st.set_page_config(page_title="Statistical Validation — CVD-GNN", page_icon="📉", layout="wide")
inject_custom_css()
setup_sidebar()

# ═══════════════════════════════════════════════════════════════════════════
render_page_header(
    "📉 Statistical Validation",
    "Paired t-tests, Wilcoxon signed-rank tests, and bootstrap confidence intervals for DPSN vs baselines"
)

# Load data
stats_df = load_csv("statistical_tests.csv")

if stats_df.empty:
    st.error("❌ Could not load statistical tests. Make sure `results/statistical_tests.csv` exists.")
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────────────────
n_significant_t = stats_df["t-test Sig."].str.contains("✓").sum() if "t-test Sig." in stats_df.columns else 0
n_significant_w = stats_df["Wilcoxon Sig."].str.contains("✓").sum() if "Wilcoxon Sig." in stats_df.columns else 0
n_total = len(stats_df)

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card("🧪", f"{n_total}", "Total Comparisons", "DPSN vs 8 baselines × 3 metrics")
with col2:
    render_kpi_card("✅", f"{n_significant_t}", "Significant (t-test)", f"p < 0.05")
with col3:
    render_kpi_card("📊", f"{n_significant_w}", "Significant (Wilcoxon)", "Non-parametric")
with col4:
    pct = round(n_significant_t / n_total * 100, 1) if n_total > 0 else 0
    render_kpi_card("📈", f"{pct}%", "Significance Rate", "Paired t-test")

st.markdown("<br>", unsafe_allow_html=True)

# ── Full Results Table ───────────────────────────────────────────────────
render_section_header("📋 Complete Statistical Tests", "DPSN vs all baselines across F1, ROC-AUC, and Accuracy")

metric_tabs = st.tabs(["📊 All Metrics", "🎯 F1 Score", "📈 ROC-AUC", "🎯 Accuracy"])

with metric_tabs[0]:
    # Style the significance columns
    def highlight_sig(val):
        if "✓" in str(val):
            return "background-color: rgba(67,160,71,0.2); color: #43A047; font-weight: 700;"
        elif "✗" in str(val):
            return "background-color: rgba(229,57,53,0.1); color: #E53935;"
        return ""

    styled = stats_df.style.applymap(
        highlight_sig, subset=["t-test Sig.", "Wilcoxon Sig."]
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=600)

for i, metric_name in enumerate(["F1", "ROC_AUC", "ACCURACY"]):
    with metric_tabs[i + 1]:
        filtered = stats_df[stats_df["Metric"] == metric_name]
        if not filtered.empty:
            styled = filtered.style.applymap(
                highlight_sig, subset=["t-test Sig.", "Wilcoxon Sig."]
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info(f"No data found for metric: {metric_name}")

render_gradient_divider()

# ── Significance Heatmap ─────────────────────────────────────────────────
render_section_header("🔥 P-Value Heatmap", "Paired t-test p-values — DPSN vs each baseline (darker = more significant)")

# Build the heatmap
metrics = stats_df["Metric"].unique().tolist()
baselines = stats_df["Baseline"].unique().tolist()

z_pvalues = []
for metric in metrics:
    row = []
    for baseline in baselines:
        mask = (stats_df["Metric"] == metric) & (stats_df["Baseline"] == baseline)
        if mask.any():
            pval = stats_df.loc[mask, "Paired t-test p"].values[0]
            row.append(pval)
        else:
            row.append(1.0)
    z_pvalues.append(row)

# Custom text with significance markers
z_text = []
for row in z_pvalues:
    text_row = []
    for p in row:
        if p < 0.001:
            text_row.append(f"{p:.4f} ***")
        elif p < 0.01:
            text_row.append(f"{p:.4f} **")
        elif p < 0.05:
            text_row.append(f"{p:.4f} *")
        else:
            text_row.append(f"{p:.4f}")
    z_text.append(text_row)

fig_pval = go.Figure(go.Heatmap(
    z=z_pvalues,
    x=baselines,
    y=metrics,
    text=z_text,
    texttemplate="%{text}",
    textfont=dict(size=11, color="#E8EAED"),
    colorscale="RdYlGn",
    reversescale=True,
    zmin=0,
    zmax=0.1,
    colorbar=dict(
        title="p-value",
        tickfont=dict(color="#9AA0A6"),
        title_font=dict(color="#9AA0A6"),
    ),
))
fig_pval.update_layout(
    title="Paired t-test P-Values (DPSN vs Baselines)",
    height=350,
    xaxis=dict(tickangle=-30),
)
fig_pval = apply_plotly_theme(fig_pval)
st.plotly_chart(fig_pval, use_container_width=True)

st.markdown("""
<div style="font-size: 12px; color: #9AA0A6; text-align: center;">
    <span class="stat-badge green">*** p &lt; 0.001</span>
    <span class="stat-badge">** p &lt; 0.01</span>
    <span class="stat-badge magenta">* p &lt; 0.05</span>
    <span style="color: #5F6368;">ns = not significant</span>
</div>
""", unsafe_allow_html=True)

render_gradient_divider()

# ── Improvement Analysis ─────────────────────────────────────────────────
render_section_header("📊 DPSN Improvement Over Baselines", "Positive = DPSN better, Negative = Baseline better")

for metric_name in metrics:
    filtered = stats_df[stats_df["Metric"] == metric_name].copy()
    if filtered.empty:
        continue

    filtered = filtered.sort_values("Improvement", ascending=True)
    imp_colors = [
        COLORS["accent_emerald"] if v >= 0 else COLORS["accent_magenta"]
        for v in filtered["Improvement"]
    ]

    fig_imp = go.Figure(go.Bar(
        y=filtered["Baseline"],
        x=filtered["Improvement"],
        orientation="h",
        marker=dict(color=imp_colors, opacity=0.88),
        text=[f"{v:+.4f}" for v in filtered["Improvement"]],
        textposition="outside",
        textfont=dict(size=11, color="#E8EAED"),
    ))
    fig_imp.update_layout(
        title=f"{metric_name} — DPSN Improvement",
        xaxis_title="Improvement (DPSN − Baseline)",
        height=350,
        xaxis=dict(zeroline=True, zerolinecolor="#9AA0A6", zerolinewidth=1),
    )
    fig_imp = apply_plotly_theme(fig_imp)
    st.plotly_chart(fig_imp, use_container_width=True)

render_gradient_divider()

# ── Confidence Intervals ─────────────────────────────────────────────────
render_section_header("📐 Confidence Intervals", "Bootstrap 95% confidence intervals for DPSN and baselines")

# Display CI data from the table
ci_cols = ["Metric", "Proposed (DPSN)", "Baseline", "Baseline Score", "DPSN 95% CI", "Baseline 95% CI"]
available_ci_cols = [c for c in ci_cols if c in stats_df.columns]

if available_ci_cols:
    ci_display = stats_df[available_ci_cols].copy()
    st.dataframe(ci_display, use_container_width=True, hide_index=True, height=400)

render_gradient_divider()

# ── Pre-generated Figures ────────────────────────────────────────────────
render_section_header("📈 Publication Figures", "Statistical significance and confidence interval visualizations")

stat_fig_tabs = st.tabs(["Confidence Intervals", "Statistical Significance"])

with stat_fig_tabs[0]:
    if figure_exists("confidence_intervals.png"):
        st.image(get_figure_path("confidence_intervals.png"), caption="Bootstrap 95% Confidence Intervals", use_container_width=True)
    else:
        st.info("Confidence intervals figure not found.")

with stat_fig_tabs[1]:
    if figure_exists("fig11_statistical_significance.png"):
        st.image(get_figure_path("fig11_statistical_significance.png"), caption="Statistical Significance Analysis", use_container_width=True)
    else:
        st.info("Statistical significance figure not found.")

render_gradient_divider()

# ── Summary ──────────────────────────────────────────────────────────────
render_section_header("📝 Statistical Summary", "Key findings from the validation analysis")

col_a, col_b = st.columns(2)

with col_a:
    render_insight(
        "Where DPSN Excels",
        "DPSN significantly outperforms <strong>GAT</strong> and <strong>GCN</strong> "
        "across all three metrics (p < 0.05 in paired t-tests). It also achieves competitive "
        "ROC-AUC (0.984), demonstrating strong discriminative ability."
    )

with col_b:
    render_warning(
        "Where DPSN Falls Short",
        "DPSN is significantly outperformed by <strong>LightGBM</strong> (F1: -0.079), "
        "<strong>Logistic Regression</strong> (F1: -0.052), and <strong>SVM</strong> (F1: -0.035). "
        "Classical ML models remain superior for this dataset size."
    )

render_insight(
    "Overall Conclusion",
    f"Out of {n_total} statistical comparisons, <strong>{n_significant_t} ({pct}%)</strong> "
    "are significant at p < 0.05 using paired t-tests. The results suggest that while DPSN "
    "shows promise in ROC-AUC and outperforms simpler GNNs, classical ensemble methods "
    "(LightGBM, XGBoost) remain the most effective for tabular CVD data."
)
