"""
Page 2: Dataset Explorer
Interactive exploration of the CVD dataset with distributions, correlations, and audit data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpers import (
    inject_custom_css, setup_sidebar, render_page_header, render_kpi_card,
    render_section_header, render_insight, render_warning, render_gradient_divider,
    load_csv, load_dataset, apply_plotly_theme, COLORS, PLOTLY_LAYOUT,
)

st.set_page_config(page_title="Dataset Explorer — CVD-GNN", page_icon="📊", layout="wide")
inject_custom_css()
setup_sidebar()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════
render_page_header("📊 Dataset Explorer", "Interactive exploration of the CVD patient dataset — 1,000 records, 14 features")

# Load data
df = load_dataset()

if df.empty:
    st.error("❌ Could not load dataset. Make sure `data/cleaned_dataset.csv` exists.")
    st.stop()

# ── Dataset Overview KPIs ────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

target_col = "Target" if "Target" in df.columns else None
n_cvd = int(df[target_col].sum()) if target_col else "N/A"
n_healthy = len(df) - n_cvd if target_col else "N/A"

with col1:
    render_kpi_card("📋", f"{len(df):,}", "Total Patients")
with col2:
    render_kpi_card("📐", f"{df.shape[1]}", "Total Features")
with col3:
    render_kpi_card("❤️", str(n_healthy), "No CVD (Class 0)")
with col4:
    render_kpi_card("🫀", str(n_cvd), "CVD (Class 1)")
with col5:
    ratio = round(n_healthy / n_cvd, 2) if isinstance(n_cvd, int) and n_cvd > 0 else "N/A"
    render_kpi_card("⚖️", str(ratio), "Imbalance Ratio")

st.markdown("<br>", unsafe_allow_html=True)

# ── Class Distribution ───────────────────────────────────────────────────
render_section_header("🎯 Target Class Distribution", "CVD vs Non-CVD patient counts")

col_pie, col_bar = st.columns(2)

if target_col:
    class_counts = df[target_col].value_counts().sort_index()
    labels = ["No CVD (0)", "CVD (1)"]
    values = [class_counts.get(0, 0), class_counts.get(1, 0)]

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=[COLORS["accent_blue"], COLORS["accent_magenta"]]),
            textinfo="label+percent+value",
            textfont=dict(size=13),
            pull=[0, 0.05],
        ))
        fig_pie.update_layout(
            title="Class Distribution",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            height=350,
        )
        fig_pie = apply_plotly_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_bar = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=[COLORS["accent_blue"], COLORS["accent_magenta"]],
                opacity=0.88,
            ),
            text=values,
            textposition="outside",
            textfont=dict(size=14, color="#E8EAED"),
        ))
        fig_bar.update_layout(title="Class Counts", yaxis_title="Count", height=350)
        fig_bar = apply_plotly_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    render_warning(
        "Class Imbalance Detected",
        f"The dataset has a {ratio}:1 imbalance ratio. SMOTE oversampling is applied "
        "within cross-validation folds to prevent data leakage."
    )

render_gradient_divider()

# ── Feature Distributions ───────────────────────────────────────────────
render_section_header("📈 Feature Distributions", "Interactive histograms split by CVD class")

continuous_features = [
    'Age', 'Blood_Pressure', 'Cholesterol', 'Heart_Rate', 'BMI',
    'Physical_Activity_HoursPerWeek', 'ECG_Score', 'Glucose',
    'Triglycerides', 'Stress_Index'
]
binary_features = ['Gender', 'Diabetes', 'Smoking', 'Family_History']

# Filter to features that exist in the dataset
continuous_features = [f for f in continuous_features if f in df.columns]
binary_features = [f for f in binary_features if f in df.columns]

tab_cont, tab_bin = st.tabs(["📊 Continuous Features", "🔘 Binary Features"])

with tab_cont:
    selected_feature = st.selectbox(
        "Select a feature to visualize:",
        continuous_features,
        key="cont_feature_select",
    )

    col_hist, col_box = st.columns(2)

    with col_hist:
        if target_col:
            fig_hist = go.Figure()
            for cls, color, name in [(0, COLORS["accent_blue"], "No CVD"), (1, COLORS["accent_magenta"], "CVD")]:
                fig_hist.add_trace(go.Histogram(
                    x=df[df[target_col] == cls][selected_feature],
                    name=name,
                    marker_color=color,
                    opacity=0.7,
                    nbinsx=30,
                ))
            fig_hist.update_layout(
                title=f"Distribution of {selected_feature}",
                xaxis_title=selected_feature,
                yaxis_title="Count",
                barmode="overlay",
                height=400,
            )
        else:
            fig_hist = px.histogram(df, x=selected_feature, nbins=30)
            fig_hist.update_layout(title=f"Distribution of {selected_feature}", height=400)
        fig_hist = apply_plotly_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        if target_col:
            fig_box = go.Figure()
            for cls, color, name in [(0, COLORS["accent_blue"], "No CVD"), (1, COLORS["accent_magenta"], "CVD")]:
                fig_box.add_trace(go.Box(
                    y=df[df[target_col] == cls][selected_feature],
                    name=name,
                    marker_color=color,
                    boxmean="sd",
                ))
            fig_box.update_layout(
                title=f"Box Plot — {selected_feature} by Class",
                yaxis_title=selected_feature,
                height=400,
            )
        else:
            fig_box = go.Figure(go.Box(y=df[selected_feature], marker_color=COLORS["accent_blue"]))
            fig_box.update_layout(title=f"Box Plot — {selected_feature}", height=400)
        fig_box = apply_plotly_theme(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

    # Summary statistics
    st.markdown(f"**Summary Statistics for `{selected_feature}`:**")
    stats = df[selected_feature].describe().to_frame().T
    st.dataframe(stats.style.format("{:.3f}"), use_container_width=True, hide_index=True)

with tab_bin:
    sel_bin = st.selectbox("Select a binary feature:", binary_features, key="bin_feature_select")

    col_a, col_b = st.columns(2)
    with col_a:
        if target_col:
            ct = pd.crosstab(df[sel_bin], df[target_col])
            ct.columns = ["No CVD", "CVD"]
            ct.index = [f"{sel_bin}=0", f"{sel_bin}=1"]
            fig_stack = go.Figure()
            fig_stack.add_trace(go.Bar(
                x=ct.index, y=ct["No CVD"], name="No CVD",
                marker_color=COLORS["accent_blue"], opacity=0.88,
            ))
            fig_stack.add_trace(go.Bar(
                x=ct.index, y=ct["CVD"], name="CVD",
                marker_color=COLORS["accent_magenta"], opacity=0.88,
            ))
            fig_stack.update_layout(
                title=f"{sel_bin} vs CVD Status",
                barmode="stack", height=400,
            )
            fig_stack = apply_plotly_theme(fig_stack)
            st.plotly_chart(fig_stack, use_container_width=True)

    with col_b:
        vc = df[sel_bin].value_counts().sort_index()
        fig_donut = go.Figure(go.Pie(
            labels=[f"{sel_bin}=0", f"{sel_bin}=1"],
            values=vc.values.tolist(),
            hole=0.5,
            marker=dict(colors=[COLORS["accent_cyan"], COLORS["accent_orange"]]),
            textinfo="label+percent+value",
        ))
        fig_donut.update_layout(title=f"{sel_bin} Distribution", height=400)
        fig_donut = apply_plotly_theme(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)

render_gradient_divider()

# ── Correlation Heatmap ──────────────────────────────────────────────────
render_section_header("🔗 Correlation Matrix", "Pearson correlations between all numeric features")

numeric_cols = [c for c in continuous_features + binary_features + ([target_col] if target_col else []) if c in df.columns]
corr_matrix = df[numeric_cols].corr()

fig_corr = go.Figure(go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns.tolist(),
    y=corr_matrix.index.tolist(),
    text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
    texttemplate="%{text}",
    textfont=dict(size=10, color="#E8EAED"),
    colorscale="RdBu_r",
    zmid=0,
    colorbar=dict(tickfont=dict(color="#9AA0A6")),
))
fig_corr.update_layout(
    title="Feature Correlation Heatmap",
    height=600,
    xaxis=dict(tickangle=-45),
)
fig_corr = apply_plotly_theme(fig_corr)
st.plotly_chart(fig_corr, use_container_width=True)

# Top correlations with target
if target_col:
    target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
    render_insight(
        "Strongest Correlations with CVD Target",
        f"Top 3: <strong>{target_corr.index[0]}</strong> ({target_corr.iloc[0]:.3f}), "
        f"<strong>{target_corr.index[1]}</strong> ({target_corr.iloc[1]:.3f}), "
        f"<strong>{target_corr.index[2]}</strong> ({target_corr.iloc[2]:.3f})"
    )

render_gradient_divider()

# ── Data Audit Report ────────────────────────────────────────────────────
render_section_header("🔍 Data Audit Report", "Feature types, unique values, and missing data analysis")

audit_df = load_csv("dataset_audit_report.csv")
if not audit_df.empty:
    st.dataframe(audit_df, use_container_width=True, hide_index=True, height=350)

# ── Outlier & Normality Analysis ─────────────────────────────────────────
col_out, col_norm = st.columns(2)

with col_out:
    render_section_header("🎯 Outlier Analysis", "IQR-based outlier detection")
    outlier_df = load_csv("outlier_analysis.csv")
    if not outlier_df.empty:
        st.dataframe(
            outlier_df.style.format({
                "Q1": "{:.2f}", "Q3": "{:.2f}", "IQR": "{:.2f}",
                "Lower_Bound": "{:.2f}", "Upper_Bound": "{:.2f}",
                "Pct_Outliers": "{:.1f}%",
            }).background_gradient(subset=["N_Outliers"], cmap="OrRd"),
            use_container_width=True, hide_index=True,
        )

with col_norm:
    render_section_header("📐 Normality Tests", "Shapiro-Wilk test results")
    normality_df = load_csv("normality_tests.csv")
    if not normality_df.empty:
        st.dataframe(normality_df, use_container_width=True, hide_index=True)

render_gradient_divider()

# ── Scaling Parameters ───────────────────────────────────────────────────
render_section_header("📏 Scaling Parameters", "StandardScaler mean and standard deviation used for feature normalization")

scaling_df = load_csv("scaling_parameters.csv")
if not scaling_df.empty:
    st.dataframe(
        scaling_df.style.format({"Mean_Original": "{:.3f}", "Std_Original": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )

render_gradient_divider()

# ── Interactive Data Table ───────────────────────────────────────────────
render_section_header("📋 Raw Data Preview", "Browse the cleaned dataset — first 200 rows")

with st.expander("🔎 Show Data Table", expanded=False):
    st.dataframe(df.head(200), use_container_width=True, hide_index=True, height=500)
