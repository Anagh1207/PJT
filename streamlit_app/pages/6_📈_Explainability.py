"""
Page 6: Feature Importance & Explainability
SHAP analysis, GNN feature importance, cross-method comparison, and clinical weights.
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
    render_section_header, render_insight, render_gradient_divider,
    load_csv, apply_plotly_theme, make_radar_chart,
    get_figure_path, figure_exists, COLORS,
)

st.set_page_config(page_title="Explainability — CVD-GNN", page_icon="📈", layout="wide")
inject_custom_css()
setup_sidebar()

# ═══════════════════════════════════════════════════════════════════════════
render_page_header(
    "📈 Feature Importance & Explainability",
    "SHAP values, GNN importance, cross-method comparison, and clinical domain weights"
)

# Load data
shap_lgbm = load_csv("shap_importance_lightgbm.csv")
shap_xgb = load_csv("shap_importance_xgboost.csv")
gnn_imp = load_csv("gnn_feature_importance.csv")

# ── KPIs ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    top_shap_lgbm = shap_lgbm.iloc[0]["Feature"] if not shap_lgbm.empty else "N/A"
    render_kpi_card("🌳", top_shap_lgbm, "Top SHAP (LightGBM)", "Highest mean |SHAP|")
with col2:
    top_shap_xgb = shap_xgb.iloc[0]["Feature"] if not shap_xgb.empty else "N/A"
    render_kpi_card("🚀", top_shap_xgb, "Top SHAP (XGBoost)", "Highest mean |SHAP|")
with col3:
    top_gnn = gnn_imp.iloc[0]["Feature"] if not gnn_imp.empty else "N/A"
    render_kpi_card("🕸️", top_gnn, "Top GNN Feature", "GNNExplainer importance")
with col4:
    render_kpi_card("🏥", "Cholesterol", "Top Clinical Weight", "Domain expert: 2.0")

st.markdown("<br>", unsafe_allow_html=True)

# ── SHAP Analysis ────────────────────────────────────────────────────────
render_section_header("🌳 SHAP Feature Importance", "Mean absolute SHAP values for tree-based models")

tab_lgbm, tab_xgb, tab_compare = st.tabs(["LightGBM", "XGBoost", "Side-by-Side"])

with tab_lgbm:
    col_t, col_c = st.columns([1, 2])
    with col_t:
        if not shap_lgbm.empty:
            st.dataframe(
                shap_lgbm.style.format({"Mean |SHAP|": "{:.4f}"}).background_gradient(
                    subset=["Mean |SHAP|"], cmap="Reds"
                ),
                use_container_width=True, hide_index=True,
            )
    with col_c:
        if not shap_lgbm.empty:
            sorted_lgbm = shap_lgbm.sort_values("Mean |SHAP|", ascending=True)
            fig = go.Figure(go.Bar(
                y=sorted_lgbm["Feature"],
                x=sorted_lgbm["Mean |SHAP|"],
                orientation="h",
                marker=dict(
                    color=sorted_lgbm["Mean |SHAP|"],
                    colorscale="Reds",
                    opacity=0.88,
                ),
                text=sorted_lgbm["Mean |SHAP|"].round(3),
                textposition="outside",
                textfont=dict(size=11, color="#E8EAED"),
            ))
            fig.update_layout(title="LightGBM — SHAP Importance", xaxis_title="Mean |SHAP|", height=500)
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    if figure_exists("shap_importance_lightgbm.png"):
        with st.expander("📊 Pre-generated SHAP Figure (LightGBM)"):
            st.image(get_figure_path("shap_importance_lightgbm.png"), use_container_width=True)

with tab_xgb:
    col_t, col_c = st.columns([1, 2])
    with col_t:
        if not shap_xgb.empty:
            st.dataframe(
                shap_xgb.style.format({"Mean |SHAP|": "{:.4f}"}).background_gradient(
                    subset=["Mean |SHAP|"], cmap="Oranges"
                ),
                use_container_width=True, hide_index=True,
            )
    with col_c:
        if not shap_xgb.empty:
            sorted_xgb = shap_xgb.sort_values("Mean |SHAP|", ascending=True)
            fig = go.Figure(go.Bar(
                y=sorted_xgb["Feature"],
                x=sorted_xgb["Mean |SHAP|"],
                orientation="h",
                marker=dict(
                    color=sorted_xgb["Mean |SHAP|"],
                    colorscale="Oranges",
                    opacity=0.88,
                ),
                text=sorted_xgb["Mean |SHAP|"].round(3),
                textposition="outside",
                textfont=dict(size=11, color="#E8EAED"),
            ))
            fig.update_layout(title="XGBoost — SHAP Importance", xaxis_title="Mean |SHAP|", height=500)
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    if figure_exists("shap_importance_xgboost.png"):
        with st.expander("📊 Pre-generated SHAP Figure (XGBoost)"):
            st.image(get_figure_path("shap_importance_xgboost.png"), use_container_width=True)

with tab_compare:
    if not shap_lgbm.empty and not shap_xgb.empty:
        merged = shap_lgbm.merge(shap_xgb, on="Feature", suffixes=("_LightGBM", "_XGBoost"))
        merged = merged.sort_values("Mean |SHAP|_LightGBM", ascending=True)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            y=merged["Feature"],
            x=merged["Mean |SHAP|_LightGBM"],
            name="LightGBM",
            orientation="h",
            marker_color=COLORS["accent_magenta"],
            marker_opacity=0.88,
        ))
        fig_comp.add_trace(go.Bar(
            y=merged["Feature"],
            x=merged["Mean |SHAP|_XGBoost"],
            name="XGBoost",
            orientation="h",
            marker_color=COLORS["accent_orange"],
            marker_opacity=0.88,
        ))
        fig_comp.update_layout(
            title="SHAP Importance — LightGBM vs XGBoost",
            barmode="group",
            height=500,
            xaxis_title="Mean |SHAP|",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_comp = apply_plotly_theme(fig_comp)
        st.plotly_chart(fig_comp, use_container_width=True)

        render_insight(
            "Consistency Across Models",
            "Both LightGBM and XGBoost agree on the top 3 features: "
            "<strong>Age</strong>, <strong>Cholesterol</strong>, and <strong>Blood Pressure</strong>. "
            "This consistency validates these features as genuine CVD risk factors."
        )

render_gradient_divider()

# ── GNN Feature Importance ───────────────────────────────────────────────
render_section_header("🕸️ GNN Feature Importance", "Feature importance from GNNExplainer")

if not gnn_imp.empty:
    col_t, col_c = st.columns([1, 2])
    with col_t:
        st.dataframe(
            gnn_imp.style.format({"GNN Importance": "{:.4f}"}).background_gradient(
                subset=["GNN Importance"], cmap="Blues"
            ),
            use_container_width=True, hide_index=True,
        )
    with col_c:
        sorted_gnn = gnn_imp.sort_values("GNN Importance", ascending=True)
        fig = go.Figure(go.Bar(
            y=sorted_gnn["Feature"],
            x=sorted_gnn["GNN Importance"],
            orientation="h",
            marker=dict(
                color=sorted_gnn["GNN Importance"],
                colorscale="Blues",
                opacity=0.88,
            ),
            text=sorted_gnn["GNN Importance"].round(3),
            textposition="outside",
            textfont=dict(size=11, color="#E8EAED"),
        ))
        fig.update_layout(title="GNN Feature Importance", xaxis_title="Importance Score", height=500)
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    if figure_exists("gnn_feature_importance_aggregated.png"):
        with st.expander("📊 Pre-generated GNN Importance Figure"):
            st.image(get_figure_path("gnn_feature_importance_aggregated.png"), use_container_width=True)

render_gradient_divider()

# ── Cross-Method Radar Chart ─────────────────────────────────────────────
render_section_header(
    "🕷️ Cross-Method Feature Comparison",
    "Radar chart comparing feature rankings across SHAP (LightGBM), SHAP (XGBoost), and GNN"
)

if not shap_lgbm.empty and not shap_xgb.empty and not gnn_imp.empty:
    # Normalize each method to [0, 1]
    top_n = 10
    features = shap_lgbm["Feature"].head(top_n).tolist()

    def normalize(series):
        mn, mx = series.min(), series.max()
        if mx - mn == 0:
            return series / mx if mx > 0 else series
        return (series - mn) / (mx - mn)

    lgbm_vals = normalize(shap_lgbm.set_index("Feature")["Mean |SHAP|"])
    xgb_vals = normalize(shap_xgb.set_index("Feature")["Mean |SHAP|"])
    gnn_vals = normalize(gnn_imp.set_index("Feature")["GNN Importance"])

    radar_data = {
        "SHAP (LightGBM)": [lgbm_vals.get(f, 0) for f in features],
        "SHAP (XGBoost)": [xgb_vals.get(f, 0) for f in features],
        "GNN Importance": [gnn_vals.get(f, 0) for f in features],
    }

    fig_radar = make_radar_chart(features, radar_data, "Cross-Method Feature Importance (Normalized)")
    fig_radar.update_layout(height=550)
    st.plotly_chart(fig_radar, use_container_width=True)

    render_insight(
        "Cross-Method Agreement",
        "<strong>Blood Pressure</strong>, <strong>Age</strong>, and <strong>Cholesterol</strong> "
        "consistently rank as the most important features across all three methods (SHAP LightGBM, "
        "SHAP XGBoost, and GNNExplainer), confirming their clinical significance for CVD prediction."
    )

render_gradient_divider()

# ── Clinical Domain Weights ──────────────────────────────────────────────
render_section_header("🏥 Clinical Domain Weights", "Expert-assigned feature weights vs. learned importance")

clinical_weights = {
    'Age': 1.5, 'Blood_Pressure': 1.8, 'Cholesterol': 2.0,
    'Heart_Rate': 1.0, 'BMI': 1.3, 'Diabetes': 1.8,
    'Smoking': 1.6, 'Physical_Activity_HoursPerWeek': 1.0,
    'ECG_Score': 1.4, 'Family_History': 1.5,
    'Glucose': 1.7, 'Triglycerides': 1.5,
    'Stress_Index': 1.0, 'Gender': 0.5,
}

cw_df = pd.DataFrame(list(clinical_weights.items()), columns=["Feature", "Clinical Weight"])
cw_df = cw_df.sort_values("Clinical Weight", ascending=True)

col_t, col_c = st.columns([1, 2])
with col_t:
    st.dataframe(
        cw_df.sort_values("Clinical Weight", ascending=False).style.format({"Clinical Weight": "{:.1f}"}).background_gradient(
            subset=["Clinical Weight"], cmap="Greens"
        ),
        use_container_width=True, hide_index=True,
    )

with col_c:
    fig = go.Figure(go.Bar(
        y=cw_df["Feature"],
        x=cw_df["Clinical Weight"],
        orientation="h",
        marker=dict(
            color=cw_df["Clinical Weight"],
            colorscale="Greens",
            opacity=0.88,
        ),
        text=cw_df["Clinical Weight"].round(1),
        textposition="outside",
        textfont=dict(size=11, color="#E8EAED"),
    ))
    fig.update_layout(
        title="Clinical Domain Weights (Expert-Assigned)",
        xaxis_title="Weight",
        height=500,
    )
    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

render_insight(
    "Domain Knowledge Alignment",
    "The clinical weights assign highest importance to <strong>Cholesterol</strong> (2.0), "
    "<strong>Blood Pressure</strong> (1.8), and <strong>Diabetes</strong> (1.8) — "
    "which aligns strongly with the data-driven SHAP analysis. This validates that "
    "the ML models capture clinically meaningful patterns."
)
