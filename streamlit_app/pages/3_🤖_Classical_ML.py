"""
Page 3: Classical ML Baselines
Metrics, ROC/PR curves, confusion matrices, and model comparison for classical ML models.
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
    load_csv, apply_plotly_theme, make_bar_chart, make_grouped_bar_chart,
    get_figure_path, figure_exists, COLORS, MODEL_COLORS,
)

st.set_page_config(page_title="Classical ML — CVD-GNN", page_icon="🤖", layout="wide")
inject_custom_css()
setup_sidebar()

# ═══════════════════════════════════════════════════════════════════════════
render_page_header(
    "🤖 Classical ML Baselines",
    "5 classical models with Stratified 5-Fold CV, SMOTE, and comprehensive evaluation"
)

# Load data
baseline_df = load_csv("baseline_metrics.csv")
comparison_df = load_csv("model_comparison.csv")

if baseline_df.empty:
    st.error("❌ Could not load baseline metrics. Make sure `results/baseline_metrics.csv` exists.")
    st.stop()

# ── Top KPIs ─────────────────────────────────────────────────────────────
best_f1_model = baseline_df.loc[baseline_df["Test_F1"].idxmax()]
best_auc_model = baseline_df.loc[baseline_df["Test_ROC_AUC"].idxmax()]
best_acc_model = baseline_df.loc[baseline_df["Test_Accuracy"].idxmax()]
fastest_model = baseline_df.loc[baseline_df["Training_Time_Sec"].idxmin()]

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card("🏆", f"{best_f1_model['Test_F1']:.3f}", "Best Test F1", best_f1_model["Model"])
with col2:
    render_kpi_card("📊", f"{best_auc_model['Test_ROC_AUC']:.3f}", "Best ROC-AUC", best_auc_model["Model"])
with col3:
    render_kpi_card("🎯", f"{best_acc_model['Test_Accuracy']:.3f}", "Best Accuracy", best_acc_model["Model"])
with col4:
    render_kpi_card("⚡", f"{fastest_model['Training_Time_Sec']:.2f}s", "Fastest Training", fastest_model["Model"])

st.markdown("<br>", unsafe_allow_html=True)

# ── Metrics Table ────────────────────────────────────────────────────────
render_section_header("📋 Complete Metrics Table", "Cross-validation (mean ± std) and test set performance")

tab_cv, tab_test = st.tabs(["📈 Cross-Validation Metrics", "🧪 Test Set Metrics"])

with tab_cv:
    cv_cols = ["Model", "CV_Accuracy_Mean", "CV_Accuracy_Std", "CV_Precision_Mean", "CV_Precision_Std",
               "CV_Recall_Mean", "CV_Recall_Std", "CV_F1_Mean", "CV_F1_Std", "CV_ROC_AUC_Mean", "CV_ROC_AUC_Std"]
    cv_display = baseline_df[cv_cols].copy()
    cv_display.columns = ["Model", "Acc Mean", "Acc Std", "Prec Mean", "Prec Std",
                          "Rec Mean", "Rec Std", "F1 Mean", "F1 Std", "AUC Mean", "AUC Std"]
    fmt = {c: "{:.4f}" for c in cv_display.columns if c != "Model"}
    st.dataframe(
        cv_display.style.format(fmt).background_gradient(
            subset=["F1 Mean", "AUC Mean"], cmap="YlOrRd", vmin=0.75, vmax=0.9
        ),
        use_container_width=True, hide_index=True,
    )

with tab_test:
    test_cols = ["Model", "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC"]
    test_display = baseline_df[test_cols].copy()
    test_display.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fmt = {c: "{:.4f}" for c in test_display.columns if c != "Model"}
    st.dataframe(
        test_display.style.format(fmt).background_gradient(
            subset=["F1", "ROC-AUC"], cmap="YlOrRd", vmin=0.8, vmax=0.95
        ).highlight_max(subset=["F1", "ROC-AUC", "Accuracy"], color="rgba(67,160,71,0.3)"),
        use_container_width=True, hide_index=True,
    )

render_gradient_divider()

# ── Interactive Metric Comparison ────────────────────────────────────────
render_section_header("📊 Model Comparison", "Select a metric to compare across models")

metric_options = {
    "Test F1 Score": "Test_F1",
    "Test ROC-AUC": "Test_ROC_AUC",
    "Test Accuracy": "Test_Accuracy",
    "Test Precision": "Test_Precision",
    "Test Recall": "Test_Recall",
    "CV F1 (Mean)": "CV_F1_Mean",
    "CV ROC-AUC (Mean)": "CV_ROC_AUC_Mean",
}

col_sel, col_chart = st.columns([1, 3])

with col_sel:
    selected_metric_name = st.radio(
        "Select Metric:",
        list(metric_options.keys()),
        index=0,
    )
    selected_metric = metric_options[selected_metric_name]

with col_chart:
    sorted_df = baseline_df.sort_values(selected_metric, ascending=False)
    fig = make_bar_chart(
        sorted_df, x="Model", y=selected_metric,
        title=f"Model Comparison — {selected_metric_name}",
        color_map=MODEL_COLORS,
        y_range=[0, 1.08],
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

# Multi-metric comparison
render_section_header("📊 Multi-Metric Comparison", "All test metrics side by side")

fig_multi = make_grouped_bar_chart(
    baseline_df, "Model",
    ["Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC"],
    "All Test Metrics — Side by Side",
)
fig_multi.update_layout(height=500)
st.plotly_chart(fig_multi, use_container_width=True)

render_gradient_divider()

# ── Training Time ────────────────────────────────────────────────────────
render_section_header("⏱️ Training Time Comparison", "Time in seconds to train each model")

time_sorted = baseline_df.sort_values("Training_Time_Sec", ascending=True)
fig_time = go.Figure(go.Bar(
    y=time_sorted["Model"],
    x=time_sorted["Training_Time_Sec"],
    orientation="h",
    marker=dict(
        color=[MODEL_COLORS.get(m, COLORS["accent_blue"]) for m in time_sorted["Model"]],
        opacity=0.88,
    ),
    text=time_sorted["Training_Time_Sec"].round(3).astype(str) + "s",
    textposition="outside",
    textfont=dict(size=12, color="#E8EAED"),
))
fig_time.update_layout(
    title="",
    xaxis_title="Training Time (seconds)",
    height=350,
)
fig_time = apply_plotly_theme(fig_time)
st.plotly_chart(fig_time, use_container_width=True)

render_insight(
    "Training Efficiency",
    f"<strong>{fastest_model['Model']}</strong> was the fastest to train ({fastest_model['Training_Time_Sec']:.3f}s), "
    f"while <strong>Random Forest</strong> took the longest ({baseline_df.loc[baseline_df['Training_Time_Sec'].idxmax(), 'Training_Time_Sec']:.3f}s)."
)

render_gradient_divider()

# ── Pre-generated Figures ────────────────────────────────────────────────
render_section_header("📈 Detailed Evaluation Plots", "Publication-quality figures from the experiment pipeline")

fig_tabs = st.tabs(["ROC Curves", "PR Curves", "Confusion Matrices", "CV Boxplots", "Confidence Intervals", "Model Comparison"])

figure_map = {
    "ROC Curves": "roc_curves.png",
    "PR Curves": "precision_recall_curves.png",
    "Confusion Matrices": "confusion_matrices.png",
    "CV Boxplots": "cv_boxplots.png",
    "Confidence Intervals": "confidence_intervals.png",
    "Model Comparison": "model_comparison_bar_chart.png",
}

for tab, (name, filename) in zip(fig_tabs, figure_map.items()):
    with tab:
        if figure_exists(filename):
            st.image(get_figure_path(filename), caption=name, use_container_width=True)
        else:
            st.info(f"Figure `{filename}` not found in the figures directory.")

render_gradient_divider()

# ── Model Hyperparameters ────────────────────────────────────────────────
render_section_header("⚙️ Model Hyperparameters", "Configuration used for each classical ML model")

hyperparams = {
    "Logistic Regression": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced", "max_iter": 1000},
    "Random Forest": {"n_estimators": 300, "max_depth": "None", "min_samples_split": 5, "min_samples_leaf": 2, "class_weight": "balanced"},
    "SVM": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced", "probability": True},
    "XGBoost": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "scale_pos_weight": 4.15},
    "LightGBM": {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 31, "subsample": 0.8, "is_unbalance": True},
}

for model_name, params in hyperparams.items():
    with st.expander(f"⚙️ {model_name}"):
        param_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
        st.dataframe(param_df, use_container_width=True, hide_index=True)
