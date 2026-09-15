"""
=============================================================================
CVD Risk Prediction Dashboard — Main Entry Point
=============================================================================
Streamlit multi-page app for the Cardiovascular Disease Risk Prediction
using Graph Neural Networks project.

Run with:  streamlit run streamlit_app/app.py
=============================================================================
"""

import streamlit as st

# ── Page Config (must be first Streamlit command) ────────────────────────
st.set_page_config(
    page_title="CVD Risk Prediction — GNN Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from helpers import inject_custom_css, setup_sidebar, render_page_header, render_kpi_card, render_gradient_divider, render_insight, load_csv, COLORS

# ── Inject CSS & Sidebar ────────────────────────────────────────────────
inject_custom_css()
setup_sidebar()

# ═══════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <h2>🧬 Cardiovascular Disease Risk Prediction</h2>
    <p>
        A comprehensive machine learning framework comparing <strong>5 classical ML models</strong>
        and <strong>4 Graph Neural Network architectures</strong> — including the novel
        <strong>Dynamic Patient Similarity Network (DPSN)</strong> — for predicting CVD risk
        from patient health records. Designed for an <strong>IEEE Conference Paper</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_kpi_card("🏥", "1,000", "Patients", "806 Healthy • 194 CVD")
with col2:
    render_kpi_card("🤖", "9", "Models Trained", "5 Classical + 4 GNN")
with col3:
    render_kpi_card("🏆", "0.894", "Best F1 Score", "LightGBM")
with col4:
    render_kpi_card("📊", "0.996", "Best ROC-AUC", "Logistic Regression")

st.markdown("<br>", unsafe_allow_html=True)

# ── Final Ranking ────────────────────────────────────────────────────────
render_page_header("📋 Final Model Ranking", "All 9 models ranked by composite performance score")

ranking_df = load_csv("final_ranking.csv")
if not ranking_df.empty:
    # Style the dataframe
    def style_ranking(df):
        styled = df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "PR-AUC": "{:.4f}",
            "MCC": "{:.4f}",
        }).background_gradient(
            subset=["F1", "ROC-AUC", "MCC"],
            cmap="YlOrRd",
            vmin=0.7, vmax=0.9,
        ).set_properties(**{
            "text-align": "center",
            "font-size": "13px",
        })
        return styled

    st.dataframe(
        style_ranking(ranking_df),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    # Insight cards
    col_a, col_b = st.columns(2)
    with col_a:
        render_insight(
            "Key Finding",
            "<strong>LightGBM</strong> achieved the best F1 score (0.894) among all models. "
            "Among GNN architectures, <strong>GraphSAGE</strong> performed best (F1 = 0.851), "
            "ranking 4th overall."
        )
    with col_b:
        render_insight(
            "DPSN Performance",
            "The novel <strong>DPSN</strong> ranked 7th (F1 = 0.813) but showed "
            "competitive ROC-AUC (0.988), demonstrating strong discriminative ability "
            "despite lower precision."
        )

render_gradient_divider()

# ── Evidence Report ──────────────────────────────────────────────────────
render_page_header("📑 Evidence Report Summary", "Key research questions and answers from the study")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="kpi-card" style="text-align: left; padding: 20px 24px;">
        <div style="font-weight: 700; color: #1E88E5; font-size: 15px; margin-bottom: 12px;">
            Q1: Does DPSN outperform all baselines?
        </div>
        <div style="color: #E8EAED; font-size: 14px; line-height: 1.8;">
            DPSN Rank: <span class="stat-badge magenta">7 / 9</span><br>
            DPSN F1: <span class="stat-badge">0.813</span><br>
            <span style="color: #9AA0A6; font-size: 12px;">
                Not the top performer, but competitive in ROC-AUC.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="kpi-card" style="text-align: left; padding: 20px 24px;">
        <div style="font-weight: 700; color: #D81B60; font-size: 15px; margin-bottom: 12px;">
            Q3: Statistically significant?
        </div>
        <div style="color: #E8EAED; font-size: 14px; line-height: 1.8;">
            Significant comparisons: <span class="stat-badge green">17 / 24</span><br>
            Significance level: <span class="stat-badge">p &lt; 0.05</span><br>
            <span style="color: #9AA0A6; font-size: 12px;">
                Paired t-tests and Wilcoxon signed-rank tests applied.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# F1 comparison against DPSN
st.markdown("""
<div class="section-header">
    <h3>📊 F1 Score Gap — DPSN vs. All Models</h3>
    <p>Positive values mean DPSN outperforms; negative values mean baseline is better</p>
</div>
""", unsafe_allow_html=True)

import plotly.graph_objects as go
from helpers import apply_plotly_theme

models = ["GAT", "GCN", "Random Forest", "XGBoost", "GraphSAGE", "SVM", "Logistic Regression", "LightGBM"]
f1_gaps = [0.039, 0.013, -0.014, -0.031, -0.037, -0.037, -0.054, -0.081]
gap_colors = [COLORS["accent_emerald"] if g >= 0 else COLORS["accent_magenta"] for g in f1_gaps]

fig = go.Figure(go.Bar(
    y=models,
    x=f1_gaps,
    orientation="h",
    marker=dict(color=gap_colors, opacity=0.88, line=dict(width=0)),
    text=[f"{g:+.3f}" for g in f1_gaps],
    textposition="outside",
    textfont=dict(size=12, color="#E8EAED"),
))
fig.update_layout(
    title="",
    xaxis_title="F1 Score Difference (DPSN − Baseline)",
    yaxis_title="",
    height=380,
    xaxis=dict(zeroline=True, zerolinecolor="#9AA0A6", zerolinewidth=1),
)
fig = apply_plotly_theme(fig)
st.plotly_chart(fig, use_container_width=True)

render_gradient_divider()

# ── Pipeline Overview ────────────────────────────────────────────────────
render_page_header("🔬 10-Phase Research Pipeline", "Complete experiment workflow from data to publication")

phases = [
    ("Phase 1", "Data Audit & Preprocessing", "Cleaned 1,000 patient records, handled missing values, outlier analysis, feature scaling"),
    ("Phase 2", "Classical ML Baselines", "Trained 5 models with Stratified 5-Fold CV + SMOTE"),
    ("Phase 3", "Graph Construction", "Built 4 patient similarity graphs (Euclidean KNN, Cosine, Clinical, Dynamic)"),
    ("Phase 4", "GNN Training", "Trained GCN, GraphSAGE, GAT across all 4 graph types"),
    ("Phase 5", "DPSN Training", "Novel Dynamic Patient Similarity Network"),
    ("Phase 6", "Ablation Studies", "5 DPSN variants to measure component contributions"),
    ("Phase 7", "Statistical Validation", "Paired t-tests, Wilcoxon signed-rank, bootstrap CIs"),
    ("Phase 8", "Explainability", "SHAP for tree models, GNNExplainer for DPSN"),
    ("Phase 9", "Publication Figures", "15 figures at 300 DPI for IEEE"),
    ("Phase 10", "Final Report", "Evidence report, rankings, LaTeX tables"),
]

cols = st.columns(5)
for i, (phase, name, desc) in enumerate(phases):
    with cols[i % 5]:
        st.markdown(f"""
        <div class="kpi-card" style="text-align: left; padding: 16px; margin-bottom: 12px; min-height: 150px;">
            <div style="font-size: 11px; font-weight: 700; color: #1E88E5; letter-spacing: 1px; text-transform: uppercase;">{phase}</div>
            <div style="font-size: 14px; font-weight: 700; color: #E8EAED; margin: 6px 0;">{name}</div>
            <div style="font-size: 12px; color: #9AA0A6; line-height: 1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
