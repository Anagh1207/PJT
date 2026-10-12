"""
=============================================================================
Step 8 - CPM "What-If" Streamlit Demo App
=============================================================================
Interactive dashboard that lets clinicians:
  1. Select or enter a patient's feature profile
  2. Generate a counterfactual (What-if: "what features would reduce risk?")
  3. See the patient migrate in graph embedding space
  4. Compare Normal CF vs Fixed-Graph vs Topology-Aware predictions
  5. View the CPM score breakdown (T, R, D_norm, F_norm effort)

Run with: streamlit run cpm/step8_streamlit_demo.py
=============================================================================
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLEANED_DATASET, SYNTHETIC_DATASET, RESULTS_DIR, FIGURES_DIR,
    ALL_FEATURES, TARGET_COLUMN, RANDOM_STATE,
    MODEL_DIR, GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT, CPM_CONFIG,
    COUNTERFACTUAL_CONFIG
)
from models.gnn_models import GraphSAGE
from graph_construction import (
    build_patient_similarity_graph, rebuild_graph_for_patient,
    compute_neighborhood_stats
)
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CPM What-If Explorer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f1117;
        color: #e0e0e0;
    }
    .main-header {
        background: linear-gradient(135deg, #1a1d2e 0%, #0d1117 100%);
        border-left: 4px solid #4e9af1;
        padding: 1.5rem 2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: #4e9af1; margin: 0; font-weight: 700; font-size: 2rem; }
    .main-header p  { color: #888; margin: 0.5rem 0 0 0; font-size: 0.95rem; }

    .metric-card {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover { border-color: #4e9af1; }
    .metric-label { color: #888; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }
    .metric-value { color: white; font-size: 1.8rem; font-weight: 700; line-height: 1; }
    .metric-delta-pos { color: #2ecc71; font-size: 0.85rem; margin-top: 0.3rem; }
    .metric-delta-neg { color: #e74c3c; font-size: 0.85rem; margin-top: 0.3rem; }

    .cpm-bar { background: #1a1d2e; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
    .risk-chip-high { background: #e74c3c22; border: 1px solid #e74c3c; color: #e74c3c; padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
    .risk-chip-low  { background: #2ecc7122; border: 1px solid #2ecc71; color: #2ecc71; padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }

    .section-header { color: #4e9af1; font-size: 1.1rem; font-weight: 700; margin: 1.5rem 0 0.8rem 0; padding-bottom: 0.4rem; border-bottom: 1px solid #2a2d3e; }
    .info-box { background: #1a1d2e; border-left: 3px solid #4e9af1; border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; color: #bbb; font-size: 0.9rem; margin: 1rem 0; }
    .warn-box  { background: #1a1d2e; border-left: 3px solid #f4a261; border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; color: #bbb; font-size: 0.9rem; margin: 1rem 0; }
    .divider { height: 1px; background: #2a2d3e; margin: 1.5rem 0; }

    div[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1a1d2e; }
    .stSlider > div > div { background: #2a2d3e; }
    .stButton > button { background: linear-gradient(135deg, #4e9af1, #2980b9); color: white; border: none; border-radius: 8px; padding: 0.6rem 1.8rem; font-weight: 600; transition: all 0.2s; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(78,154,241,0.3); }
    .stSelectbox > div > div { background: #1a1d2e; border-color: #2a2d3e; }
</style>
""", unsafe_allow_html=True)


# ── Resource loading ────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    df_real = pd.read_csv(CLEANED_DATASET)
    df_syn  = pd.read_csv(SYNTHETIC_DATASET)
    df_full = pd.concat([df_real, df_syn], ignore_index=True)

    X_real = df_real[ALL_FEATURES].values.astype(float)
    y_real = df_real[TARGET_COLUMN].values.astype(int)

    idx_all = np.arange(len(df_real))
    idx_tv, idx_test, y_tv, _ = train_test_split(
        idx_all, y_real, test_size=0.20, stratify=y_real, random_state=RANDOM_STATE)
    idx_train, idx_val, _, _ = train_test_split(
        idx_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RANDOM_STATE)
    idx_dev = np.concatenate([idx_train, idx_val])
    X_dev = X_real[idx_dev]
    y_dev = y_real[idx_dev]

    ck = torch.load(os.path.join(MODEL_DIR, "graphsage_optimal.pt"),
                    map_location='cpu', weights_only=False)
    optimal_k = ck.get('optimal_k', CPM_CONFIG['graph_k'])
    model = GraphSAGE(len(ALL_FEATURES), GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    X_full = df_full[ALL_FEATURES].values.astype(float)
    y_full = df_full[TARGET_COLUMN].values.astype(int)
    edge_index_full, edge_weight_full, _ = build_patient_similarity_graph(
        X_full, k=optimal_k, feature_names=ALL_FEATURES, metric='clinical')

    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
    lgb_model.fit(X_dev, y_dev)

    pop_min = X_dev.min(axis=0)
    pop_max = X_dev.max(axis=0)
    pop_range = pop_max - pop_min
    pop_range[pop_range == 0] = 1.0

    with torch.no_grad():
        embeddings = model.get_embeddings(
            torch.tensor(X_full, dtype=torch.float), edge_index_full).numpy()

    return {
        'X_full': X_full, 'y_full': y_full,
        'X_dev': X_dev, 'y_dev': y_dev,
        'df_real': df_real, 'df_full': df_full,
        'model': model, 'lgb_model': lgb_model,
        'edge_index_full': edge_index_full,
        'edge_weight_full': edge_weight_full,
        'optimal_k': optimal_k,
        'pop_min': pop_min, 'pop_max': pop_max, 'pop_range': pop_range,
        'embeddings': embeddings,
    }


def _generate_cf(x_orig, lgb_model, pop_min, pop_max, pop_range, n_steps=150):
    """Generate counterfactual via range-scaled perturbation."""
    x_cf = x_orig.copy()
    lower_indices   = [ALL_FEATURES.index(f) for f in
                       ['Blood_Pressure', 'Cholesterol', 'BMI', 'Glucose',
                        'Triglycerides', 'Stress_Index', 'Heart_Rate', 'Smoking', 'Diabetes']
                       if f in ALL_FEATURES and f in COUNTERFACTUAL_CONFIG['modifiable_features']]
    increase_indices = [ALL_FEATURES.index(f) for f in ['Physical_Activity_HoursPerWeek']
                        if f in ALL_FEATURES and f in COUNTERFACTUAL_CONFIG['modifiable_features']]
    for _ in range(n_steps):
        p = lgb_model.predict_proba(x_cf.reshape(1, -1))[0, 1]
        if p <= 0.50:
            break
        for j in lower_indices:
            x_cf[j] = max(pop_min[j], x_cf[j] - 0.02 * pop_range[j])
        for j in increase_indices:
            x_cf[j] = min(pop_max[j], x_cf[j] + 0.02 * pop_range[j])
    return x_cf


def _gnn_prob(model, X, edge_index, local_idx):
    with torch.no_grad():
        return torch.softmax(
            model(torch.tensor(X, dtype=torch.float), edge_index)[local_idx], dim=0)[1].item()


def _make_comparison_fig(a1_orig, a1_cf, a2_orig, a2_cf, a3_orig, a3_cf):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d2e')
    labels = ['Normal CF\n(LightGBM)', 'Fixed-Graph\n(GNN, frozen)', 'Topology-Aware\n(GNN + rebuild)']
    orig_risks = [a1_orig, a2_orig, a3_orig]
    cf_risks   = [a1_cf,   a2_cf,   a3_cf]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, orig_risks, w, label='Original', color='#e74c3c', alpha=0.8)
    ax.bar(x + w/2, cf_risks,   w, label='Counterfactual', color='#2ecc71', alpha=0.8)
    for i, (o, c) in enumerate(zip(orig_risks, cf_risks)):
        ax.annotate('', xy=(i + w/2, c + 0.01), xytext=(i - w/2, o + 0.01),
                    arrowprops=dict(arrowstyle='->', color='#4e9af1', lw=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='#ccc', fontsize=9)
    ax.set_ylabel('CVD Risk Probability', color='#ccc')
    ax.set_ylim(0, 1.1)
    ax.legend(facecolor='#1a1d2e', labelcolor='#ccc', framealpha=0.8)
    ax.tick_params(colors='#ccc')
    ax.spines[:].set_color('#444')
    ax.set_title('Risk Before vs After Counterfactual by Approach', color='white', fontsize=10)
    return fig


def _make_cpm_radar(T, R, D, F):
    labels = ['Turnover\n(T)', 'Risk Shift\n(R)', 'Emb Disp\n(D)', 'CF Effort\n(F)']
    vals   = [T, R, D, F]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals_c = vals + [vals[0]]
    angles_c = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d2e')
    ax.plot(angles_c, vals_c, color='#4e9af1', linewidth=2)
    ax.fill(angles_c, vals_c, color='#4e9af1', alpha=0.2)
    ax.set_thetagrids(np.degrees(angles), labels, color='#ccc', fontsize=9)
    ax.set_ylim(0, 1)
    ax.tick_params(colors='#666')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(color='#333', linestyle='--', alpha=0.5)
    ax.set_title('CPM Component Breakdown', color='white', fontsize=10, pad=20)
    return fig


def _make_embedding_fig(embeddings, y_full, patient_idx, cf_embedding):
    """2D t-SNE embedding with patient before/after highlighted."""
    subset = min(500, len(embeddings))
    np.random.seed(42)
    sample_idx = np.random.choice(len(embeddings), size=subset, replace=False)
    if patient_idx not in sample_idx:
        sample_idx[0] = patient_idx
    emb_sub = embeddings[sample_idx]
    y_sub   = y_full[sample_idx]

    # Add CF embedding
    all_emb = np.vstack([emb_sub, cf_embedding.reshape(1, -1)])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=400)
    coords = tsne.fit_transform(all_emb)

    orig_coord = coords[np.where(sample_idx == patient_idx)[0][0]]
    cf_coord   = coords[-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d2e')
    colors_bg = ['#e74c3c' if y == 1 else '#4e9af1' for y in y_sub]
    ax.scatter(coords[:-1, 0], coords[:-1, 1],
               c=colors_bg, alpha=0.35, s=15, zorder=1)
    ax.scatter(*orig_coord, color='white', s=200, zorder=5,
               marker='*', label='Original position', edgecolors='#f4a261', linewidths=1.5)
    ax.scatter(*cf_coord, color='#2ecc71', s=200, zorder=5,
               marker='*', label='CF position (Topo-Aware)', edgecolors='white', linewidths=1.5)
    ax.annotate('', xy=cf_coord, xytext=orig_coord,
                arrowprops=dict(arrowstyle='->', color='#4e9af1', lw=1.5))

    handles = [
        mpatches.Patch(color='#e74c3c', label='CVD+ population', alpha=0.5),
        mpatches.Patch(color='#4e9af1', label='CVD- population', alpha=0.5),
        plt.Line2D([0], [0], marker='*', color='w', markersize=10, label='Original'),
        plt.Line2D([0], [0], marker='*', color='#2ecc71', markersize=10, label='Counterfactual'),
    ]
    ax.legend(handles=handles, facecolor='#1a1d2e', labelcolor='#ccc', framealpha=0.8, fontsize=8)
    ax.tick_params(colors='#666')
    ax.spines[:].set_color('#444')
    ax.set_title('Patient Graph Embedding Migration', color='white', fontsize=10)
    return fig


# ── Main app ───────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🫀 CPM What-If Explorer</h1>
        <p>Counterfactual Patient Migration · CVD Risk Reduction Pipeline · GNN + Graph-Aware Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Load
    with st.spinner("Loading model and data..."):
        res = load_resources()

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 👤 Patient Profile")
        st.markdown("Adjust feature values or pick a high-risk patient.")

        # Patient selector
        step5_csv = os.path.join(RESULTS_DIR, "cpm_step5_scores.csv")
        if os.path.exists(step5_csv):
            cpm_df = pd.read_csv(step5_csv).head(10)
            patient_options = [f"Patient {int(r['global_idx'])} (CPM={r['CPM_score']:.3f})"
                               for _, r in cpm_df.iterrows()]
            patient_options = ["Custom entry"] + patient_options
        else:
            patient_options = ["Custom entry"]

        selected_patient = st.selectbox("Load example patient", patient_options)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Feature sliders
        X_full = res['X_full']
        pop_min, pop_max = res['pop_min'], res['pop_max']
        feature_vals = {}

        # Determine defaults
        defaults = {}
        if selected_patient != "Custom entry" and os.path.exists(step5_csv):
            pat_idx = int(selected_patient.split("Patient ")[1].split(" ")[0])
            defaults = dict(zip(ALL_FEATURES, X_full[pat_idx]))

        for feat in ALL_FEATURES:
            lo = float(pop_min[ALL_FEATURES.index(feat)])
            hi = float(pop_max[ALL_FEATURES.index(feat)])
            default_val = defaults.get(feat, float(np.median(X_full[:, ALL_FEATURES.index(feat)])))
            if hi - lo < 5:
                step = 0.01
            elif hi - lo < 50:
                step = 0.1
            else:
                step = 1.0
            feature_vals[feat] = st.slider(
                feat, min_value=float(lo), max_value=float(hi),
                value=float(np.clip(default_val, lo, hi)), step=step,
                key=f"slider_{feat}"
            )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        run_btn = st.button("🔬 Run CPM Analysis", use_container_width=True)

    # ── Main content ───────────────────────────────────────────────────────
    if not run_btn:
        st.markdown("""
        <div class="info-box">
            <b>How to use:</b><br>
            1. Select a high-risk patient from the dropdown (or manually set feature values)<br>
            2. Click <b>Run CPM Analysis</b> to generate a counterfactual and compute the CPM score<br>
            3. Compare how the 3 CF approaches differ in estimated risk reduction<br>
            4. Inspect the patient's graph migration in embedding space
        </div>
        """, unsafe_allow_html=True)

        # Show population stats
        col1, col2, col3, col4 = st.columns(4)
        y_full = res['y_full']
        for col, (label, value) in zip([col1, col2, col3, col4], [
            ("Total Patients", f"{len(X_full):,}"),
            ("CVD+ Rate", f"{y_full.mean():.1%}"),
            ("Graph K*", str(res['optimal_k'])),
            ("CPM Formula", "0.40·T + 0.40·R + 0.20·D"),
        ]):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>""", unsafe_allow_html=True)
        return

    # ── Run analysis ───────────────────────────────────────────────────────
    x_orig = np.array([feature_vals[f] for f in ALL_FEATURES], dtype=float)

    # Find closest real patient in full population for global index
    dists = np.linalg.norm(X_full - x_orig, axis=1)
    closest_idx = int(np.argmin(dists))

    with st.spinner("Generating counterfactual..."):
        x_cf = _generate_cf(x_orig, res['lgb_model'], pop_min, pop_max, res['pop_range'])

    cf_success = res['lgb_model'].predict_proba(x_cf.reshape(1, -1))[0, 1] <= 0.50

    with st.spinner("Computing GNN risks and graph rebuild..."):
        # A1: Normal CF
        a1_orig = res['lgb_model'].predict_proba(x_orig.reshape(1, -1))[0, 1]
        a1_cf   = res['lgb_model'].predict_proba(x_cf.reshape(1, -1))[0, 1]

        # A2: Fixed-Graph
        a2_orig = _gnn_prob(res['model'], X_full, res['edge_index_full'], closest_idx)
        X_full_mod = X_full.copy()
        X_full_mod[closest_idx] = x_cf
        a2_cf = _gnn_prob(res['model'], X_full_mod, res['edge_index_full'], closest_idx)

        # A3: Topology-Aware
        edge_new, _, _ = rebuild_graph_for_patient(
            X_full, closest_idx, x_cf, k=res['optimal_k'],
            feature_names=ALL_FEATURES, metric='clinical')
        a3_orig = a2_orig
        a3_cf = _gnn_prob(res['model'], X_full_mod, edge_new, closest_idx)

        # Neighborhood stats
        stats_o = compute_neighborhood_stats(res['edge_index_full'], res['y_full'], closest_idx, k=res['optimal_k'])
        stats_n = compute_neighborhood_stats(edge_new, res['y_full'], closest_idx, k=res['optimal_k'])

        # CPM components
        set_o, set_n = set(stats_o['neighbors']), set(stats_n['neighbors'])
        inter = len(set_o & set_n)
        union_sz = len(set_o | set_n)
        T = 1.0 - (inter / union_sz if union_sz > 0 else 0)
        R = max(0.0, stats_o['cvd_risk_ratio'] - stats_n['cvd_risk_ratio'])

        X_full_t = torch.tensor(X_full, dtype=torch.float)
        X_full_mod_t = torch.tensor(X_full_mod, dtype=torch.float)
        with torch.no_grad():
            emb_orig = res['model'].get_embeddings(X_full_t, res['edge_index_full'])[closest_idx].numpy()
            emb_cf   = res['model'].get_embeddings(X_full_mod_t, edge_new)[closest_idx].numpy()
        D_raw = float(np.linalg.norm(emb_cf - emb_orig))

        d_vals = [float(np.linalg.norm(res['embeddings'][i] - res['embeddings'][0]))
                  for i in range(min(100, len(res['embeddings'])))]
        d_norm_ref = np.array(d_vals)
        D_norm = float(np.clip((D_raw - d_norm_ref.min()) / (d_norm_ref.max() - d_norm_ref.min() + 1e-9), 0, 1))

        l2 = float(np.linalg.norm(x_cf - x_orig))
        F_raw = l2 / np.sqrt(len(x_orig))
        F_norm = min(1.0, F_raw)

        CPM = 0.40 * T + 0.40 * R + 0.20 * D_norm

    # ── Display results ────────────────────────────────────────────────────
    if not cf_success:
        st.warning("⚠️ Counterfactual could not fully reduce risk below 0.50. "
                   "Results show partial migration. Consider adjusting feature values.")

    # Key metrics row
    st.markdown('<p class="section-header">📊 CPM Score Breakdown</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    risk_orig = a1_orig
    risk_after = a1_cf
    risk_diff = risk_orig - risk_after

    for col, (label, val, delta) in zip([c1, c2, c3, c4, c5], [
        ("CPM Score",       f"{CPM:.3f}",  None),
        ("T (Turnover)",    f"{T:.3f}",    None),
        ("R (Risk Shift)",  f"{R:.3f}",    None),
        ("D (Emb Disp)",   f"{D_norm:.3f}", None),
        ("F (CF Effort)",  f"{F_norm:.3f}", "standalone diagnostic"),
    ]):
        delta_html = f'<div class="metric-delta-pos">{delta}</div>' if delta else ""
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            {delta_html}
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Three-way comparison
    st.markdown('<p class="section-header">🔬 Three-Way Risk Comparison</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, (approach, orig_r, cf_r) in zip([c1, c2, c3], [
        ("Normal CF (LGB)",     a1_orig, a1_cf),
        ("Fixed-Graph (GNN)",   a2_orig, a2_cf),
        ("Topo-Aware (CPM)",    a3_orig, a3_cf),
    ]):
        reduction = orig_r - cf_r
        delta_cls = "metric-delta-pos" if reduction > 0 else "metric-delta-neg"
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{approach}</div>
            <div class="metric-value">{cf_r:.3f}</div>
            <div class="{delta_cls}">{'-' if reduction > 0 else '+'}{abs(reduction):.3f} vs orig ({orig_r:.3f})</div>
        </div>""", unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown('<p class="section-header">📈 Risk Before vs After by Approach</p>', unsafe_allow_html=True)
        fig_comp = _make_comparison_fig(a1_orig, a1_cf, a2_orig, a2_cf, a3_orig, a3_cf)
        st.pyplot(fig_comp, use_container_width=True)
        plt.close(fig_comp)

    with col_r:
        st.markdown('<p class="section-header">🎯 CPM Component Radar</p>', unsafe_allow_html=True)
        fig_radar = _make_cpm_radar(T, R, D_norm, F_norm)
        st.pyplot(fig_radar, use_container_width=True)
        plt.close(fig_radar)

    # Embedding migration
    st.markdown('<p class="section-header">🗺️ Graph Embedding Migration</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    t-SNE projection of patient embeddings. The star markers show where this patient sits before (white) 
    and after (green) the counterfactual. A large shift away from the red cluster indicates strong migration.
    </div>""", unsafe_allow_html=True)
    with st.spinner("Computing embedding projection..."):
        fig_emb = _make_embedding_fig(res['embeddings'], res['y_full'], closest_idx, emb_cf)
        st.pyplot(fig_emb, use_container_width=True)
        plt.close(fig_emb)

    # Feature change table
    st.markdown('<p class="section-header">📋 Feature Changes (Original → Counterfactual)</p>',
                unsafe_allow_html=True)
    changes = []
    for i, feat in enumerate(ALL_FEATURES):
        orig_v = x_orig[i]
        cf_v   = x_cf[i]
        diff   = cf_v - orig_v
        if abs(diff) > 1e-6:
            changes.append({'Feature': feat,
                            'Original': round(orig_v, 3),
                            'Counterfactual': round(cf_v, 3),
                            'Change': f"{'▼' if diff < 0 else '▲'} {abs(diff):.3f}"})
    if changes:
        ch_df = pd.DataFrame(changes)
        st.dataframe(ch_df, use_container_width=True, hide_index=True)
    else:
        st.info("No features were changed — patient already meets risk threshold.")

    # Neighborhood comparison
    st.markdown('<p class="section-header">🏘️ Neighborhood CVD Composition</p>', unsafe_allow_html=True)
    nc1, nc2 = st.columns(2)
    nc1.metric("CVD% Before (original graph)", f"{stats_o['cvd_risk_ratio']:.1%}")
    nc2.metric("CVD% After (rebuilt graph)", f"{stats_n['cvd_risk_ratio']:.1%}",
               delta=f"{stats_n['cvd_risk_ratio'] - stats_o['cvd_risk_ratio']:.1%}")

    # Limitation note
    st.markdown("""
    <div class="warn-box">
    <b>Research note:</b> This demo uses a development cohort for illustrative purposes.
    The n=10 held-out test split is reported separately in Step 7 results with bootstrap CIs.
    CPM scores should be interpreted directionally, not as absolute clinical recommendations.
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
