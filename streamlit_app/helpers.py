"""
=============================================================================
Streamlit Dashboard — Shared Helpers & Utilities
=============================================================================
Cached data loaders, custom CSS injection, reusable Plotly chart builders,
and KPI card renderers for the CVD Risk Prediction dashboard.

Author: AI Research Assistant
Date: August 2026
=============================================================================
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
DATA_DIR = PROJECT_ROOT / "data"

# ═══════════════════════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg_primary": "#0E1117",
    "bg_card": "#1B1F2A",
    "bg_card_hover": "#242938",
    "accent_blue": "#1E88E5",
    "accent_magenta": "#D81B60",
    "accent_emerald": "#43A047",
    "accent_purple": "#8E24AA",
    "accent_orange": "#F18F01",
    "accent_cyan": "#00BCD4",
    "accent_amber": "#FFB300",
    "text_primary": "#E8EAED",
    "text_secondary": "#9AA0A6",
    "text_muted": "#5F6368",
    "border": "#2D3348",
    "positive": "#43A047",
    "negative": "#E53935",
    "gradient_start": "#1E88E5",
    "gradient_end": "#D81B60",
}

MODEL_COLORS = {
    "Logistic Regression": "#2E86AB",
    "Random Forest": "#A23B72",
    "SVM": "#3B1F2B",
    "XGBoost": "#F18F01",
    "LightGBM": "#C73E1D",
    "GCN": "#1E88E5",
    "GraphSAGE": "#43A047",
    "GAT": "#8E24AA",
    "DPSN": "#D81B60",
}

GRAPH_COLORS = {
    "Euclidean KNN": "#1565C0",
    "Cosine Similarity": "#2E7D32",
    "Clinical Similarity": "#E65100",
    "Dynamic PSG": "#AD1457",
}


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS INJECTION
# ═══════════════════════════════════════════════════════════════════════════

def inject_custom_css():
    """Inject premium dark-theme CSS with glassmorphism and gradient accents."""
    st.markdown("""
    <style>
    /* ── Import Google Fonts ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Overrides ────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #151B28 50%, #0E1117 100%);
    }

    /* ── Sidebar Styling ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0E1117 100%) !important;
        border-right: 1px solid rgba(30, 136, 229, 0.15) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h1 {
        background: linear-gradient(135deg, #1E88E5, #D81B60);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    /* ── KPI Card ────────────────────────────────────────────────── */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.08) 0%, rgba(216, 27, 96, 0.08) 100%);
        border: 1px solid rgba(30, 136, 229, 0.2);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #1E88E5, #D81B60);
        border-radius: 16px 16px 0 0;
    }

    .kpi-card:hover {
        border-color: rgba(30, 136, 229, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(30, 136, 229, 0.15);
    }

    .kpi-icon {
        font-size: 28px;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #1E88E5, #00BCD4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-bottom: 6px;
    }

    .kpi-label {
        font-size: 13px;
        font-weight: 500;
        color: #9AA0A6;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .kpi-sublabel {
        font-size: 11px;
        color: #5F6368;
        margin-top: 4px;
    }

    /* ── Section Headers ─────────────────────────────────────────── */
    .section-header {
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.06) 0%, rgba(142, 36, 170, 0.06) 100%);
        border: 1px solid rgba(30, 136, 229, 0.12);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 24px 0 16px 0;
        backdrop-filter: blur(8px);
    }

    .section-header h3 {
        margin: 0 !important;
        padding: 0 !important;
        font-weight: 700 !important;
        font-size: 20px !important;
        color: #E8EAED !important;
    }

    .section-header p {
        margin: 4px 0 0 0 !important;
        font-size: 13px;
        color: #9AA0A6;
    }

    /* ── Insight Card ────────────────────────────────────────────── */
    .insight-card {
        background: linear-gradient(135deg, rgba(67, 160, 71, 0.1) 0%, rgba(30, 136, 229, 0.05) 100%);
        border: 1px solid rgba(67, 160, 71, 0.25);
        border-left: 4px solid #43A047;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
    }

    .insight-card .insight-title {
        font-weight: 700;
        color: #43A047;
        font-size: 14px;
        margin-bottom: 6px;
    }

    .insight-card .insight-text {
        color: #E8EAED;
        font-size: 14px;
        line-height: 1.5;
    }

    /* ── Warning Card ────────────────────────────────────────────── */
    .warning-card {
        background: linear-gradient(135deg, rgba(255, 179, 0, 0.1) 0%, rgba(241, 143, 1, 0.05) 100%);
        border: 1px solid rgba(255, 179, 0, 0.25);
        border-left: 4px solid #FFB300;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
    }

    .warning-card .warning-title {
        font-weight: 700;
        color: #FFB300;
        font-size: 14px;
        margin-bottom: 6px;
    }

    .warning-card .warning-text {
        color: #E8EAED;
        font-size: 14px;
        line-height: 1.5;
    }

    /* ── Stat Badge ──────────────────────────────────────────────── */
    .stat-badge {
        display: inline-block;
        background: rgba(30, 136, 229, 0.12);
        border: 1px solid rgba(30, 136, 229, 0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 600;
        color: #1E88E5;
        margin: 2px 4px;
    }

    .stat-badge.green {
        background: rgba(67, 160, 71, 0.12);
        border-color: rgba(67, 160, 71, 0.3);
        color: #43A047;
    }

    .stat-badge.magenta {
        background: rgba(216, 27, 96, 0.12);
        border-color: rgba(216, 27, 96, 0.3);
        color: #D81B60;
    }

    /* ── DataFrames ──────────────────────────────────────────────── */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* ── Tabs ────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(30, 136, 229, 0.04);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-weight: 500 !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.2), rgba(142, 36, 170, 0.15)) !important;
        border: 1px solid rgba(30, 136, 229, 0.3) !important;
    }

    /* ── Expander ────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 15px !important;
        color: #E8EAED !important;
        background: rgba(30, 136, 229, 0.05) !important;
        border-radius: 10px !important;
    }

    /* ── Metrics ─────────────────────────────────────────────────── */
    [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        font-size: 28px !important;
    }

    /* ── Page Title ──────────────────────────────────────────────── */
    .page-title {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #1E88E5 0%, #8E24AA 50%, #D81B60 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }

    .page-subtitle {
        font-size: 15px;
        color: #9AA0A6;
        margin-bottom: 24px;
        font-weight: 400;
    }

    /* ── Divider ─────────────────────────────────────────────────── */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #1E88E5, #D81B60, transparent);
        margin: 24px 0;
        border: none;
        border-radius: 2px;
    }

    /* ── Hero Banner ──────────────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.12) 0%, rgba(216, 27, 96, 0.12) 100%);
        border: 1px solid rgba(30, 136, 229, 0.2);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(30, 136, 229, 0.08) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-banner h2 {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(135deg, #E8EAED, #9AA0A6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero-banner p {
        color: #9AA0A6;
        font-size: 15px;
        line-height: 1.6;
    }

    /* ── Rank badges ──────────────────────────────────────────────── */
    .rank-gold { color: #FFD700; font-weight: 800; }
    .rank-silver { color: #C0C0C0; font-weight: 800; }
    .rank-bronze { color: #CD7F32; font-weight: 800; }

    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT RENDERERS
# ═══════════════════════════════════════════════════════════════════════════

def render_page_header(title: str, subtitle: str = ""):
    """Render a gradient page title with optional subtitle."""
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


def render_kpi_card(icon: str, value: str, label: str, sublabel: str = ""):
    """Render a single KPI card with glassmorphism styling."""
    sublabel_html = f'<div class="kpi-sublabel">{sublabel}</div>' if sublabel else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {sublabel_html}
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, description: str = ""):
    """Render a styled section header."""
    desc_html = f"<p>{description}</p>" if description else ""
    st.markdown(f"""
    <div class="section-header">
        <h3>{title}</h3>
        {desc_html}
    </div>
    """, unsafe_allow_html=True)


def render_insight(title: str, text: str):
    """Render a green insight/finding card."""
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">💡 {title}</div>
        <div class="insight-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_warning(title: str, text: str):
    """Render an amber warning card."""
    st.markdown(f"""
    <div class="warning-card">
        <div class="warning-title">⚠️ {title}</div>
        <div class="warning-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_gradient_divider():
    """Render a gradient divider line."""
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADERS (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the results directory."""
    filepath = RESULTS_DIR / filename
    if filepath.exists():
        return pd.read_csv(filepath)
    return pd.DataFrame()


@st.cache_data
def load_dataset(filename: str = "cleaned_dataset.csv") -> pd.DataFrame:
    """Load the cleaned dataset."""
    filepath = DATA_DIR / filename
    if filepath.exists():
        return pd.read_csv(filepath)
    return pd.DataFrame()


@st.cache_data
def load_processed_dataset() -> pd.DataFrame:
    """Load the processed (scaled) dataset."""
    filepath = DATA_DIR / "processed_dataset.csv"
    if filepath.exists():
        return pd.read_csv(filepath)
    return pd.DataFrame()


def get_figure_path(filename: str) -> str:
    """Get the full path to a figure file."""
    return str(FIGURES_DIR / filename)


def figure_exists(filename: str) -> bool:
    """Check if a figure file exists."""
    return (FIGURES_DIR / filename).exists()


# ═══════════════════════════════════════════════════════════════════════════
# PLOTLY THEME & CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#E8EAED", size=13),
    title_font=dict(size=18, color="#E8EAED"),
    legend=dict(
        bgcolor="rgba(27,31,42,0.8)",
        bordercolor="rgba(30,136,229,0.2)",
        borderwidth=1,
        font=dict(size=12),
    ),
    xaxis=dict(
        gridcolor="rgba(45,51,72,0.5)",
        zerolinecolor="rgba(45,51,72,0.5)",
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor="rgba(45,51,72,0.5)",
        zerolinecolor="rgba(45,51,72,0.5)",
        tickfont=dict(size=11),
    ),
    margin=dict(l=60, r=30, t=60, b=60),
    hoverlabel=dict(
        bgcolor="#1B1F2A",
        bordercolor="#1E88E5",
        font=dict(family="Inter", size=13, color="#E8EAED"),
    ),
)


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Apply the dashboard's dark theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def make_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color_map: dict = None,
    horizontal: bool = False,
    show_values: bool = True,
    y_range: list = None,
) -> go.Figure:
    """Create a styled bar chart."""
    colors = [
        (color_map or MODEL_COLORS).get(name, COLORS["accent_blue"])
        for name in df[x]
    ]

    if horizontal:
        fig = go.Figure(go.Bar(
            y=df[x], x=df[y],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0), opacity=0.88),
            text=df[y].round(4) if show_values else None,
            textposition="outside",
            textfont=dict(size=12, color="#E8EAED"),
        ))
    else:
        fig = go.Figure(go.Bar(
            x=df[x], y=df[y],
            marker=dict(color=colors, line=dict(width=0), opacity=0.88),
            text=df[y].round(4) if show_values else None,
            textposition="outside",
            textfont=dict(size=12, color="#E8EAED"),
        ))

    fig.update_layout(title=title, showlegend=False)
    if y_range:
        if horizontal:
            fig.update_layout(xaxis=dict(range=y_range))
        else:
            fig.update_layout(yaxis=dict(range=y_range))

    return apply_plotly_theme(fig)


def make_grouped_bar_chart(
    df: pd.DataFrame,
    x: str,
    y_columns: list,
    title: str,
    color_list: list = None,
) -> go.Figure:
    """Create a grouped bar chart for multiple metrics."""
    if color_list is None:
        color_list = [
            COLORS["accent_blue"], COLORS["accent_magenta"],
            COLORS["accent_emerald"], COLORS["accent_purple"],
            COLORS["accent_orange"], COLORS["accent_cyan"],
        ]

    fig = go.Figure()
    for i, col in enumerate(y_columns):
        fig.add_trace(go.Bar(
            name=col.replace("_", " "),
            x=df[x],
            y=df[col],
            marker_color=color_list[i % len(color_list)],
            marker_opacity=0.88,
            text=df[col].round(3),
            textposition="outside",
            textfont=dict(size=10),
        ))

    fig.update_layout(
        title=title,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return apply_plotly_theme(fig)


def make_radar_chart(
    categories: list,
    values_dict: dict,
    title: str,
) -> go.Figure:
    """Create a radar/spider chart for multi-dimensional comparison."""
    fig = go.Figure()

    color_list = [
        COLORS["accent_blue"], COLORS["accent_magenta"],
        COLORS["accent_emerald"], COLORS["accent_purple"],
        COLORS["accent_orange"],
    ]

    for i, (name, values) in enumerate(values_dict.items()):
        color = color_list[i % len(color_list)]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba") if "rgb" in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
            line=dict(color=color, width=2),
            name=name,
        ))

    fig.update_layout(
        title=title,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(45,51,72,0.5)",
                tickfont=dict(size=10, color="#9AA0A6"),
            ),
            angularaxis=dict(
                gridcolor="rgba(45,51,72,0.3)",
                tickfont=dict(size=11, color="#E8EAED"),
            ),
        ),
    )
    return apply_plotly_theme(fig)


def make_heatmap(
    z: list,
    x_labels: list,
    y_labels: list,
    title: str,
    colorscale: str = "Viridis",
    text_format: str = ".3f",
) -> go.Figure:
    """Create a styled heatmap."""
    text = [[f"{val:{text_format}}" for val in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12, color="#E8EAED"),
        colorscale=colorscale,
        hoverongaps=False,
        colorbar=dict(
            tickfont=dict(color="#9AA0A6"),
            title=dict(font=dict(color="#9AA0A6")),
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-30),
    )
    return apply_plotly_theme(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR SETUP
# ═══════════════════════════════════════════════════════════════════════════

def setup_sidebar():
    """Configure the sidebar with project info."""
    with st.sidebar:
        st.markdown("# 🧬 CVD-GNN")
        st.markdown("---")
        st.markdown("""
        <div style="padding: 8px 0; font-size: 13px; color: #9AA0A6; line-height: 1.8;">
            <strong style="color: #E8EAED;">Project:</strong> CVD Risk Prediction<br>
            <strong style="color: #E8EAED;">Method:</strong> Graph Neural Networks<br>
            <strong style="color: #E8EAED;">Target:</strong> IEEE Conference Paper<br>
            <strong style="color: #E8EAED;">Dataset:</strong> 1,000 patients<br>
            <strong style="color: #E8EAED;">Models:</strong> 9 (5 Classical + 4 GNN)<br>
            <strong style="color: #E8EAED;">Phases:</strong> 10 completed
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            '<div style="font-size: 11px; color: #5F6368; text-align: center;">'
            "Built with Streamlit • Aug 2026"
            "</div>",
            unsafe_allow_html=True,
        )
