"""
=============================================================================
Phase 1 — Task 3: Exploratory Data Analysis (EDA)
=============================================================================
Cardiovascular Disease Risk Prediction using Graph Neural Networks
IEEE Conference Paper — Phase 1 Foundation

Generates 13+ publication-quality figures (300 DPI) suitable for
IEEE conference submissions.

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = r"c:\Users\Asus\OneDrive\Desktop\GNN project"
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR  = os.path.join(BASE_DIR, "figures")

# ── Style Configuration ────────────────────────────────────────────────────
# IEEE-friendly style: clean, professional, high contrast
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette: medical/clinical theme
COLOR_NEGATIVE  = '#2196F3'   # Blue  — No CVD (class 0)
COLOR_POSITIVE  = '#E53935'   # Red   — CVD (class 1)
COLOR_PALETTE   = [COLOR_NEGATIVE, COLOR_POSITIVE]
CONTINUOUS_CMAP = 'RdYlBu_r'
HEATMAP_CMAP    = 'coolwarm'

# ── Load Data ──────────────────────────────────────────────────────────────
df = pd.read_excel(os.path.join(BASE_DIR, "GNN DATASET.xlsx"))
df_clean = df.drop(columns=['Patient_ID'])

binary_cols     = ['Gender', 'Diabetes', 'Smoking', 'Family_History']
continuous_cols = ['Age', 'Blood_Pressure', 'Cholesterol', 'Heart_Rate', 'BMI',
                   'Physical_Activity_HoursPerWeek', 'ECG_Score', 'Glucose',
                   'Triglycerides', 'Stress_Index']
target_col      = 'Target'

# Create label mapping for readable plots
target_labels = {0: 'No CVD (0)', 1: 'CVD (1)'}
df_clean['Target_Label'] = df_clean[target_col].map(target_labels)

print("=" * 70)
print("  PHASE 1 — TASK 3: EXPLORATORY DATA ANALYSIS")
print("  Generating publication-quality figures (300 DPI)")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: TARGET DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
def plot_target_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Bar chart
    ax = axes[0]
    counts = df_clean[target_col].value_counts().sort_index()
    bars = ax.bar(['No CVD\n(Class 0)', 'CVD\n(Class 1)'], counts.values,
                  color=COLOR_PALETTE, edgecolor='white', linewidth=1.5, width=0.55)
    for bar, count in zip(bars, counts.values):
        pct = count / len(df_clean) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom',
                fontweight='bold', fontsize=11)
    ax.set_ylabel('Number of Patients')
    ax.set_title('(a) Class Distribution', fontweight='bold')
    ax.set_ylim(0, max(counts.values) * 1.2)

    # Pie chart
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=['No CVD', 'CVD'],
        colors=COLOR_PALETTE, autopct='%1.1f%%',
        startangle=90, explode=(0, 0.05),
        textprops={'fontsize': 11}, pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for t in autotexts:
        t.set_fontweight('bold')
    ax.set_title('(b) Class Proportion', fontweight='bold')

    fig.suptitle('Figure 1: Target Variable Distribution — Cardiovascular Disease',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "01_target_distribution.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: FEATURE TYPE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
def plot_feature_type_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Feature type breakdown
    ax = axes[0]
    types = ['Continuous', 'Binary']
    counts = [len(continuous_cols), len(binary_cols)]
    colors = ['#1976D2', '#FF7043']
    bars = ax.barh(types, counts, color=colors, edgecolor='white', height=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontweight='bold', fontsize=12)
    ax.set_xlabel('Number of Features')
    ax.set_title('(a) Feature Type Breakdown', fontweight='bold')
    ax.set_xlim(0, max(counts) + 2)

    # Feature names per type
    ax = axes[1]
    ax.axis('off')
    table_data = []
    for c in continuous_cols:
        table_data.append(['Continuous', c])
    for c in binary_cols:
        table_data.append(['Binary', c])

    table = ax.table(cellText=table_data, colLabels=['Type', 'Feature Name'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    # Color header
    for j in range(2):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(continuous_cols) + 1):
        table[i, 0].set_facecolor('#E3F2FD')
    for i in range(len(continuous_cols) + 1, len(table_data) + 1):
        table[i, 0].set_facecolor('#FBE9E7')

    ax.set_title('(b) Feature Inventory', fontweight='bold')

    fig.suptitle('Figure 2: Feature Type Distribution',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "02_feature_type_distribution.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: HISTOGRAMS OF NUMERICAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════
def plot_histograms():
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(continuous_cols):
        ax = axes[i]
        ax.hist(df_clean[df_clean[target_col]==0][col], bins=30, alpha=0.6,
                color=COLOR_NEGATIVE, label='No CVD', density=True, edgecolor='white')
        ax.hist(df_clean[df_clean[target_col]==1][col], bins=30, alpha=0.6,
                color=COLOR_POSITIVE, label='CVD', density=True, edgecolor='white')
        ax.set_title(col.replace('_', ' '), fontweight='bold', fontsize=10)
        ax.set_ylabel('Density' if i % 5 == 0 else '')
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Figure 3: Distribution of Numerical Features by Target Class',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "03_numerical_histograms.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: KDE PLOTS
# ═══════════════════════════════════════════════════════════════════════════
def plot_kde():
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(continuous_cols):
        ax = axes[i]
        for label, color in zip([0, 1], COLOR_PALETTE):
            subset = df_clean[df_clean[target_col] == label][col]
            subset.plot.kde(ax=ax, color=color, linewidth=2,
                           label=target_labels[label])
        ax.set_title(col.replace('_', ' '), fontweight='bold', fontsize=10)
        ax.set_ylabel('Density' if i % 5 == 0 else '')
        ax.set_xlabel('')
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Figure 4: Kernel Density Estimation — Class Comparison',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "04_kde_plots.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: BOXPLOTS BY TARGET
# ═══════════════════════════════════════════════════════════════════════════
def plot_boxplots():
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(continuous_cols):
        ax = axes[i]
        data_0 = df_clean[df_clean[target_col]==0][col]
        data_1 = df_clean[df_clean[target_col]==1][col]
        bp = ax.boxplot([data_0, data_1], labels=['No CVD', 'CVD'],
                        patch_artist=True, widths=0.5,
                        medianprops={'color': 'black', 'linewidth': 2},
                        flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5})
        bp['boxes'][0].set_facecolor(COLOR_NEGATIVE)
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(COLOR_POSITIVE)
        bp['boxes'][1].set_alpha(0.7)
        ax.set_title(col.replace('_', ' '), fontweight='bold', fontsize=10)

    fig.suptitle('Figure 5: Boxplots of Numerical Features by Target Class',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "05_boxplots.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: VIOLIN PLOTS BY TARGET
# ═══════════════════════════════════════════════════════════════════════════
def plot_violins():
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(continuous_cols):
        ax = axes[i]
        sns.violinplot(x=target_col, y=col, data=df_clean, ax=ax,
                       palette=COLOR_PALETTE, inner='box', linewidth=1.2,
                       cut=0, density_norm='width')
        ax.set_xlabel('')
        ax.set_xticklabels(['No CVD', 'CVD'])
        ax.set_title(col.replace('_', ' '), fontweight='bold', fontsize=10)
        ax.set_ylabel('')

    fig.suptitle('Figure 6: Violin Plots — Feature Distribution by Target',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "06_violin_plots.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: PEARSON CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
def plot_pearson_heatmap():
    fig, ax = plt.subplots(figsize=(12, 10))

    features = continuous_cols + binary_cols + [target_col]
    corr = df_clean[features].corr(method='pearson')

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    short_labels = [c.replace('Physical_Activity_HoursPerWeek', 'Phys_Activity')
                      .replace('Family_History', 'Family_Hist')
                      .replace('Blood_Pressure', 'Blood_Press')
                      .replace('Triglycerides', 'Triglyc')
                      .replace('Stress_Index', 'Stress_Idx')
                    for c in features]

    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=HEATMAP_CMAP,
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                xticklabels=short_labels, yticklabels=short_labels,
                ax=ax, annot_kws={'size': 8},
                cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
    ax.set_title('Figure 7: Pearson Correlation Heatmap', fontweight='bold', fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    path = os.path.join(FIG_DIR, "07_pearson_correlation.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8: SPEARMAN CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
def plot_spearman_heatmap():
    fig, ax = plt.subplots(figsize=(12, 10))

    features = continuous_cols + binary_cols + [target_col]
    corr = df_clean[features].corr(method='spearman')

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    short_labels = [c.replace('Physical_Activity_HoursPerWeek', 'Phys_Activity')
                      .replace('Family_History', 'Family_Hist')
                      .replace('Blood_Pressure', 'Blood_Press')
                      .replace('Triglycerides', 'Triglyc')
                      .replace('Stress_Index', 'Stress_Idx')
                    for c in features]

    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=HEATMAP_CMAP,
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                xticklabels=short_labels, yticklabels=short_labels,
                ax=ax, annot_kws={'size': 8},
                cbar_kws={'shrink': 0.8, 'label': 'Spearman ρ'})
    ax.set_title('Figure 8: Spearman Rank Correlation Heatmap', fontweight='bold', fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    path = os.path.join(FIG_DIR, "08_spearman_correlation.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 9: FEATURE VS TARGET — GROUPED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def plot_feature_vs_target():
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    axes = axes.flatten()

    for i, col in enumerate(continuous_cols):
        ax = axes[i]
        means = df_clean.groupby(target_col)[col].mean()
        stds  = df_clean.groupby(target_col)[col].std()
        bars = ax.bar(['No CVD', 'CVD'], means.values, yerr=stds.values,
                      color=COLOR_PALETTE, edgecolor='white', linewidth=1.2,
                      capsize=5, width=0.5, alpha=0.85)
        # Add mean values on bars
        for bar, mean_val in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{mean_val:.1f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)
        ax.set_title(col.replace('_', ' '), fontweight='bold', fontsize=10)
        ax.set_ylabel('Mean ± SD' if i % 5 == 0 else '')

    fig.suptitle('Figure 9: Mean Feature Values by Target Class (± Std Dev)',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "09_feature_vs_target.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 10: BINARY FEATURES — PREVALENCE BY TARGET
# ═══════════════════════════════════════════════════════════════════════════
def plot_binary_vs_target():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for i, col in enumerate(binary_cols):
        ax = axes[i]
        ct = pd.crosstab(df_clean[col], df_clean[target_col], normalize='columns') * 100
        ct.plot(kind='bar', ax=ax, color=COLOR_PALETTE, edgecolor='white',
                linewidth=1.2, rot=0, width=0.6)
        ax.set_title(col.replace('_', ' '), fontweight='bold', fontsize=11)
        ax.set_ylabel('Percentage (%)' if i == 0 else '')
        ax.set_xlabel('')
        ax.set_xticklabels(['0 (No)', '1 (Yes)'])
        ax.legend(['No CVD', 'CVD'], fontsize=8, loc='upper right')
        ax.set_ylim(0, 100)

    fig.suptitle('Figure 10: Binary Feature Prevalence by Target Class',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "10_binary_vs_target.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 11: RISK FACTOR ANALYSIS — CORRELATION RANKING
# ═══════════════════════════════════════════════════════════════════════════
def plot_risk_factor_ranking():
    fig, ax = plt.subplots(figsize=(10, 6))

    all_features = continuous_cols + binary_cols
    corr = df_clean[all_features + [target_col]].corr()[target_col].drop(target_col)
    corr_sorted = corr.abs().sort_values(ascending=True)
    corr_original = corr[corr_sorted.index]

    colors = [COLOR_POSITIVE if v > 0 else COLOR_NEGATIVE for v in corr_original.values]
    
    short_names = [c.replace('Physical_Activity_HoursPerWeek', 'Physical Activity')
                    .replace('Family_History', 'Family History')
                    .replace('Blood_Pressure', 'Blood Pressure')
                    .replace('Stress_Index', 'Stress Index')
                    .replace('ECG_Score', 'ECG Score')
                    .replace('Heart_Rate', 'Heart Rate')
                   for c in corr_sorted.index]
    
    bars = ax.barh(range(len(corr_sorted)), corr_original.values,
                   color=colors, edgecolor='white', linewidth=0.8, height=0.6)
    ax.set_yticks(range(len(corr_sorted)))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel('Correlation with CVD Target', fontsize=12)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    ax.axvline(x=0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(x=-0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(x=0.4, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(x=-0.4, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    # Annotate values
    for bar_idx, (bar, val) in enumerate(zip(bars, corr_original.values)):
        offset = 0.01 if val >= 0 else -0.01
        ha = 'left' if val >= 0 else 'right'
        ax.text(val + offset, bar_idx, f'{val:.3f}', va='center', ha=ha,
                fontsize=9, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLOR_POSITIVE, label='Positive (increases risk)'),
                       Patch(facecolor=COLOR_NEGATIVE, label='Negative (decreases risk)')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax.set_title('Figure 11: Risk Factor Ranking — Correlation with CVD',
                 fontweight='bold', fontsize=13, pad=15)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "11_risk_factor_analysis.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 12: CLASS IMBALANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def plot_class_imbalance():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    counts = df_clean[target_col].value_counts().sort_index()

    # 1. Absolute counts
    ax = axes[0]
    bars = ax.bar(['No CVD (0)', 'CVD (1)'], counts.values,
                  color=COLOR_PALETTE, edgecolor='white', width=0.5)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', fontweight='bold', fontsize=12)
    ax.set_ylabel('Count')
    ax.set_title('(a) Absolute Counts', fontweight='bold')

    # 2. Imbalance ratio visualization
    ax = axes[1]
    ratio = counts[0] / counts[1]
    ax.barh(['Imbalance\nRatio'], [ratio], color='#FF7043', edgecolor='white', height=0.4)
    ax.barh(['Ideal\nRatio'], [1.0], color='#66BB6A', edgecolor='white', height=0.4)
    ax.text(ratio + 0.1, 0, f'{ratio:.2f}:1', va='center', fontweight='bold', fontsize=12)
    ax.text(1.1, 1, '1.00:1', va='center', fontweight='bold', fontsize=12)
    ax.set_xlabel('Ratio (Majority:Minority)')
    ax.set_title('(b) Imbalance Ratio', fontweight='bold')

    # 3. Cumulative distribution
    ax = axes[2]
    pcts = counts / counts.sum() * 100
    ax.barh(['No CVD', 'CVD'], pcts.values, color=COLOR_PALETTE,
            edgecolor='white', height=0.4)
    ax.set_xlabel('Percentage of Total Dataset (%)')
    ax.set_title('(c) Class Proportions', fontweight='bold')
    for idx, (pct, cnt) in enumerate(zip(pcts.values, counts.values)):
        ax.text(pct + 0.5, idx, f'{pct:.1f}% (n={cnt})', va='center',
                fontweight='bold', fontsize=10)
    ax.set_xlim(0, 100)

    fig.suptitle('Figure 12: Class Imbalance Analysis',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "12_class_imbalance_analysis.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 13: OUTLIER ANALYSIS — BOXPLOTS WITH IQR BOUNDS
# ═══════════════════════════════════════════════════════════════════════════
def plot_outlier_analysis():
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(continuous_cols):
        ax = axes[i]
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()

        bp = ax.boxplot(df_clean[col], patch_artist=True, widths=0.5,
                        boxprops={'facecolor': '#BBDEFB', 'edgecolor': '#1565C0'},
                        medianprops={'color': '#D32F2F', 'linewidth': 2},
                        flierprops={'marker': 'o', 'markerfacecolor': COLOR_POSITIVE,
                                    'markersize': 4, 'alpha': 0.6})
        ax.axhline(y=lower, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=upper, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title(f'{col.replace("_", " ")}\n(n_outliers={n_outliers})',
                     fontweight='bold', fontsize=9)
        ax.set_xticklabels([''])

    fig.suptitle('Figure 13: Outlier Detection — IQR Method (1.5×IQR bounds in orange)',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "13_outlier_analysis.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTE ALL PLOTS
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nGenerating figures...\n")

    plot_target_distribution()
    plot_feature_type_distribution()
    plot_histograms()
    plot_kde()
    plot_boxplots()
    plot_violins()
    plot_pearson_heatmap()
    plot_spearman_heatmap()
    plot_feature_vs_target()
    plot_binary_vs_target()
    plot_risk_factor_ranking()
    plot_class_imbalance()
    plot_outlier_analysis()

    print(f"\n{'=' * 70}")
    print(f"  ✓ ALL 13 EDA FIGURES GENERATED SUCCESSFULLY")
    print(f"  ✓ Saved to: {FIG_DIR}")
    print(f"{'=' * 70}")
