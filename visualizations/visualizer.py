"""
=============================================================================
Visualization Module — Publication-Quality Figures
=============================================================================
Generates all required visualization types for IEEE conference publication:

  1. Correlation heatmap
  2. Class distribution plots
  3. Feature distribution plots
  4. ROC curves (all models)
  5. Precision-Recall curves (all models)
  6. Confusion matrices (all models)
  7. Model comparison bar charts
  8. Feature importance plots (tree-based models)
  9. CV box plots (metric stability)

All figures are saved at 300 DPI into the /visualizations folder.

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score
)

from config import (
    MODEL_COLORS, MODEL_MARKERS, COLOR_NEGATIVE, COLOR_POSITIVE,
    COLOR_PALETTE, HEATMAP_CMAP, PLOT_STYLE, PLOT_DPI,
    VISUALIZATIONS_DIR, CONTINUOUS_FEATURES, BINARY_FEATURES,
    TARGET_COLUMN, CLASS_NAMES
)

# Apply global plot style
plt.rcParams.update(PLOT_STYLE)


# ═══════════════════════════════════════════════════════════════════════════
# 1. CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(df, save_dir=None):
    """
    Generate a Pearson correlation heatmap for all features + target.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target column.
    save_dir : str, optional
        Directory to save the figure. Defaults to VISUALIZATIONS_DIR.
    """
    save_dir = save_dir or VISUALIZATIONS_DIR
    fig, ax = plt.subplots(figsize=(12, 10))

    features = CONTINUOUS_FEATURES + BINARY_FEATURES + [TARGET_COLUMN]
    corr = df[features].corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    short_labels = [c.replace('Physical_Activity_HoursPerWeek', 'Phys_Activity')
                     .replace('Family_History', 'Fam_Hist')
                     .replace('Blood_Pressure', 'BP')
                     .replace('Triglycerides', 'Triglyc')
                     .replace('Stress_Index', 'Stress')
                    for c in features]

    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=HEATMAP_CMAP,
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                xticklabels=short_labels, yticklabels=short_labels,
                ax=ax, annot_kws={'size': 8},
                cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
    ax.set_title('Pearson Correlation Heatmap', fontweight='bold',
                 fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    path = os.path.join(save_dir, 'correlation_heatmap.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Correlation heatmap: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. CLASS DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════

def plot_class_distribution(y, save_dir=None):
    """
    Generate class distribution bar and pie charts.

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Target vector.
    save_dir : str, optional
        Directory to save the figure.
    """
    save_dir = save_dir or VISUALIZATIONS_DIR
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    counts = pd.Series(y).value_counts().sort_index()

    # Bar chart
    ax = axes[0]
    bars = ax.bar(['No CVD\n(Class 0)', 'CVD\n(Class 1)'], counts.values,
                  color=COLOR_PALETTE, edgecolor='white', linewidth=1.5, width=0.55)
    for bar, count in zip(bars, counts.values):
        pct = count / len(y) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom',
                fontweight='bold', fontsize=11)
    ax.set_ylabel('Number of Patients')
    ax.set_title('(a) Class Distribution', fontweight='bold')
    ax.set_ylim(0, max(counts.values) * 1.25)

    # Pie chart
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=['No CVD', 'CVD'],
        colors=COLOR_PALETTE, autopct='%1.1f%%',
        startangle=90, explode=(0, 0.05),
        textprops={'fontsize': 11}, pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    for t in autotexts:
        t.set_fontweight('bold')
    ax.set_title('(b) Class Proportion', fontweight='bold')

    fig.suptitle('Target Variable Distribution — Cardiovascular Disease',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()

    path = os.path.join(save_dir, 'class_distribution.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Class distribution: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_distributions(df, save_dir=None):
    """
    Generate histograms of all continuous features split by target class.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    save_dir : str, optional
        Directory to save the figure.
    """
    save_dir = save_dir or VISUALIZATIONS_DIR
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(CONTINUOUS_FEATURES):
        ax = axes[i]
        ax.hist(df[df[TARGET_COLUMN]==0][col], bins=30, alpha=0.6,
                color=COLOR_NEGATIVE, label='No CVD', density=True,
                edgecolor='white')
        ax.hist(df[df[TARGET_COLUMN]==1][col], bins=30, alpha=0.6,
                color=COLOR_POSITIVE, label='CVD', density=True,
                edgecolor='white')
        ax.set_title(col.replace('_', ' '), fontweight='bold', fontsize=10)
        ax.set_ylabel('Density' if i % 5 == 0 else '')
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Feature Distributions by Target Class',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(save_dir, 'feature_distributions.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Feature distributions: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. ROC CURVES
# ═══════════════════════════════════════════════════════════════════════════

def plot_roc_curves(test_results, y_test, save_dir=None):
    """
    Generate combined ROC curves for all models.

    Parameters
    ----------
    test_results : dict
        model_name → results dict (must contain 'y_prob' and 'roc_auc').
    y_test : array-like
        True test labels.
    save_dir : str, optional
        Directory to save the figure.
    """
    save_dir = save_dir or VISUALIZATIONS_DIR
    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name, results in test_results.items():
        fpr, tpr, _ = roc_curve(y_test, results['y_prob'])
        color = MODEL_COLORS.get(model_name, '#333333')
        marker = MODEL_MARKERS.get(model_name, 'o')

        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{model_name} (AUC = {results['roc_auc']:.4f})",
                marker=marker, markevery=max(1, len(fpr)//8),
                markersize=6, alpha=0.9)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5,
            label='Random (AUC = 0.5000)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 − Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True,
              fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    path = os.path.join(save_dir, 'roc_curves.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] ROC curves: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. PRECISION-RECALL CURVES
# ═══════════════════════════════════════════════════════════════════════════

def plot_precision_recall_curves(test_results, y_test, save_dir=None):
    """
    Generate Precision-Recall curves — especially important for
    imbalanced datasets like ours (~19.4% positive class).

    Parameters
    ----------
    test_results : dict
        model_name → results dict.
    y_test : array-like
        True test labels.
    save_dir : str, optional
        Directory to save the figure.
    """
    save_dir = save_dir or VISUALIZATIONS_DIR
    fig, ax = plt.subplots(figsize=(8, 7))
    baseline = sum(np.array(y_test) == 1) / len(y_test)

    for model_name, results in test_results.items():
        prec, rec, _ = precision_recall_curve(y_test, results['y_prob'])
        ap = average_precision_score(y_test, results['y_prob'])
        color = MODEL_COLORS.get(model_name, '#333333')
        marker = MODEL_MARKERS.get(model_name, 'o')

        ax.plot(rec, prec, color=color, linewidth=2.2,
                label=f"{model_name} (AP = {ap:.4f})",
                marker=marker, markevery=max(1, len(rec)//8),
                markersize=6, alpha=0.9)

    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1, alpha=0.5,
               label=f'No-Skill ({baseline:.3f})')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontweight='bold')
    ax.set_ylabel('Precision (PPV)', fontweight='bold')
    ax.set_title('Precision-Recall Curves — Model Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
              fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    path = os.path.join(save_dir, 'precision_recall_curves.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Precision-Recall curves: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. CONFUSION MATRICES
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(test_results, save_dir=None):
    """
    Generate confusion matrix heatmaps for each model.

    Parameters
    ----------
    test_results : dict
        model_name → results dict (must contain 'confusion_matrix').
    save_dir : str, optional
        Directory to save the figure.
    """
    save_dir = save_dir or VISUALIZATIONS_DIR
    n_models = len(test_results)
    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = LinearSegmentedColormap.from_list(
        'ieee_blue', ['#FFFFFF', '#E8EDF2', '#A8C6E0', '#4A90B8', '#1B5E7A'])

    for idx, (model_name, results) in enumerate(test_results.items()):
        ax = axes[idx]
        cm = results['confusion_matrix']
        cm_pct = cm.astype(float) / cm.sum() * 100

        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'

        sns.heatmap(cm, annot=annot, fmt='', cmap=cmap,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=1.5, linecolor='white', cbar=True,
                    ax=ax, square=True)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        color = MODEL_COLORS.get(model_name, '#333333')
        ax.set_title(model_name, fontsize=12, fontweight='bold', color=color)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Confusion Matrices — Test Set',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(save_dir, 'confusion_matrices.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Confusion matrices: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. MODEL COMPARISON BAR CHART
# ═══════════════════════════════════════════════════════════════════════════

def plot_model_comparison(cv_results, save_dir=None):
    """
    Generate grouped bar chart comparing all models across all metrics.

    Parameters
    ----------
    cv_results : dict
        model_name → CV results dict (mean/std for each metric).
    save_dir : str, optional
        Directory to save the figure.
    """
    save_dir = save_dir or VISUALIZATIONS_DIR
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    model_names = list(cv_results.keys())

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(metrics))
    n_models = len(model_names)
    width = 0.8 / n_models
    offsets = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)

    for i, model_name in enumerate(model_names):
        means = [cv_results[model_name][f'{m}_mean'] for m in metrics]
        stds = [cv_results[model_name][f'{m}_std'] for m in metrics]
        color = MODEL_COLORS.get(model_name, '#333333')

        bars = ax.bar(x + offsets[i], means, width, yerr=stds,
                      label=model_name, color=color, edgecolor='white',
                      capsize=3, alpha=0.88, zorder=3,
                      error_kw={'linewidth': 1.0, 'capthick': 1.0})

        for bar, mv in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mv:.3f}', ha='center', va='bottom', fontsize=5.5,
                    fontweight='bold', rotation=90)

    ax.set_xlabel('Metric', fontweight='bold', fontsize=13)
    ax.set_ylabel('Score', fontweight='bold', fontsize=13)
    ax.set_title('Model Comparison — Stratified 5-Fold CV (Mean ± Std)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim([0.0, 1.18])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=min(n_models, 4), frameon=True, fancybox=True,
              shadow=True, fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)

    path = os.path.join(save_dir, 'model_comparison_bar_chart.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Model comparison: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 8. FEATURE IMPORTANCE (TREE-BASED MODELS)
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_importance(trained_models, feature_names, save_dir=None):
    """
    Generate feature importance plots for tree-based models.

    Supported: Random Forest, Decision Tree, XGBoost, LightGBM.

    Parameters
    ----------
    trained_models : dict
        model_name → fitted model.
    feature_names : list
        Feature column names.
    save_dir : str, optional
        Directory to save figures.
    """
    save_dir = save_dir or VISUALIZATIONS_DIR
    tree_models = {k: v for k, v in trained_models.items()
                   if hasattr(v, 'feature_importances_')}

    if not tree_models:
        print("  [SKIP] No tree-based models found for feature importance.")
        return

    n = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for idx, (model_name, model) in enumerate(tree_models.items()):
        ax = axes[idx]
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        color = MODEL_COLORS.get(model_name, '#333333')

        short_names = [f.replace('Physical_Activity_HoursPerWeek', 'Phys_Activity')
                        .replace('Family_History', 'Fam_Hist')
                        .replace('Blood_Pressure', 'BP')
                       for f in np.array(feature_names)[sorted_idx]]

        ax.barh(range(len(sorted_idx)), importances[sorted_idx],
                color=color, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels(short_names, fontsize=9)
        ax.set_xlabel('Feature Importance', fontweight='bold')
        ax.set_title(model_name, fontweight='bold', fontsize=12, color=color)

        # Annotate values
        for j, (val, pos) in enumerate(zip(importances[sorted_idx],
                                           range(len(sorted_idx)))):
            ax.text(val + 0.002, pos, f'{val:.3f}', va='center',
                    fontsize=7.5, fontweight='bold')

    fig.suptitle('Feature Importance — Tree-Based Models',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(save_dir, 'feature_importance.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Feature importance: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. CV BOX PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_cv_boxplots(cv_results, save_dir=None):
    """
    Generate box plots showing metric distributions across CV folds.

    Parameters
    ----------
    cv_results : dict
        model_name → CV results with per-fold values.
    save_dir : str, optional
        Directory to save the figure.
    """
    save_dir = save_dir or VISUALIZATIONS_DIR
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    model_names = list(cv_results.keys())

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        data = [cv_results[mn][f'{metric}_folds'] for mn in model_names]
        colors_list = [MODEL_COLORS.get(mn, '#333333') for mn in model_names]

        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2))
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')

        ax.set_title(label, fontweight='bold', fontsize=12)
        ax.set_xticklabels(
            [n.replace(' ', '\n') for n in model_names], fontsize=6.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle('Cross-Validation Metric Distributions (5 Folds)',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()

    path = os.path.join(save_dir, 'cv_boxplots.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] CV box plots: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# GENERATE ALL VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_visualizations(df_clean, y, cv_results, test_results,
                                 trained_models, feature_names, y_test,
                                 save_dir=None):
    """
    Generate all visualization types in a single call.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned dataset (unscaled) for EDA plots.
    y : pd.Series
        Full target vector for class distribution.
    cv_results : dict
        Cross-validation results.
    test_results : dict
        Test set results.
    trained_models : dict
        model_name → fitted model.
    feature_names : list
        Feature column names.
    y_test : array-like
        Test set true labels.
    save_dir : str, optional
        Output directory.
    """
    save_dir = save_dir or VISUALIZATIONS_DIR

    print("\n  Generating all visualizations...")

    plot_correlation_heatmap(df_clean, save_dir)
    plot_class_distribution(y, save_dir)
    plot_feature_distributions(df_clean, save_dir)
    plot_roc_curves(test_results, y_test, save_dir)
    plot_precision_recall_curves(test_results, y_test, save_dir)
    plot_confusion_matrices(test_results, save_dir)
    plot_model_comparison(cv_results, save_dir)
    plot_feature_importance(trained_models, feature_names, save_dir)
    plot_cv_boxplots(cv_results, save_dir)

    print(f"\n  [OK] All 9 visualizations saved to: {save_dir}")
