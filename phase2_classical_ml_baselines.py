"""
=============================================================================
Phase 2 -- Classical Machine Learning Baselines
=============================================================================
Cardiovascular Disease Risk Prediction using Graph Neural Networks
IEEE Conference Paper -- Phase 2 Benchmark Baselines

This script implements classical ML baselines to serve as benchmark
comparisons for the proposed Graph Neural Network model. Each model
is trained using Stratified 5-Fold Cross-Validation with SMOTE
oversampling applied within training folds to handle class imbalance.

Models:
  1. Logistic Regression       (Linear baseline)
  2. Random Forest              (Ensemble, bagging)
  3. XGBoost                    (Ensemble, gradient boosting)
  4. LightGBM                   (Ensemble, gradient boosting)
  5. Support Vector Machine     (Kernel-based)

Evaluation Metrics:
  - Accuracy, Precision, Recall, F1 Score, ROC-AUC

Outputs:
  - Publication-quality figures (confusion matrices, ROC, PR curves)
  - CSV result tables (baseline_metrics.csv, model_comparison.csv)
  - Serialized model files (.pkl)
  - Final ranking table

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

import os
import sys
import io
import warnings
import time
import pickle
import numpy as np
import pandas as pd

# Scikit-learn core
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Imbalanced learning -- SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script usage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# Set global random seed for reproducibility (IEEE requirement)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Project paths
BASE_DIR    = r"c:\Users\Asus\OneDrive\Desktop\GNN project"
DATA_DIR    = os.path.join(BASE_DIR, "data")
FIG_DIR     = os.path.join(BASE_DIR, "figures")
RES_DIR     = os.path.join(BASE_DIR, "results")
MODEL_DIR   = os.path.join(BASE_DIR, "models")

# Create directories
for d in [DATA_DIR, FIG_DIR, RES_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# Cross-validation configuration
N_FOLDS         = 5           # Number of stratified folds
TEST_SIZE       = 0.20        # 80-20 train-test split
SMOTE_STRATEGY  = 'minority'  # SMOTE oversampling strategy

# Visualization settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# IEEE-friendly color palette
COLORS = {
    'Logistic Regression':  '#2E86AB',   # Steel blue
    'Random Forest':        '#A23B72',   # Rich magenta
    'XGBoost':              '#F18F01',   # Amber orange
    'LightGBM':             '#C73E1D',   # Crimson
    'SVM':                  '#3B1F2B',   # Dark plum
}

MODEL_MARKERS = {
    'Logistic Regression': 'o',
    'Random Forest': 's',
    'XGBoost': '^',
    'LightGBM': 'D',
    'SVM': 'v',
}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING AND PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def load_and_prepare_data(data_path):
    """
    Load the processed dataset and separate features from target.

    The processed dataset (from Phase 1) contains:
      - 10 continuous features (standardized via StandardScaler)
      - 4 binary features (0/1 encoded)
      - 1 binary target: 0 = No CVD, 1 = CVD

    Parameters
    ----------
    data_path : str
        Path to the processed_dataset.csv file.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (n_samples × n_features).
    y : pd.Series
        Binary target vector.
    feature_names : list
        List of feature column names.
    """
    df = pd.read_csv(data_path)

    # Separate features and target
    target_col = 'Target'
    feature_names = [col for col in df.columns if col != target_col]

    X = df[feature_names]
    y = df[target_col]

    return X, y, feature_names


def create_train_test_split(X, y):
    """
    Perform stratified train-test split (80-20).

    Stratification ensures that both the training and test sets
    maintain the same class distribution as the original dataset,
    which is critical for imbalanced classification problems.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    X_train, X_test, y_train, y_test : array-like
        Stratified train-test split components.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_models():
    """
    Define the classical ML models for baseline comparison.

    Each model is configured with:
      - class_weight='balanced' where supported (LR, RF, SVM)
      - scale_pos_weight for XGBoost (equivalent to balanced weighting)
      - is_positive_class_weight for LightGBM
      - Fixed random_state=42 for full reproducibility

    The selection covers the canonical ML paradigms:
      - Linear:           Logistic Regression
      - Bagging Ensemble: Random Forest
      - Boosting Ensemble: XGBoost, LightGBM
      - Kernel Method:    Support Vector Machine

    Returns
    -------
    models : dict
        Dictionary mapping model names to (estimator, filename) tuples.
    """
    models = {
        'Logistic Regression': (
            LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                class_weight='balanced',
                max_iter=1000,
                random_state=RANDOM_STATE
            ),
            'logistic_regression.pkl'
        ),

        'Random Forest': (
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=1
            ),
            'random_forest.pkl'
        ),

        'XGBoost': (
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                # scale_pos_weight handles imbalance:
                # ratio = n_negative / n_positive ≈ 806/194 ≈ 4.15
                scale_pos_weight=4.15,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0
            ),
            'xgboost.pkl'
        ),

        'LightGBM': (
            LGBMClassifier(
                n_estimators=300,
                max_depth=-1,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                is_unbalance=True,      # Handles imbalanced datasets
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=-1
            ),
            'lightgbm.pkl'
        ),

        'SVM': (
            SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                probability=True,       # Required for ROC-AUC computation
                random_state=RANDOM_STATE
            ),
            'svm.pkl'
        ),
    }

    return models


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: TRAINING WITH STRATIFIED K-FOLD CV + SMOTE
# ═══════════════════════════════════════════════════════════════════════════

def train_and_evaluate_cv(models, X_train, y_train, n_folds=N_FOLDS):
    """
    Train and evaluate each model using Stratified K-Fold Cross-Validation
    with SMOTE applied within each training fold.

    SMOTE is applied ONLY to the training fold to prevent data leakage --
    a critical methodological requirement for unbiased evaluation. The
    validation fold remains untouched and reflects the true class distribution.

    Parameters
    ----------
    models : dict
        Dictionary of model_name → (estimator, filename) tuples.
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target vector.
    n_folds : int
        Number of cross-validation folds.

    Returns
    -------
    cv_results : dict
        Per-model cross-validation results containing fold-level metrics.
    """
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    smote = SMOTE(
        sampling_strategy=SMOTE_STRATEGY,
        random_state=RANDOM_STATE
    )

    cv_results = {}

    for model_name, (model, _) in models.items():
        print(f"\n  Training: {model_name}")
        print(f"  {'─' * 50}")

        fold_metrics = {
            'accuracy': [], 'precision': [], 'recall': [],
            'f1': [], 'roc_auc': []
        }

        fold_start = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            # Split into fold training and validation sets
            X_fold_train = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            X_fold_val   = X_train.iloc[val_idx]   if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_fold_val   = y_train.iloc[val_idx]   if hasattr(y_train, 'iloc') else y_train[val_idx]

            # Apply SMOTE ONLY to the training fold (prevents data leakage)
            X_fold_resampled, y_fold_resampled = smote.fit_resample(
                X_fold_train, y_fold_train
            )

            # Clone model to ensure fresh training each fold
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_fold_resampled, y_fold_resampled)

            # Predict on the original (un-resampled) validation fold
            y_pred = fold_model.predict(X_fold_val)
            y_prob = fold_model.predict_proba(X_fold_val)[:, 1]

            # Compute fold-level metrics
            fold_metrics['accuracy'].append(accuracy_score(y_fold_val, y_pred))
            fold_metrics['precision'].append(precision_score(y_fold_val, y_pred, zero_division=0))
            fold_metrics['recall'].append(recall_score(y_fold_val, y_pred, zero_division=0))
            fold_metrics['f1'].append(f1_score(y_fold_val, y_pred, zero_division=0))
            fold_metrics['roc_auc'].append(roc_auc_score(y_fold_val, y_prob))

            print(f"    Fold {fold_idx}: "
                  f"Acc={fold_metrics['accuracy'][-1]:.4f}  "
                  f"F1={fold_metrics['f1'][-1]:.4f}  "
                  f"AUC={fold_metrics['roc_auc'][-1]:.4f}")

        elapsed = time.time() - fold_start

        # Compute mean ± std for each metric
        cv_summary = {}
        for metric_name, values in fold_metrics.items():
            cv_summary[f'{metric_name}_mean'] = np.mean(values)
            cv_summary[f'{metric_name}_std']  = np.std(values)
            cv_summary[f'{metric_name}_folds'] = values

        cv_summary['training_time'] = elapsed
        cv_results[model_name] = cv_summary

        print(f"  ── CV Summary ──")
        print(f"    Accuracy:  {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}")
        print(f"    Precision: {cv_summary['precision_mean']:.4f} ± {cv_summary['precision_std']:.4f}")
        print(f"    Recall:    {cv_summary['recall_mean']:.4f} ± {cv_summary['recall_std']:.4f}")
        print(f"    F1 Score:  {cv_summary['f1_mean']:.4f} ± {cv_summary['f1_std']:.4f}")
        print(f"    ROC-AUC:   {cv_summary['roc_auc_mean']:.4f} ± {cv_summary['roc_auc_std']:.4f}")
        print(f"    Time:      {elapsed:.2f}s")

    return cv_results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: FINAL MODEL TRAINING ON FULL TRAINING SET
# ═══════════════════════════════════════════════════════════════════════════

def train_final_models(models, X_train, y_train):
    """
    Train final models on the full training set with SMOTE applied.

    These are the models that will be:
      1. Evaluated on the held-out test set for unbiased performance estimates
      2. Serialized to .pkl files for future comparison with GNN models
      3. Used to generate publication-quality figures

    Parameters
    ----------
    models : dict
        Dictionary of model_name → (estimator, filename) tuples.
    X_train : array-like
        Full training feature matrix.
    y_train : array-like
        Full training target vector.

    Returns
    -------
    trained_models : dict
        Dictionary of model_name → fitted_estimator.
    """
    smote = SMOTE(
        sampling_strategy=SMOTE_STRATEGY,
        random_state=RANDOM_STATE
    )

    # Apply SMOTE to the full training set
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"\n  SMOTE Applied to Training Set:")
    print(f"    Original:   {len(y_train)} samples "
          f"(Class 0: {sum(y_train == 0)}, Class 1: {sum(y_train == 1)})")
    print(f"    Resampled:  {len(y_resampled)} samples "
          f"(Class 0: {sum(y_resampled == 0)}, Class 1: {sum(y_resampled == 1)})")

    trained_models = {}

    for model_name, (model, _) in models.items():
        from sklearn.base import clone
        final_model = clone(model)
        final_model.fit(X_resampled, y_resampled)
        trained_models[model_name] = final_model
        print(f"    [OK] {model_name} trained on resampled data.")

    return trained_models


def evaluate_on_test_set(trained_models, X_test, y_test):
    """
    Evaluate all trained models on the held-out test set.

    This provides the unbiased, final performance estimate that will be
    reported in the IEEE paper and compared against the GNN model.

    Parameters
    ----------
    trained_models : dict
        Dictionary of model_name → fitted_estimator.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        Test target vector.

    Returns
    -------
    test_results : dict
        Per-model test set results including predictions and probabilities.
    """
    test_results = {}

    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
            'roc_auc':   roc_auc_score(y_test, y_prob),
            'y_pred':    y_pred,
            'y_prob':    y_prob,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        test_results[model_name] = results

    return test_results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: PUBLICATION-QUALITY VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(test_results, y_test, save_dir):
    """
    Generate confusion matrix heatmaps for each model.

    Each confusion matrix is annotated with both absolute counts and
    percentages, following IEEE visualization standards.

    Parameters
    ----------
    test_results : dict
        Per-model test results containing confusion matrices.
    y_test : array-like
        True test labels (used for annotation).
    save_dir : str
        Directory to save the figure.
    """
    n_models = len(test_results)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Custom colormap: white → deep blue
    cmap = LinearSegmentedColormap.from_list(
        'ieee_blue', ['#FFFFFF', '#E8EDF2', '#A8C6E0', '#4A90B8', '#1B5E7A']
    )

    class_labels = ['No CVD (0)', 'CVD (1)']

    for idx, (model_name, results) in enumerate(test_results.items()):
        ax = axes[idx]
        cm = results['confusion_matrix']
        cm_percent = cm.astype(float) / cm.sum() * 100

        # Create annotation text: count + percentage
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

        sns.heatmap(
            cm, annot=annot, fmt='', cmap=cmap,
            xticklabels=class_labels,
            yticklabels=class_labels,
            linewidths=1.5, linecolor='white',
            cbar=True, cbar_kws={'label': 'Count'},
            ax=ax, square=True
        )

        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold',
                     color=COLORS[model_name], pad=10)

    # Hide the 6th subplot (only 5 models)
    axes[5].axis('off')

    fig.suptitle('Confusion Matrices — Classical ML Baselines (Test Set)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'confusion_matrices.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Confusion matrices saved to: {save_path}")


def plot_roc_curves(test_results, y_test, save_dir):
    """
    Generate a combined ROC curve plot for all models.

    Includes:
      - Individual ROC curves per model with distinct colors/styles
      - Diagonal reference line (random classifier baseline)
      - AUC values in the legend
      - Professional formatting for IEEE publication

    Parameters
    ----------
    test_results : dict
        Per-model test results containing predicted probabilities.
    y_test : array-like
        True test labels.
    save_dir : str
        Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name, results in test_results.items():
        fpr, tpr, _ = roc_curve(y_test, results['y_prob'])
        auc_val = results['roc_auc']

        ax.plot(
            fpr, tpr,
            color=COLORS[model_name],
            linewidth=2.2,
            label=f'{model_name} (AUC = {auc_val:.4f})',
            marker=MODEL_MARKERS[model_name],
            markevery=max(1, len(fpr) // 8),
            markersize=6,
            alpha=0.9
        )

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, alpha=0.5,
            label='Random Classifier (AUC = 0.5000)')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 − Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_title('ROC Curves — Classical ML Baselines',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fancybox=True,
              shadow=True, fontsize=9.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    save_path = os.path.join(save_dir, 'roc_curves.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] ROC curves saved to: {save_path}")


def plot_precision_recall_curves(test_results, y_test, save_dir):
    """
    Generate Precision-Recall curves for all models.

    PR curves are especially informative for imbalanced datasets
    (such as ours with ~19.4% positive class) because they focus
    on the minority class performance.

    Parameters
    ----------
    test_results : dict
        Per-model test results containing predicted probabilities.
    y_test : array-like
        True test labels.
    save_dir : str
        Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Baseline: ratio of positive class
    baseline = sum(y_test == 1) / len(y_test)

    for model_name, results in test_results.items():
        precision_vals, recall_vals, _ = precision_recall_curve(
            y_test, results['y_prob']
        )
        ap_score = average_precision_score(y_test, results['y_prob'])

        ax.plot(
            recall_vals, precision_vals,
            color=COLORS[model_name],
            linewidth=2.2,
            label=f'{model_name} (AP = {ap_score:.4f})',
            marker=MODEL_MARKERS[model_name],
            markevery=max(1, len(recall_vals) // 8),
            markersize=6,
            alpha=0.9
        )

    # Baseline: no-skill classifier
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.0,
               alpha=0.5, label=f'No-Skill Baseline ({baseline:.3f})')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontweight='bold')
    ax.set_ylabel('Precision (Positive Predictive Value)', fontweight='bold')
    ax.set_title('Precision-Recall Curves — Classical ML Baselines',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', frameon=True, fancybox=True,
              shadow=True, fontsize=9.5)
    ax.grid(True, alpha=0.3, linestyle='--')

    save_path = os.path.join(save_dir, 'precision_recall_curves.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Precision-Recall curves saved to: {save_path}")


def plot_model_comparison_bar_chart(cv_results, save_dir):
    """
    Generate a grouped bar chart comparing all models across all metrics.

    Includes error bars (±1 standard deviation from cross-validation)
    to visualize model stability across folds.

    Parameters
    ----------
    cv_results : dict
        Cross-validation results with mean and std for each metric.
    save_dir : str
        Directory to save the figure.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    model_names = list(cv_results.keys())

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(metrics))
    n_models = len(model_names)
    width = 0.15  # Bar width
    offsets = np.linspace(-(n_models - 1) * width / 2,
                          (n_models - 1) * width / 2, n_models)

    for i, model_name in enumerate(model_names):
        means = [cv_results[model_name][f'{m}_mean'] for m in metrics]
        stds  = [cv_results[model_name][f'{m}_std'] for m in metrics]

        bars = ax.bar(
            x + offsets[i], means, width,
            yerr=stds,
            label=model_name,
            color=COLORS[model_name],
            edgecolor='white',
            linewidth=0.5,
            capsize=3,
            error_kw={'linewidth': 1.0, 'capthick': 1.0},
            alpha=0.88,
            zorder=3
        )

        # Add value labels on bars
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{mean_val:.3f}',
                ha='center', va='bottom', fontsize=6.5,
                fontweight='bold', rotation=90
            )

    ax.set_xlabel('Evaluation Metric', fontweight='bold', fontsize=13)
    ax.set_ylabel('Score', fontweight='bold', fontsize=13)
    ax.set_title('Model Comparison — Stratified 5-Fold Cross-Validation\n'
                 '(Mean ± Std, SMOTE within training folds)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim([0.0, 1.15])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=n_models, frameon=True, fancybox=True, shadow=True,
              fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3, zorder=0)

    save_path = os.path.join(save_dir, 'model_comparison_bar_chart.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Model comparison bar chart saved to: {save_path}")


def plot_cv_boxplots(cv_results, save_dir):
    """
    Generate box plots showing metric distributions across CV folds.

    Box plots reveal the variance and stability of each model,
    which is important for assessing reliability in clinical applications.

    Parameters
    ----------
    cv_results : dict
        Cross-validation results with per-fold metric values.
    save_dir : str
        Directory to save the figure.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    model_names = list(cv_results.keys())

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        data = []
        colors_list = []

        for model_name in model_names:
            fold_values = cv_results[model_name][f'{metric}_folds']
            data.append(fold_values)
            colors_list.append(COLORS[model_name])

        bp = ax.boxplot(
            data, patch_artist=True, widths=0.6,
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(markersize=5),
        )

        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)

        ax.set_title(label, fontweight='bold', fontsize=12)
        ax.set_xticklabels(
            [n.replace(' ', '\n') for n in model_names],
            fontsize=7.5, rotation=0
        )
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylabel('Score' if idx == 0 else '')

    fig.suptitle('Cross-Validation Metric Distributions (5 Folds)',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'cv_boxplots.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] CV box plots saved to: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: RESULTS SAVING AND RANKING
# ═══════════════════════════════════════════════════════════════════════════

def save_baseline_metrics(cv_results, test_results, save_dir):
    """
    Save comprehensive baseline metrics to CSV files.

    Two files are generated:
      1. baseline_metrics.csv: Detailed per-metric CV and test results
      2. model_comparison.csv: Side-by-side comparison summary

    Parameters
    ----------
    cv_results : dict
        Cross-validation results.
    test_results : dict
        Test set evaluation results.
    save_dir : str
        Directory to save CSV files.
    """
    # ── baseline_metrics.csv: Detailed results ────────────────────────────
    rows = []
    for model_name in cv_results.keys():
        cv = cv_results[model_name]
        test = test_results[model_name]

        row = {
            'Model': model_name,
            # CV results (mean ± std)
            'CV_Accuracy_Mean':  cv['accuracy_mean'],
            'CV_Accuracy_Std':   cv['accuracy_std'],
            'CV_Precision_Mean': cv['precision_mean'],
            'CV_Precision_Std':  cv['precision_std'],
            'CV_Recall_Mean':    cv['recall_mean'],
            'CV_Recall_Std':     cv['recall_std'],
            'CV_F1_Mean':        cv['f1_mean'],
            'CV_F1_Std':         cv['f1_std'],
            'CV_ROC_AUC_Mean':   cv['roc_auc_mean'],
            'CV_ROC_AUC_Std':    cv['roc_auc_std'],
            # Test set results
            'Test_Accuracy':     test['accuracy'],
            'Test_Precision':    test['precision'],
            'Test_Recall':       test['recall'],
            'Test_F1':           test['f1'],
            'Test_ROC_AUC':      test['roc_auc'],
            # Training time
            'Training_Time_Sec': cv['training_time']
        }
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(save_dir, 'baseline_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False, float_format='%.6f')
    print(f"  [OK] Baseline metrics saved to: {metrics_path}")

    # ── model_comparison.csv: Summary comparison ──────────────────────────
    comparison_rows = []
    for model_name in cv_results.keys():
        cv = cv_results[model_name]
        test = test_results[model_name]

        comparison_rows.append({
            'Model': model_name,
            'CV_Accuracy':  f"{cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}",
            'CV_Precision': f"{cv['precision_mean']:.4f} ± {cv['precision_std']:.4f}",
            'CV_Recall':    f"{cv['recall_mean']:.4f} ± {cv['recall_std']:.4f}",
            'CV_F1':        f"{cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}",
            'CV_ROC_AUC':   f"{cv['roc_auc_mean']:.4f} ± {cv['roc_auc_std']:.4f}",
            'Test_Accuracy':  f"{test['accuracy']:.4f}",
            'Test_F1':        f"{test['f1']:.4f}",
            'Test_ROC_AUC':   f"{test['roc_auc']:.4f}",
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = os.path.join(save_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"  [OK] Model comparison saved to: {comparison_path}")

    return metrics_df


def save_trained_models(trained_models, models_config, save_dir):
    """
    Serialize trained models to .pkl files for future loading.

    These saved models can be loaded in Phase 3 (GNN) for direct
    head-to-head comparison without retraining.

    Parameters
    ----------
    trained_models : dict
        Dictionary of model_name → fitted_estimator.
    models_config : dict
        Dictionary containing filenames for each model.
    save_dir : str
        Directory to save .pkl files.
    """
    for model_name, model in trained_models.items():
        _, filename = models_config[model_name]
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"  [OK] {model_name} saved to: {filepath}")


def generate_ranking_table(cv_results, test_results):
    """
    Generate a comprehensive ranking table for all models.

    Models are ranked by:
      1. CV F1 Score (primary — best for imbalanced classification)
      2. CV ROC-AUC (secondary — discrimination capability)
      3. Test F1 Score (tertiary — generalization check)

    The ranking methodology follows established ML benchmarking
    practices for medical diagnosis applications.

    Parameters
    ----------
    cv_results : dict
        Cross-validation results.
    test_results : dict
        Test set evaluation results.

    Returns
    -------
    ranking_df : pd.DataFrame
        Final ranking table sorted by composite score.
    """
    ranking_data = []
    for model_name in cv_results.keys():
        cv = cv_results[model_name]
        test = test_results[model_name]

        # Composite ranking score:
        # 40% CV F1 + 30% CV ROC-AUC + 20% Test F1 + 10% Test ROC-AUC
        composite = (
            0.40 * cv['f1_mean'] +
            0.30 * cv['roc_auc_mean'] +
            0.20 * test['f1'] +
            0.10 * test['roc_auc']
        )

        ranking_data.append({
            'Model': model_name,
            'CV_F1_Mean':     cv['f1_mean'],
            'CV_F1_Std':      cv['f1_std'],
            'CV_ROC_AUC':     cv['roc_auc_mean'],
            'Test_F1':        test['f1'],
            'Test_ROC_AUC':   test['roc_auc'],
            'Test_Accuracy':  test['accuracy'],
            'Composite_Score': composite,
            'Training_Time':  cv['training_time']
        })

    ranking_df = pd.DataFrame(ranking_data)
    ranking_df = ranking_df.sort_values('Composite_Score', ascending=False)
    ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
    ranking_df = ranking_df.reset_index(drop=True)

    return ranking_df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Main execution pipeline for Phase 2 — Classical ML Baselines.

    Orchestrates the complete workflow:
      1. Data loading and preparation
      2. Stratified train-test split
      3. Model definition
      4. Cross-validation with SMOTE
      5. Final model training and test evaluation
      6. Publication-quality visualization
      7. Results saving and model serialization
      8. Final ranking generation
    """
    print("=" * 80)
    print("  PHASE 2: CLASSICAL MACHINE LEARNING BASELINES")
    print("  Cardiovascular Disease Risk Prediction using GNN")
    print("  IEEE Conference Paper — Benchmark Evaluation")
    print("=" * 80)

    # ── Step 1: Load Data ─────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  STEP 1: DATA LOADING AND PREPARATION")
    print("═" * 80)

    data_path = os.path.join(DATA_DIR, "processed_dataset.csv")
    X, y, feature_names = load_and_prepare_data(data_path)

    print(f"\n  [OK] Dataset loaded: {data_path}")
    print(f"  Samples:  {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Features: {feature_names}")
    print(f"  Target distribution:")
    print(f"    Class 0 (No CVD): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    print(f"    Class 1 (CVD):    {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")

    # ── Step 2: Stratified Train-Test Split ───────────────────────────────
    print("\n" + "═" * 80)
    print("  STEP 2: STRATIFIED TRAIN-TEST SPLIT (80-20)")
    print("═" * 80)

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    print(f"\n  Training set: {X_train.shape[0]} samples")
    print(f"    Class 0: {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"    Class 1: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")
    print(f"  Test set:     {X_test.shape[0]} samples")
    print(f"    Class 0: {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
    print(f"    Class 1: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")

    # ── Step 3: Define Models ─────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  STEP 3: MODEL DEFINITIONS")
    print("═" * 80)

    models = get_models()

    for model_name, (model, filename) in models.items():
        print(f"\n  {model_name}:")
        print(f"    Type:     {type(model).__name__}")
        print(f"    Filename: {filename}")

    # ── Step 4: Stratified 5-Fold Cross-Validation with SMOTE ─────────────
    print("\n" + "═" * 80)
    print("  STEP 4: STRATIFIED 5-FOLD CROSS-VALIDATION WITH SMOTE")
    print("═" * 80)
    print(f"\n  Configuration:")
    print(f"    Folds:          {N_FOLDS}")
    print(f"    SMOTE:          Applied within training folds only")
    print(f"    Stratification: Preserves class ratio in each fold")
    print(f"    Random State:   {RANDOM_STATE}")

    cv_results = train_and_evaluate_cv(models, X_train, y_train)

    # ── Step 5: Final Model Training on Full Training Set ─────────────────
    print("\n" + "═" * 80)
    print("  STEP 5: FINAL MODEL TRAINING ON FULL TRAINING SET")
    print("═" * 80)

    trained_models = train_final_models(models, X_train, y_train)

    # ── Step 6: Test Set Evaluation ───────────────────────────────────────
    print("\n" + "═" * 80)
    print("  STEP 6: TEST SET EVALUATION (HELD-OUT, UNSEEN DATA)")
    print("═" * 80)

    test_results = evaluate_on_test_set(trained_models, X_test, y_test)

    print(f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print(f"  {'─' * 75}")
    for model_name, results in test_results.items():
        print(f"  {model_name:<25} {results['accuracy']:>10.4f} "
              f"{results['precision']:>10.4f} {results['recall']:>10.4f} "
              f"{results['f1']:>10.4f} {results['roc_auc']:>10.4f}")

    # ── Step 7: Publication-Quality Figures ────────────────────────────────
    print("\n" + "═" * 80)
    print("  STEP 7: GENERATING PUBLICATION-QUALITY FIGURES")
    print("═" * 80)

    plot_confusion_matrices(test_results, y_test, FIG_DIR)
    plot_roc_curves(test_results, y_test, FIG_DIR)
    plot_precision_recall_curves(test_results, y_test, FIG_DIR)
    plot_model_comparison_bar_chart(cv_results, FIG_DIR)
    plot_cv_boxplots(cv_results, FIG_DIR)

    # ── Step 8: Save Results ──────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  STEP 8: SAVING RESULTS")
    print("═" * 80)

    metrics_df = save_baseline_metrics(cv_results, test_results, RES_DIR)

    # ── Step 9: Save Trained Models ───────────────────────────────────────
    print("\n" + "═" * 80)
    print("  STEP 9: SAVING TRAINED MODELS")
    print("═" * 80)

    save_trained_models(trained_models, models, MODEL_DIR)

    # ── Step 10: Final Ranking Table ──────────────────────────────────────
    print("\n" + "═" * 80)
    print("  STEP 10: FINAL MODEL RANKING")
    print("═" * 80)

    ranking_df = generate_ranking_table(cv_results, test_results)

    # Save ranking table
    ranking_path = os.path.join(RES_DIR, 'model_ranking.csv')
    ranking_df.to_csv(ranking_path, index=False, float_format='%.6f')

    # Print formatted ranking
    print(f"\n  Ranking Methodology:")
    print(f"    Composite Score = 40% CV_F1 + 30% CV_AUC + 20% Test_F1 + 10% Test_AUC")
    print(f"\n  {'Rank':<6} {'Model':<25} {'CV F1':>12} {'CV AUC':>10} "
          f"{'Test F1':>10} {'Test AUC':>10} {'Composite':>12}")
    print(f"  {'─' * 85}")
    for _, row in ranking_df.iterrows():
        rank_marker = '★' if row['Rank'] == 1 else ' '
        print(f"  {rank_marker} {int(row['Rank']):<4} {row['Model']:<25} "
              f"{row['CV_F1_Mean']:>8.4f}±{row['CV_F1_Std']:.4f} "
              f"{row['CV_ROC_AUC']:>10.4f} "
              f"{row['Test_F1']:>10.4f} "
              f"{row['Test_ROC_AUC']:>10.4f} "
              f"{row['Composite_Score']:>12.4f}")

    print(f"\n  [OK] Ranking table saved to: {ranking_path}")

    # ── Classification Reports ────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  DETAILED CLASSIFICATION REPORTS (TEST SET)")
    print("═" * 80)

    for model_name, results in test_results.items():
        print(f"\n  ── {model_name} ──")
        report = classification_report(
            y_test, results['y_pred'],
            target_names=['No CVD (0)', 'CVD (1)'],
            digits=4
        )
        for line in report.split('\n'):
            print(f"    {line}")

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  [OK] PHASE 2 — CLASSICAL ML BASELINES COMPLETE")
    print("=" * 80)
    print(f"\n  Generated Files:")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Results:")
    print(f"    • {os.path.join(RES_DIR, 'baseline_metrics.csv')}")
    print(f"    • {os.path.join(RES_DIR, 'model_comparison.csv')}")
    print(f"    • {os.path.join(RES_DIR, 'model_ranking.csv')}")
    print(f"  Figures:")
    print(f"    • {os.path.join(FIG_DIR, 'confusion_matrices.png')}")
    print(f"    • {os.path.join(FIG_DIR, 'roc_curves.png')}")
    print(f"    • {os.path.join(FIG_DIR, 'precision_recall_curves.png')}")
    print(f"    • {os.path.join(FIG_DIR, 'model_comparison_bar_chart.png')}")
    print(f"    • {os.path.join(FIG_DIR, 'cv_boxplots.png')}")
    print(f"  Models:")
    for model_name, (_, filename) in models.items():
        print(f"    • {os.path.join(MODEL_DIR, filename)}")
    print(f"\n  Best Model: {ranking_df.iloc[0]['Model']} "
          f"(Composite Score: {ranking_df.iloc[0]['Composite_Score']:.4f})")
    print(f"\n  Next Step: Phase 3 — Graph Neural Network Implementation")
    print(f"    The GNN model will be compared against these baselines.")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
