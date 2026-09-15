"""
=============================================================================
Configuration Module
=============================================================================
Cardiovascular Disease Risk Prediction using Graph Neural Networks
IEEE Conference Paper — Central Configuration

All project-wide settings: paths, model hyperparameters, evaluation
parameters, visualization options, and experiment tracking configuration.

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os

# ═══════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "data")
MODEL_DIR        = os.path.join(BASE_DIR, "models")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")
FIGURES_DIR      = os.path.join(BASE_DIR, "figures")

# Data files
RAW_DATASET      = os.path.join(BASE_DIR, "GNN DATASET.xlsx")
CLEANED_DATASET  = os.path.join(DATA_DIR, "cleaned_dataset.csv")
PROCESSED_DATASET = os.path.join(DATA_DIR, "processed_dataset.csv")

# Results files
METRICS_CSV      = os.path.join(RESULTS_DIR, "metrics.csv")
METRICS_JSON     = os.path.join(RESULTS_DIR, "metrics.json")
RANKING_CSV      = os.path.join(RESULTS_DIR, "model_ranking.csv")
COMPARISON_CSV   = os.path.join(RESULTS_DIR, "model_comparison.csv")

# Create directories
for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR, VISUALIZATIONS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATASET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

TARGET_COLUMN = "Target"

# Column classification from Phase 1 audit
BINARY_FEATURES = ['Gender', 'Diabetes', 'Smoking', 'Family_History']

CONTINUOUS_FEATURES = [
    'Age', 'Blood_Pressure', 'Cholesterol', 'Heart_Rate', 'BMI',
    'Physical_Activity_HoursPerWeek', 'ECG_Score', 'Glucose',
    'Triglycerides', 'Stress_Index'
]

ALL_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES

CLASS_LABELS = {0: 'No CVD', 1: 'CVD'}
CLASS_NAMES  = ['No CVD (0)', 'CVD (1)']


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

RANDOM_STATE    = 42
TEST_SIZE       = 0.20       # 80-20 stratified split
N_FOLDS         = 5          # Stratified K-Fold CV
SMOTE_STRATEGY  = 'minority' # SMOTE oversampling strategy

# Imbalance ratio (from Phase 1 audit: 806/194 ≈ 4.15)
IMBALANCE_RATIO = 4.15


# ═══════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PARAMS = {
    'Logistic Regression': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'class_weight': 'balanced',
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
    },

    'Random Forest': {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': 1,
    },

    'Decision Tree': {
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
    },

    'SVM': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'class_weight': 'balanced',
        'probability': True,
        'random_state': RANDOM_STATE,
    },

    'KNN': {
        'n_neighbors': 7,
        'weights': 'distance',
        'metric': 'minkowski',
        'n_jobs': 1,
    },

    'XGBoost': {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': IMBALANCE_RATIO,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
        'n_jobs': 1,
        'verbosity': 0,
    },

    'LightGBM': {
        'n_estimators': 300,
        'max_depth': -1,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'is_unbalance': True,
        'random_state': RANDOM_STATE,
        'n_jobs': 1,
        'verbose': -1,
    },
}

# Model save filenames
MODEL_FILENAMES = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Random Forest':       'random_forest.pkl',
    'Decision Tree':       'decision_tree.pkl',
    'SVM':                 'svm.pkl',
    'KNN':                 'knn.pkl',
    'XGBoost':             'xgboost.pkl',
    'LightGBM':            'lightgbm.pkl',
}


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# IEEE-friendly color palette for models
MODEL_COLORS = {
    'Logistic Regression': '#2E86AB',
    'Random Forest':       '#A23B72',
    'Decision Tree':       '#6B4226',
    'SVM':                 '#3B1F2B',
    'KNN':                 '#1B998B',
    'XGBoost':             '#F18F01',
    'LightGBM':            '#C73E1D',
}

MODEL_MARKERS = {
    'Logistic Regression': 'o',
    'Random Forest':       's',
    'Decision Tree':       'P',
    'SVM':                 'v',
    'KNN':                 'd',
    'XGBoost':             '^',
    'LightGBM':            'D',
}

# Clinical color scheme
COLOR_NEGATIVE = '#2196F3'   # Blue  — No CVD (class 0)
COLOR_POSITIVE = '#E53935'   # Red   — CVD    (class 1)
COLOR_PALETTE  = [COLOR_NEGATIVE, COLOR_POSITIVE]
HEATMAP_CMAP   = 'coolwarm'

# Plot settings
PLOT_DPI       = 300
PLOT_STYLE     = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': PLOT_DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}


# ═══════════════════════════════════════════════════════════════════════════
# RANKING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Composite ranking weights
RANKING_WEIGHTS = {
    'cv_f1':       0.40,
    'cv_roc_auc':  0.30,
    'test_f1':     0.20,
    'test_roc_auc': 0.10,
}


# ═══════════════════════════════════════════════════════════════════════════
# GNN CONFIGURATION — Phases 3–10
# ═══════════════════════════════════════════════════════════════════════════

# Additional directories
GNN_DIR          = os.path.join(BASE_DIR, "models", "gnn_checkpoints")
GRAPH_DIR        = os.path.join(DATA_DIR, "graphs")
PAPER_ASSETS_DIR = os.path.join(BASE_DIR, "paper_assets")

for d in [GNN_DIR, GRAPH_DIR, PAPER_ASSETS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Graph Construction Parameters ─────────────────────────────────────────

KNN_K              = 10          # neighbours for KNN graph
COSINE_THRESHOLD   = 0.85        # threshold for cosine similarity graph
CLINICAL_THRESHOLD = 0.80        # threshold for clinical similarity graph

# Domain-specific clinical risk-factor weights (used by Clinical Similarity Graph)
CLINICAL_WEIGHTS = {
    'Age':                            1.5,
    'Blood_Pressure':                 1.8,
    'Cholesterol':                    2.0,
    'Heart_Rate':                     1.0,
    'BMI':                            1.3,
    'Diabetes':                       1.8,
    'Smoking':                        1.6,
    'Physical_Activity_HoursPerWeek': 1.0,
    'ECG_Score':                      1.4,
    'Family_History':                 1.5,
    'Glucose':                        1.7,
    'Triglycerides':                  1.5,
    'Stress_Index':                   1.0,
    'Gender':                         0.5,
}

# ── GNN Training Hyperparameters ──────────────────────────────────────────

GNN_HIDDEN_DIM     = 64
GNN_EMBED_DIM      = 32
GNN_LEARNING_RATE  = 0.005
GNN_WEIGHT_DECAY   = 5e-4
GNN_EPOCHS         = 200
GNN_PATIENCE       = 20         # early stopping patience
GNN_DROPOUT        = 0.3
GAT_HEADS          = 4
GNN_NUM_SEEDS      = 5          # number of random seeds for statistical validation

# Train / Validation / Test split ratios for node-level GNN
GNN_TRAIN_RATIO = 0.60
GNN_VAL_RATIO   = 0.20
GNN_TEST_RATIO  = 0.20

# ── DPSN-Specific Settings ────────────────────────────────────────────────

DPSN_TOPK            = 15       # top-K edges to keep per node in dynamic graph
DPSN_EDGE_HIDDEN     = 32      # hidden dim for edge scoring MLP
DPSN_TEMPERATURE     = 0.5     # temperature for edge weight softmax

# ── Ablation Variant Definitions ──────────────────────────────────────────

ABLATION_VARIANTS = {
    'Full DPSN':                 {'dynamic_edges': True,  'clinical_sim': True,  'attention': True,  'feature_weight': True},
    'A: No Dynamic Edges':       {'dynamic_edges': False, 'clinical_sim': True,  'attention': True,  'feature_weight': True},
    'B: No Clinical Similarity': {'dynamic_edges': True,  'clinical_sim': False, 'attention': True,  'feature_weight': True},
    'C: No Attention':           {'dynamic_edges': True,  'clinical_sim': True,  'attention': False, 'feature_weight': True},
    'D: No Feature Weighting':   {'dynamic_edges': True,  'clinical_sim': True,  'attention': True,  'feature_weight': False},
}

# ── GNN Model Colors & Markers (for visualizations) ──────────────────────

GNN_MODEL_COLORS = {
    'GCN':          '#1E88E5',
    'GraphSAGE':    '#43A047',
    'GAT':          '#8E24AA',
    'DPSN':         '#D81B60',
}

GNN_MODEL_MARKERS = {
    'GCN':          'o',
    'GraphSAGE':    's',
    'GAT':          '^',
    'DPSN':         'D',
}

GRAPH_COLORS = {
    'Euclidean KNN':      '#1565C0',
    'Cosine Similarity':  '#2E7D32',
    'Clinical Similarity':'#E65100',
    'Dynamic PSG':        '#AD1457',
}

# Merge all model colors for combined plots
ALL_MODEL_COLORS = {**MODEL_COLORS, **GNN_MODEL_COLORS}
ALL_MODEL_MARKERS = {**MODEL_MARKERS, **GNN_MODEL_MARKERS}


# ═══════════════════════════════════════════════════════════════════════════
# RESTRUCTURED FRAMEWORK CONFIGURATION (Benchmark, Synthetic, Topology-CF)
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK_DATASET = os.path.join(DATA_DIR, "benchmark_dataset.csv")
SYNTHETIC_DATASET = os.path.join(DATA_DIR, "synthetic_dataset.csv")

# K-Value Sweep for Patient Similarity Graph
K_VALUES = [5, 10, 15, 20]
DEFAULT_K = 10

BENCHMARK_CONFIG = {
    'primary_source': 'framingham',  # Framingham Heart Study dataset
    'fallback_source': 'kaggle_cvd',  # Kaggle CVD dataset
    'framingham_url': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/framingham.csv',
    'kaggle_url': 'https://raw.githubusercontent.com/datasets/cardiovascular-disease/main/cardio_train.csv',
    'output_path': BENCHMARK_DATASET,
}

SYNTHETIC_CONFIG = {
    'method': 'ctgan',              # 'ctgan' or 'tvae'
    'sample_size': 2000,            # 2x original dataset size
    'epochs': 300,
    'batch_size': 100,
    'output_path': SYNTHETIC_DATASET,
    'validation_thresholds': {
        'ks_p_value_min': 0.05,     # KS test p-value threshold for similarity
        'correlation_diff_max': 0.15,# Max Frobenius norm diff on corr matrix
    }
}

TOPOLOGY_CONFIG = {
    'k_values': K_VALUES,
    'similarity_metric': 'clinical', # 'clinical', 'cosine', or 'euclidean'
    'default_k': DEFAULT_K,
}

COUNTERFACTUAL_CONFIG = {
    'immutable_features': ['Age', 'Gender', 'Family_History'],
    'modifiable_features': [
        'Blood_Pressure', 'Cholesterol', 'Heart_Rate', 'BMI',
        'Physical_Activity_HoursPerWeek', 'ECG_Score', 'Glucose',
        'Triglycerides', 'Stress_Index', 'Smoking', 'Diabetes'
    ],
    'max_iterations': 500,
    'step_size': 0.05,
    'target_risk_reduction': 0.50,  # Target risk score reduction (or target class 0)
    'realism_z_score_max': 3.0,     # Max allowed z-score distance from population mean
}

EXPERIMENT_CONFIG = {
    'approaches': ['Normal CF', 'Fixed-Graph CF', 'Topology-Aware CF'],
    'num_test_patients': 50,         # Number of high-risk patients to generate CFs for
    'metrics': [
        'risk_reduction',
        'n_features_changed',
        'l2_feature_distance',
        'jaccard_neighbor_overlap',
        'neighborhood_risk_change',
        'embedding_displacement'
    ]
}

