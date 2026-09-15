import os, sys, warnings, time, pickle, json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report,
    matthews_corrcoef)
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import clone

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# ── Visualization Libraries ────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import altair as alt

import missingno as msno

from plotnine import (ggplot, aes, geom_boxplot, geom_bar, geom_col,
    geom_point, geom_errorbar, labs, theme_minimal, theme,
    element_text, scale_fill_manual, coord_flip, position_dodge,
    facet_wrap, geom_tile, scale_fill_gradient2)

from bokeh.plotting import figure as bk_figure, show as bk_show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, HoverTool, BoxAnnotation
from bokeh.layouts import gridplot

# PyTorch + PyG
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, BatchNorm

warnings.filterwarnings('ignore')
output_notebook()  # Enable Bokeh inline

# ── Project paths ──────────────────────────────────────────────────────
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR  = os.path.join(BASE_DIR, "figures")
RES_DIR  = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")
GNN_DIR   = os.path.join(MODEL_DIR, "gnn_checkpoints")
PAPER_DIR = os.path.join(BASE_DIR, "paper_assets")
GRAPH_DIR = os.path.join(DATA_DIR, "graphs")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# IEEE-friendly style
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Color palette
COLOR_NEG = '#2196F3'   # Blue  — No CVD
COLOR_POS = '#E53935'   # Red   — CVD
PALETTE   = [COLOR_NEG, COLOR_POS]

MODEL_COLORS = {
    'Logistic Regression': '#2E86AB', 'Random Forest': '#A23B72',
    'XGBoost': '#F18F01', 'LightGBM': '#C73E1D', 'SVM': '#3B1F2B',
    'GCN': '#1E88E5', 'GraphSAGE': '#43A047', 'GAT': '#8E24AA', 'DPSN': '#D81B60',
}

alt.data_transformers.disable_max_rows()

print("✅ All libraries loaded successfully")
print(f"   PyTorch: {torch.__version__}")
print(f"   NumPy:   {np.__version__}")
print(f"   Pandas:  {pd.__version__}")
