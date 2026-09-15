"""
=============================================================================
Model Definitions — Baseline Classifiers
=============================================================================
Factory functions that instantiate all baseline models with configured
hyperparameters. Each model handles class imbalance via built-in
mechanisms (class_weight, scale_pos_weight, is_unbalance).

Supported Models:
  - Logistic Regression  (Linear baseline)
  - Random Forest        (Bagging ensemble)
  - Decision Tree        (Single tree baseline)
  - SVM                  (Kernel-based)
  - KNN                  (Instance-based)
  - XGBoost              (Gradient boosting)
  - LightGBM             (Gradient boosting)

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Optional dependencies — graceful fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from config import MODEL_PARAMS, MODEL_FILENAMES


def get_model(model_name):
    """
    Instantiate a model by name using configured hyperparameters.

    Parameters
    ----------
    model_name : str
        One of: 'Logistic Regression', 'Random Forest', 'Decision Tree',
        'SVM', 'KNN', 'XGBoost', 'LightGBM'.

    Returns
    -------
    model : estimator
        Scikit-learn compatible estimator.

    Raises
    ------
    ValueError
        If model_name is not recognized or dependency is missing.
    """
    if model_name not in MODEL_PARAMS:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: {list(MODEL_PARAMS.keys())}")

    params = MODEL_PARAMS[model_name]

    if model_name == 'Logistic Regression':
        return LogisticRegression(**params)

    elif model_name == 'Random Forest':
        return RandomForestClassifier(**params)

    elif model_name == 'Decision Tree':
        return DecisionTreeClassifier(**params)

    elif model_name == 'SVM':
        return SVC(**params)

    elif model_name == 'KNN':
        return KNeighborsClassifier(**params)

    elif model_name == 'XGBoost':
        if not HAS_XGBOOST:
            raise ValueError("XGBoost is not installed. "
                             "Install with: pip install xgboost")
        return XGBClassifier(**params)

    elif model_name == 'LightGBM':
        if not HAS_LIGHTGBM:
            raise ValueError("LightGBM is not installed. "
                             "Install with: pip install lightgbm")
        return LGBMClassifier(**params)

    else:
        raise ValueError(f"Model '{model_name}' not implemented.")


def get_all_models():
    """
    Return a dictionary of all available models.

    Returns
    -------
    models : dict
        Dictionary mapping model_name → (estimator, filename).
        Only includes models whose dependencies are available.
    """
    available = {}

    for name in MODEL_PARAMS.keys():
        try:
            model = get_model(name)
            filename = MODEL_FILENAMES.get(name, f"{name.lower().replace(' ', '_')}.pkl")
            available[name] = (model, filename)
        except ValueError as e:
            print(f"  [SKIP] {name}: {e}")

    return available


def get_tree_based_models():
    """
    Return names of tree-based models (support feature importance).

    Returns
    -------
    list of str
        Names of tree-based models available in the project.
    """
    tree_models = ['Random Forest', 'Decision Tree', 'XGBoost', 'LightGBM']
    return [m for m in tree_models if m in MODEL_PARAMS]
