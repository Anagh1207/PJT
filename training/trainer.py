"""
=============================================================================
Training Pipeline — Stratified K-Fold CV with SMOTE
=============================================================================
Modular training pipeline that handles:
  1. Stratified K-Fold Cross-Validation
  2. SMOTE oversampling within each training fold (no data leakage)
  3. Final model training on full training set
  4. Model serialization to .pkl files

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import sys
import time
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE

from config import RANDOM_STATE, N_FOLDS, SMOTE_STRATEGY


def cross_validate_with_smote(model, X_train, y_train, n_folds=N_FOLDS,
                               verbose=True):
    """
    Perform Stratified K-Fold Cross-Validation with SMOTE applied
    within each training fold to prevent data leakage.

    Parameters
    ----------
    model : estimator
        Scikit-learn compatible classifier.
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training target vector.
    n_folds : int
        Number of CV folds.
    verbose : bool
        Whether to print fold-level metrics.

    Returns
    -------
    cv_results : dict
        Contains mean, std, and per-fold values for each metric,
        plus total training time.
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

    fold_metrics = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'roc_auc': []
    }

    fold_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        # Split into fold
        X_fold_train = (X_train.iloc[train_idx] if hasattr(X_train, 'iloc')
                        else X_train[train_idx])
        y_fold_train = (y_train.iloc[train_idx] if hasattr(y_train, 'iloc')
                        else y_train[train_idx])
        X_fold_val = (X_train.iloc[val_idx] if hasattr(X_train, 'iloc')
                      else X_train[val_idx])
        y_fold_val = (y_train.iloc[val_idx] if hasattr(y_train, 'iloc')
                      else y_train[val_idx])

        # SMOTE only on training fold — critical for unbiased evaluation
        X_resampled, y_resampled = smote.fit_resample(X_fold_train, y_fold_train)

        # Train a fresh clone
        fold_model = clone(model)
        fold_model.fit(X_resampled, y_resampled)

        # Predict on untouched validation fold
        y_pred = fold_model.predict(X_fold_val)
        y_prob = fold_model.predict_proba(X_fold_val)[:, 1]

        # Record metrics
        fold_metrics['accuracy'].append(accuracy_score(y_fold_val, y_pred))
        fold_metrics['precision'].append(
            precision_score(y_fold_val, y_pred, zero_division=0))
        fold_metrics['recall'].append(
            recall_score(y_fold_val, y_pred, zero_division=0))
        fold_metrics['f1'].append(
            f1_score(y_fold_val, y_pred, zero_division=0))
        fold_metrics['roc_auc'].append(roc_auc_score(y_fold_val, y_prob))

        if verbose:
            print(f"    Fold {fold_idx}: "
                  f"Acc={fold_metrics['accuracy'][-1]:.4f}  "
                  f"F1={fold_metrics['f1'][-1]:.4f}  "
                  f"AUC={fold_metrics['roc_auc'][-1]:.4f}")

    elapsed = time.time() - fold_start

    # Aggregate results
    cv_results = {'training_time': elapsed}
    for metric_name, values in fold_metrics.items():
        cv_results[f'{metric_name}_mean'] = np.mean(values)
        cv_results[f'{metric_name}_std'] = np.std(values)
        cv_results[f'{metric_name}_folds'] = values

    return cv_results


def train_final_model(model, X_train, y_train, apply_smote=True):
    """
    Train a final model on the full training set.

    Parameters
    ----------
    model : estimator
        Scikit-learn compatible classifier (will be cloned).
    X_train : array-like
        Full training feature matrix.
    y_train : array-like
        Full training target vector.
    apply_smote : bool
        Whether to apply SMOTE before training.

    Returns
    -------
    trained_model : estimator
        The fitted model.
    training_time : float
        Time taken for training in seconds.
    """
    final_model = clone(model)

    if apply_smote:
        smote = SMOTE(
            sampling_strategy=SMOTE_STRATEGY,
            random_state=RANDOM_STATE
        )
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_resampled, y_resampled = X_train, y_train

    start = time.time()
    final_model.fit(X_resampled, y_resampled)
    training_time = time.time() - start

    return final_model, training_time


def save_model(model, filepath):
    """
    Serialize a trained model to disk.

    Parameters
    ----------
    model : estimator
        Fitted model to save.
    filepath : str
        Destination .pkl path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    """
    Load a serialized model from disk.

    Parameters
    ----------
    filepath : str
        Path to .pkl file.

    Returns
    -------
    model : estimator
        The loaded model.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)
