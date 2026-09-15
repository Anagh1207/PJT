"""
=============================================================================
Evaluation Pipeline — Metrics Computation & Classification Reports
=============================================================================
Modular evaluation system that computes all standard classification metrics
and generates structured output for experiment tracking.

Metrics:
  - Accuracy, Precision, Recall, F1 Score, ROC-AUC
  - Confusion Matrix
  - Full classification report

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from config import CLASS_NAMES


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on a test set.

    Computes all standard classification metrics and measures
    inference time for benchmarking.

    Parameters
    ----------
    model : estimator
        Fitted scikit-learn compatible classifier.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True test labels.

    Returns
    -------
    results : dict
        Dictionary containing:
        - accuracy, precision, recall, f1, roc_auc (float)
        - y_pred (np.ndarray): predicted labels
        - y_prob (np.ndarray): predicted probabilities for positive class
        - confusion_matrix (np.ndarray): 2×2 confusion matrix
        - inference_time (float): time for prediction in seconds
    """
    # Measure inference time
    start = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start

    results = {
        'accuracy':         accuracy_score(y_test, y_pred),
        'precision':        precision_score(y_test, y_pred, zero_division=0),
        'recall':           recall_score(y_test, y_pred, zero_division=0),
        'f1':               f1_score(y_test, y_pred, zero_division=0),
        'roc_auc':          roc_auc_score(y_test, y_prob),
        'y_pred':           y_pred,
        'y_prob':           y_prob,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'inference_time':   inference_time,
    }

    return results


def print_test_results(all_results):
    """
    Print a formatted comparison table of test results.

    Parameters
    ----------
    all_results : dict
        Dictionary of model_name → results dict.
    """
    header = (f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} "
              f"{'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print(header)
    print(f"  {'─' * 75}")
    for model_name, r in all_results.items():
        print(f"  {model_name:<25} {r['accuracy']:>10.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>10.4f} "
              f"{r['f1']:>10.4f} {r['roc_auc']:>10.4f}")


def print_classification_reports(all_results, y_test):
    """
    Print detailed classification reports for all models.

    Parameters
    ----------
    all_results : dict
        Dictionary of model_name → results dict.
    y_test : array-like
        True test labels.
    """
    for model_name, results in all_results.items():
        print(f"\n  ── {model_name} ──")
        report = classification_report(
            y_test, results['y_pred'],
            target_names=CLASS_NAMES,
            digits=4
        )
        for line in report.split('\n'):
            print(f"    {line}")


def compute_ranking(cv_results, test_results, weights=None):
    """
    Compute a composite ranking of all models.

    Default weights:
      40% CV F1 + 30% CV AUC + 20% Test F1 + 10% Test AUC

    Parameters
    ----------
    cv_results : dict
        model_name → CV results dict.
    test_results : dict
        model_name → test results dict.
    weights : dict, optional
        Custom ranking weights.

    Returns
    -------
    ranking : list of dict
        Sorted ranking data (best first).
    """
    if weights is None:
        from config import RANKING_WEIGHTS
        weights = RANKING_WEIGHTS

    ranking = []
    for model_name in cv_results.keys():
        cv = cv_results[model_name]
        test = test_results[model_name]

        composite = (
            weights.get('cv_f1', 0.40) * cv['f1_mean'] +
            weights.get('cv_roc_auc', 0.30) * cv['roc_auc_mean'] +
            weights.get('test_f1', 0.20) * test['f1'] +
            weights.get('test_roc_auc', 0.10) * test['roc_auc']
        )

        ranking.append({
            'Model': model_name,
            'CV_F1_Mean': cv['f1_mean'],
            'CV_F1_Std': cv['f1_std'],
            'CV_ROC_AUC': cv['roc_auc_mean'],
            'Test_F1': test['f1'],
            'Test_ROC_AUC': test['roc_auc'],
            'Test_Accuracy': test['accuracy'],
            'Composite_Score': composite,
            'Training_Time': cv.get('training_time', 0),
        })

    ranking.sort(key=lambda x: x['Composite_Score'], reverse=True)

    # Add rank numbers
    for i, entry in enumerate(ranking, 1):
        entry['Rank'] = i

    return ranking
