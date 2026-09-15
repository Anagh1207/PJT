"""
=============================================================================
GNN Evaluation Pipeline
=============================================================================
Phase 4: Evaluation functions for GNN models.

Computes:
  Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC
  Confusion matrix, predictions, probabilities, embeddings

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report
)


def evaluate_gnn(model, x, edge_index, y, test_mask, edge_weight=None):
    """
    Evaluate a trained GNN on the test set.

    Parameters
    ----------
    model : BaseGNN
    x : torch.FloatTensor (N, F)
    edge_index : torch.LongTensor (2, E)
    y : torch.LongTensor (N,)
    test_mask : torch.BoolTensor (N,)
    edge_weight : torch.FloatTensor (E,), optional

    Returns
    -------
    results : dict
        Contains accuracy, precision, recall, f1, roc_auc, pr_auc, mcc,
        y_pred, y_prob, confusion_matrix, inference_time, embeddings
    """
    model.eval()

    start = time.time()
    with torch.no_grad():
        logits = model(x, edge_index, edge_weight)
        probs = F.softmax(logits, dim=1)
        embeddings = model.get_embeddings(x, edge_index, edge_weight)
    inference_time = time.time() - start

    # Extract test predictions
    y_true = y[test_mask].numpy()
    y_pred = logits[test_mask].argmax(dim=1).numpy()
    y_prob = probs[test_mask][:, 1].numpy()
    embeds = embeddings[test_mask].numpy()

    results = {
        'accuracy':         accuracy_score(y_true, y_pred),
        'precision':        precision_score(y_true, y_pred, zero_division=0),
        'recall':           recall_score(y_true, y_pred, zero_division=0),
        'f1':               f1_score(y_true, y_pred, zero_division=0),
        'roc_auc':          roc_auc_score(y_true, y_prob),
        'pr_auc':           average_precision_score(y_true, y_prob),
        'mcc':              matthews_corrcoef(y_true, y_pred),
        'y_true':           y_true,
        'y_pred':           y_pred,
        'y_prob':           y_prob,
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'inference_time':   inference_time,
        'embeddings':       embeds,
        'all_embeddings':   embeddings.numpy(),
        'all_labels':       y.numpy(),
    }

    return results


def evaluate_gnn_multi_seed(model_class, in_channels, x, edge_index, y,
                            edge_weight=None, num_seeds=5, verbose=True,
                            **train_kwargs):
    """
    Train and evaluate a GNN across multiple random seeds for
    statistical validation.

    Parameters
    ----------
    model_class : type
        GNN model class (GCN, GraphSAGE, GAT)
    in_channels : int
    x, edge_index, y : tensors
    edge_weight : optional tensor
    num_seeds : int
    train_kwargs : passed to train_gnn

    Returns
    -------
    all_results : list of dict
    summary : dict with mean/std for each metric
    """
    from training.gnn_trainer import train_gnn, create_masks

    all_results = []
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1',
                    'roc_auc', 'pr_auc', 'mcc']

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx * 10

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model_class(in_channels)
        train_mask, val_mask, test_mask = create_masks(y, random_state=seed)

        if verbose:
            print(f"\n    Seed {seed_idx + 1}/{num_seeds} (seed={seed})")

        model, history = train_gnn(
            model, x, edge_index, y, train_mask, val_mask,
            edge_weight=edge_weight, verbose=verbose, **train_kwargs
        )

        results = evaluate_gnn(model, x, edge_index, y, test_mask, edge_weight)
        results['seed'] = seed
        results['history'] = history
        all_results.append(results)

        if verbose:
            print(f"      Acc={results['accuracy']:.4f} "
                  f"F1={results['f1']:.4f} "
                  f"AUC={results['roc_auc']:.4f} "
                  f"MCC={results['mcc']:.4f}")

    # Aggregate
    summary = {}
    for key in metrics_keys:
        values = [r[key] for r in all_results]
        summary[f'{key}_mean'] = np.mean(values)
        summary[f'{key}_std'] = np.std(values)
        summary[f'{key}_values'] = values

    return all_results, summary


def print_gnn_results(all_results):
    """Print formatted comparison table for GNN models."""
    header = (f"  {'Model':<30} {'Accuracy':>10} {'Precision':>10} "
              f"{'Recall':>10} {'F1':>10} {'ROC-AUC':>10} "
              f"{'PR-AUC':>10} {'MCC':>10}")
    print(header)
    print(f"  {'─' * 100}")
    for model_name, r in all_results.items():
        if isinstance(r, dict) and 'accuracy' in r:
            print(f"  {model_name:<30} {r['accuracy']:>10.4f} "
                  f"{r['precision']:>10.4f} {r['recall']:>10.4f} "
                  f"{r['f1']:>10.4f} {r['roc_auc']:>10.4f} "
                  f"{r.get('pr_auc', 0):>10.4f} {r.get('mcc', 0):>10.4f}")
