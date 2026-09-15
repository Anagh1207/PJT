"""
=============================================================================
GNN Training Pipeline
=============================================================================
Phase 4: Training loop for GNN models with:
  - Stratified train/val/test masks
  - Class-weighted loss
  - Early stopping
  - Learning rate scheduling
  - Per-epoch metric logging
  - Model checkpointing

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import sys
import time
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

from config import (
    RANDOM_STATE, GNN_LEARNING_RATE, GNN_WEIGHT_DECAY,
    GNN_EPOCHS, GNN_PATIENCE, GNN_TRAIN_RATIO, GNN_VAL_RATIO,
    GNN_TEST_RATIO, GNN_DIR
)


def create_masks(y, train_ratio=GNN_TRAIN_RATIO, val_ratio=GNN_VAL_RATIO,
                 random_state=RANDOM_STATE):
    """
    Create stratified train/val/test boolean masks.

    Parameters
    ----------
    y : np.ndarray or torch.Tensor
        Target labels (N,)
    train_ratio : float
    val_ratio : float
    random_state : int

    Returns
    -------
    train_mask, val_mask, test_mask : torch.BoolTensor (N,)
    """
    if isinstance(y, torch.Tensor):
        y_np = y.numpy()
    else:
        y_np = np.array(y)

    n = len(y_np)
    indices = np.arange(n)

    # First split: train vs (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - train_ratio,
                                  random_state=random_state)
    train_idx, rest_idx = next(sss1.split(indices, y_np))

    # Second split: val vs test from the rest
    val_fraction = val_ratio / (val_ratio + (1.0 - train_ratio - val_ratio))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_fraction,
                                  random_state=random_state)
    val_idx_local, test_idx_local = next(sss2.split(rest_idx, y_np[rest_idx]))
    val_idx = rest_idx[val_idx_local]
    test_idx = rest_idx[test_idx_local]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def compute_class_weights(y, mask=None):
    """
    Compute inverse-frequency class weights for balanced loss.

    Parameters
    ----------
    y : torch.Tensor
    mask : torch.BoolTensor, optional

    Returns
    -------
    weights : torch.FloatTensor of shape (num_classes,)
    """
    if mask is not None:
        labels = y[mask]
    else:
        labels = y

    classes, counts = torch.unique(labels, return_counts=True)
    total = counts.sum().float()
    weights = total / (len(classes) * counts.float())
    return weights


def train_gnn(model, x, edge_index, y, train_mask, val_mask,
              edge_weight=None, epochs=GNN_EPOCHS, lr=GNN_LEARNING_RATE,
              weight_decay=GNN_WEIGHT_DECAY, patience=GNN_PATIENCE,
              verbose=True):
    """
    Train a GNN model with early stopping.

    Parameters
    ----------
    model : BaseGNN
    x : torch.FloatTensor (N, F)
    edge_index : torch.LongTensor (2, E)
    y : torch.LongTensor (N,)
    train_mask : torch.BoolTensor (N,)
    val_mask : torch.BoolTensor (N,)
    edge_weight : torch.FloatTensor (E,), optional
    epochs : int
    lr : float
    weight_decay : float
    patience : int
    verbose : bool

    Returns
    -------
    model : trained model (best state)
    history : dict with per-epoch metrics
    """
    # Class weights
    class_weights = compute_class_weights(y, train_mask)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'lr': [],
    }

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index, edge_weight)

        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # ── Evaluation ────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, edge_weight)

            train_loss = criterion(out[train_mask], y[train_mask]).item()
            val_loss = criterion(out[val_mask], y[val_mask]).item()

            train_pred = out[train_mask].argmax(dim=1)
            val_pred = out[val_mask].argmax(dim=1)

            train_acc = (train_pred == y[train_mask]).float().mean().item()
            val_acc = (val_pred == y[val_mask]).float().mean().item()

            # F1 score (manual computation to avoid sklearn import overhead)
            def _f1(pred, true):
                tp = ((pred == 1) & (true == 1)).sum().float()
                fp = ((pred == 1) & (true == 0)).sum().float()
                fn = ((pred == 0) & (true == 1)).sum().float()
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                return (2 * prec * rec / (prec + rec + 1e-8)).item()

            train_f1 = _f1(train_pred, y[train_mask])
            val_f1 = _f1(val_pred, y[val_mask])

        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)

        scheduler.step(val_loss)

        # ── Early Stopping ────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"    Epoch {epoch:>4d} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.6f}")

        if patience_counter >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch} "
                      f"(patience={patience})")
            break

    elapsed = time.time() - start_time
    history['training_time'] = elapsed
    history['best_epoch'] = epoch - patience_counter
    history['total_epochs'] = epoch

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if verbose:
        print(f"    Training complete: {elapsed:.2f}s, "
              f"Best epoch: {history['best_epoch']}")

    return model, history


def save_gnn_model(model, filepath):
    """Save a GNN model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_gnn_model(model, filepath):
    """Load a GNN model checkpoint."""
    model.load_state_dict(torch.load(filepath, weights_only=True))
    return model
