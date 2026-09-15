"""
=============================================================================
GraphSAGE Training Pipeline — K-Value Sweep & Model Optimization
=============================================================================
Trains GraphSAGE on patient similarity graphs across K values (K ∈ {5, 10, 15, 20}),
selects optimal K based on validation F1/ROC-AUC, and exports trained model.

Author: AI Research Assistant
Date: September 2026
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    K_VALUES, GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_LEARNING_RATE,
    GNN_WEIGHT_DECAY, GNN_EPOCHS, GNN_PATIENCE, GNN_TRAIN_RATIO,
    GNN_VAL_RATIO, GNN_TEST_RATIO, RANDOM_STATE, MODEL_DIR
)
from models.gnn_models import GraphSAGE
from graph_construction import build_patient_similarity_graph


def train_single_graphsage(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    epochs: int = GNN_EPOCHS,
    lr: float = GNN_LEARNING_RATE,
    weight_decay: float = GNN_WEIGHT_DECAY,
    patience: int = GNN_PATIENCE,
    seed: int = RANDOM_STATE
):
    """Trains a GraphSAGE model on a given graph structure."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    in_channels = X_tensor.shape[1]
    model = GraphSAGE(in_channels=in_channels, hidden_channels=GNN_HIDDEN_DIM, embed_channels=GNN_EMBED_DIM)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Calculate class weights for imbalanced cross entropy
    n_pos = (y_tensor[train_mask] == 1).sum().item()
    n_neg = (y_tensor[train_mask] == 0).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float, device=X_tensor.device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, float(pos_weight.item())], device=X_tensor.device))
    
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X_tensor, edge_index, edge_weight)
        loss = criterion(logits[train_mask], y_tensor[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_tensor, edge_index, edge_weight)[val_mask]
            val_loss = criterion(val_logits, y_tensor[val_mask]).item()
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            val_f1 = f1_score(y_tensor[val_mask].cpu().numpy(), val_preds, zero_division=0)
            
            history['train_loss'].append(float(loss.item()))
            history['val_loss'].append(float(val_loss))
            history['val_f1'].append(float(val_f1))
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Evaluate final model on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(X_tensor, edge_index, edge_weight)[test_mask]
        test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)
        y_test_np = y_tensor[test_mask].cpu().numpy()
        
        test_metrics = {
            'accuracy': float(accuracy_score(y_test_np, test_preds)),
            'precision': float(precision_score(y_test_np, test_preds, zero_division=0)),
            'recall': float(recall_score(y_test_np, test_preds, zero_division=0)),
            'f1_score': float(f1_score(y_test_np, test_preds, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test_np, test_probs)) if len(np.unique(y_test_np)) > 1 else 0.5,
            'best_val_f1': float(best_val_f1),
            'epochs_trained': epoch,
            'history': history
        }
        
    return model, test_metrics


def select_optimal_k_and_train_graphsage(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    k_values: list = K_VALUES,
    seed: int = RANDOM_STATE
):
    """
    Sweeps K ∈ {5, 10, 15, 20} for patient similarity graph construction,
    trains GraphSAGE for each K, and selects the optimal K.
    """
    print("\n" + "="*70)
    print("  GRAPHSAGE TRAINING & K-VALUE OPTIMIZATION SWEEP")
    print("="*70)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    # Stratified 60 / 20 / 20 split
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y, test_size=(GNN_VAL_RATIO + GNN_TEST_RATIO), stratify=y, random_state=seed
    )
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    
    train_mask = torch.zeros(n_samples, dtype=torch.bool)
    val_mask = torch.zeros(n_samples, dtype=torch.bool)
    test_mask = torch.zeros(n_samples, dtype=torch.bool)
    
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    
    X_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    sweep_results = []
    best_k = k_values[0]
    best_val_f1 = -1.0
    best_model = None
    best_graph = None
    
    for k in k_values:
        print(f"\n[GraphSAGE Sweep] Evaluating K = {k}...")
        edge_index, edge_weight, stats = build_patient_similarity_graph(
            X, k=k, feature_names=feature_names, metric='clinical'
        )
        
        model, metrics = train_single_graphsage(
            X_tensor, y_tensor, edge_index, edge_weight,
            train_mask, val_mask, test_mask, seed=seed
        )
        
        metrics['k'] = k
        metrics['num_edges'] = stats['num_edges']
        sweep_results.append(metrics)
        
        print(f"  K = {k:2d} | Val F1: {metrics['best_val_f1']:.4f} | Test F1: {metrics['f1_score']:.4f} | Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
        if metrics['best_val_f1'] > best_val_f1:
            best_val_f1 = metrics['best_val_f1']
            best_k = k
            best_model = model
            best_graph = (edge_index, edge_weight, stats)
            
    sweep_df = pd.DataFrame(sweep_results)
    print(f"\n[Optimal K Selected]: K* = {best_k} (Validation F1 = {best_val_f1:.4f})")
    
    # Save optimal model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_DIR, "graphsage_optimal.pt")
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'optimal_k': best_k,
        'feature_names': feature_names,
        'sweep_df': sweep_df
    }, model_save_path)
    print(f"[GraphSAGE] Saved optimal model checkpoint to: {model_save_path}")
    
    return {
        'optimal_k': best_k,
        'model': best_model,
        'graph': best_graph,
        'sweep_df': sweep_df,
        'masks': (train_mask, val_mask, test_mask),
        'tensors': (X_tensor, y_tensor)
    }


if __name__ == "__main__":
    from data.benchmark_data import prepare_benchmark_dataset
    df = prepare_benchmark_dataset()
    from config import ALL_FEATURES, TARGET_COLUMN
    X = df[ALL_FEATURES].values
    y = df[TARGET_COLUMN].values
    results = select_optimal_k_and_train_graphsage(X, y, ALL_FEATURES)
