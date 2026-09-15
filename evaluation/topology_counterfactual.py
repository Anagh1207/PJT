"""
=============================================================================
Topology-Aware Counterfactual Engine — Core Methodological Innovation
=============================================================================
Generates counterfactual explanations for high-risk CVD patients and evaluates
the impact of clinical feature changes on patient similarity graph topology.

Three-Way Comparison Paradigm:
  1. Normal CF (Feature perturbation, no graph model — LightGBM baseline)
  2. Fixed-Graph CF (Feature perturbation, frozen graph topology — GraphSAGE)
  3. Topology-Aware CF (Feature perturbation, graph topology regenerated — GraphSAGE)

Author: AI Research Assistant
Date: September 2026
=============================================================================
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import torch

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    COUNTERFACTUAL_CONFIG, ALL_FEATURES, CONTINUOUS_FEATURES,
    BINARY_FEATURES, TARGET_COLUMN
)
from graph_construction import (
    rebuild_graph_for_patient, compute_neighborhood_stats
)
from evaluation.topology_metrics import compute_composite_topology_metrics

def generate_counterfactual_vector(
    model,
    X: np.ndarray,
    patient_idx: int,
    edge_index: torch.Tensor = None,
    edge_weight: torch.Tensor = None,
    model_type: str = 'graphsage',
    target_risk: float = 0.30,
    max_iter: int = 300,
    step_size: float = 0.05
) -> np.ndarray:
    """
    Finds a minimal, realistic counterfactual feature vector for patient_idx to achieve target risk.
    Holds immutable features (Age, Gender, Family_History) fixed.
    """
    x_orig = X[patient_idx].copy()
    x_cf = x_orig.copy()
    
    immutable = COUNTERFACTUAL_CONFIG['immutable_features']
    modifiable = COUNTERFACTUAL_CONFIG['modifiable_features']
    
    immutable_indices = [ALL_FEATURES.index(f) for f in immutable if f in ALL_FEATURES]
    modifiable_indices = [ALL_FEATURES.index(f) for f in modifiable if f in ALL_FEATURES]
    
    # Clinical direction guidelines: reduce risk factors
    # e.g., lower BP, Cholesterol, BMI, Glucose, Triglycerides, Stress, Smoking, Diabetes; increase Activity
    lower_risk_factors = ['Blood_Pressure', 'Cholesterol', 'BMI', 'Glucose', 'Triglycerides', 'Stress_Index', 'Heart_Rate', 'Smoking', 'Diabetes']
    increase_factors = ['Physical_Activity_HoursPerWeek']
    
    lower_indices = [ALL_FEATURES.index(f) for f in lower_risk_factors if f in ALL_FEATURES]
    increase_indices = [ALL_FEATURES.index(f) for f in increase_factors if f in ALL_FEATURES]
    
    pop_min = X.min(axis=0)
    pop_max = X.max(axis=0)
    
    X_tensor = torch.tensor(X, dtype=torch.float)
    
    for iteration in range(max_iter):
        # Predict current risk
        if model_type == 'graphsage':
            X_curr = torch.tensor(X, dtype=torch.float)
            X_curr[patient_idx] = torch.tensor(x_cf, dtype=torch.float)
            with torch.no_grad():
                logits = model(X_curr, edge_index, edge_weight)
                prob = torch.softmax(logits[patient_idx], dim=0)[1].item()
        else:
            # Tabular model (LightGBM / RF)
            prob = float(model.predict_proba(x_cf.reshape(1, -1))[0, 1])
            
        if prob <= target_risk:
            break
            
        # Step modifiable features in beneficial clinical direction
        for idx in lower_indices:
            if idx in modifiable_indices:
                x_cf[idx] = max(pop_min[idx], x_cf[idx] - step_size)
                
        for idx in increase_indices:
            if idx in modifiable_indices:
                x_cf[idx] = min(pop_max[idx], x_cf[idx] + step_size)
                
        # Re-enforce immutable features strictly
        for idx in immutable_indices:
            x_cf[idx] = x_orig[idx]
            
    return x_cf


def run_topology_aware_counterfactual_analysis(
    model,
    X: np.ndarray,
    y: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    patient_idx: int,
    feature_names: list = ALL_FEATURES,
    k: int = 10
) -> dict:
    """
    Executes complete topology-aware counterfactual analysis for a single patient:
      1. Original state, predictions, graph neighbors, and embeddings
      2. Fixed-Graph CF state and predictions
      3. Topology-Aware CF state, graph rebuild, new neighbors, and embedding shift
    """
    X_tensor = torch.tensor(X, dtype=torch.float)
    
    # 1. Original State
    model.eval()
    with torch.no_grad():
        orig_logits = model(X_tensor, edge_index, edge_weight)
        orig_prob = torch.softmax(orig_logits[patient_idx], dim=0)[1].item()
        orig_emb = model.get_embeddings(X_tensor, edge_index, edge_weight)[patient_idx].numpy()
        
    orig_neighborhood = compute_neighborhood_stats(edge_index, y, patient_idx, k=k)
    
    # 2. Generate Counterfactual Vector
    x_cf = generate_counterfactual_vector(
        model, X, patient_idx, edge_index, edge_weight, model_type='graphsage'
    )
    
    # 3. Fixed-Graph Prediction (Frozen Topology)
    X_mod = X.copy()
    X_mod[patient_idx] = x_cf
    X_mod_tensor = torch.tensor(X_mod, dtype=torch.float)
    
    with torch.no_grad():
        fixed_logits = model(X_mod_tensor, edge_index, edge_weight)
        fixed_prob = torch.softmax(fixed_logits[patient_idx], dim=0)[1].item()
        fixed_emb = model.get_embeddings(X_mod_tensor, edge_index, edge_weight)[patient_idx].numpy()
        
    # 4. Topology-Aware Rebuilt Graph Prediction
    edge_index_new, edge_weight_new, stats_new = rebuild_graph_for_patient(
        X, patient_idx, x_cf, k=k, feature_names=feature_names, metric='clinical'
    )
    
    with torch.no_grad():
        rebuilt_logits = model(X_mod_tensor, edge_index_new, edge_weight_new)
        rebuilt_prob = torch.softmax(rebuilt_logits[patient_idx], dim=0)[1].item()
        rebuilt_emb = model.get_embeddings(X_mod_tensor, edge_index_new, edge_weight_new)[patient_idx].numpy()
        
    new_neighborhood = compute_neighborhood_stats(edge_index_new, y, patient_idx, k=k)
    
    # 5. Compute Topology Metrics
    topo_metrics = compute_composite_topology_metrics(
        patient_idx=patient_idx,
        neighbors_orig=orig_neighborhood['neighbors'],
        neighbors_new=new_neighborhood['neighbors'],
        stats_orig=orig_neighborhood,
        stats_new=new_neighborhood,
        emb_orig=orig_emb,
        emb_new=rebuilt_emb
    )
    
    # Feature difference summary
    n_changed = int((np.abs(x_cf - X[patient_idx]) > 1e-4).sum())
    l2_dist = float(np.linalg.norm(x_cf - X[patient_idx]))
    
    result = {
        'patient_idx': patient_idx,
        'orig_risk_prob': orig_prob,
        'fixed_graph_risk_prob': fixed_prob,
        'topology_aware_risk_prob': rebuilt_prob,
        'risk_reduction_fixed': orig_prob - fixed_prob,
        'risk_reduction_topo': orig_prob - rebuilt_prob,
        'n_features_changed': n_changed,
        'l2_feature_distance': l2_dist,
        'x_orig': X[patient_idx],
        'x_cf': x_cf,
        'topology_metrics': topo_metrics
    }
    
    return result


if __name__ == "__main__":
    print("Topology-aware counterfactual module initialized.")
