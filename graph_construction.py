"""
=============================================================================
Graph Construction Module — Patient Similarity Graphs
=============================================================================
Cardiovascular Disease Risk Prediction using Graph Neural Networks
IEEE Conference Paper — Phase 3: Graph Construction

Implements 4 graph generation strategies:
  1. Euclidean KNN Graph
  2. Cosine Similarity Graph
  3. Clinical Similarity Graph (domain-weighted)
  4. Dynamic Patient Similarity Graph (initial topology)

Each function returns:
  - edge_index  (2 × E, LongTensor)
  - edge_weight (E, FloatTensor)
  - stats       (dict of graph statistics)

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    KNN_K, COSINE_THRESHOLD, CLINICAL_THRESHOLD,
    CLINICAL_WEIGHTS, ALL_FEATURES, RANDOM_STATE
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _compute_graph_statistics(edge_index, edge_weight, num_nodes):
    """
    Compute comprehensive graph statistics.

    Parameters
    ----------
    edge_index : torch.LongTensor, shape (2, E)
    edge_weight : torch.FloatTensor, shape (E,)
    num_nodes : int

    Returns
    -------
    stats : dict
    """
    num_edges = edge_index.shape[1]

    # Degree distribution (undirected — count each direction)
    src = edge_index[0].numpy()
    degrees = np.bincount(src, minlength=num_nodes)

    # Connected components via sparse adjacency
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.ones(len(row))
    adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    n_components, labels = connected_components(adj, directed=False)

    # Density
    max_edges = num_nodes * (num_nodes - 1)  # directed
    density = num_edges / max_edges if max_edges > 0 else 0

    # Edge weight statistics
    ew = edge_weight.numpy()

    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges // 2,  # undirected edge count
        'num_directed_edges': num_edges,
        'density': density,
        'avg_degree': degrees.mean(),
        'std_degree': degrees.std(),
        'min_degree': int(degrees.min()),
        'max_degree': int(degrees.max()),
        'median_degree': float(np.median(degrees)),
        'num_connected_components': n_components,
        'largest_component_size': int(Counter(labels).most_common(1)[0][1]),
        'edge_weight_mean': float(ew.mean()),
        'edge_weight_std': float(ew.std()),
        'edge_weight_min': float(ew.min()),
        'edge_weight_max': float(ew.max()),
        'degree_distribution': degrees.tolist(),
    }
    return stats


def _make_symmetric(edge_index, edge_weight):
    """Ensure graph is undirected by adding reverse edges."""
    rev_edge_index = edge_index.flip(0)
    edge_index_sym = torch.cat([edge_index, rev_edge_index], dim=1)
    edge_weight_sym = torch.cat([edge_weight, edge_weight], dim=0)

    # Remove duplicates
    edge_set = set()
    unique_src, unique_dst, unique_w = [], [], []
    for i in range(edge_index_sym.shape[1]):
        s, d = int(edge_index_sym[0, i]), int(edge_index_sym[1, i])
        if (s, d) not in edge_set:
            edge_set.add((s, d))
            unique_src.append(s)
            unique_dst.append(d)
            unique_w.append(float(edge_weight_sym[i]))

    ei = torch.tensor([unique_src, unique_dst], dtype=torch.long)
    ew = torch.tensor(unique_w, dtype=torch.float)
    return ei, ew


# ═══════════════════════════════════════════════════════════════════════════
# 1. EUCLIDEAN KNN GRAPH
# ═══════════════════════════════════════════════════════════════════════════

def build_euclidean_knn_graph(X, k=KNN_K):
    """
    Build a KNN graph using Euclidean distance.

    Each patient is connected to its K nearest neighbours.
    Edge weight = 1 / (1 + distance).

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
        Feature matrix (standardized).
    k : int
        Number of nearest neighbours.

    Returns
    -------
    edge_index : torch.LongTensor (2, E)
    edge_weight : torch.FloatTensor (E,)
    stats : dict
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    src_list, dst_list, weight_list = [], [], []
    for i in range(X.shape[0]):
        for j_idx in range(1, k + 1):  # skip self (index 0)
            neighbor = indices[i, j_idx]
            dist = distances[i, j_idx]
            src_list.append(i)
            dst_list.append(neighbor)
            weight_list.append(1.0 / (1.0 + dist))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float)

    # Make symmetric
    edge_index, edge_weight = _make_symmetric(edge_index, edge_weight)

    stats = _compute_graph_statistics(edge_index, edge_weight, X.shape[0])
    stats['graph_type'] = 'Euclidean KNN'
    stats['k'] = k

    return edge_index, edge_weight, stats


# ═══════════════════════════════════════════════════════════════════════════
# 2. COSINE SIMILARITY GRAPH
# ═══════════════════════════════════════════════════════════════════════════

def build_cosine_similarity_graph(X, threshold=COSINE_THRESHOLD):
    """
    Build a graph from cosine similarity with thresholding.

    Patients with cosine similarity >= threshold are connected.
    Edge weight = cosine similarity score.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    threshold : float
        Minimum cosine similarity to form an edge.

    Returns
    -------
    edge_index, edge_weight, stats
    """
    sim_matrix = cosine_similarity(X)
    np.fill_diagonal(sim_matrix, 0)  # no self-loops

    # Adaptive threshold: lower if no edges found
    current_threshold = threshold
    rows, cols = np.where(sim_matrix >= current_threshold)
    while len(rows) == 0 and current_threshold > 0.1:
        current_threshold -= 0.05
        rows, cols = np.where(sim_matrix >= current_threshold)

    if len(rows) == 0:
        # Fallback: use top-K per node
        k = min(10, X.shape[0] - 1)
        src_list, dst_list, weight_list = [], [], []
        for i in range(X.shape[0]):
            top_k = np.argsort(sim_matrix[i])[-k:]
            for j in top_k:
                if sim_matrix[i, j] > 0:
                    src_list.append(i)
                    dst_list.append(j)
                    weight_list.append(sim_matrix[i, j])
        rows = np.array(src_list)
        cols = np.array(dst_list)
        weights = np.array(weight_list)
        print(f"    [INFO] Cosine graph: threshold too strict, using top-{k} fallback")
    else:
        weights = sim_matrix[rows, cols]
        if current_threshold != threshold:
            print(f"    [INFO] Cosine graph: lowered threshold {threshold:.2f} → {current_threshold:.2f}")

    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)

    stats = _compute_graph_statistics(edge_index, edge_weight, X.shape[0])
    stats['graph_type'] = 'Cosine Similarity'
    stats['threshold'] = current_threshold

    return edge_index, edge_weight, stats


# ═══════════════════════════════════════════════════════════════════════════
# 3. CLINICAL SIMILARITY GRAPH
# ═══════════════════════════════════════════════════════════════════════════

def build_clinical_similarity_graph(X, feature_names, threshold=CLINICAL_THRESHOLD):
    """
    Build a graph using domain-weighted clinical risk factors.

    Distance = weighted Euclidean distance where weights come from
    clinical domain knowledge (e.g., cholesterol and BP weighted higher).
    Similarity = 1 / (1 + weighted_distance). Edge if similarity >= threshold.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    feature_names : list of str
    threshold : float

    Returns
    -------
    edge_index, edge_weight, stats
    """
    # Build weight vector aligned with feature order
    weight_vector = np.array([
        CLINICAL_WEIGHTS.get(f, 1.0) for f in feature_names
    ])
    weight_vector = weight_vector / weight_vector.sum()  # normalize

    # Weighted features
    X_weighted = X * np.sqrt(weight_vector)

    # Pairwise Euclidean distance on weighted features
    dist_matrix = cdist(X_weighted, X_weighted, metric='euclidean')

    # Convert to similarity
    sim_matrix = 1.0 / (1.0 + dist_matrix)
    np.fill_diagonal(sim_matrix, 0)

    # Adaptive threshold: lower if no edges found
    current_threshold = threshold
    rows, cols = np.where(sim_matrix >= current_threshold)
    while len(rows) == 0 and current_threshold > 0.1:
        current_threshold -= 0.05
        rows, cols = np.where(sim_matrix >= current_threshold)
    
    if len(rows) == 0:
        # Fallback: use top-K per node
        k = min(10, X.shape[0] - 1)
        src_list, dst_list, weight_list = [], [], []
        for i in range(X.shape[0]):
            top_k = np.argsort(sim_matrix[i])[-k:]
            for j in top_k:
                if sim_matrix[i, j] > 0:
                    src_list.append(i)
                    dst_list.append(j)
                    weight_list.append(sim_matrix[i, j])
        rows = np.array(src_list)
        cols = np.array(dst_list)
        weights = np.array(weight_list)
        print(f"    [INFO] Clinical graph: threshold too strict, using top-{k} fallback")
    else:
        weights = sim_matrix[rows, cols]
        if current_threshold != threshold:
            print(f"    [INFO] Clinical graph: lowered threshold {threshold:.2f} → {current_threshold:.2f}")

    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)

    stats = _compute_graph_statistics(edge_index, edge_weight, X.shape[0])
    stats['graph_type'] = 'Clinical Similarity'
    stats['threshold'] = current_threshold

    stats['clinical_weights'] = {f: CLINICAL_WEIGHTS.get(f, 1.0) for f in feature_names}

    return edge_index, edge_weight, stats


# ═══════════════════════════════════════════════════════════════════════════
# 4. DYNAMIC PATIENT SIMILARITY GRAPH (Initial Topology)
# ═══════════════════════════════════════════════════════════════════════════

def build_dynamic_patient_similarity_graph(X, feature_names, k=None):
    """
    Build the initial topology for the Dynamic Patient Similarity Network.

    Combines clinical-weighted similarity with adaptive KNN:
      1. Compute clinical-weighted similarity
      2. For each patient, keep top-K most similar neighbours
      3. Edge weight = combined similarity score

    This initial graph will be refined during DPSN training (Phase 5).

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    feature_names : list of str
    k : int, optional
        Top-K neighbours per node. Defaults to DPSN_TOPK from config.

    Returns
    -------
    edge_index, edge_weight, stats
    """
    from config import DPSN_TOPK
    if k is None:
        k = DPSN_TOPK

    # Clinical-weighted distance
    weight_vector = np.array([
        CLINICAL_WEIGHTS.get(f, 1.0) for f in feature_names
    ])
    weight_vector = weight_vector / weight_vector.sum()
    X_weighted = X * np.sqrt(weight_vector)

    # Also incorporate cosine similarity component
    cos_sim = cosine_similarity(X)
    euc_dist = cdist(X_weighted, X_weighted, metric='euclidean')
    euc_sim = 1.0 / (1.0 + euc_dist)

    # Combined similarity: 60% clinical-weighted + 40% cosine
    combined_sim = 0.6 * euc_sim + 0.4 * cos_sim
    np.fill_diagonal(combined_sim, 0)

    # Top-K per node
    src_list, dst_list, weight_list = [], [], []
    for i in range(X.shape[0]):
        top_k_indices = np.argsort(combined_sim[i])[-k:]
        for j in top_k_indices:
            if combined_sim[i, j] > 0:
                src_list.append(i)
                dst_list.append(j)
                weight_list.append(combined_sim[i, j])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float)

    # Make symmetric
    edge_index, edge_weight = _make_symmetric(edge_index, edge_weight)

    stats = _compute_graph_statistics(edge_index, edge_weight, X.shape[0])
    stats['graph_type'] = 'Dynamic PSG'
    stats['k'] = k

    return edge_index, edge_weight, stats


# ═══════════════════════════════════════════════════════════════════════════
# BUILD ALL GRAPHS
# ═══════════════════════════════════════════════════════════════════════════

def build_all_graphs(X, feature_names):
    """
    Build all 4 graph types and return them in a dictionary.

    Parameters
    ----------
    X : np.ndarray
    feature_names : list of str

    Returns
    -------
    graphs : dict
        graph_name → (edge_index, edge_weight, stats)
    """
    print("\n  Building graphs...")

    print("    [1/4] Euclidean KNN Graph...")
    g1 = build_euclidean_knn_graph(X)

    print("    [2/4] Cosine Similarity Graph...")
    g2 = build_cosine_similarity_graph(X)

    print("    [3/4] Clinical Similarity Graph...")
    g3 = build_clinical_similarity_graph(X, feature_names)

    print("    [4/4] Dynamic Patient Similarity Graph...")
    g4 = build_dynamic_patient_similarity_graph(X, feature_names)

    graphs = {
        'Euclidean KNN':       g1,
        'Cosine Similarity':   g2,
        'Clinical Similarity': g3,
        'Dynamic PSG':         g4,
    }

    return graphs


def save_graph(edge_index, edge_weight, stats, filepath):
    """Save a graph to disk as a .pt file."""
    torch.save({
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'stats': stats,
    }, filepath)


def load_graph(filepath):
    """Load a graph from a .pt file."""
    data = torch.load(filepath, weights_only=False)
    return data['edge_index'], data['edge_weight'], data['stats']


# ═══════════════════════════════════════════════════════════════════════════
# CANONICAL GRAPH BUILDER & REBUILDER (FOR COUNTERFACTUAL TOPOLOGY ANALYSIS)
# ═══════════════════════════════════════════════════════════════════════════

def build_patient_similarity_graph(X, k=10, feature_names=None, metric='clinical'):
    """
    Canonical patient similarity graph construction supporting K parameter sweeps.
    Used uniformly across GraphSAGE training, evaluation, and counterfactual experiments.
    """
    if feature_names is None:
        feature_names = ALL_FEATURES

    if metric == 'clinical':
        return build_dynamic_patient_similarity_graph(X, feature_names, k=k)
    elif metric == 'euclidean':
        return build_euclidean_knn_graph(X, k=k)
    else:
        return build_cosine_similarity_graph(X)


def rebuild_graph_for_patient(X, patient_idx, modified_patient_vector, k=10, feature_names=None, metric='clinical'):
    """
    Rebuilds the entire patient similarity graph after replacing patient_idx's features
    with modified_patient_vector (the counterfactual state).

    Parameters
    ----------
    X : np.ndarray (N, F)
    patient_idx : int
    modified_patient_vector : np.ndarray (F,)
    k : int
    feature_names : list
    metric : str

    Returns
    -------
    edge_index, edge_weight, stats
    """
    X_mod = X.copy()
    X_mod[patient_idx] = modified_patient_vector
    return build_patient_similarity_graph(X_mod, k=k, feature_names=feature_names, metric=metric)


def compute_neighborhood_stats(edge_index, y, node_idx, k=None):
    """
    Extracts neighbors and computes CVD-risk composition of node_idx's neighborhood.

    Parameters
    ----------
    edge_index : torch.LongTensor (2, E)
    y : np.ndarray or torch.Tensor (N,) - Class labels (0 or 1)
    node_idx : int
    k : int, optional

    Returns
    -------
    stats : dict
        - neighbors: list of neighbor node IDs
        - n_neighbors: int
        - cvd_count: int
        - non_cvd_count: int
        - cvd_risk_ratio: float (% of neighbors with CVD)
    """
    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()
    else:
        y_np = np.array(y)

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    # Find neighbors where src == node_idx
    neighbors = dst[src == node_idx].tolist()
    unique_neighbors = sorted(list(set(neighbors)))

    if len(unique_neighbors) == 0:
        return {
            'neighbors': [],
            'n_neighbors': 0,
            'cvd_count': 0,
            'non_cvd_count': 0,
            'cvd_risk_ratio': 0.0
        }

    neighbor_labels = y_np[unique_neighbors]
    cvd_count = int((neighbor_labels == 1).sum())
    non_cvd_count = int((neighbor_labels == 0).sum())
    cvd_risk_ratio = float(cvd_count / len(unique_neighbors))

    return {
        'neighbors': unique_neighbors,
        'n_neighbors': len(unique_neighbors),
        'cvd_count': cvd_count,
        'non_cvd_count': non_cvd_count,
        'cvd_risk_ratio': cvd_risk_ratio
    }

