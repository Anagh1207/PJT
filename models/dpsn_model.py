"""
=============================================================================
Dynamic Patient Similarity Network (DPSN)
=============================================================================
Phase 5: Novel GNN model that learns patient-to-patient similarity
dynamically during training.

Architecture:
  1. Feature Encoder: Linear → BN → ReLU → Dropout
  2. Learnable Feature Weights: softmax-normalized per-feature importance
  3. Attention-Based Edge Learning: bilinear scoring → dynamic edge weights
  4. Dynamic Edge Reweighting: MLP(|h_i - h_j| ∘ w_feat) → edge scores
  5. GNN Backbone: GAT-based message passing (2 layers)
  6. Classifier: Linear → 2 classes

Key Innovations:
  - Learnable clinical feature weighting
  - Dynamic graph topology adaptation during training
  - Attention-based edge importance scoring
  - Clinically-informed similarity

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, BatchNorm
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from config import (
    GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT, GAT_HEADS,
    DPSN_TOPK, DPSN_EDGE_HIDDEN, DPSN_TEMPERATURE
)


class SimilarityLearner(nn.Module):
    """
    Learns pairwise patient similarity from node embeddings.

    Given node embeddings h_i, h_j:
      score(i,j) = MLP(|h_i - h_j| * feature_weights)

    This produces dynamic edge weights that are updated each forward pass.
    """

    def __init__(self, embed_dim, hidden_dim=DPSN_EDGE_HIDDEN):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_src, h_dst, feature_weights=None):
        """
        Compute similarity scores for edges.

        Parameters
        ----------
        h_src : (E, D) source node embeddings
        h_dst : (E, D) destination node embeddings
        feature_weights : (D,) optional per-feature weights

        Returns
        -------
        scores : (E,) edge similarity scores in [0, 1]
        """
        diff = torch.abs(h_src - h_dst)
        if feature_weights is not None:
            diff = diff * feature_weights.unsqueeze(0)
        return self.mlp(diff).squeeze(-1)


class DynamicPatientSimilarityNetwork(nn.Module):
    """
    Dynamic Patient Similarity Network (DPSN).

    The complete architecture that dynamically adjusts graph topology
    based on learned patient similarity during training.
    """

    def __init__(self, in_channels, hidden_channels=GNN_HIDDEN_DIM,
                 embed_channels=GNN_EMBED_DIM, dropout=GNN_DROPOUT,
                 heads=GAT_HEADS, topk=DPSN_TOPK,
                 use_dynamic_edges=True, use_clinical_sim=True,
                 use_attention=True, use_feature_weight=True):
        super().__init__()

        self.topk = topk
        self.use_dynamic_edges = use_dynamic_edges
        self.use_clinical_sim = use_clinical_sim
        self.use_attention = use_attention
        self.use_feature_weight = use_feature_weight

        # ── 1. Feature Encoder ────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── 2. Learnable Feature Weights ──────────────────────────
        if use_feature_weight:
            self.feature_weights_raw = nn.Parameter(
                torch.ones(in_channels))
        else:
            self.register_buffer(
                'feature_weights_raw',
                torch.ones(in_channels) / in_channels)

        # ── 3. Similarity Learner ─────────────────────────────────
        if use_dynamic_edges:
            self.similarity_learner = SimilarityLearner(
                hidden_channels, DPSN_EDGE_HIDDEN)

        # ── 4. GNN Backbone ───────────────────────────────────────
        if use_attention:
            self.gnn1 = GATConv(hidden_channels, hidden_channels // heads,
                                heads=heads, dropout=dropout)
            self.gnn2 = GATConv(hidden_channels, embed_channels,
                                heads=1, concat=False, dropout=dropout)
        else:
            self.gnn1 = GCNConv(hidden_channels, hidden_channels)
            self.gnn2 = GCNConv(hidden_channels, embed_channels)

        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(embed_channels)
        self.dropout = dropout

        # ── 5. Classifier ─────────────────────────────────────────
        self.classifier = nn.Linear(embed_channels, 2)

        # Storage for visualization / analysis
        self._learned_edge_weights = None
        self._learned_feature_weights = None

    @property
    def feature_weights(self):
        """Normalized feature weights (softmax)."""
        if self.use_feature_weight:
            return F.softmax(self.feature_weights_raw, dim=0)
        else:
            return self.feature_weights_raw

    def _reweight_edges(self, h, edge_index):
        """
        Dynamically recompute edge weights using the similarity learner.

        Parameters
        ----------
        h : (N, D) node embeddings
        edge_index : (2, E) edge indices

        Returns
        -------
        edge_weight : (E,) learned edge weights
        """
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]
        h_dst = h[dst]

        fw = self.feature_weights if self.use_clinical_sim else None

        # Pad or truncate feature weights to match hidden dim
        if fw is not None and fw.shape[0] != h_src.shape[1]:
            # Project feature weights to hidden dim
            fw_proj = torch.ones(h_src.shape[1], device=h.device)
            fw_proj[:min(fw.shape[0], h_src.shape[1])] = \
                fw[:min(fw.shape[0], h_src.shape[1])]
            fw = fw_proj

        scores = self.similarity_learner(h_src, h_dst, fw)

        # Apply temperature scaling
        scores = scores / DPSN_TEMPERATURE

        self._learned_edge_weights = scores.detach()
        return scores

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass with dynamic edge reweighting.

        Parameters
        ----------
        x : (N, F) node features
        edge_index : (2, E) edge indices
        edge_weight : (E,) initial edge weights (optional)

        Returns
        -------
        logits : (N, 2)
        """
        # Save feature weights for analysis
        self._learned_feature_weights = self.feature_weights.detach()

        # 1. Feature encoding
        h = self.encoder(x)

        # 2. Dynamic edge reweighting
        if self.use_dynamic_edges and hasattr(self, 'similarity_learner'):
            edge_weight = self._reweight_edges(h, edge_index)

        # 3. GNN message passing
        if self.use_attention:
            h = self.gnn1(h, edge_index)
        else:
            h = self.gnn1(h, edge_index, edge_weight)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        if self.use_attention:
            h = self.gnn2(h, edge_index)
        else:
            h = self.gnn2(h, edge_index, edge_weight)
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # 4. Classification
        return self.classifier(h)

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Return node embeddings from the penultimate layer."""
        h = self.encoder(x)

        if self.use_dynamic_edges and hasattr(self, 'similarity_learner'):
            edge_weight = self._reweight_edges(h, edge_index)

        if self.use_attention:
            h = self.gnn1(h, edge_index)
        else:
            h = self.gnn1(h, edge_index, edge_weight)
        h = self.bn1(h)
        h = F.elu(h)

        if self.use_attention:
            h = self.gnn2(h, edge_index)
        else:
            h = self.gnn2(h, edge_index, edge_weight)
        h = self.bn2(h)
        h = F.elu(h)
        return h

    def get_learned_feature_weights(self):
        """Return the current learned feature importance weights."""
        return self._learned_feature_weights

    def get_learned_edge_weights(self):
        """Return the most recently computed edge weights."""
        return self._learned_edge_weights
