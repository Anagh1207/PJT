"""
=============================================================================
GNN Model Definitions — GCN, GraphSAGE, GAT
=============================================================================
Phase 4: Standard GNN models for node classification using PyTorch Geometric.

All models share a common interface:
  forward(x, edge_index, edge_weight=None) → logits
  get_embeddings(x, edge_index, edge_weight=None) → node embeddings

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
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, BatchNorm

from config import (
    GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_DROPOUT, GAT_HEADS
)


class BaseGNN(nn.Module):
    """Base class for all GNN models with a shared interface."""

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Return node embeddings from the penultimate layer."""
        raise NotImplementedError

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class GCN(BaseGNN):
    """
    Graph Convolutional Network (Kipf & Welling, 2017).

    Architecture:
      Input(F) → GCNConv(Hidden) → BN → ReLU → Dropout
               → GCNConv(Embed)  → BN → ReLU → Dropout
               → Linear(2)
    """

    def __init__(self, in_channels, hidden_channels=GNN_HIDDEN_DIM,
                 embed_channels=GNN_EMBED_DIM, dropout=GNN_DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embed_channels)
        self.bn2 = BatchNorm(embed_channels)
        self.classifier = nn.Linear(embed_channels, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_weight)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h)

    def get_embeddings(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.conv2(h, edge_index, edge_weight)
        h = self.bn2(h)
        h = F.relu(h)
        return h


class GraphSAGE(BaseGNN):
    """
    GraphSAGE (Hamilton et al., 2017).

    Architecture:
      Input(F) → SAGEConv(Hidden) → BN → ReLU → Dropout
               → SAGEConv(Embed)  → BN → ReLU → Dropout
               → Linear(2)
    """

    def __init__(self, in_channels, hidden_channels=GNN_HIDDEN_DIM,
                 embed_channels=GNN_EMBED_DIM, dropout=GNN_DROPOUT):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embed_channels)
        self.bn2 = BatchNorm(embed_channels)
        self.classifier = nn.Linear(embed_channels, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h)

    def get_embeddings(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        return h


class GAT(BaseGNN):
    """
    Graph Attention Network (Velickovic et al., 2018).

    Architecture:
      Input(F) → GATConv(Hidden, heads=4) → BN → ELU → Dropout
               → GATConv(Embed, heads=1)  → BN → ELU → Dropout
               → Linear(2)
    """

    def __init__(self, in_channels, hidden_channels=GNN_HIDDEN_DIM,
                 embed_channels=GNN_EMBED_DIM, dropout=GNN_DROPOUT,
                 heads=GAT_HEADS):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads,
                             heads=heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GATConv(hidden_channels, embed_channels,
                             heads=1, concat=False, dropout=dropout)
        self.bn2 = BatchNorm(embed_channels)
        self.classifier = nn.Linear(embed_channels, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h)

    def get_embeddings(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        return h


def get_gnn_model(model_name, in_channels):
    """
    Factory function for GNN models.

    Parameters
    ----------
    model_name : str
        One of 'GCN', 'GraphSAGE', 'GAT'
    in_channels : int
        Number of input features

    Returns
    -------
    model : BaseGNN
    """
    models = {
        'GCN': GCN,
        'GraphSAGE': GraphSAGE,
        'GAT': GAT,
    }
    if model_name not in models:
        raise ValueError(f"Unknown GNN model: {model_name}. "
                         f"Available: {list(models.keys())}")
    return models[model_name](in_channels)
