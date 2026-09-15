"""
=============================================================================
DPSN Ablation Variants
=============================================================================
Phase 6: Model variants with specific components removed for ablation study.

Variants:
  A. No Dynamic Edges — fixed initial graph edges
  B. No Clinical Similarity — Euclidean-only initialization
  C. No Attention — GCN backbone instead of GAT
  D. No Feature Weighting — uniform feature weights

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dpsn_model import DynamicPatientSimilarityNetwork
from config import ABLATION_VARIANTS


def get_ablation_model(variant_name, in_channels):
    """
    Create a DPSN ablation variant by name.

    Parameters
    ----------
    variant_name : str
        One of the keys in ABLATION_VARIANTS config.
    in_channels : int
        Number of input features.

    Returns
    -------
    model : DynamicPatientSimilarityNetwork
    """
    if variant_name not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant_name}. "
                         f"Available: {list(ABLATION_VARIANTS.keys())}")

    flags = ABLATION_VARIANTS[variant_name]
    model = DynamicPatientSimilarityNetwork(
        in_channels=in_channels,
        use_dynamic_edges=flags['dynamic_edges'],
        use_clinical_sim=flags['clinical_sim'],
        use_attention=flags['attention'],
        use_feature_weight=flags['feature_weight'],
    )
    return model


def get_all_ablation_models(in_channels):
    """
    Create all ablation variants.

    Returns
    -------
    models : dict
        variant_name → model
    """
    models = {}
    for name in ABLATION_VARIANTS:
        models[name] = get_ablation_model(name, in_channels)
    return models
