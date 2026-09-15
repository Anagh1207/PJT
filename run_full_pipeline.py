"""
=============================================================================
Master Orchestrator — Full Topology-Aware Counterfactual Pipeline
=============================================================================
Executes the complete restructured three-dataset, topology-aware counterfactual
framework for CVD risk prediction using GraphSAGE.

Phases Executed:
  Phase A: Benchmark Data Prep & CTGAN Synthetic Generation
  Phase B: Synthetic Quality & ML Utility Validation
  Phase C: Patient Similarity Graph Sweep & GraphSAGE K Optimization
  Phase D: Core Topology-Aware Counterfactual Analysis (Three-Way Experiment)
  Phase E: Cross-Dataset Benchmark Validation
  Phase F: Publication Assets & Summary Report Generation

Author: AI Research Assistant
Date: September 2026
=============================================================================
"""

import os
import sys
import time
import pandas as pd
import torch

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CLEANED_DATASET, ALL_FEATURES, TARGET_COLUMN,
    RESULTS_DIR, VISUALIZATIONS_DIR, FIGURES_DIR
)
from data.benchmark_data import prepare_benchmark_dataset, inspect_benchmark_dataset
from data.synthetic_data import generate_synthetic_dataset
from evaluation.synthetic_validation import validate_synthetic_dataset
from graph_construction import build_patient_similarity_graph
from training.graphsage_pipeline import select_optimal_k_and_train_graphsage
from experiments.three_way_experiment import run_three_way_experiment
from experiments.benchmark_validation import run_benchmark_validation_experiment


def main():
    start_time = time.time()
    print("="*80)
    print("  TOPOLOGY-AWARE COUNTERFACTUAL CVD RISK PREDICTION PIPELINE")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # 1. Load Original Dataset
    # -------------------------------------------------------------------------
    print("\n[Phase 1] Loading original patient dataset...")
    if not os.path.exists(CLEANED_DATASET):
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATASET}")
    original_df = pd.read_csv(CLEANED_DATASET)
    print(f"  Original dataset shape: {original_df.shape}")
    
    # -------------------------------------------------------------------------
    # 2. Benchmark Dataset Preparation
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Preparing Framingham/Kaggle CVD Benchmark reference dataset...")
    benchmark_df = prepare_benchmark_dataset()
    bench_stats = inspect_benchmark_dataset(benchmark_df)
    print(f"  Benchmark reference population shape: {benchmark_df.shape}")
    
    # -------------------------------------------------------------------------
    # 3. Synthetic Dataset Generation (CTGAN)
    # -------------------------------------------------------------------------
    print("\n[Phase 3] Generating CTGAN synthetic patient population (N=2000)...")
    synthetic_df = generate_synthetic_dataset(benchmark_df, n_samples=2000, epochs=100)
    
    # -------------------------------------------------------------------------
    # 4. Synthetic Data Validation
    # -------------------------------------------------------------------------
    print("\n[Phase 4] Validating synthetic data statistical fidelity & ML utility...")
    syn_val_summary = validate_synthetic_dataset(benchmark_df, synthetic_df)
    
    # -------------------------------------------------------------------------
    # 5. Graph Construction & GraphSAGE Training with K Sweep
    # -------------------------------------------------------------------------
    print("\n[Phase 5] Building patient similarity graph & optimizing GraphSAGE across K sweep...")
    X_orig = original_df[ALL_FEATURES].values
    y_orig = original_df[TARGET_COLUMN].values
    
    gs_results = select_optimal_k_and_train_graphsage(X_orig, y_orig, ALL_FEATURES)
    optimal_k = gs_results['optimal_k']
    graphsage_model = gs_results['model']
    edge_index, edge_weight, graph_stats = gs_results['graph']
    
    # -------------------------------------------------------------------------
    # 6. Core Innovation: Three-Way Counterfactual Experiment
    # -------------------------------------------------------------------------
    print("\n[Phase 6] Running Three-Way Counterfactual Experiment...")
    three_way_summary = run_three_way_experiment(
        graphsage_model, X_orig, y_orig, edge_index, edge_weight,
        feature_names=ALL_FEATURES, k=optimal_k, n_patients=50
    )
    
    # -------------------------------------------------------------------------
    # 7. Cross-Dataset Validation Experiment
    # -------------------------------------------------------------------------
    print("\n[Phase 7] Running Cross-Dataset Benchmark Validation Experiment...")
    bench_val_summary = run_benchmark_validation_experiment(
        original_df, synthetic_df, benchmark_df
    )
    
    # -------------------------------------------------------------------------
    # 8. Generate Summary Execution Artifact
    # -------------------------------------------------------------------------
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"  PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed_time:.2f} SECONDS!")
    print("="*80)
    print(f"  - Benchmark reference dataset: {benchmark_df.shape[0]} patients")
    print(f"  - CTGAN synthetic dataset  : {synthetic_df.shape[0]} patients")
    print(f"  - Optimal GraphSAGE K      : K* = {optimal_k}")
    print(f"  - Results saved to         : {RESULTS_DIR}")
    print(f"  - Visualizations saved to  : {VISUALIZATIONS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
