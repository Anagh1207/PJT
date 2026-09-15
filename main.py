"""
=============================================================================
Main Entry Point — ML Experimentation Framework
=============================================================================
Cardiovascular Disease Risk Prediction using Graph Neural Networks
IEEE Conference Paper — Complete Baseline Evaluation Pipeline

This script orchestrates the entire experimentation workflow:
  1. Data loading and preparation
  2. Stratified train-test split
  3. Model training with Stratified 5-Fold CV + SMOTE
  4. Final model training and test evaluation
  5. Publication-quality visualizations
  6. Experiment tracking (CSV + JSON)
  7. Model serialization
  8. Ranking and comparison

Usage:
    python main.py

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import sys
import io
import warnings

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

# Project modules
from config import (
    RANDOM_STATE, TEST_SIZE, N_FOLDS, PROCESSED_DATASET,
    CLEANED_DATASET, MODEL_DIR, RESULTS_DIR, VISUALIZATIONS_DIR,
    MODEL_FILENAMES, TARGET_COLUMN
)
from utils.data_utils import load_dataset, stratified_split, print_split_summary
from utils.logger import ExperimentLogger, print_section_header, print_subsection_header
from models.baseline_models import get_all_models
from training.trainer import cross_validate_with_smote, train_final_model, save_model
from evaluation.evaluator import evaluate_model, print_test_results, print_classification_reports, compute_ranking
from visualizations.visualizer import generate_all_visualizations
from dashboard.dashboard_utils import DashboardData

# Reproducibility
np.random.seed(RANDOM_STATE)


def main():
    """
    Main execution pipeline.

    Runs the complete ML experimentation framework from data loading
    through final ranking and visualization generation.
    """

    print("=" * 80)
    print("  CARDIOVASCULAR DISEASE RISK PREDICTION")
    print("  ML Experimentation Framework — Baseline Evaluation")
    print("  IEEE Conference Paper")
    print("=" * 80)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: DATA LOADING
    # ══════════════════════════════════════════════════════════════════
    print_section_header("STEP 1: DATA LOADING")

    X, y, feature_names = load_dataset(PROCESSED_DATASET, TARGET_COLUMN)

    print(f"  Dataset:  {PROCESSED_DATASET}")
    print(f"  Samples:  {X.shape[0]}")
    print(f"  Features: {X.shape[1]} → {feature_names}")
    print(f"  Target:   Class 0 = {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%), "
          f"Class 1 = {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")

    # Load cleaned dataset (unscaled) for EDA visualizations
    df_clean = pd.read_csv(CLEANED_DATASET)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: STRATIFIED TRAIN-TEST SPLIT
    # ══════════════════════════════════════════════════════════════════
    print_section_header("STEP 2: STRATIFIED TRAIN-TEST SPLIT (80-20)")

    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print_split_summary(y_train, y_test)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: MODEL DEFINITIONS
    # ══════════════════════════════════════════════════════════════════
    print_section_header("STEP 3: MODEL DEFINITIONS")

    models = get_all_models()
    print(f"\n  {len(models)} models available:")
    for name, (model, fname) in models.items():
        print(f"    • {name:<25} → {fname}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: STRATIFIED 5-FOLD CROSS-VALIDATION WITH SMOTE
    # ══════════════════════════════════════════════════════════════════
    print_section_header(f"STEP 4: STRATIFIED {N_FOLDS}-FOLD CV WITH SMOTE")

    print(f"\n  Configuration:")
    print(f"    Folds:          {N_FOLDS}")
    print(f"    SMOTE:          Applied within training folds only")
    print(f"    Stratification: Preserves class ratio in each fold")
    print(f"    Random State:   {RANDOM_STATE}")

    cv_results = {}
    for model_name, (model, _) in models.items():
        print_subsection_header(f"Training: {model_name}")
        cv = cross_validate_with_smote(model, X_train, y_train,
                                        n_folds=N_FOLDS, verbose=True)
        cv_results[model_name] = cv

        print(f"\n  CV Summary:")
        print(f"    Accuracy:  {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}")
        print(f"    Precision: {cv['precision_mean']:.4f} ± {cv['precision_std']:.4f}")
        print(f"    Recall:    {cv['recall_mean']:.4f} ± {cv['recall_std']:.4f}")
        print(f"    F1 Score:  {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}")
        print(f"    ROC-AUC:   {cv['roc_auc_mean']:.4f} ± {cv['roc_auc_std']:.4f}")
        print(f"    Time:      {cv['training_time']:.2f}s")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: FINAL MODEL TRAINING
    # ══════════════════════════════════════════════════════════════════
    print_section_header("STEP 5: FINAL MODEL TRAINING (FULL TRAINING SET + SMOTE)")

    trained_models = {}
    training_times = {}

    for model_name, (model, _) in models.items():
        fitted, t_time = train_final_model(model, X_train, y_train,
                                            apply_smote=True)
        trained_models[model_name] = fitted
        training_times[model_name] = t_time
        print(f"  [OK] {model_name} — {t_time:.3f}s")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: TEST SET EVALUATION
    # ══════════════════════════════════════════════════════════════════
    print_section_header("STEP 6: TEST SET EVALUATION")

    test_results = {}
    inference_times = {}

    for model_name, model in trained_models.items():
        results = evaluate_model(model, X_test, y_test)
        test_results[model_name] = results
        inference_times[model_name] = results['inference_time']

    print()
    print_test_results(test_results)

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════
    print_section_header("STEP 7: GENERATING VISUALIZATIONS")

    generate_all_visualizations(
        df_clean=df_clean,
        y=y,
        cv_results=cv_results,
        test_results=test_results,
        trained_models=trained_models,
        feature_names=feature_names,
        y_test=y_test,
        save_dir=VISUALIZATIONS_DIR
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 8: EXPERIMENT TRACKING
    # ══════════════════════════════════════════════════════════════════
    print_section_header("STEP 8: EXPERIMENT TRACKING")

    logger = ExperimentLogger(RESULTS_DIR)

    for model_name in models.keys():
        from config import MODEL_PARAMS
        cv = cv_results[model_name]
        test = test_results[model_name]

        # Build flat CV metrics for logging
        cv_metrics_flat = {
            'accuracy_mean': cv['accuracy_mean'],
            'accuracy_std': cv['accuracy_std'],
            'precision_mean': cv['precision_mean'],
            'precision_std': cv['precision_std'],
            'recall_mean': cv['recall_mean'],
            'recall_std': cv['recall_std'],
            'f1_mean': cv['f1_mean'],
            'f1_std': cv['f1_std'],
            'roc_auc_mean': cv['roc_auc_mean'],
            'roc_auc_std': cv['roc_auc_std'],
        }

        test_metrics_flat = {
            'accuracy': test['accuracy'],
            'precision': test['precision'],
            'recall': test['recall'],
            'f1': test['f1'],
            'roc_auc': test['roc_auc'],
        }

        logger.log_experiment(
            model_name=model_name,
            hyperparameters=MODEL_PARAMS.get(model_name, {}),
            cv_metrics=cv_metrics_flat,
            test_metrics=test_metrics_flat,
            training_time=training_times.get(model_name, cv['training_time']),
            inference_time=inference_times.get(model_name, 0)
        )

    logger.save()

    # ══════════════════════════════════════════════════════════════════
    # STEP 9: SAVE TRAINED MODELS
    # ══════════════════════════════════════════════════════════════════
    print_section_header("STEP 9: SAVING TRAINED MODELS")

    for model_name, model in trained_models.items():
        fname = MODEL_FILENAMES.get(model_name,
                                    f"{model_name.lower().replace(' ', '_')}.pkl")
        fpath = os.path.join(MODEL_DIR, fname)
        save_model(model, fpath)
        print(f"  [OK] {model_name} → {fpath}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 10: FINAL RANKING
    # ══════════════════════════════════════════════════════════════════
    print_section_header("STEP 10: FINAL MODEL RANKING")

    ranking = compute_ranking(cv_results, test_results)
    ranking_df = pd.DataFrame(ranking)
    ranking_df.to_csv(os.path.join(RESULTS_DIR, 'model_ranking.csv'),
                      index=False, float_format='%.6f')

    print(f"\n  Ranking: Composite = 40% CV_F1 + 30% CV_AUC + 20% Test_F1 + 10% Test_AUC\n")
    print(f"  {'Rank':<6} {'Model':<25} {'CV F1':>12} {'CV AUC':>10} "
          f"{'Test F1':>10} {'Test AUC':>10} {'Composite':>12}")
    print(f"  {'─' * 85}")

    for entry in ranking:
        marker = '★' if entry['Rank'] == 1 else ' '
        print(f"  {marker} {entry['Rank']:<4} {entry['Model']:<25} "
              f"{entry['CV_F1_Mean']:>8.4f}±{entry['CV_F1_Std']:.4f} "
              f"{entry['CV_ROC_AUC']:>10.4f} "
              f"{entry['Test_F1']:>10.4f} "
              f"{entry['Test_ROC_AUC']:>10.4f} "
              f"{entry['Composite_Score']:>12.4f}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 11: CLASSIFICATION REPORTS
    # ══════════════════════════════════════════════════════════════════
    print_section_header("DETAILED CLASSIFICATION REPORTS (TEST SET)")
    print_classification_reports(test_results, y_test)

    # ══════════════════════════════════════════════════════════════════
    # STEP 12: DASHBOARD DATA PREPARATION
    # ══════════════════════════════════════════════════════════════════
    print_section_header("STEP 12: DASHBOARD DATA PREPARED")

    dash = DashboardData(cv_results, test_results, trained_models,
                         feature_names, y_test)
    metrics_table = dash.get_metrics_table()
    print("\n  Dashboard metrics table (preview):")
    print(metrics_table.to_string(index=False))

    feature_imp = dash.get_feature_importance_figure()
    for name, df in feature_imp.items():
        print(f"\n  Top 5 features — {name}:")
        print(df.head().to_string(index=False))

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  [OK] ML EXPERIMENTATION FRAMEWORK — COMPLETE")
    print("=" * 80)
    print(f"\n  Generated Outputs:")
    print(f"  {'─' * 55}")
    print(f"  Results:")
    print(f"    • {os.path.join(RESULTS_DIR, 'metrics.csv')}")
    print(f"    • {os.path.join(RESULTS_DIR, 'metrics.json')}")
    print(f"    • {os.path.join(RESULTS_DIR, 'model_ranking.csv')}")
    print(f"  Visualizations ({VISUALIZATIONS_DIR}):")
    viz_files = ['correlation_heatmap.png', 'class_distribution.png',
                 'feature_distributions.png', 'roc_curves.png',
                 'precision_recall_curves.png', 'confusion_matrices.png',
                 'model_comparison_bar_chart.png', 'feature_importance.png',
                 'cv_boxplots.png']
    for vf in viz_files:
        print(f"    • {vf}")
    print(f"  Models ({MODEL_DIR}):")
    for name, (_, fname) in models.items():
        print(f"    • {fname}")
    print(f"\n  Best Model: {ranking[0]['Model']} "
          f"(Composite: {ranking[0]['Composite_Score']:.4f})")
    print(f"\n  Next Step: Phase 3 — GNN Implementation")
    print("=" * 80)


if __name__ == "__main__":
    main()
