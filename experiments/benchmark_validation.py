"""
=============================================================================
Benchmark Validation Experiment — Cross-Dataset Generalization
=============================================================================
Evaluates model generalization across three dataset regimes:
  Regime 1: Train on Synthetic Dataset -> Test on Real Benchmark Dataset
  Regime 2: Train on Original Dataset  -> Test on Real Benchmark Dataset
  Regime 3: Train on Combined Dataset  -> Test on Real Benchmark Dataset

Author: AI Research Assistant
Date: September 2026
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ALL_FEATURES, TARGET_COLUMN, RESULTS_DIR

def run_benchmark_validation_experiment(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    benchmark_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluates ML models across training dataset regimes and validates on benchmark reference data.
    """
    print("\n" + "="*70)
    print("  RUNNING CROSS-DATASET BENCHMARK VALIDATION EXPERIMENT")
    print("="*70)
    
    # Feature & Target extraction
    X_orig = original_df[ALL_FEATURES].values
    y_orig = original_df[TARGET_COLUMN].values
    
    X_syn = synthetic_df[ALL_FEATURES].values
    y_syn = synthetic_df[TARGET_COLUMN].values
    
    combined_df = pd.concat([original_df, synthetic_df], axis=0).reset_index(drop=True)
    X_comb = combined_df[ALL_FEATURES].values
    y_comb = combined_df[TARGET_COLUMN].values
    
    X_bench = benchmark_df[ALL_FEATURES].values
    y_bench = benchmark_df[TARGET_COLUMN].values
    
    regimes = {
        'Train on Original (N=1000)': (X_orig, y_orig),
        'Train on Synthetic (N=2000)': (X_syn, y_syn),
        'Train on Combined (N=3000)': (X_comb, y_comb),
    }
    
    models = {
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    
    for regime_name, (X_tr, y_tr) in regimes.items():
        for model_name, model in models.items():
            model.fit(X_tr, y_tr)
            
            preds = model.predict(X_bench)
            probs = model.predict_proba(X_bench)[:, 1]
            
            acc = accuracy_score(y_bench, preds)
            prec = precision_score(y_bench, preds, zero_division=0)
            rec = recall_score(y_bench, preds, zero_division=0)
            f1 = f1_score(y_bench, preds, zero_division=0)
            auc = roc_auc_score(y_bench, probs) if len(np.unique(y_bench)) > 1 else 0.5
            
            results.append({
                'Training Regime': regime_name,
                'Model': model_name,
                'Benchmark Accuracy': acc,
                'Benchmark Precision': prec,
                'Benchmark Recall': rec,
                'Benchmark F1': f1,
                'Benchmark ROC-AUC': auc
            })
            
            print(f"[{regime_name}] {model_name:15s} | Bench F1: {f1:.4f} | Bench AUC: {auc:.4f}")
            
    res_df = pd.DataFrame(results)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "benchmark_validation_results.csv")
    res_df.to_csv(csv_path, index=False)
    
    latex_path = os.path.join(RESULTS_DIR, "benchmark_validation_table.tex")
    res_df.to_latex(latex_path, index=False, caption="Cross-Dataset Validation Results on Real Benchmark Reference Data")
    
    print(f"\n[BenchmarkVal] Exported results to {csv_path}")
    return res_df


if __name__ == "__main__":
    from data.benchmark_data import prepare_benchmark_dataset
    from data.synthetic_data import generate_synthetic_dataset
    
    bench_df = prepare_benchmark_dataset()
    syn_df = generate_synthetic_dataset(bench_df, n_samples=500, epochs=20)
    
    res = run_benchmark_validation_experiment(bench_df.iloc[:1000], syn_df, bench_df)
