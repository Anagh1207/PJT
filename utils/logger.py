"""
=============================================================================
Utility Functions — Logging & Experiment Tracking
=============================================================================
Functions for logging experiment results to CSV and JSON, and printing
formatted console output.

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import json
import time
import csv
from datetime import datetime
import pandas as pd


class ExperimentLogger:
    """
    Centralized experiment logging system.

    Records model name, hyperparameters, training/inference time,
    and all performance metrics. Supports CSV and JSON output.

    Usage
    -----
    >>> logger = ExperimentLogger(results_dir='results/')
    >>> logger.log_experiment('Random Forest', params, metrics, train_time, infer_time)
    >>> logger.save()
    """

    def __init__(self, results_dir):
        """
        Initialize the logger.

        Parameters
        ----------
        results_dir : str
            Directory where metrics.csv and metrics.json will be saved.
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.experiments = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log_experiment(self, model_name, hyperparameters, cv_metrics,
                       test_metrics, training_time, inference_time):
        """
        Log a single experiment (model run).

        Parameters
        ----------
        model_name : str
            Name of the model.
        hyperparameters : dict
            Model hyperparameters.
        cv_metrics : dict
            Cross-validation metrics (mean ± std for each metric).
        test_metrics : dict
            Test set metrics.
        training_time : float
            Time taken for training (seconds).
        inference_time : float
            Time taken for inference on test set (seconds).
        """
        record = {
            'model_name': model_name,
            'timestamp': self.timestamp,
            'hyperparameters': hyperparameters,
            'training_time_sec': round(training_time, 4),
            'inference_time_sec': round(inference_time, 4),
            'cv_metrics': {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in cv_metrics.items()
            },
            'test_metrics': {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in test_metrics.items()
            },
        }
        self.experiments.append(record)

    def save(self):
        """Save all logged experiments to CSV and JSON."""
        self._save_csv()
        self._save_json()

    def _save_csv(self):
        """Save experiments as a flat CSV table."""
        rows = []
        for exp in self.experiments:
            row = {
                'Model': exp['model_name'],
                'Timestamp': exp['timestamp'],
                'Training_Time_Sec': exp['training_time_sec'],
                'Inference_Time_Sec': exp['inference_time_sec'],
            }
            # Flatten CV metrics
            for k, v in exp['cv_metrics'].items():
                row[f'CV_{k}'] = v
            # Flatten test metrics
            for k, v in exp['test_metrics'].items():
                row[f'Test_{k}'] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.results_dir, 'metrics.csv')
        df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"  [OK] Metrics CSV saved: {csv_path}")

    def _save_json(self):
        """Save experiments as structured JSON (preserves hierarchy)."""
        # Convert hyperparameters to JSON-serializable format
        serializable = []
        for exp in self.experiments:
            exp_copy = exp.copy()
            exp_copy['hyperparameters'] = {
                k: str(v) if not isinstance(v, (int, float, bool, str, type(None)))
                else v
                for k, v in exp_copy['hyperparameters'].items()
            }
            serializable.append(exp_copy)

        json_path = os.path.join(self.results_dir, 'metrics.json')
        with open(json_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"  [OK] Metrics JSON saved: {json_path}")

    def get_summary_dataframe(self):
        """Return a summary DataFrame for dashboard use."""
        rows = []
        for exp in self.experiments:
            row = {'Model': exp['model_name']}
            row['Training_Time'] = exp['training_time_sec']
            row['Inference_Time'] = exp['inference_time_sec']
            for k, v in exp['test_metrics'].items():
                row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)


def print_section_header(title, char='═', width=80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_subsection_header(title, char='─', width=60):
    """Print a formatted subsection header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")
