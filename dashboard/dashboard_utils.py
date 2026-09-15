"""
=============================================================================
Dashboard Preparation Module — Streamlit-Ready Functions
=============================================================================
Functions that return structured data suitable for frontend integration
(Streamlit, Gradio, etc.). Each function returns either DataFrames,
figure objects, or formatted dictionaries.

This module does NOT run a server — it provides the building blocks
that a Streamlit app can import and call directly.

Usage (future Streamlit integration):
    from dashboard.dashboard_utils import DashboardData
    dash = DashboardData(cv_results, test_results, trained_models, ...)
    st.dataframe(dash.get_metrics_table())
    st.pyplot(dash.get_roc_figure())

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

from config import MODEL_COLORS, MODEL_MARKERS, CLASS_NAMES


class DashboardData:
    """
    Dashboard-ready data provider.

    Encapsulates all experiment results and provides methods that return
    structured data for UI rendering.

    Parameters
    ----------
    cv_results : dict
        Cross-validation results per model.
    test_results : dict
        Test set evaluation results per model.
    trained_models : dict
        Fitted model instances.
    feature_names : list
        Feature column names.
    y_test : array-like
        True test set labels.
    """

    def __init__(self, cv_results, test_results, trained_models,
                 feature_names, y_test):
        self.cv_results = cv_results
        self.test_results = test_results
        self.trained_models = trained_models
        self.feature_names = feature_names
        self.y_test = y_test

    # ── Metrics Table ─────────────────────────────────────────────────

    def get_metrics_table(self):
        """
        Return a DataFrame with CV and test metrics for all models.

        Returns
        -------
        pd.DataFrame
            Columns: Model, CV_Accuracy, CV_F1, CV_ROC_AUC,
            Test_Accuracy, Test_F1, Test_ROC_AUC, etc.
        """
        rows = []
        for name in self.cv_results:
            cv = self.cv_results[name]
            test = self.test_results[name]
            rows.append({
                'Model': name,
                'CV Accuracy': f"{cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}",
                'CV Precision': f"{cv['precision_mean']:.4f} ± {cv['precision_std']:.4f}",
                'CV Recall': f"{cv['recall_mean']:.4f} ± {cv['recall_std']:.4f}",
                'CV F1': f"{cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}",
                'CV ROC-AUC': f"{cv['roc_auc_mean']:.4f} ± {cv['roc_auc_std']:.4f}",
                'Test Accuracy': f"{test['accuracy']:.4f}",
                'Test F1': f"{test['f1']:.4f}",
                'Test ROC-AUC': f"{test['roc_auc']:.4f}",
            })
        return pd.DataFrame(rows)

    def get_metrics_numeric(self):
        """
        Return a numeric DataFrame (no ± formatting) for plotting.

        Returns
        -------
        pd.DataFrame
            All metrics as float values.
        """
        rows = []
        for name in self.cv_results:
            cv = self.cv_results[name]
            test = self.test_results[name]
            rows.append({
                'Model': name,
                'CV_Accuracy': cv['accuracy_mean'],
                'CV_F1': cv['f1_mean'],
                'CV_ROC_AUC': cv['roc_auc_mean'],
                'Test_Accuracy': test['accuracy'],
                'Test_Precision': test['precision'],
                'Test_Recall': test['recall'],
                'Test_F1': test['f1'],
                'Test_ROC_AUC': test['roc_auc'],
            })
        return pd.DataFrame(rows)

    # ── Comparison Chart ──────────────────────────────────────────────

    def get_comparison_chart(self, metric='f1'):
        """
        Return a matplotlib figure showing a bar chart for a given metric.

        Parameters
        ----------
        metric : str
            One of 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        models = list(self.cv_results.keys())
        values = [self.cv_results[m][f'{metric}_mean'] for m in models]
        stds = [self.cv_results[m][f'{metric}_std'] for m in models]
        colors = [MODEL_COLORS.get(m, '#333333') for m in models]

        bars = ax.bar(models, values, yerr=stds, color=colors,
                      edgecolor='white', capsize=5, alpha=0.88)
        ax.set_ylabel(f'{metric.replace("_", " ").title()} Score')
        ax.set_title(f'Model Comparison — {metric.replace("_", " ").title()}',
                     fontweight='bold')
        ax.set_ylim([0, 1.1])
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        return fig

    # ── ROC Figure ────────────────────────────────────────────────────

    def get_roc_figure(self):
        """
        Return a matplotlib figure with ROC curves for all models.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        for name, res in self.test_results.items():
            fpr, tpr, _ = roc_curve(self.y_test, res['y_prob'])
            color = MODEL_COLORS.get(name, '#333333')
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name} ({res['roc_auc']:.4f})")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC Curves', fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_aspect('equal')
        plt.tight_layout()
        return fig

    # ── Feature Importance ────────────────────────────────────────────

    def get_feature_importance_figure(self, model_name=None):
        """
        Return feature importance as a DataFrame or figure.

        Parameters
        ----------
        model_name : str, optional
            Specific model. If None, returns for all tree-based models.

        Returns
        -------
        dict
            model_name → pd.DataFrame with Feature and Importance columns.
        """
        result = {}
        models_to_check = ({model_name: self.trained_models[model_name]}
                           if model_name else self.trained_models)

        for name, model in models_to_check.items():
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': imp
                }).sort_values('Importance', ascending=False)
                result[name] = df

        return result

    # ── Ranking Table ─────────────────────────────────────────────────

    def get_ranking_table(self):
        """
        Return the final ranking table as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Sorted by composite score.
        """
        from evaluation.evaluator import compute_ranking
        ranking = compute_ranking(self.cv_results, self.test_results)
        return pd.DataFrame(ranking)
