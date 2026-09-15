"""
=============================================================================
Calibration Analysis Module
=============================================================================
Phase 8a: Calibration analysis for DPSN and baseline models.

Implements:
  - Reliability diagram (calibration curve) for all models
  - Brier score computation
  - F1-optimal threshold search (sweep 0.01–0.99)
  - Before/after comparison table

Key Question: DPSN has ROC-AUC=0.988 but F1=0.813 — does recalibration
help close the gap with LightGBM's F1=0.894?

Author: AI Research Assistant
Date: August 2026
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
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score
from sklearn.isotonic import IsotonicRegression

from config import PLOT_DPI, PLOT_STYLE, FIGURES_DIR, RESULTS_DIR

plt.rcParams.update(PLOT_STYLE)


def compute_brier_score(y_true, y_prob):
    """Compute Brier score (lower is better, 0 = perfect)."""
    return brier_score_loss(y_true, y_prob)


def find_optimal_threshold(y_true, y_prob, metric='f1', steps=199):
    """
    Sweep thresholds to find the one that maximizes the given metric.

    Parameters
    ----------
    y_true : array-like, true binary labels
    y_prob : array-like, predicted probabilities for positive class
    metric : str, one of 'f1', 'precision', 'recall'
    steps : int, number of threshold steps to try

    Returns
    -------
    best_threshold : float
    best_score : float
    all_thresholds : np.ndarray
    all_scores : np.ndarray
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    scores = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        if metric == 'f1':
            s = f1_score(y_true, y_pred_t, zero_division=0)
        elif metric == 'precision':
            s = precision_score(y_true, y_pred_t, zero_division=0)
        elif metric == 'recall':
            s = recall_score(y_true, y_pred_t, zero_division=0)
        else:
            s = f1_score(y_true, y_pred_t, zero_division=0)
        scores.append(s)

    scores = np.array(scores)
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx], thresholds, scores


def apply_isotonic_calibration(y_true_cal, y_prob_cal, y_prob_test):
    """
    Apply isotonic regression calibration.

    Parameters
    ----------
    y_true_cal : array, true labels for calibration set
    y_prob_cal : array, predicted probs for calibration set
    y_prob_test : array, predicted probs for test set to calibrate

    Returns
    -------
    calibrated_probs : array, calibrated probabilities
    """
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_prob_cal, y_true_cal)
    return iso.predict(y_prob_test)


def plot_reliability_diagram(model_probs, y_true_dict, save_dir=None,
                             n_bins=10):
    """
    Plot reliability diagram (calibration curve) for multiple models.

    Parameters
    ----------
    model_probs : dict {model_name: y_prob}
    y_true_dict : dict {model_name: y_true}
    save_dir : str
    n_bins : int
    """
    save_dir = save_dir or FIGURES_DIR

    colors = {
        'DPSN': '#D81B60', 'LightGBM': '#C73E1D', 'XGBoost': '#F18F01',
        'Random Forest': '#A23B72', 'Logistic Regression': '#2E86AB',
        'SVM': '#3B1F2B', 'GCN': '#1E88E5', 'GraphSAGE': '#43A047',
        'GAT': '#8E24AA',
        'DPSN (calibrated)': '#FF6F00',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly calibrated')

    for name, y_prob in model_probs.items():
        y_true = y_true_dict[name]
        try:
            fraction_pos, mean_pred = calibration_curve(
                y_true, y_prob, n_bins=n_bins, strategy='uniform')
            color = colors.get(name, '#333333')
            ax.plot(mean_pred, fraction_pos, 's-', label=name,
                    color=color, markersize=5, linewidth=1.5)
        except Exception:
            continue

    ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontweight='bold')
    ax.set_title('Reliability Diagram (Calibration Curve)',
                 fontweight='bold', fontsize=13)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # Right: Brier score comparison bar chart
    ax2 = axes[1]
    brier_scores = {}
    for name, y_prob in model_probs.items():
        y_true = y_true_dict[name]
        try:
            brier_scores[name] = brier_score_loss(y_true, y_prob)
        except Exception:
            continue

    if brier_scores:
        sorted_models = sorted(brier_scores.items(), key=lambda x: x[1])
        names = [s[0] for s in sorted_models]
        scores = [s[1] for s in sorted_models]
        bar_colors = [colors.get(n, '#333') for n in names]

        bars = ax2.barh(names, scores, color=bar_colors, edgecolor='white',
                        alpha=0.85)
        ax2.set_xlabel('Brier Score (lower = better)', fontweight='bold')
        ax2.set_title('Brier Score Comparison', fontweight='bold', fontsize=13)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels
        for bar, score in zip(bars, scores):
            ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                     f'{score:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'calibration_reliability_diagram.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Reliability diagram: {path}")
    return brier_scores


def plot_threshold_tuning(y_true, y_prob, model_name='DPSN', save_dir=None):
    """
    Plot F1 vs threshold curve and mark the optimal threshold.

    Parameters
    ----------
    y_true : array
    y_prob : array
    model_name : str
    save_dir : str

    Returns
    -------
    best_threshold : float
    best_f1 : float
    """
    save_dir = save_dir or FIGURES_DIR

    best_t, best_f1, thresholds, f1_scores = find_optimal_threshold(
        y_true, y_prob, metric='f1')

    # Also compute precision and recall curves
    _, _, _, prec_scores = find_optimal_threshold(
        y_true, y_prob, metric='precision')
    _, _, _, rec_scores = find_optimal_threshold(
        y_true, y_prob, metric='recall')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1_scores, '-', color='#D81B60', linewidth=2,
            label='F1 Score')
    ax.plot(thresholds, prec_scores, '--', color='#1E88E5', linewidth=1.5,
            label='Precision', alpha=0.7)
    ax.plot(thresholds, rec_scores, '--', color='#43A047', linewidth=1.5,
            label='Recall', alpha=0.7)

    # Mark optimal threshold
    ax.axvline(x=best_t, color='#D81B60', linestyle=':', alpha=0.7)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5,
               label='Default (0.5)')
    ax.plot(best_t, best_f1, 'o', color='#D81B60', markersize=12,
            zorder=5, markeredgecolor='white', markeredgewidth=2)

    # Compute F1 at default threshold
    y_pred_default = (y_prob >= 0.5).astype(int)
    f1_default = f1_score(y_true, y_pred_default, zero_division=0)

    ax.annotate(f'Optimal: t={best_t:.3f}\nF1={best_f1:.4f}',
                xy=(best_t, best_f1),
                xytext=(best_t + 0.1, best_f1 - 0.05),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#D81B60'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#D81B60', alpha=0.9))

    ax.annotate(f'Default: t=0.500\nF1={f1_default:.4f}',
                xy=(0.5, f1_default),
                xytext=(0.5 + 0.12, f1_default + 0.06),
                fontsize=10, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.9))

    ax.set_xlabel('Decision Threshold', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(f'{model_name} — Threshold Tuning\n'
                 f'Improvement: F1 {f1_default:.4f} → {best_f1:.4f} '
                 f'(+{best_f1 - f1_default:.4f})',
                 fontweight='bold', fontsize=13)
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    path = os.path.join(save_dir,
                        f'threshold_tuning_{model_name.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Threshold tuning: {path}")

    return best_t, best_f1


def run_calibration_analysis(all_results, save_dir=None):
    """
    Run full calibration analysis across all models.

    Parameters
    ----------
    all_results : dict {model_name: {'y_true': ..., 'y_prob': ...}}
    save_dir : str

    Returns
    -------
    calibration_df : pd.DataFrame with calibration metrics
    """
    save_dir = save_dir or FIGURES_DIR

    print("\n  ── Calibration Analysis ──────────────────────────────────")

    # Collect probabilities and true labels
    model_probs = {}
    y_true_dict = {}
    rows = []

    for name, r in all_results.items():
        if 'y_prob' in r and 'y_true' in r:
            y_true = r['y_true']
            y_prob = r['y_prob']
            model_probs[name] = y_prob
            y_true_dict[name] = y_true

            # Brier score
            brier = compute_brier_score(y_true, y_prob)

            # F1 at default threshold
            y_pred_default = (y_prob >= 0.5).astype(int)
            f1_default = f1_score(y_true, y_pred_default, zero_division=0)

            # Optimal threshold
            best_t, best_f1, _, _ = find_optimal_threshold(y_true, y_prob)

            rows.append({
                'Model': name,
                'Brier Score': brier,
                'F1 @ 0.5': f1_default,
                'Optimal Threshold': best_t,
                'F1 @ Optimal': best_f1,
                'F1 Improvement': best_f1 - f1_default,
            })

            print(f"  {name:25s}  Brier={brier:.4f}  "
                  f"F1@0.5={f1_default:.4f}  "
                  f"F1@{best_t:.3f}={best_f1:.4f}  "
                  f"(Δ={best_f1 - f1_default:+.4f})")

    # Reliability diagram + Brier chart
    if model_probs:
        plot_reliability_diagram(model_probs, y_true_dict, save_dir)

    # Threshold tuning plot for DPSN specifically
    if 'DPSN' in model_probs:
        dpsn_best_t, dpsn_best_f1 = plot_threshold_tuning(
            y_true_dict['DPSN'], model_probs['DPSN'],
            model_name='DPSN', save_dir=save_dir)

        # Also apply isotonic calibration to DPSN
        try:
            y_true_dpsn = y_true_dict['DPSN']
            y_prob_dpsn = model_probs['DPSN']

            # Use half for calibration, half for evaluation
            n = len(y_true_dpsn)
            cal_idx = np.arange(0, n, 2)
            test_idx = np.arange(1, n, 2)

            calibrated_probs = apply_isotonic_calibration(
                y_true_dpsn[cal_idx], y_prob_dpsn[cal_idx],
                y_prob_dpsn[test_idx])

            # F1 with calibrated probs
            y_pred_cal = (calibrated_probs >= 0.5).astype(int)
            f1_calibrated = f1_score(y_true_dpsn[test_idx], y_pred_cal,
                                     zero_division=0)

            # Optimal threshold on calibrated probs
            best_t_cal, best_f1_cal, _, _ = find_optimal_threshold(
                y_true_dpsn[test_idx], calibrated_probs)

            brier_cal = compute_brier_score(y_true_dpsn[test_idx],
                                             calibrated_probs)

            rows.append({
                'Model': 'DPSN (isotonic calibrated)',
                'Brier Score': brier_cal,
                'F1 @ 0.5': f1_calibrated,
                'Optimal Threshold': best_t_cal,
                'F1 @ Optimal': best_f1_cal,
                'F1 Improvement': best_f1_cal - f1_calibrated,
            })

            print(f"\n  DPSN (calibrated):          Brier={brier_cal:.4f}  "
                  f"F1@0.5={f1_calibrated:.4f}  "
                  f"F1@{best_t_cal:.3f}={best_f1_cal:.4f}")

            # Add calibrated probs to reliability diagram
            model_probs['DPSN (calibrated)'] = np.full(n, np.nan)
            model_probs['DPSN (calibrated)'][test_idx] = calibrated_probs
            y_true_dict['DPSN (calibrated)'] = y_true_dpsn

            # Re-plot with calibrated
            cal_probs = {k: v for k, v in model_probs.items()
                         if not np.any(np.isnan(v))}
            cal_true = {k: y_true_dict[k] for k in cal_probs}
            if 'DPSN (calibrated)' not in cal_probs:
                cal_probs['DPSN (calibrated)'] = calibrated_probs
                cal_true['DPSN (calibrated)'] = y_true_dpsn[test_idx]
            plot_reliability_diagram(cal_probs, cal_true, save_dir)

        except Exception as e:
            print(f"  [WARN] Isotonic calibration failed: {e}")

    # Save results
    cal_df = pd.DataFrame(rows)
    cal_path = os.path.join(RESULTS_DIR, 'calibration_analysis.csv')
    cal_df.to_csv(cal_path, index=False, float_format='%.6f')
    print(f"\n  [OK] Calibration analysis saved: {cal_path}")

    return cal_df
