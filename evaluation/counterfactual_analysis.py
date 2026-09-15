"""
=============================================================================
Counterfactual / What-If Analysis
=============================================================================
Phase 8b: For test patients predicted as high-risk (CVD positive),
find the minimal feature changes that drop predicted risk below 0.5
or reduce risk by >= 20%.

Produces actionable clinical recommendations per patient.

Author: AI Research Assistant
Date: August 2026
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    PLOT_DPI, PLOT_STYLE, FIGURES_DIR, RESULTS_DIR,
    CONTINUOUS_FEATURES, BINARY_FEATURES
)

plt.rcParams.update(PLOT_STYLE)

# Modifiable features and their clinically desirable direction
# direction: 'decrease' = lower is better, 'increase' = higher is better
MODIFIABLE_FEATURES = {
    'BMI':                             {'direction': 'decrease', 'label': 'BMI'},
    'Physical_Activity_HoursPerWeek':  {'direction': 'increase', 'label': 'Physical Activity'},
    'Smoking':                         {'direction': 'decrease', 'label': 'Smoking'},
    'Stress_Index':                    {'direction': 'decrease', 'label': 'Stress Index'},
    'Cholesterol':                     {'direction': 'decrease', 'label': 'Cholesterol'},
    'Glucose':                         {'direction': 'decrease', 'label': 'Glucose'},
    'Triglycerides':                   {'direction': 'decrease', 'label': 'Triglycerides'},
    'Blood_Pressure':                  {'direction': 'decrease', 'label': 'Blood Pressure'},
    'Heart_Rate':                      {'direction': 'decrease', 'label': 'Heart Rate'},
}


def get_model_probability(model, x, edge_index, node_idx, edge_weight=None):
    """
    Get predicted CVD probability for a single node supporting both DPSN and GraphSAGE models.
    """
    model.eval()
    with torch.no_grad():
        try:
            logits = model(x, edge_index, edge_weight)
        except TypeError:
            logits = model(x, edge_index)
        probs = F.softmax(logits, dim=1)
        return probs[node_idx, 1].item()


# Backward compatibility alias
get_dpsn_probability = get_model_probability



def sweep_single_feature(model, x, edge_index, node_idx, feature_idx,
                          feature_name, feature_info, population_stats,
                          edge_weight=None, n_steps=50):
    """
    Sweep a single feature value and track risk change.

    Parameters
    ----------
    model : trained model
    x : tensor (N, F), original features
    edge_index : tensor
    node_idx : int
    feature_idx : int, index of feature to sweep
    feature_name : str
    feature_info : dict with 'direction'
    population_stats : dict with 'mean', 'min', 'max', 'median'
    edge_weight : tensor, optional
    n_steps : int

    Returns
    -------
    result : dict with sweep results
    """
    x_modified = x.clone()
    original_value = x[node_idx, feature_idx].item()
    original_risk = get_dpsn_probability(model, x, edge_index, node_idx,
                                          edge_weight)

    # Determine sweep direction
    direction = feature_info['direction']
    pop_median = population_stats['median']
    pop_min = population_stats['min']
    pop_max = population_stats['max']

    if direction == 'decrease':
        # Sweep from current value down to population minimum
        target_end = pop_min
    else:
        # Sweep from current value up to population maximum
        target_end = pop_max

    sweep_values = np.linspace(original_value, target_end, n_steps)
    risks = []

    for val in sweep_values:
        x_modified[node_idx, feature_idx] = val
        risk = get_dpsn_probability(model, x_modified, edge_index,
                                     node_idx, edge_weight)
        risks.append(risk)

    risks = np.array(risks)

    # Find threshold crossings
    # 1. Absolute threshold: risk < 0.5
    below_05 = np.where(risks < 0.5)[0]
    threshold_value_abs = None
    if len(below_05) > 0:
        threshold_value_abs = sweep_values[below_05[0]]

    # 2. Relative threshold: 20% reduction
    target_risk = original_risk * 0.80
    below_20pct = np.where(risks < target_risk)[0]
    threshold_value_rel = None
    if len(below_20pct) > 0:
        threshold_value_rel = sweep_values[below_20pct[0]]

    # Best achievable risk
    min_risk = risks.min()
    min_risk_value = sweep_values[np.argmin(risks)]

    # Reset
    x_modified[node_idx, feature_idx] = original_value

    return {
        'feature': feature_name,
        'original_value': original_value,
        'original_risk': original_risk,
        'direction': direction,
        'threshold_value_abs': threshold_value_abs,
        'threshold_value_rel': threshold_value_rel,
        'min_achievable_risk': min_risk,
        'min_risk_feature_value': min_risk_value,
        'change_for_abs': (threshold_value_abs - original_value)
            if threshold_value_abs is not None else None,
        'change_for_rel': (threshold_value_rel - original_value)
            if threshold_value_rel is not None else None,
        'sweep_values': sweep_values,
        'sweep_risks': risks,
    }


def run_counterfactual_for_patient(model, x, edge_index, node_idx,
                                     feature_names, population_stats_dict,
                                     edge_weight=None):
    """
    Run counterfactual analysis for a single patient.

    Parameters
    ----------
    model : trained DPSN
    x : tensor
    edge_index : tensor
    node_idx : int
    feature_names : list
    population_stats_dict : dict {feature_name: {'mean', 'min', 'max', 'median'}}
    edge_weight : tensor, optional

    Returns
    -------
    results : list of dicts, one per modifiable feature
    """
    results = []

    for feat_name, feat_info in MODIFIABLE_FEATURES.items():
        if feat_name in feature_names:
            feat_idx = feature_names.index(feat_name)
            pop_stats = population_stats_dict.get(feat_name, {
                'mean': 0, 'min': -3, 'max': 3, 'median': 0
            })

            result = sweep_single_feature(
                model, x, edge_index, node_idx, feat_idx,
                feat_name, feat_info, pop_stats, edge_weight)
            results.append(result)

    return results


def plot_counterfactual_tornado(patient_results, node_idx, save_dir=None):
    """
    Plot tornado chart showing which features are most actionable.
    """
    save_dir = save_dir or FIGURES_DIR

    # Filter to features where some risk reduction was achieved
    actionable = []
    for r in patient_results:
        risk_reduction = r['original_risk'] - r['min_achievable_risk']
        if risk_reduction > 0.001:
            actionable.append({
                'Feature': MODIFIABLE_FEATURES[r['feature']]['label'],
                'Risk Reduction': risk_reduction,
                'Original Risk': r['original_risk'],
                'Min Risk': r['min_achievable_risk'],
                'Change': r.get('change_for_abs') or r.get('change_for_rel') or 0,
            })

    if not actionable:
        print(f"  [WARN] No actionable features found for patient {node_idx}")
        return None

    df = pd.DataFrame(actionable).sort_values('Risk Reduction', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.6 + 1)))

    # Color gradient by risk reduction
    max_reduction = df['Risk Reduction'].max()
    colors = [plt.cm.RdYlGn(0.3 + 0.6 * r / max(max_reduction, 0.01))
              for r in df['Risk Reduction']]

    bars = ax.barh(df['Feature'], df['Risk Reduction'],
                   color=colors, edgecolor='white', alpha=0.9)

    # Add percentage labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        pct = row['Risk Reduction'] / row['Original Risk'] * 100
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{row["Risk Reduction"]:.3f} ({pct:.1f}%)',
                va='center', fontsize=9, fontweight='bold')

    original_risk = patient_results[0]['original_risk']
    ax.set_xlabel('Risk Reduction (absolute)', fontweight='bold')
    ax.set_title(f'Counterfactual Analysis — Patient {node_idx}\n'
                 f'Current Risk: {original_risk:.4f} | '
                 f'Most Actionable Features',
                 fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    path = os.path.join(save_dir, f'counterfactual_tornado_p{node_idx}.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Tornado chart: {path}")
    return path


def plot_counterfactual_summary(all_csv_rows, save_dir=None):
    """
    Plot a summary heatmap of counterfactual recommendations across patients.
    """
    save_dir = save_dir or FIGURES_DIR

    if not all_csv_rows:
        return

    df = pd.DataFrame(all_csv_rows)

    # Pivot: patients × features → risk reduction
    pivot_df = df.pivot_table(
        index='Patient_ID', columns='Feature',
        values='Risk_Reduction_Abs', aggfunc='first')

    if pivot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot_df) * 0.5 + 2)))

    # Replace NaN with 0 for display
    pivot_display = pivot_df.fillna(0)

    im = ax.imshow(pivot_display.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(pivot_display.columns)))
    ax.set_xticklabels(pivot_display.columns, rotation=45, ha='right',
                       fontsize=9)
    ax.set_yticks(range(len(pivot_display.index)))
    ax.set_yticklabels([f'Patient {i}' for i in pivot_display.index],
                       fontsize=9)

    # Add text annotations
    for i in range(len(pivot_display.index)):
        for j in range(len(pivot_display.columns)):
            val = pivot_display.values[i, j]
            if val > 0:
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8, color='white' if val > 0.1 else 'black')

    plt.colorbar(im, ax=ax, label='Risk Reduction', fraction=0.03)
    ax.set_title('Counterfactual Risk Reduction — All Patients × Features',
                 fontweight='bold', fontsize=13)

    plt.tight_layout()
    path = os.path.join(save_dir, 'counterfactual_summary_heatmap.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Summary heatmap: {path}")


def run_counterfactual_analysis(model, x, edge_index, y, test_mask,
                                 y_pred, y_prob, feature_names,
                                 edge_weight=None, n_patients=10,
                                 save_dir=None):
    """
    Run counterfactual analysis for high-risk test patients.

    Parameters
    ----------
    model : trained DPSN model
    x : tensor (N, F)
    edge_index : tensor (2, E)
    y : tensor or array
    test_mask : tensor
    y_pred : array, predictions for test set
    y_prob : array, predicted probabilities for test set
    feature_names : list of str
    edge_weight : tensor, optional
    n_patients : int, number of patients to analyze
    save_dir : str

    Returns
    -------
    results_df : pd.DataFrame with counterfactual recommendations
    """
    save_dir = save_dir or FIGURES_DIR

    print("\n  ── Counterfactual / What-If Analysis ─────────────────────")

    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)
    test_indices = test_mask.nonzero(as_tuple=True)[0].numpy()

    # Compute population statistics for each feature (from all data)
    population_stats = {}
    x_np = x.numpy()
    for i, fname in enumerate(feature_names):
        population_stats[fname] = {
            'mean': float(np.mean(x_np[:, i])),
            'min': float(np.min(x_np[:, i])),
            'max': float(np.max(x_np[:, i])),
            'median': float(np.median(x_np[:, i])),
            'std': float(np.std(x_np[:, i])),
        }

    # Select high-risk patients (predicted positive, sorted by probability)
    positive_mask = y_pred == 1
    if not positive_mask.any():
        print("  [WARN] No positive predictions in test set.")
        return pd.DataFrame()

    positive_indices = np.where(positive_mask)[0]
    sorted_by_prob = positive_indices[np.argsort(y_prob[positive_indices])[::-1]]
    selected = sorted_by_prob[:min(n_patients, len(sorted_by_prob))]

    print(f"  Analyzing {len(selected)} high-risk patients...")

    all_csv_rows = []

    for local_idx in selected:
        global_idx = test_indices[local_idx]
        risk = y_prob[local_idx]
        true_label = y_np[global_idx]

        print(f"\n  Patient {global_idx}: Risk={risk:.4f}, "
              f"True={'CVD' if true_label == 1 else 'No CVD'}")

        # Run counterfactual sweep
        patient_results = run_counterfactual_for_patient(
            model, x, edge_index, global_idx,
            feature_names, population_stats, edge_weight)

        # Record results
        for r in patient_results:
            risk_reduction = r['original_risk'] - r['min_achievable_risk']
            csv_row = {
                'Patient_ID': global_idx,
                'True_Label': int(true_label),
                'Feature': r['feature'],
                'Current_Value': r['original_value'],
                'Risk_Before': r['original_risk'],
                'Direction': r['direction'],
                'Min_Achievable_Risk': r['min_achievable_risk'],
                'Risk_Reduction_Abs': risk_reduction,
                'Risk_Reduction_Pct': risk_reduction / r['original_risk'] * 100
                    if r['original_risk'] > 0 else 0,
            }

            if r['threshold_value_abs'] is not None:
                csv_row['Value_for_Risk_Below_0.5'] = r['threshold_value_abs']
                csv_row['Change_for_Risk_Below_0.5'] = r['change_for_abs']
            else:
                csv_row['Value_for_Risk_Below_0.5'] = 'N/A'
                csv_row['Change_for_Risk_Below_0.5'] = 'N/A'

            if r['threshold_value_rel'] is not None:
                csv_row['Value_for_20pct_Reduction'] = r['threshold_value_rel']
                csv_row['Change_for_20pct_Reduction'] = r['change_for_rel']
            else:
                csv_row['Value_for_20pct_Reduction'] = 'N/A'
                csv_row['Change_for_20pct_Reduction'] = 'N/A'

            all_csv_rows.append(csv_row)

            if risk_reduction > 0.01:
                print(f"    {r['feature']:35s}  "
                      f"Δrisk={risk_reduction:+.4f} "
                      f"({risk_reduction/r['original_risk']*100:.1f}%)")

        # Plot tornado chart for this patient
        plot_counterfactual_tornado(patient_results, global_idx, save_dir)

    # Save CSV
    results_df = pd.DataFrame(all_csv_rows)
    csv_path = os.path.join(RESULTS_DIR, 'counterfactual_recommendations.csv')
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n  [OK] Counterfactual CSV saved: {csv_path}")

    # Summary heatmap
    plot_counterfactual_summary(all_csv_rows, save_dir)

    print(f"\n  [OK] Counterfactual analysis complete.")
    return results_df
