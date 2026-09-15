"""
=============================================================================
Dataset Provenance Disclosure
=============================================================================
Checks whether GNN DATASET.xlsx is synthetic or derived from a real clinical
source. Generates a provenance report with limitations for clinical claims.

Author: AI Research Assistant
Date: August 2026
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats as scipy_stats

from config import RAW_DATASET, RESULTS_DIR


def analyze_dataset_provenance(filepath=None):
    """
    Analyze dataset characteristics to determine likely provenance.

    Parameters
    ----------
    filepath : str, path to raw dataset

    Returns
    -------
    report : str, provenance report text
    findings : dict, structured findings
    """
    filepath = filepath or RAW_DATASET

    # Load raw dataset
    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    findings = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,
        'features': list(df.columns),
        'indicators': [],
    }

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATASET PROVENANCE ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nDataset: {os.path.basename(filepath)}")
    report_lines.append(f"Samples: {len(df)}")
    report_lines.append(f"Features: {len(df.columns) - 1}")
    report_lines.append(f"Target column: Target")
    report_lines.append("")

    # ── Check 1: Sample count ───────────────────────────────────────
    report_lines.append("─" * 60)
    report_lines.append("CHECK 1: Sample Count")
    if len(df) in [100, 200, 500, 1000, 2000, 5000, 10000]:
        findings['indicators'].append(
            f'Round sample count ({len(df)}) — common in synthetic data')
        report_lines.append(f"  ⚠ Round number ({len(df)}) — unusual for "
                            "real clinical datasets")
    else:
        report_lines.append(f"  ✓ Non-round sample count ({len(df)})")

    # ── Check 2: Missing values ─────────────────────────────────────
    report_lines.append("\n" + "─" * 60)
    report_lines.append("CHECK 2: Missing Values")
    missing_total = df.isnull().sum().sum()
    if missing_total == 0:
        findings['indicators'].append(
            'Zero missing values — extremely rare in real clinical data')
        report_lines.append("  ⚠ Zero missing values — real clinical data "
                            "almost always has missingness")
    else:
        report_lines.append(f"  Total missing values: {missing_total}")

    # ── Check 3: Feature naming conventions ─────────────────────────
    report_lines.append("\n" + "─" * 60)
    report_lines.append("CHECK 3: Feature Naming")
    clean_names = all('_' in col or col.isalpha() for col in df.columns)
    if clean_names:
        findings['indicators'].append(
            'Uniformly clean feature names — suggests designed schema')
        report_lines.append("  ⚠ All feature names follow a uniform, clean "
                            "convention (CamelCase/snake_case)")
        report_lines.append("    Real EHR data typically has messy, "
                            "inconsistent naming")
    else:
        report_lines.append("  ✓ Mixed naming conventions")

    # ── Check 4: Distribution analysis ──────────────────────────────
    report_lines.append("\n" + "─" * 60)
    report_lines.append("CHECK 4: Distribution Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_normal = 0
    for col in numeric_cols:
        if col == 'Target':
            continue
        values = df[col].dropna().values
        if len(np.unique(values)) > 5:  # Skip binary features
            _, p_val = scipy_stats.normaltest(values)
            if p_val > 0.05:
                n_normal += 1
                report_lines.append(f"  {col}: appears normally distributed "
                                    f"(p={p_val:.4f})")

    if n_normal >= 3:
        findings['indicators'].append(
            f'{n_normal} features appear normally distributed — '
            'suggests Gaussian generation')
        report_lines.append(f"\n  ⚠ {n_normal} features follow near-normal "
                            "distributions — common in synthetic data "
                            "generated from Gaussian priors")

    # ── Check 5: Duplicate rows ─────────────────────────────────────
    report_lines.append("\n" + "─" * 60)
    report_lines.append("CHECK 5: Duplicate Rows")
    n_dupes = df.duplicated().sum()
    report_lines.append(f"  Exact duplicates: {n_dupes}")
    if n_dupes == 0:
        report_lines.append("  ✓ No exact duplicates")

    # ── Check 6: Value precision ────────────────────────────────────
    report_lines.append("\n" + "─" * 60)
    report_lines.append("CHECK 6: Value Precision / Rounding")
    for col in ['Age', 'Blood_Pressure', 'Cholesterol', 'Heart_Rate', 'BMI']:
        if col in df.columns:
            values = df[col].dropna().values
            # Check if values have suspicious decimal patterns
            decimals = values - np.floor(values)
            unique_decimals = len(np.unique(np.round(decimals, 4)))
            report_lines.append(f"  {col}: {unique_decimals} unique decimal "
                                f"patterns, range [{values.min():.1f}, "
                                f"{values.max():.1f}]")

    # ── Check 7: Class balance ──────────────────────────────────────
    report_lines.append("\n" + "─" * 60)
    report_lines.append("CHECK 7: Class Balance")
    if 'Target' in df.columns:
        class_counts = df['Target'].value_counts()
        for cls, cnt in class_counts.items():
            pct = cnt / len(df) * 100
            report_lines.append(f"  Class {cls}: {cnt} ({pct:.1f}%)")
        ratio = class_counts.max() / class_counts.min()
        report_lines.append(f"  Imbalance ratio: {ratio:.2f}:1")

    # ── Check 8: Correlation structure ──────────────────────────────
    report_lines.append("\n" + "─" * 60)
    report_lines.append("CHECK 8: Clinical Correlation Structure")
    expected_correlations = [
        ('Age', 'Blood_Pressure', 'positive', 'Age-BP correlation'),
        ('BMI', 'Cholesterol', 'positive', 'BMI-Cholesterol'),
        ('Diabetes', 'Glucose', 'positive', 'Diabetes-Glucose'),
        ('Smoking', 'Blood_Pressure', 'positive', 'Smoking-BP'),
    ]
    for col1, col2, expected_dir, desc in expected_correlations:
        if col1 in df.columns and col2 in df.columns:
            corr = df[col1].corr(df[col2])
            matches = (expected_dir == 'positive' and corr > 0) or \
                      (expected_dir == 'negative' and corr < 0)
            status = '✓' if matches else '⚠'
            report_lines.append(f"  {status} {desc}: r={corr:.4f} "
                                f"(expected {expected_dir})")

    # ── CONCLUSION ──────────────────────────────────────────────────
    report_lines.append("\n" + "=" * 80)
    report_lines.append("CONCLUSION")
    report_lines.append("=" * 80)

    n_indicators = len(findings['indicators'])
    if n_indicators >= 3:
        findings['conclusion'] = 'LIKELY SYNTHETIC'
        report_lines.append(
            f"\n  ASSESSMENT: LIKELY SYNTHETIC/SIMULATED DATASET"
        )
        report_lines.append(
            f"\n  {n_indicators} indicators of synthetic origin detected:"
        )
        for i, ind in enumerate(findings['indicators'], 1):
            report_lines.append(f"    {i}. {ind}")
    elif n_indicators >= 1:
        findings['conclusion'] = 'POSSIBLY SYNTHETIC'
        report_lines.append(
            f"\n  ASSESSMENT: POSSIBLY SYNTHETIC (inconclusive)"
        )
    else:
        findings['conclusion'] = 'LIKELY REAL'
        report_lines.append(
            f"\n  ASSESSMENT: No strong indicators of synthetic origin"
        )

    # ── PROVENANCE DISCLOSURE STATEMENT ─────────────────────────────
    report_lines.append("\n" + "=" * 80)
    report_lines.append("RECOMMENDED PROVENANCE DISCLOSURE (for paper)")
    report_lines.append("=" * 80)

    disclosure = generate_disclosure_statement(findings)
    report_lines.append(disclosure)

    report = '\n'.join(report_lines)
    return report, findings


def generate_disclosure_statement(findings):
    """Generate a provenance disclosure statement for the paper."""

    if findings['conclusion'] == 'LIKELY SYNTHETIC':
        statement = """
  "The dataset used in this study ('GNN DATASET.xlsx') contains
  {n_samples} patient records with {n_features} clinical features.
  Based on provenance analysis — including its round sample count,
  absence of missing values, uniform feature naming, and distributional
  properties — the dataset appears to be synthetic or simulated rather
  than derived from a real Electronic Health Record (EHR) system.

  LIMITATIONS: As a synthetic dataset, results should be interpreted
  as a proof-of-concept demonstrating the methodological framework
  rather than as clinically validated findings. The following caveats
  apply:

  1. Performance metrics may not generalize to real clinical populations
  2. Feature distributions may not reflect true clinical variability
  3. Inter-feature correlations may not capture real comorbidity patterns
  4. The class imbalance ratio ({n_samples} total) may differ from
     real-world CVD prevalence

  Future work should validate these methods on established clinical
  datasets (e.g., Framingham Heart Study, UK Biobank, MIMIC-III) to
  establish clinical validity."
""".format(**findings)
    else:
        statement = """
  "The dataset contains {n_samples} patient records with {n_features}
  clinical features. The exact provenance and collection methodology
  should be documented for reproducibility."
""".format(**findings)

    return statement


def run_provenance_analysis():
    """
    Run the full provenance analysis and save report.
    """
    print("\n  ── Dataset Provenance Analysis ───────────────────────────")

    report, findings = analyze_dataset_provenance()

    # Save report
    report_path = os.path.join(RESULTS_DIR, 'dataset_provenance_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n  [OK] Provenance report saved: {report_path}")

    return report, findings
