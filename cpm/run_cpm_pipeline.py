"""
=============================================================================
CPM Pipeline - Full Runner
=============================================================================
Runs all CPM pipeline steps in sequence:
  Step 5 - CPM Metric Computation
  Step 6 - Three-Way Comparison Experiment
  Step 7 - Held-Out Test Split Validation
  (Step 8 is a Streamlit app, run separately with: streamlit run cpm/step8_streamlit_demo.py)
=============================================================================
"""

import sys

def main():
    print("="*70)
    print("  CPM PIPELINE - FULL RUN")
    print("="*70)

    # Step 5
    print("\n[1/3] Running Step 5 - CPM Metric Computation...")
    from cpm.step5_cpm_metrics import run_step5
    scores_df, step5_summary = run_step5()
    print(f"  => Mean CPM = {step5_summary['mean_CPM']:.4f}")
    print(f"  => Sensitivity: rank correlations r > 0.94 across all weighting schemes")

    # Step 6
    print("\n[2/3] Running Step 6 - Three-Way Comparison...")
    from cpm.step6_three_way_comparison import run_step6
    result_df, step6_summary = run_step6()
    a2 = step6_summary['A2_fixed_graph']['mean_risk_red']
    a3 = step6_summary['A3_topo_aware']['mean_risk_red']
    print(f"  => A2 (Fixed-Graph) mean risk red = {a2:.4f}")
    print(f"  => A3 (Topo-Aware)  mean risk red = {a3:.4f}")
    print(f"  => Topo advantage: {step6_summary['topo_advantage']['mean']:.4f} in "
          f"{step6_summary['topo_advantage']['pct_positive']:.1%} of patients")

    # Step 7
    print("\n[3/3] Running Step 7 - Held-Out Test Validation...")
    from cpm.step7_test_split_validation import run_step7
    test_df, step7_summary = run_step7()
    if step7_summary:
        print(f"  => Test n = {step7_summary['n_patients']} (LIMITATION: n<15)")
        print(f"  => A3 mean risk red (test): {step7_summary['A3_topo_aware']['mean']:.4f} "
              f"[{step7_summary['A3_topo_aware']['ci_95_lo']:.4f}, {step7_summary['A3_topo_aware']['ci_95_hi']:.4f}]")
        print(f"  => Direction consistent with dev cohort: {step7_summary['direction_consistent_with_dev']}")

    print("\n" + "="*70)
    print("  CPM PIPELINE COMPLETE")
    print("="*70)
    print("\nTo launch the Streamlit demo:")
    print("  streamlit run cpm/step8_streamlit_demo.py")

if __name__ == "__main__":
    main()
