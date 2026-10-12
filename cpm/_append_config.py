"""Append CPM config block to config.py"""
import os

cpm_block = """

# =============================================================================
# CPM (COUNTERFACTUAL PATIENT MIGRATION) PIPELINE CONFIGURATION
# =============================================================================

CPM_DIR   = os.path.join(BASE_DIR, "cpm")
CPM_CACHE = os.path.join(DATA_DIR, "cpm_demo_cache.pkl")

os.makedirs(CPM_DIR, exist_ok=True)

CPM_CONFIG = {
    # High-risk threshold strategy:
    #   'percentile_75_cvd' - 75th-percentile of predicted probs among true CVD patients
    #   'fixed_0.5'         - prob > 0.5 AND true label = 1  (fallback for small cohorts)
    "high_risk_threshold_strategy": "percentile_75_cvd",

    # Synthetic generation (Track A) - fitted on training split only
    "synthetic_n": 2000,
    "synthetic_method": "ctgan",
    "synthetic_epochs": 300,

    # Graph parameters - must match the existing GraphSAGE checkpoint
    "graph_k": 15,               # K* selected by sweep (models/graphsage_optimal.pt)
    "graph_metric": "clinical",  # clinical similarity metric

    # CPM composite score weights (proposed formula - hard gate before use)
    # CPM = T*w_t + R*w_r + D_norm*w_d + F_norm*w_f
    #   T      = neighbor turnover  (1 - Jaccard)
    #   R      = neighbourhood CVD prevalence shift (delta CVD ratio)
    #   D_norm = embedding displacement, min-max normalised to [0,1] within cohort
    #   F_norm = normalised feature L2 distance (/ sqrt(n_features)), min-max to [0,1]
    "cpm_weights": {
        "turnover":     0.35,
        "risk_shift":   0.35,
        "embedding":    0.20,
        "feature_dist": 0.10,
    },

    # Minimum cohort size to consider Step 7 statistically meaningful
    "min_test_cohort_for_step7": 15,
}
"""

with open("config.py", "a") as f:
    f.write(cpm_block)

print("CPM config block appended successfully.")
