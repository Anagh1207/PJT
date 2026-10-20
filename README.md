# CardioMigrate — Counterfactual-Induced Patient Migration (CPM) for Graph-Based Cardiovascular Disease Risk Prediction

[![IEEE Conference Research](https://img.shields.io/badge/Research-IEEE%20Format-blue.svg)](paper_assets/IEEE_Dual_Layer_Cardiovascular_Paper.pdf)
[![Live Interactive Dashboard](https://img.shields.io/badge/Demo-Interactive%20Platform-emerald.svg)](https://anagh1207.github.io/PJT/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-GraphSAGE%20%7C%20DPSN-orange.svg)](https://pyg.org/)

This repository contains the complete research pipeline, model weights, evaluation suite, and interactive clinical visualization platform for **Counterfactual Patient Migration (CPM)** on patient similarity networks.

---

## 🌟 Interactive Clinical Dashboard (Live Deployment)

The interactive multi-graph platform is configured to deploy directly on **GitHub Pages**:
🔗 **[Launch Interactive CPM Simulator](https://anagh1207.github.io/PJT/)**

### Dashboard Capabilities:
- **Patient-Specific Micro-Topologies:** Local subgraphs showing real baseline high-risk peers and healthy destination cohorts for specific clinical archetypes (*Ramesh (58M)*, *Priya (64F)*, *Anil (52M)*, *George (72M)*).
- **Macro 5-Cluster Manifold:** Visualizes global population dynamics (*Severe Hypertensives*, *Diabetic Metabolic*, *Transitional Corridor*, *Controlled Primary Care*, *Healthy Reference Cohort*).
- **6-Axis Polar Biomarker Radar:** Real-time physiological fingerprint comparing baseline risk vs. target state.
- **Live Mathematical Proof Engine:** Real-time computation of domain-weighted distance, Jaccard turnover ($T$), risk shift ($R$), and latent displacement ($D_{\text{norm}}$).
- **Physician Counseling Scripts:** Translates complex graph mathematics into empathetic doctor-patient dialogue.

---

## 🫀 Core Research Innovation: Counterfactual Patient Migration (CPM)

### The Missing Clinical Question
Standard CVD risk models (Logistic Regression, LightGBM, XGBoost) and conventional counterfactuals answer:
> *"If blood pressure and cholesterol drop, by how much does the predicted risk score decrease?"*

However, traditional models compute risk in total isolation. CPM addresses the **topological question**:
> *"When a patient improves actionable biomarkers, do they structurally migrate into a healthier peer community, and does that neighborhood change carry information beyond the risk score alone?"*

### The Approved CPM Formula
For high-risk patient $i$:
$$\mathbf{CPM}_i = 0.40 \cdot T_i + 0.40 \cdot R_i + 0.20 \cdot D_{i,\text{norm}}$$

- **$T_i$ (Neighbor Turnover):** $1 - \text{Jaccard}(\mathcal{N}_{\text{orig}}, \mathcal{N}_{\text{new}}) \in [0, 1]$
- **$R_i$ (Prevalence Shift):** $\max(0, \text{Prevalence}_{\text{orig}} - \text{Prevalence}_{\text{new}}) \in [0, 1]$
- **$D_{i,\text{norm}}$ (GraphSAGE Latent Displacement):** Normalized Euclidean displacement across the learned 32D manifold.
- **$F_{i,\text{norm}}$ (Intervention Effort):** Normalized feature modification distance, reported separately to decouple clinical difficulty from topological relocation.

---

## 📊 Empirical Findings & Performance Summary

| Architecture / Configuration | Accuracy | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **GraphSAGE (Clinical Similarity Graph)** | **0.9350** | **0.9487** | **0.8506** | **0.9892** |
| LightGBM Baseline | 0.9550 | 0.8710 | 0.8941 | 0.9893 |
| Logistic Regression | 0.9400 | 0.9742 | 0.8667 | 0.9960 |
| Dynamic Patient Similarity Network (DPSN) | 0.9125 | 0.8910 | 0.8132 | 0.9880 |
| GCN Baseline | 0.9050 | 0.9743 | 0.8000 | 0.9909 |
| GAT Baseline | 0.8700 | 0.8718 | 0.7234 | 0.9552 |

### Three-Way Counterfactual Comparison (Novelty Evidence)
- **Normal Tabular Counterfactual:** Risk-score change only (no peer context).
- **Fixed-Graph Counterfactual:** Neighbors held static ($\Delta \text{Risk} = 0.0000$).
- **Topology-Aware Counterfactual (CPM):** Yields a positive topological advantage for **$71.79\%$ of patients** ($p < 0.001$, paired Wilcoxon test).

---

## 📁 Repository Structure

```
├── .github/workflows/deploy.yml   # GitHub Pages deployment automation
├── dashboard/                     # Interactive multi-graph web application
│   ├── index.html                 # 60FPS fluid physics animation & KaTeX proof engine
│   ├── cpm_multigraph_data.json   # Precomputed patient topologies and coordinates
│   └── build_spacious_ui.py       # UI generator script
├── cpm/                           # Complete Counterfactual Patient Migration pipeline
│   ├── track_a_synthetic_validation.py # CTGAN synthetic validation (N=2,000)
│   ├── track_b_high_risk_cohort.py     # High-risk cohort selection
│   ├── step3_counterfactual_generation.py # Constrained clinical counterfactuals
│   ├── step4_graph_rebuild.py          # Dynamic K-NN neighborhood recomputation
│   ├── step5_cpm_metrics.py            # CPM score computation & sensitivity checks
│   ├── step6_three_way_comparison.py   # Statistical tests & three-way verification
│   └── step7_test_split_validation.py  # Held-out bootstrap test validation
├── models/                        # GNN and ML models
│   ├── graphsage_optimal.pt       # Trained GraphSAGE optimal checkpoint
│   ├── dpsn_model.py              # Dynamic Patient Similarity Network implementation
│   ├── gnn_models.py              # GCN, GraphSAGE, GAT PyG modules
│   └── baseline_models.py         # Classical baselines (LightGBM, XGBoost, SVM, etc.)
├── paper_assets/                  # IEEE Conference paper & LaTeX artifacts
│   ├── IEEE_Dual_Layer_Cardiovascular_Paper.pdf
│   └── table_*.tex                # LaTeX formatted result tables
├── docs/                          # Comprehensive reports & documentation
│   └── BCE497J_Comprehensive_Project_Report_Final.docx
├── graph_construction.py          # 4 Patient similarity graph strategies
├── config.py                      # Global parameters, clinical weights, and thresholds
└── main.py                        # Pipeline execution entry point
```

---

## 🚀 Local Quickstart

### 1. Run the Interactive Dashboard Locally
```bash
# Start a local HTTP server
python -m http.server 8088 --directory dashboard
# Open in your browser: http://localhost:8088
```

### 2. Run the Full CPM Pipeline
```bash
python cpm/run_cpm_pipeline.py
```

---

## 📜 Citation & Authors
- **Research Owners:** Aryan Verma & Anagh Vasishtha
- **Affiliation:** Research Project / IEEE Conference Submission
- **Target Paper:** *Counterfactual-Induced Patient Migration for Graph-Based Cardiovascular Disease Risk Prediction*
