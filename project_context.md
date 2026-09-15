# Project Context: Cardiovascular Disease Risk Prediction using Graph Neural Networks

**Target Publication:** IEEE Conference Paper

This document serves as a summary of the project's current state, detailing what has been implemented so far and outlining the immediate next steps.

## 1. Project Overview
The primary objective of this project is to predict Cardiovascular Disease (CVD) risk. The workflow is designed as a rigorous machine learning experimentation framework, structured in phases to establish strong classical ML baselines before introducing the novel Graph Neural Network (GNN) approach.

## 2. Implemented Phases (Completed)

### Phase 1: Data Audit, Preprocessing, and EDA
**Files:** `phase1_01_audit_and_preprocessing.py`, `phase1_02_eda.py`
* **Data Processing:** The dataset (`GNN DATASET.xlsx`) has been audited and cleaned. Missing values have been handled, outliers analyzed, and features scaled appropriately.
* **Outputs:** Generated reports such as `dataset_audit_report.csv`, `outlier_analysis.csv`, and `scaling_parameters.csv` stored in the `results/` directory.

### Phase 2: Classical Machine Learning Baselines
**Files:** `phase2_classical_ml_baselines.py`, `main.py`, `config.py`
* **Orchestration:** `main.py` serves as the central orchestrator, executing a 12-step pipeline.
* **Data Splitting:** Applied an 80-20 stratified train-test split to maintain class distributions.
* **Class Imbalance Handling:** Integrated SMOTE within the training folds during Cross-Validation to handle imbalanced data effectively without data leakage.
* **Models Trained:** 
  * Logistic Regression
  * Random Forest
  * Support Vector Machine (SVM)
  * LightGBM
  * XGBoost
* **Evaluation:** Models were evaluated using Stratified 5-Fold Cross-Validation, followed by a final test set evaluation. A composite ranking system (using F1-Score and ROC-AUC) was used to compare models.
* **Outputs:** 
  * Trained models serialized as `.pkl` files in the `models/` directory.
  * Evaluation metrics and `model_ranking.csv` saved in the `results/` directory.
  * Comprehensive plots (ROC curves, PR curves, Confusion Matrices, Feature Importance, etc.) generated in the `visualizations/` directory.
  * Detailed classification reports and a dashboard metrics summary prepared.

## 3. Directory Structure and Modules
The codebase is highly modular, organized into specific directories for scalability:
* `data/`: Contains raw and processed dataset files.
* `models/`: Stores serialized trained baseline models (`.pkl` files).
* `training/`: Contains `trainer.py` handling model training and cross-validation loops.
* `evaluation/`: Contains `evaluator.py` for metrics calculation and model ranking.
* `visualizations/`: Contains `visualizer.py` for generating publication-quality plots.
* `dashboard/`: Contains utilities (`dashboard_utils.py`) for summarizing experiment results.
* `utils/`: Contains data handling and logging utilities (`data_utils.py`, `logger.py`).
* `results/`: Stores all CSV and JSON tracking metrics, rankings, and data audit logs.
* `figures/`: Additional directory reserved for plots/diagrams.

## 4. Current State — ALL PHASES COMPLETE
All 10 phases of the project pipeline have been fully executed. The framework has:
* **Phase 3:** Built 4 patient similarity graphs (Euclidean KNN, Cosine, Clinical, Dynamic)
* **Phase 4:** Trained 3 GNN architectures (GCN, GraphSAGE, GAT) across 4 graph types
* **Phase 5:** Trained the novel Dynamic Patient Similarity Network (DPSN)
* **Phase 6:** Completed ablation studies (5 DPSN variants)
* **Phase 7:** Statistical validation (paired t-tests, Wilcoxon, bootstrap CIs)
* **Phase 8:** Explainability analysis (SHAP for tree models, GNNExplainer for DPSN)
* **Phase 9:** Generated 15 publication-quality figures at 300 DPI
* **Phase 10:** Final evidence report, model rankings, and 3 LaTeX tables for IEEE paper

**Key Finding:** LightGBM achieved the best F1 (0.894), with DPSN ranking 7th (F1=0.813) but showing competitive ROC-AUC (0.988). GraphSAGE was the best GNN variant (F1=0.851).

