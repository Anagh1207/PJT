Product Requirements Document (v2.0 — Updated for Current Project State)
Counterfactual-Induced Patient Migration for Graph-Based Cardiovascular Disease Risk Prediction

Project: Dynamic Patient Similarity Network (DPSN) — CVD Risk Prediction Document owners: Aryan and Anagh Status: Draft v2.0 — supersedes v1.0 Target output: IEEE conference paper + supporting Streamlit demo

Changelog from v1.0
No second/benchmark dataset. v1.0 assumed a separate benchmark dataset would be mapped onto the existing schema. That's no longer the plan — a single existing dataset is used for everything, including final validation via its held-out test split.
Data prep, graph construction, and the risk GNN are already built, from the original (pre-CPM) project pipeline. This PRD no longer treats them as new work — they're inputs to reuse, not phases to execute.
Phases renumbered and restructured around what's actually new work, with two early steps parallelized to save time.
Frontend scope trimmed given time constraints — the three-way comparison view is now explicitly optional/cut-if-needed rather than core.
1. Background

The existing DPSN project has completed data cleaning, classical ML baselines, graph construction (4 similarity metrics), and GNN training (GCN, GraphSAGE, GAT) for CVD risk prediction, plus the DPSN model itself, ablations, statistical validation, explainability, and publication figures. Counterfactual explanation methods for tabular risk models are well established, and graph-based patient similarity networks are also established. What is not established in prior work is what happens to a patient's position within the similarity graph after a counterfactual intervention — whether a hypothetical healthier version of a patient would also be structurally re-embedded among a healthier peer group, and whether that structural shift carries additional signal beyond the risk score change alone.

This PRD defines the new pipeline component — Counterfactual Patient Migration (CPM) — built on top of the already-completed infrastructure. This is the paper's core novel contribution.

2. Objective

Build and validate a pipeline that:

Generates realistic, constrained counterfactuals for high-risk patients.
Recomputes each counterfactual patient's position in the patient similarity graph rather than assuming their original neighborhood still applies.
Quantifies how much a patient's graph neighborhood changes as a result ("migration").
Demonstrates that this migration signal is informative beyond risk-score change alone, through a controlled three-way comparison.
Validates that migration patterns discovered on the development/synthetic population also hold on a held-out test split of the same dataset.
3. Non-Goals
The GraphSAGE architecture itself is not the novelty — reuse the existing trained model rather than re-architecting it.
Real-time or production deployment is out of scope — this is a research pipeline producing paper-ready results, plus a demo frontend.
Exhaustive frontend polish is out of scope given time constraints — see Section 8.
4. What's Already Done (Reuse — Do Not Rebuild)
Component	Status	Source
Data cleaning, missing-value handling, splits	Done	phase1_01_audit_and_preprocessing.py, phase1_02_eda.py
Patient similarity graph (KNN, 4 metrics: Euclidean, Cosine, Clinical, Dynamic)	Done	graph_construction.py
GraphSAGE CVD risk model	Done	training/graphsage_pipeline.py
Classical ML baselines (LightGBM best, F1 0.894)	Done	phase2_classical_ml_baselines.py, main.py
DPSN model (F1 0.813, ROC-AUC 0.988)	Done	models/dpsn_model.py
Ablations, stats, explainability, figures, paper assets	Done	models/dpsn_ablations.py, evaluation/, visualizations/publication_figures.py, paper_assets/

Note: the existing GraphSAGE checkpoint should only be retrained if adding synthetic patients (Section 5, Track A) meaningfully changes the graph structure enough to warrant it — this is a judgment call to flag and confirm, not something to do by default.

5. New Work — Execution Plan
Track A and Track B run in parallel (no dependency between them)

Track A — Synthetic Patient Generation

Fit CTGAN or TVAE on the training split only (no leakage into validation/test).
Generate a synthetic population at a scale sufficient to strengthen graph density (N tracked as a config parameter).
Hard gate: validate against real data — univariate distributions (KS test), pairwise correlations, class balance, clinical range plausibility. Must pass before this output is used downstream, but does not block Track B.
Output artifacts: synthetic patient dataset, validation report, fitted generator.

Track B — High-Risk Patient Identification

Define and justify a threshold/percentile cutoff on the existing GraphSAGE model's predicted probabilities (does not wait on Track A).
Select the high-risk cohort for counterfactual analysis.
Output artifacts: high-risk cohort list, selection criteria, cohort size.
Sequential from here — each step depends on the last

Step 3 — Constrained Counterfactual Generation

For the Track B cohort, generate modified patient versions changing only mutable features (define mutable vs. immutable explicitly, documented before generation), within physiologically plausible ranges.
Objective: minimal change that reduces predicted risk below a defined threshold.
Track and report success rate.
Output artifacts: counterfactual records, per-patient success/failure flag, magnitude of change per feature.

Step 4 — Graph Position Rebuild (Core Novelty)

For every successful counterfactual, recompute similarity against the full population (real + Track A synthetic) using the existing metric/K — genuine recomputation, not an approximation.
Determine new K-nearest neighbors.
Output artifacts: before/after neighbor sets, before/after GraphSAGE embeddings.

Step 5 — Patient Migration Measurement (Additional Novelty)

Compute per patient: neighbor turnover, directional movement, neighborhood prevalence shift, representation-space distance moved.
Hard gate: propose the exact CPM metric formula/components (single composite score vs. reported separately) and get sign-off before implementing or moving to Step 6.
Output artifacts: per-patient CPM values, aggregate distribution, illustrative before/after visualizations.

Step 6 — Three-Way Comparison

Compare: normal counterfactual (risk change only), fixed-graph counterfactual (risk change, neighbors held fixed), topology-aware counterfactual (risk change + migration).
Report with statistical rigor (effect sizes, not just point estimates). This is the primary novelty evidence for the paper.
Output artifacts: comparison table/figure, statistical test results.

Step 7 — Validation on Held-Out Test Split

Repeat Steps 2–6 (Track B high-risk identification through the three-way comparison) on the test split rather than the synthetic-augmented development population.
Confirm the test split has enough high-risk patients for this to be statistically meaningful before running the full analysis — flag if too small rather than proceeding anyway.
Report divergences from development-population findings, not just confirmations.
Output artifacts: validation report, side-by-side development-vs-test-split comparison.
6. System Flow Summary
[Existing dataset, cleaned] ─┬─→ Track A: Synthetic generation (CTGAN/TVAE) ─┐
                              └─→ Track B: High-risk identification           │
                                          (using existing GraphSAGE model)    │
                                                     │                        │
                                                     ▼                        ▼
                              Step 3: Constrained counterfactual generation
                                                     │
                                                     ▼
                    Step 4: Graph position rebuild (real + synthetic population)
                                                     │
                                                     ▼
                         Step 5: Patient migration quantification (CPM)
                                                     │
                                                     ▼
                    Step 6: Three-way comparison (normal / fixed-graph / topology-aware)
                                                     │
                                                     ▼
                         Step 7: Validation on held-out test split
                                                     │
                                                     ▼
                              Frontend: CPM "what-if" visualizer
7. Key Risks / Open Questions
Counterfactual generation may have a low success rate under tight mutability constraints — report failure rate as a finding, not a bug to hide.
CPM metric definition risk: avoid an arbitrary-feeling composite; report components separately if a single score proves hard to justify (Step 5 hard gate exists for this reason).
Synthetic data fidelity (Track A): if validation fails, the graph built on it is unreliable — hard gate, not a soft warning.
Test-split size risk (Step 7): a single dataset now serves both development and final validation — if the test split has too few high-risk patients, the validation step won't be statistically meaningful. Must be checked before running, not after.
GraphSAGE retrain decision: adding synthetic patients might warrant retraining the existing model — this needs an explicit judgment call, not an automatic default either way.
Time constraint trade-offs: given the compressed timeline, any step where the "ideal" PRD version would take significantly longer than a simplified version should be flagged as a trade-off decision, not silently resolved either direction.
8. Frontend Scope (Given Time Constraints)

Core (must-have):

Patient selection (cohort dropdown + random pick)
Before-view: risk score + local subgraph + neighbor stats
Intervention controls scoped to mutable features only
After-view: recomputed risk score + recomputed subgraph + neighbor stats
Migration summary panel with a plain-language explanation sentence

Optional (cut first if time runs short):

Three-way comparison tab (Step 6 results surfaced in the UI)
Fully custom/live patient input (beyond the pre-selected cohort)

Build as a new page in streamlit_app/pages/, following the existing page pattern. Precompute and cache results for the demo cohort — live computation should not be relied on for the presentation itself.

9. Success Criteria
Track A and Track B both complete and pass their respective checks.
Steps 3–6 produce a working, reproducible counterfactual → migration → comparison chain on the development population.
Step 7 confirms (or honestly reports divergence in) the migration pattern on the held-out test split.
The core frontend view (Section 8) runs smoothly for a live demo.
Every hard gate (Track A validation, CPM metric sign-off, test-split size check) was actually paused on and confirmed, not silently passed through.

10. Deliverables Checklist
 Track A: synthetic population + validation report
 Track B: high-risk cohort + selection criteria
 Step 3: counterfactual records + success rate
 Step 4: before/after neighbor sets + embeddings
 Step 5: CPM metric (approved formula) + per-patient results
 Step 6: three-way comparison results with statistical rigor
 Step 7: held-out test split validation report
 Frontend: core before/after + migration summary view (Section 8 core scope)
 (Optional, if time allows) Frontend three-way comparison tab
 Paper-ready figures/tables summarizing the above