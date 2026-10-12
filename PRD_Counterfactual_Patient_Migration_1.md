# Product Requirements Document

## Counterfactual-Induced Patient Migration for Graph-Based Cardiovascular Disease Risk Prediction

**Project:** Dynamic Patient Similarity Network (DPSN) — CVD Risk Prediction
**Document owner:** Aryan and Anagh
**Status:** Draft v1.0
**Target output:** IEEE conference paper + supporting Streamlit demo

---

## 1. Background

The existing DPSN project has completed classical ML baselines, graph construction strategies, and benchmark GNN training (GCN, GraphSAGE, GAT) for CVD risk prediction. Counterfactual explanation methods for tabular risk models are well established (change a patient's modifiable features to see how their predicted risk would change), and graph-based patient similarity networks are also established. What is *not* established in prior work is what happens to a patient's **position within the similarity graph** after a counterfactual intervention — i.e., whether a hypothetical healthier version of a patient would also be structurally re-embedded among a healthier peer group, and whether that structural shift carries additional predictive or explanatory signal beyond the risk score change alone.

This PRD defines a new pipeline component — **Counterfactual Patient Migration (CPM)** — built on top of the existing DPSN infrastructure. This is the core novel contribution the paper will center on, with the existing risk-prediction work serving as its foundation.

## 2. Objective

Build and validate a pipeline that:

1. Generates realistic, constrained counterfactuals for high-risk patients.
2. Recomputes each counterfactual patient's position in the patient similarity graph rather than assuming their original neighborhood still applies.
3. Quantifies how much a patient's graph neighborhood changes as a result of the counterfactual intervention ("migration").
4. Demonstrates that this migration signal is informative beyond risk-score change alone, through a controlled three-way comparison.
5. Validates that migration patterns discovered on the development/synthetic population generalize to an independent benchmark dataset.

## 3. Non-Goals

- The GraphSAGE architecture itself is **not** the novelty and should remain a standard, well-documented implementation — no architectural innovation is required or expected here.
- This PRD does not cover the panel presentation or Streamlit UI polish; those are tracked separately, though the CPM metric is expected to become the centerpiece of the Streamlit "what-if" demo.
- Real-time or production deployment is out of scope — this is a research pipeline producing paper-ready results and figures.

## 4. Success Criteria

- A working end-to-end pipeline from raw benchmark data to final validated migration metrics, reproducible via `run_full_pipeline.py` or equivalent.
- Synthetic patient population statistically resembles the benchmark data (distributions, correlations, class balance, and clinically plausible ranges all pass validation checks defined in Phase 2).
- At least one clearly defined, quantifiable Counterfactual Patient Migration metric with a defensible computation method.
- A three-way comparison (normal counterfactual vs. fixed-graph counterfactual vs. topology-aware counterfactual) that shows a measurable, non-trivial difference attributable to graph migration.
- Migration patterns observed in the synthetic/development population also appear — even if attenuated — in the independent benchmark dataset.
- All results are reproducible from a documented, checkpointed pipeline with clear intermediate artifacts (so any phase can be re-run independently).

## 5. Pipeline Requirements

### Phase 1 — Data Preparation
**Input:** Benchmark CVD dataset (to be provided) + existing project dataset.
**Requirements:**
- Map the benchmark dataset's CVD-related features onto the existing dataset's feature schema; document every mapping decision (renamed fields, unit conversions, encoding differences) in a data dictionary.
- Remove all patient identifiers.
- Handle missing values with a documented, justified strategy per feature (not a single blanket method) — e.g., clinical-plausible imputation for vitals vs. flagging/dropping for high-missingness fields.
- Produce a train/validation/test split with fixed random seed, stratified on CVD outcome to preserve class balance across splits.
**Output artifacts:** Cleaned dataset, data dictionary, split indices, a data-quality report (missingness %, feature ranges before/after cleaning).

### Phase 2 — Synthetic Patient Generation
**Requirements:**
- Fit a tabular generative model (CTGAN or TVAE) on the *training split only* of the benchmark data, to avoid leakage into validation/test.
- Generate a synthetic patient population at a scale sufficient to strengthen graph density and statistical power (exact N to be determined empirically, tracked as a config parameter, not hardcoded).
- Validate synthetic data against real data on: univariate distributions (e.g., KS test per feature), pairwise correlations, class balance, and clinical range plausibility (no synthetic patient outside physiologically possible bounds).
**Output artifacts:** Synthetic patient dataset, validation report with pass/fail per check, and the fitted generator (for reproducibility/regeneration).

### Phase 3 — Patient Similarity Network Construction
**Requirements:**
- Represent each patient (real + synthetic) as a graph node.
- Construct a KNN-based similarity graph using a clinically justified distance/similarity metric over relevant features.
- Test multiple K values; select the best K using validation-set performance (not test-set, to avoid leakage), with the selection criterion explicitly defined (e.g., downstream GNN validation AUROC, or graph modularity/quality metric — decide and document which).
**Output artifacts:** Graph construction module output (adjacency structure) per candidate K, a K-selection report with the comparison metric across all tested K values.

### Phase 4 — CVD Risk GNN
**Requirements:**
- Train a standard GraphSAGE model on the constructed patient graph to predict CVD risk probability.
- Architecture stays standard/off-the-shelf — this is infrastructure, not the contribution being claimed.
- Report standard classification metrics (AUROC, F1, precision/recall, calibration) on held-out test data.
**Output artifacts:** Trained GraphSAGE model checkpoint, evaluation report.
**Dependency note:** This phase can likely reuse/extend the existing benchmark GNN work already completed in the base DPSN project rather than starting from scratch.

### Phase 5 — High-Risk Patient Identification
**Requirements:**
- Define and document a clear, justified threshold (or top-percentile cutoff) for "high risk" based on the trained model's predicted probabilities.
- Select the cohort of high-risk patients that will be subjects for counterfactual analysis.
**Output artifacts:** High-risk patient cohort list with selection criteria documented.

### Phase 6 — Constrained Counterfactual Generation
**Requirements:**
- For each high-risk patient, generate a hypothetical modified version by changing only permitted/modifiable features (e.g., blood pressure, cholesterol, lifestyle factors) while holding immutable features fixed (e.g., age, sex, genetic markers).
- Explicitly define, per feature, whether it is mutable or immutable, and the plausible range of change for mutable features (i.e., no physiologically impossible jumps).
- Objective: find the minimal feature change that reduces predicted CVD risk below a defined threshold, using an appropriate counterfactual-generation method (e.g., gradient-based search, DiCE, or a custom constrained optimizer).
- Track and report the success rate (not every patient will yield a valid, minimal counterfactual within constraints).
**Output artifacts:** Counterfactual patient records (original + modified feature vectors), per-patient success/failure flag, and the magnitude of change per feature.

### Phase 7 — Graph Position Rebuild (Core Novelty)
**Requirements:**
- For every successfully generated counterfactual, recompute that patient's similarity to the *entire* population using the same similarity metric and K from Phase 3 — do not assume or retain the original neighbor set.
- Determine the new K-nearest neighbors for the counterfactual version of the patient.
- This must be a genuine recomputation against the full graph, not an approximation or incremental patch, to ensure validity of the migration measurement.
**Output artifacts:** Before/after neighbor sets per patient, before/after graph embeddings (from the trained GraphSAGE model) per patient.

### Phase 8 — Patient Migration Measurement (Additional Novelty)
**Requirements:** Define and compute, per patient:
- **Neighbor turnover:** count/percentage of neighbors that changed between original and counterfactual positions (e.g., Jaccard distance between neighbor sets).
- **Directional movement:** whether the patient's new neighborhood has lower average CVD risk/prevalence than the original neighborhood.
- **Neighborhood prevalence shift:** the magnitude of change in neighborhood CVD prevalence (original vs. new).
- **Representation shift:** distance moved in the learned GraphSAGE embedding space (original vs. new embedding), using a defined distance metric (e.g., cosine or Euclidean).
- Combine these into a single, well-defined **Counterfactual Patient Migration (CPM)** score or a small set of component metrics — the exact formula must be explicit and justified, not an ad hoc combination.
**Output artifacts:** Per-patient CPM metric values, aggregate distribution of CPM across the high-risk cohort, visualizations (e.g., before/after neighborhood plots for illustrative case studies).

### Phase 9 — Critical Comparison
**Requirements:** Run and compare three conditions on the same high-risk cohort:
1. **Normal counterfactual:** risk change only, no graph consideration at all.
2. **Fixed-graph counterfactual:** risk change computed while keeping the patient's original graph neighbors fixed (i.e., ignoring migration).
3. **Topology-aware counterfactual:** risk change **plus** graph migration (the full CPM pipeline).
- Report differences across all three in both predictive terms (does considering topology change which counterfactuals are judged "successful" or "minimal"?) and interpretive terms (does the CPM condition surface information the other two miss?).
- This comparison is the primary evidence for the paper's novelty claim, so results must be presented with statistical rigor (effect sizes, not just point estimates).
**Output artifacts:** Comparison table/figure across the three conditions, statistical test results supporting any claimed difference.

### Phase 10 — Benchmark Validation
**Requirements:**
- Repeat the counterfactual + migration measurement pipeline (Phases 5–9) on the independent benchmark dataset (not the synthetic-augmented development population).
- Assess whether the migration patterns found in development also appear in the benchmark data — direction of effect should be consistent, even if magnitude differs.
- Explicitly report cases where benchmark results diverge from development findings, rather than only reporting confirmatory results.
**Output artifacts:** Benchmark validation report, side-by-side comparison of development vs. benchmark migration patterns.

## 6. System Flow Summary

```
Benchmark data → Data prep → Synthetic population (CTGAN/TVAE)
   → Patient Similarity Graph (KNN, K selected on validation)
   → GraphSAGE risk model
   → High-risk patient identification
   → Constrained counterfactual generation
   → Graph position rebuild (recompute neighbors — core novelty)
   → Patient migration quantification (CPM metric — additional novelty)
   → Three-way comparison (normal vs. fixed-graph vs. topology-aware)
   → Benchmark dataset validation
```

## 7. Key Risks / Open Questions

- **Counterfactual generation may have a low success rate** for some high-risk patients under tight mutability constraints — need a fallback plan (e.g., relax constraints, report failure rate as a finding rather than treating it as a bug).
- **CPM metric definition risk:** combining neighbor turnover, prevalence shift, and embedding distance into one coherent metric needs care to avoid an arbitrary-feeling composite — consider reporting components separately if a single score proves hard to justify.
- **Synthetic data fidelity:** if CTGAN/TVAE-generated patients don't pass validation (Phase 2), the graph built on them will be unreliable — this is a hard gate, not something to proceed past on partial validation.
- **K selection criterion** needs to be locked down early (Phase 3) since it affects every downstream phase.
- **Benchmark dataset is not yet provided** — data mapping in Phase 1 cannot be finalized until it's available; treat the data dictionary as provisional until then.
- **Compute/runtime cost:** recomputing full-graph similarity (Phase 7) for every counterfactual patient could be expensive at scale — needs a performance check early rather than discovering it late.

## 8. Deliverables Checklist

- [ ] Data dictionary + cleaned dataset + splits (Phase 1)
- [ ] Synthetic population + validation report (Phase 2)
- [ ] Patient similarity graph + K-selection report (Phase 3)
- [ ] Trained GraphSAGE risk model + evaluation report (Phase 4)
- [ ] High-risk cohort with documented selection criteria (Phase 5)
- [ ] Constrained counterfactual generator + generated counterfactuals (Phase 6)
- [ ] Graph rebuild module (Phase 7)
- [ ] CPM metric implementation + per-patient results (Phase 8)
- [ ] Three-way comparison results (Phase 9)
- [ ] Benchmark validation report (Phase 10)
- [ ] Paper-ready figures and tables summarizing the above
