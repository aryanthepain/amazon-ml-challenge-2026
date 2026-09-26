# Roadmap: Amazon ML Challenge 2026 Business Entity Resolution

## Overview

Build, optimize, and package an end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026 Business Entity Resolution challenge. The pipeline progresses from local data validation and remote Kaggle compute setup to high-recall cascade blocking, precision-tuned LightGBM scoring on 24 engineered features, and full-scale 1.73M test inference producing an audited, compliant `<team_name>_submission.zip`.

## Phases

- [x] **Phase 1: Foundation, Ingestion & Compute Infrastructure** - Set up data pipelines, local smoke tests, official submission validation harness, and Kaggle GPU automation. (completed 2026-09-27)
- [x] **Phase 2: Text Preprocessing & Cascade Blocking Engine** - Implement Unicode NFKD cleaning, corporate suffix stripping, country-partitioned cascade inverted indexes, and emit `candidate_pairs.tsv`. (completed 2026-09-27)
- [ ] **Phase 3: Feature Extraction, GBDT Modeling & Macro $F_{0.5}$ Tuning** - Extract 24-dim similarity/conflict features, train LightGBM classifier, and tune decision threshold $\theta^* \approx 0.70$ for Macro $F_{0.5}$.
- [ ] **Phase 4: Full Test Set Inference, Verification & Submission Packaging** - Execute full 1.73M test inference via Kaggle/local GPUs, verify `validate_submission.py` exit 0, and build `<team_name>_submission.zip`.

## Phase Details

### Phase 1: Foundation, Ingestion & Compute Infrastructure

**Goal**: Establish verified data ingestion, offline validation harness, stratified holdout splits, and automated Kaggle dual-T4 GPU kernel execution.
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03, REMOTE-01, REMOTE-02
**Success Criteria** (what must be TRUE):

  1. Datasets in `student_resource/dataset/` load cleanly without column distortion or line-merging errors.
  2. `validate_submission.py` runs on a dummy submission and emits exit code 0 (`PASS`).
  3. Stratified 80/20 train/validation split is generated with zero Source 1 entity overlap.
  4. Kaggle CLI script connects, pushes a test kernel, and checks remote execution status.

**Plans**: 2 plans

Plans:

- [x] 01-01: Ingestion loaders, stratified split generation, and baseline validation runner.
- [x] 01-02: Remote Kaggle dual-T4 GPU dispatch and artifact synchronization scripts.

---

### Phase 2: Text Preprocessing & Cascade Blocking Engine

**Goal**: Build a scalable text normalization and country-partitioned cascade blocking engine that shrinks 17.6T pair space to high-probability candidates.
**Depends on**: Phase 1
**Requirements**: PREP-01, PREP-02, PREP-03, BLOCK-01, BLOCK-02, BLOCK-03, BLOCK-04
**Success Criteria** (what must be TRUE):

  1. Text normalizer cleanly strips corporate suffixes (`LLC`, `Corp`, `Pvt Ltd`) and handles French accents / Indian transliterations.
  2. Blocking strictly enforces country boundaries (`US`, `India`, `France`), eliminating cross-country comparisons.
  3. Multi-key inverted index cascade (`by_both`, `by_name`, `by_addr`, `by_pref`) achieves $\ge 95\%$ pair recall ceiling on validation split with $>99.9\%$ space reduction.
  4. Exporter emits valid `output/candidate_pairs.tsv` conforming to official competition format.

**Plans**: 2 plans

Plans:
**Wave 1**

- [x] 02-01: Fast text preprocessing, normalization, and corporate suffix stripping.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02-02: Country-partitioned cascade inverted indexing, candidate generation, and recall ceiling auditing.

---

### Phase 3: Feature Extraction, GBDT Modeling & Macro $F_{0.5}$ Tuning

**Goal**: Engineer 24 similarity and conflict features, train a LightGBM classifier, and tune the decision threshold specifically for Macro $F_{0.5}$.
**Depends on**: Phase 2
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04, MODEL-01, MODEL-02, MODEL-03, MODEL-04
**Success Criteria** (what must be TRUE):

  1. RapidFuzz-accelerated feature extractor computes 24-dimensional feature vectors including Jaro-Winkler, Levenshtein, token containment, and character 3-grams.
  2. Street number conflict detector accurately flags contradicting street numbers between candidate records.
  3. LightGBM classifier trains on balanced positive/hard-negative pairs with validation ROC-AUC $\ge 0.99$.
  4. Grid search threshold tuning selects $\theta^* \approx 0.70$, maximizing Macro $F_{0.5}$ and protecting singleton accuracy (>95%).

**Plans**: 2 plans

Plans:

- [ ] 03-01: 24-dimensional feature extraction engine and pair dataset builder.
- [ ] 03-02: LightGBM model training, threshold calibration, and validation evaluation.

---

### Phase 4: Full Test Set Inference, Verification & Submission Packaging

**Goal**: Execute full-scale test inference across all 1,732,544 test entities, achieve official validation pass, and package `<team_name>_submission.zip`.
**Depends on**: Phase 3
**Requirements**: SUB-01, SUB-02, SUB-03
**Success Criteria** (what must be TRUE):

  1. Streaming chunked inference produces `output/matching_results.tsv` and `output/candidate_pairs.tsv` with exactly 1,732,544 rows.
  2. `student_resource/utils/validate_submission.py` outputs `PASS` with exit code 0.
  3. `package_submission.py` creates `Antigravity_AI_submission.zip` containing `output/`, self-contained `code/business_entity_resolution/`, and completed `Documentation_template.md`.

**Plans**: 2 plans

Plans:

- [ ] 04-01: Full 1.73M test inference execution (local/Kaggle dual-T4) and streaming TSV exporter.
- [ ] 04-02: Official validation audit, documentation writeup, and submission zip packaging.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation, Ingestion & Compute Infrastructure | 2/2 | Complete    | 2026-09-27 |
| 2. Text Preprocessing & Cascade Blocking Engine | 2/2 | Complete    | 2026-09-27 |
| 3. Feature Extraction, GBDT Modeling & Macro $F_{0.5}$ Tuning | 0/2 | Not started | - |
| 4. Full Test Set Inference, Verification & Submission Packaging | 0/2 | Not started | - |
