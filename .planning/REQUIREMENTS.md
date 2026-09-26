# Requirements: Amazon ML Challenge 2026 Business Entity Resolution

**Defined:** 2026-09-27  
**Core Value:** Maximize macro $F_{0.5}$ score through country-partitioned cascade blocking and precision-optimized LightGBM classification, strictly adhering to competition rules and submission packaging formats.

## v1 Requirements

### Environment & Ingestion (ENV)

- [ ] **ENV-01**: Ingestion and validation of training files (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`) and test files (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`) without line-merging truncation.
- [ ] **ENV-02**: Offline validation harness running `student_resource/utils/validate_submission.py` on dummy/baseline submissions to establish deterministic exit 0 checks.
- [ ] **ENV-03**: Stratified holdout split module (80/20 train/val partitioned strictly by Source 1 entity IDs) ensuring zero data leakage.

### Text Preprocessing & Cleaning (PREP)

- [ ] **PREP-01**: Unicode NFKD normalization decomposing diacritics, multilingual accents, and non-ASCII glyphs.
- [ ] **PREP-02**: Corporate legal suffix remover stripping 30+ business extensions (`LLC`, `INC`, `CORP`, `PVT LTD`, `LTD`, `SA`, `SAS`, `GMBH`, `CO`) to expose root business entity names.
- [ ] **PREP-03**: Address normalizer expanding common abbreviations (`ST`, `RD`, `AVE`, `BLVD`) and standardizing whitespace and punctuation.

### Cascade Blocking & Candidate Generation (BLOCK)

- [ ] **BLOCK-01**: 100% strict country partitioning separating records by open-set country strings (`US`, `India`, `France`), eliminating cross-country comparisons.
- [ ] **BLOCK-02**: Multi-key inverted index cascade (`by_both`, `by_name`, `by_addr`, `by_pref`) generating high-recall candidates for every Source 1 entity.
- [ ] **BLOCK-03**: Candidate generation exporter producing `output/candidate_pairs.tsv` conforming to official format rules (one row per S1 entity, comma-separated candidate IDs).
- [ ] **BLOCK-04**: Empirical recall ceiling measurement on validation split demonstrating $\ge 95\%$ pair recall ceiling with $>99.9\%$ comparison space reduction.

### Feature Extraction (FEAT)

- [ ] **FEAT-01**: Fast C++ SIMD string similarity metric computation via RapidFuzz (Jaro-Winkler, Levenshtein ratio, Token Jaccard, Token Containment).
- [ ] **FEAT-02**: Character 3-gram Jaccard similarity computation for typo and transliteration resilience.
- [ ] **FEAT-03**: Numeric and street conflict detection parsing digits/street numbers via regex, computing numeric Jaccard and setting `has_number_conflict` penalty flags.
- [ ] **FEAT-04**: Missing address indicator flags (`s1_has_addr`, `cand_has_addr`, `both_have_addr`) ensuring missing addresses do not unfairly penalize valid name matches.

### Model Training & Threshold Calibration (MODEL)

- [ ] **MODEL-01**: Training set pair generator constructing balanced positive and hard-negative pairs from candidate generation.
- [ ] **MODEL-02**: LightGBM binary classifier training with early stopping on validation ROC-AUC and PR-AUC.
- [ ] **MODEL-03**: Macro $F_{0.5}$ threshold tuner optimizing decision threshold $\theta^* \in [0.40, 0.90]$ to balance precision $2\times$ over recall and protect singleton entities.
- [ ] **MODEL-04**: Validation evaluation module calculating exact macro $F_{0.5}$ score including singleton bonus across all validation Source 1 entities.

### Remote Compute & Acceleration (REMOTE)

- [ ] **REMOTE-01**: Kaggle automated kernel packager bundling modular code and dataset inputs for execution on Kaggle dual-T4 GPUs.
- [ ] **REMOTE-02**: Kernel status poller and artifact synchronizer downloading outputs back to local repository (`output/` and model checkpoints).

### Submission Packaging & Audit (SUB)

- [ ] **SUB-01**: Full 1.73M test set inference pipeline generating `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- [ ] **SUB-02**: Official validation verification executing `validate_submission.py` and obtaining clean exit code 0 (`PASS`).
- [ ] **SUB-03**: Packaging script creating `<team_name>_submission.zip` with `output/`, self-contained `code/business_entity_resolution/`, and completed `Documentation_template.md`.

## v2 Requirements (Deferred Enhancements)

- **DENSE-01**: Fine-tuned Siamese bi-encoder embeddings (e.g. `bge-small-en-v1.5` or `all-MiniLM-L6-v2`) generating cosine similarity features for ambiguous pairs.
- **ENSEMBLE-01**: Blending CatBoost and XGBoost rankers with LightGBM predictions.

## Out of Scope

| Feature | Reason |
|---------|--------|
| External geocoding APIs | Strictly forbidden by competition rules; leads to immediate disqualification |
| External business registries | Strictly forbidden by competition rules; leads to immediate disqualification |
| Deep neural networks > 8B parameters | Exceeds competition model size limit (max 8B parameters) |
| $O(N^2)$ all-pairs exhaustive scoring | Computationally intractable ($>17.6$ trillion pairs) |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Pending |
| ENV-02 | Phase 1 | Pending |
| ENV-03 | Phase 1 | Pending |
| REMOTE-01 | Phase 1 | Pending |
| REMOTE-02 | Phase 1 | Pending |
| PREP-01 | Phase 2 | Pending |
| PREP-02 | Phase 2 | Pending |
| PREP-03 | Phase 2 | Pending |
| BLOCK-01 | Phase 2 | Pending |
| BLOCK-02 | Phase 2 | Pending |
| BLOCK-03 | Phase 2 | Pending |
| BLOCK-04 | Phase 2 | Pending |
| FEAT-01 | Phase 3 | Pending |
| FEAT-02 | Phase 3 | Pending |
| FEAT-03 | Phase 3 | Pending |
| FEAT-04 | Phase 3 | Pending |
| MODEL-01 | Phase 3 | Pending |
| MODEL-02 | Phase 3 | Pending |
| MODEL-03 | Phase 3 | Pending |
| MODEL-04 | Phase 3 | Pending |
| SUB-01 | Phase 4 | Pending |
| SUB-02 | Phase 4 | Pending |
| SUB-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-27*  
*Last updated: 2026-09-27 after initialization*
