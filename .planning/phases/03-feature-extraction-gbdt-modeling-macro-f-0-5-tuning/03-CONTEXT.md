# Phase 3: Feature Extraction, GBDT Modeling & Macro $F_{0.5}$ Tuning - Context

**Gathered:** 2026-09-27
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers high-performance 24-dimensional feature extraction (RapidFuzz C++ SIMD string metrics, character 3-grams, street number conflict detection, missing address indicators), balanced training pair dataset construction with hard negative mining, LightGBM binary classifier training with ROC-AUC early stopping, and grid-search calibration of the decision threshold $\theta^* \approx 0.70$ optimized specifically for the competition's Macro $F_{0.5}$ metric while strictly protecting singleton accuracy.

</domain>

<decisions>
## Implementation Decisions

### Negative Sampling & Hard Negative Mining Strategy
- **D-01:** Mine hard negatives directly from Phase 2 candidate blocking results for training Source 1 entities. If an entity has true match in candidate pool, the remaining top candidates serve as hard negatives. Non-candidate random negatives are used only if candidates are insufficient.
- **D-02:** Standardize on a 1:4 positive-to-negative training pair ratio. This balances learning signal without skewing gradient descent or creating unwieldy pairwise datasets.
- **D-03:** Validation pairs are generated from the official 20% validation split entities without downsampling, reflecting realistic inference candidate distributions.

### 24-Dimensional Feature Vector Architecture
- **D-04:** Engineer exactly 24 dense, highly discriminative features across five logical families:
  1. *Name String Similarity (5 features):* RapidFuzz Jaro-Winkler, Levenshtein ratio, Token Sort ratio, Token Set ratio, Token Containment.
  2. *Address String Similarity (4 features):* RapidFuzz Jaro-Winkler, Levenshtein ratio, Token Sort ratio, Token Set ratio.
  3. *Composite Text Similarity (3 features):* Name + Address concatenated Jaro-Winkler, Levenshtein ratio, Token Jaccard.
  4. *Character N-Gram Jaccard (2 features):* Char 3-gram Jaccard on name, Char 3-gram Jaccard on address.
  5. *Street Number & Numeric Conflict Detection (3 features):* Extracted digits Jaccard similarity, `has_exact_number_match` boolean flag, `has_number_conflict` penalty flag (1 if both records contain street numbers and they disagree, 0 otherwise).
  6. *Missing Field & Length Ratio Indicators (5 features):* `s1_has_addr`, `cand_has_addr`, `both_have_addr`, `name_len_diff_ratio`, `addr_len_diff_ratio`.
  7. *Blocking Prior Signals (2 features):* Blocking similarity score (TF-IDF dot product) and candidate rank index (1 to 25).
- **D-05:** RapidFuzz C++ SIMD bindings (`rapidfuzz.distance`) must be used for all string metrics to maintain high throughput (>50,000 pair evaluations/sec).

### GBDT Classifier Architecture & Training
- **D-06:** Model selection: `lightgbm.LGBMClassifier` trained with binary cross-entropy (`objective="binary"`).
- **D-07:** Early stopping on validation ROC-AUC and PR-AUC with 50-round patience.
- **D-08:** Default hyperparameters: `learning_rate=0.05`, `n_estimators=1000`, `num_leaves=63`, `max_depth=7`, `min_child_samples=50`, `colsample_bytree=0.8`, `subsample=0.8`.

### Decision Threshold Tuning & Macro $F_{0.5}$ Optimization
- **D-09:** Grid search threshold $\theta^* \in [0.40, 0.90]$ with step 0.02 evaluated against the exact competition validation metric (Macro $F_{0.5}$ with $\beta=0.5$).
- **D-10:** Strict single-match assignment during inference: For any Source 1 entity, among candidate pairs with $P(\text{match}) \ge \theta^*$, assign the candidate with $\arg\max P(\text{match})$. If all candidates fall below $\theta^*$, predict singleton (`None` / empty prediction).
- **D-11:** Singleton protection: Because singletons comprise >95% of test queries and false positives severely penalize precision in $F_{0.5}$, threshold $\theta^*$ is calibrated conservative-first to ensure validation singleton accuracy >95%.

### the agent's Discretion
- User deferred interactive Q&A directly to planning ("just move to planning"). The agent selected proven, high-performance ML engineering defaults aligned directly with REQUIREMENTS.md (FEAT-01..FEAT-04 and MODEL-01..MODEL-04) and ROADMAP.md Phase 3 deliverables.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Specifications & Data Contracts
- `.planning/PROJECT.md` — Competition overview, Macro $F_{0.5}$ metric formulation, and system constraints.
- `.planning/REQUIREMENTS.md` — Formal requirements FEAT-01 to FEAT-04 and MODEL-01 to MODEL-04.
- `.planning/ROADMAP.md` — Phase 3 deliverables, success criteria, and verification targets.

### Existing Architecture & Pipelines
- `src/preprocessing.py` — Text cleaning, corporate suffix normalization, address canonicalization.
- `src/blocking.py` — Inverted index and char-ngram cascade blocking candidate generator.
- `scripts/split_dataset.py` — Official 80/20 train/validation split indices.
- `scripts/verify_submission.py` — Submission schema validator and duplicate check.

### External Libraries & APIs
- `rapidfuzz.distance` — High-speed C++ SIMD Levenshtein, Jaro-Winkler, and Token Sort implementations.
- `lightgbm.LGBMClassifier` — GBDT classification, early stopping, and feature importance.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src.preprocessing.clean_corporate_name`: Standardized corporate text without suffixes/diacritics.
- `src.preprocessing.clean_address_text`: Cleaned address string with canonical abbreviations.
- `src.blocking.CandidateBlockingEngine`: Generator of candidate pairs with similarity scores and ranks.
- `data/splits/train_ids.parquet` and `data/splits/val_ids.parquet`: Deterministic 80/20 train/validation split.

### Established Patterns
- High-throughput pandas/pyarrow ingestion with `sep="\t"` and empty string fill for missing text.
- Memory-efficient batch processing for large candidate pools.
- Deterministic random seed (`random_state=42`) across all splitting and model training routines.

### Integration Points
- Input: Preprocessed records from Phase 2 + candidate pairs from `CandidateBlockingEngine`.
- Output: `src/features.py` (24-dim extractor), `src/model.py` (LightGBM trainer/predictor), `scripts/train_gbdt.py`, and `scripts/tune_threshold.py`.

</code_context>

<specifics>
## Specific Ideas

- Focus heavily on street number conflict penalties: Two entities with identical or near-identical names on the same street but different building numbers (e.g., "102 Main St" vs "108 Main St") must receive a strong negative penalty to avoid false positive merges.
- Missing address fields should not zero-out name match scores; indicator flags (`s1_has_addr`, `cand_has_addr`, `both_have_addr`) allow LightGBM to adjust its decision boundary based on available metadata.

</specifics>

<deferred>
## Deferred Ideas

- Cross-encoder / bi-encoder deep learning reranker (e.g. DeBERTa-v3) — deferred to Phase 4 (Deep Learning Embeddings & Neural Reranking).
- Graph community detection / transitive closure resolution — deferred to Phase 5 (Ensemble, Post-Processing & Graph Transitivity).

</deferred>

---

*Phase: 03-feature-extraction-gbdt-modeling-macro-f-0-5-tuning*
*Context gathered: 2026-09-27*
