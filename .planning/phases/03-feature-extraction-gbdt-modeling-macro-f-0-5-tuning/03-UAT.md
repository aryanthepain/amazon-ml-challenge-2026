---
status: complete
phase: 03-feature-extraction-gbdt-modeling-macro-f-0-5-tuning
source:
  - 03-01-SUMMARY.md
  - 03-02-SUMMARY.md
started: 2026-09-27T03:35:00Z
updated: 2026-09-27T03:38:00Z
---

## Current Test

[testing complete]

## Tests

### 1. RapidFuzz SIMD String Metrics & 24-Feature Vector Assembly
expected: Computes Jaro-Winkler, Levenshtein, token sort, token set, char 3-gram Jaccard, normalized strictly to [0.0, 1.0].
result: pass
source: automated
verification_target: pytest tests/test_phase3_features.py::test_feature_names_count && test_extract_24_features_full

### 2. Street Number Conflict Disambiguation
expected: Isolates street numbers with regex; flags contradicting building numerals (e.g. 102 vs 108 Main St) with has_number_conflict == 1.0.
result: pass
source: automated
verification_target: pytest tests/test_phase3_features.py::test_street_number_conflict_detection

### 3. Pairwise Dataset Assembly with 1:4 Hard Negative Mining
expected: Assembles positive ground truth pairs and mines top candidate negatives from blocking at 1:4 ratio with country-partitioned backfilling.
result: pass
source: automated
verification_target: pytest tests/test_phase3_features.py::test_dataset_pair_builder_and_negative_mining

### 4. LightGBM Binary Classifier & Booster Persistence
expected: EntityMatcherLGBM trains with binary cross-entropy, validation ROC-AUC early stopping (15-50 rounds), and C++ text booster export.
result: pass
source: automated
verification_target: pytest tests/test_phase3_model.py::test_lgbm_matcher_fit_predict_and_persistence

### 5. Exact Competition Macro F0.5 Metric Calculation
expected: Implements precision-weighted Macro F0.5 formula (beta=0.5, precision weighted 4x over recall) rewarding singletons correctly.
result: pass
source: automated
verification_target: pytest tests/test_phase3_model.py::test_compute_macro_f05

### 6. Grid Search Threshold Calibration & Singleton Protection
expected: Scans theta in [0.40, 0.90] to select optimal theta* ~= 0.70 enforcing singleton accuracy >= 95%.
result: pass
source: automated
verification_target: pytest tests/test_phase3_model.py::test_grid_search_threshold

### 7. Single Match Resolver & Submission Compliance
expected: SingleMatchResolver assigns highest probability candidate exceeding threshold, assigning empty strings to singletons.
result: pass
source: automated
verification_target: pytest tests/test_phase3_model.py::test_single_match_resolver_behavior

## Summary

total: 7
passed: 7
issues: 0
pending: 0
skipped: 0

## Gaps

[none]
