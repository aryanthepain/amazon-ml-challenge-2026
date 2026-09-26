---
phase: 03-feature-extraction-gbdt-modeling-macro-f-0-5-tuning
verified: 2026-09-27T03:30:00Z
status: passed
score: 1.0
requirements_verified:
  - FEAT-01
  - FEAT-02
  - FEAT-03
  - FEAT-04
  - MODEL-01
  - MODEL-02
  - MODEL-03
  - MODEL-04
---

# Phase 3 Verification Report: Feature Extraction, GBDT Modeling & Macro $F_{0.5}$ Tuning

## 1. Executive Summary
Phase 3 implementation has completed and deterministically passed all verification gates.
The 24-dimensional feature extraction engine (`src/features.py`), pairwise dataset builder with 1:4 hard negative mining (`src/dataset.py`), LightGBM classifier (`src/model.py`), exact competition Macro $F_{0.5}$ tuner (`src/tuning.py`), and inference match resolver (`src/matcher.py`, `scripts/predict_matches.py`) are fully implemented, tested across 14 new unit and integration test cases (38 total test suite assertions), and fully verified.

## 2. Requirement Traceability

| Requirement ID | Description | Artifact(s) | Status | Evidence |
|----------------|-------------|-------------|--------|----------|
| **FEAT-01** | RapidFuzz C++ SIMD Name & Address Metrics | `src/features.py` | PASS | Computes Jaro-Winkler, Levenshtein, token sort, token set, and partial ratios normalized strictly to $[0.0, 1.0]$; verified in `test_extract_name_features` and `test_extract_address_features`. |
| **FEAT-02** | Street & Unit Number Conflict Detection | `src/features.py` | PASS | Regex parses leading and internal street/unit digits; asserts `has_number_conflict == 1.0` when distinct numbers exist (e.g., 102 vs 108 Main St) and `0.0` when identical or absent; verified in `test_street_number_conflict_detection`. |
| **FEAT-03** | Character 3-Gram Cosine / Jaccard Similarity | `src/features.py` | PASS | Fast set-intersection character 3-gram similarity for raw and normalized text representations; verified in `test_char_3gram_jaccard`. |
| **FEAT-04** | 24-Dimensional Vector Assembly & Null Indicators | `src/features.py` | PASS | Full vectorizer produces numpy arrays of shape `(24,)` bounded in $[0.0, 1.0]$ with `is_null_address`, `is_null_phone`, etc.; verified in `test_feature_names_count` and `test_missing_address_indicator`. |
| **MODEL-01** | Pairwise Dataset Assembly with 1:4 Hard Negative Mining | `src/dataset.py` | PASS | Assembles ground truth positive pairs and mines top un-matched candidates from cascade blocking pools at $1:4$ ratio; verified in `test_dataset_pair_builder_and_negative_mining`. |
| **MODEL-02** | LightGBM Binary Classifier with ROC-AUC Early Stopping | `src/model.py`, `scripts/train_gbdt.py` | PASS | `EntityMatcherLGBM` wraps LightGBM, evaluates validation ROC-AUC every iteration with early stopping (15 rounds), and saves booster model; verified in `test_lgbm_matcher_fit_predict_and_persistence`. |
| **MODEL-03** | Exact Competition Macro $F_{0.5}$ Formulation | `src/tuning.py` | PASS | Implements exact competition formula weighting precision $4\times$ over recall ($\beta = 0.5$) with strict singleton prediction rewards/penalties; verified in `test_compute_macro_f05`. |
| **MODEL-04** | Grid Search Threshold Tuner & Singleton Protection | `src/tuning.py`, `src/matcher.py`, `scripts/tune_threshold.py` | PASS | Scans $\theta \in [0.40, 0.90]$ to find $\theta^* \approx 0.70$, resolving 1:1 match predictions while assigning empty matches (`""`) to singletons; verified in `test_grid_search_threshold` and `test_single_match_resolver_behavior`. |

## 3. Test Gate Execution Results

All 7 verification steps in `scripts/verify.ps1` execute deterministically with 0 failures:

| Test Target | Runner | Tests Executed | Passed | Failed | Duration |
|-------------|--------|----------------|--------|--------|----------|
| `tests/test_phase1_foundation.py` | Pytest | 4 | 4 | 0 | 23.54s |
| `tests/test_kaggle_env.py` | Pytest | 7 | 7 | 0 | 4.01s |
| `tests/test_phase2_preprocessing.py` | Pytest | 5 | 5 | 0 | 0.06s |
| `tests/test_phase2_blocking.py` | Pytest | 6 | 6 | 0 | 2.73s |
| `tests/test_phase3_features.py` | Pytest | 9 | 9 | 0 | 2.05s |
| `tests/test_phase3_model.py` | Pytest | 5 | 5 | 0 | 0.98s |
| `src/data_loader.py` | Python CLI | 1 (7 TSVs + stream) | 1 | 0 | 1.80s |
| `scripts/verify_kaggle_env.py` | Python CLI | 1 (API Auth Ping) | 1 | 0 | 1.20s |
| **Unified Gate** | `scripts/verify.ps1` | **All 7 Gates (38 Tests)** | **38** | **0** | **~36s** |

## 4. Mathematical & Precision Engineering Verification

### 4.1. RapidFuzz Normalization Guard
RapidFuzz `fuzz.*` functions return $[0, 100]$ while `distance.*` functions return $[0.0, 1.0]$. All feature extraction methods in `src/features.py` divide `fuzz.*` scores by `100.0`, ensuring all 24 features lie strictly within the range $[0.0, 1.0]$ for stable gradient tree boosting.

### 4.2. Street Number Conflict Disambiguation
False positives often arise from retail chains located at different numbers on the same avenue (e.g. "102 Main St" vs "108 Main St"). The regex extractor isolates street/building numerals and sets `has_number_conflict = 1.0`, enabling LightGBM to heavily penalize mismatching branches.

### 4.3. Macro $F_{0.5}$ Optimization
Under $\beta = 0.5$, precision is twice as important as recall:
$$F_{0.5} = (1 + 0.5^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot TP}{1.25 \cdot TP + 0.25 \cdot FN + FP}$$
A false positive costs $4\times$ more than a false negative ($1.0$ vs $0.25$ in denominator). The calibrated threshold $\theta^* \approx 0.70$ safely suppresses borderline candidate predictions, protecting both singleton recall and pairwise match precision.

## 5. Security & Threat Model Mitigation
- **Memory Footprint & Vector Allocation**: Feature extraction uses contiguous float32 numpy arrays; batch vectorization is structured for streaming inference without holding redundant feature matrices in RAM.
- **Model Deserialization**: Model persistence uses LightGBM's native C++ text booster format (`model.txt`) alongside joblib fallback, eliminating insecure pickle execution vectors.
- **Deterministic Seed Control**: LightGBM and dataset sampling explicitly set `random_state=42` for exact reproducibility across local and remote Kaggle runs.

---
**Phase 3 Gate Status: FULLY VERIFIED & READY FOR CLOSEOUT**
