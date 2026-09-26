# Feature Research

**Domain:** Cross-Source Business Entity Resolution (Amazon ML Challenge 2026)  
**Researched:** 2026-09-27  
**Confidence:** HIGH  

## Feature Landscape

### Table Stakes (Required for Scoring & Feasibility)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Strict Country Partitioning | 0.00% cross-country ground truth matches; reduces search space by >70% | LOW | Partition S1, S2, S3 by `country` (US, India, France) |
| Unicode NFKD Normalization | Handles diacritics, multilingual characters, accents in French/Indian text | LOW | Standardizes `café` -> `cafe`, decomposing combined glyphs |
| Corporate Suffix Stripping | Prevents legal entities (`LLC`, `Corp`, `Pvt Ltd`) from skewing name similarities | LOW | Regex word-boundary replacement for 30+ global legal terms |
| Multi-Key Inverted Index Cascade | Compresses 17.6 trillion pair space to ~1-5 candidates per entity | MEDIUM | Keys: `by_both`, `by_name`, `by_addr`, `by_pref` |
| Macro $F_{0.5}$ Threshold Tuner | Default 0.50 threshold fails on precision-heavy $F_{0.5}$ metric | MEDIUM | Grid search $\theta \in [0.4, 0.9]$; optimal $\theta^* \approx 0.70$ |
| Format Validation Gate | Invalid format leads to disqualified zero score | LOW | Verify with `validate_submission.py` locally |

### Differentiators (Score Maximizers)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Street Number Conflict Detector | Addresses with conflicting digits (`101 Main St` vs `103 Main St`) are distinct entities | MEDIUM | Extract digit tokens; set binary conflict flag and numeric Jaccard |
| Missing Address Indicator Flags | 3.3% of S2/S3 have empty addresses; model must not penalize address absence | LOW | Binary features: `s1_has_addr`, `cand_has_addr`, `both_have_addr` |
| Token Containment Metric | Catches abbreviations and parent company names where one is a strict subset of other | LOW | `len(A & B) / min(len(A), len(B))` |
| Character 3-Gram Jaccard | Resilient to severe typos, transliterations, and character swaps | MEDIUM | Computed over character trigrams of business names |
| Dual-Engine Compute Runner | Enables fast local prototyping while using Kaggle dual T4 GPUs for full 1.73M test set | HIGH | Kaggle API automated kernel packaging, dispatch, and sync |

### Anti-Features (Avoid These)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| External Geocoding / Web APIs | Seems like an easy way to verify addresses | Explicit grounds for immediate disqualification under competition fair play | In-memory token parsing and numeric conflict detection |
| Heavy Transformer Cross-Encoder for All Pairs | Theoretical high accuracy | $O(N \cdot M)$ inference over 5M pairs would take days/weeks | LightGBM over engineered features (sub-second per 10k pairs) |
| Monolithic Brute-Force Cartesian Join | Simple to implement | 17.6 trillion pairs causes immediate Out-Of-Memory (OOM) crash | Country-partitioned inverted index blocking |

## Feature Dependencies

```
[Text Preprocessing & Normalization]
    └──requires──> [Country-Partitioned Cascade Blocking]
                       └──emits──> [Candidate Pairs TSV]
                                       └──requires──> [Feature Extraction Engine]
                                                          └──requires──> [LightGBM Classifier]
                                                                             └──requires──> [Macro F0.5 Threshold Tuning]
                                                                                                └──emits──> [Final matching_results.tsv]
```

## MVP Definition (Phase 1 Baseline -> Final Submission)

### Launch With (v1 Baseline)
- [ ] Fast text preprocessor (`preprocessing.py`)
- [ ] Country-partitioned cascade blocking (`blocking.py`)
- [ ] 24-dimensional feature extractor (`feature_extractor.py`)
- [ ] LightGBM binary classifier (`train_matcher.py`)
- [ ] Threshold tuner maximizing Macro $F_{0.5}$ (`tune_threshold.py`)
- [ ] Format validator check (`validate_submission.py` -> exit 0)

### Add After Validation (v1.x Enhancements)
- [ ] Dense bi-encoder embeddings as auxiliary features via Kaggle GPU
- [ ] Kaggle automated remote dispatch and result synchronization script
- [ ] Submission zip packaging script (`package_submission.py`) and `Documentation_template.md`

---
*Feature research for: Amazon ML Challenge 2026 Business Entity Resolution*  
*Researched: 2026-09-27*
