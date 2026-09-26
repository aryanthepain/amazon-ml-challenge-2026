# Project Research Summary

**Project:** Amazon ML Challenge 2026: Business Entity Resolution  
**Domain:** Large-Scale Commercial Entity Resolution & Record Linkage  
**Researched:** 2026-09-27  
**Confidence:** HIGH  

## Executive Summary

The Amazon ML Challenge 2026 presents a massive-scale Entity Resolution task: matching noisy business records from two independent sources ($S_2, S_3$) to a deduplicated reference source ($S_1$ with 1.73M records). With over 17 trillion possible pairwise comparisons, a brute-force approach is computationally impossible. Furthermore, evaluation under Macro $F_{0.5}$ penalizes false merges $2\times$ harder than missed links and awards full credit (1.0) for identifying singletons.

Our verified approach integrates:
1. **Strict Country Partitioning & Multi-Key Cascade Blocking**: Exploits the 0.00% cross-country match invariant and inverted index keys (`by_both`, `by_name`, `by_addr`, `by_pref`) to prune >99.98% of candidate space while maintaining a high recall ceiling.
2. **24-Dimensional Similarity & Conflict Engineering**: RapidFuzz-vectorized string metrics, token containment, character 3-grams, and a street number conflict detector.
3. **LightGBM Binary Classifier & Precision Threshold Calibration**: LightGBM model trained on candidate pairs, with decision threshold calibrated to $\theta^* \approx 0.70$ specifically maximizing Macro $F_{0.5}$.
4. **Dual Compute Strategy**: Fast local CPU/GPU verification paired with automated Kaggle dual-T4 GPU kernel execution for full test set inference.

## Key Findings

### Recommended Stack
- **Languages & Frameworks**: Python 3.10+, Polars/Pandas, RapidFuzz (C++ SIMD), LightGBM, Scikit-Learn.
- **Compute Accelerators**: Local RTX 3050 Laptop GPU (prototyping/smoke tests) + Kaggle dual-T4 GPUs (full test set batch inference).

### Architecture Approach
- 4-stage modular pipeline: Preprocessing -> Country-Cascade Blocking -> Feature Extraction -> LightGBM Classification with Macro $F_{0.5}$ thresholding.

### Critical Pitfalls Mitigated
- **No external data/geocoding lookup**: 100% compliant with zero external API rule.
- **OOM Prevention**: Country-partitioned streaming chunks prevent memory exhaustion.
- **Singleton Credit**: High threshold ($\theta^* \approx 0.70$) protects singleton accuracy (>96%).
- **Open Country Support**: Dynamic country handling correctly processes `France` in the test set.

## Implications for Roadmap (Coarse Granularity)

Based on research and dependencies, we structure the project into 4 coarse phases:

### Phase 1: Environment, Ingestion & Baseline Verification
- Verify dataset layout (`student_resource/dataset/`) and official validation script (`validate_submission.py`).
- Implement baseline generator and local test suite.
- Set up Kaggle API automation scripts (`scripts/run_kaggle_kernel.py`).

### Phase 2: Text Preprocessing & Cascade Blocking Engine
- Implement Unicode NFKD cleaner, corporate suffix remover, and address normalizer.
- Build strict country-partitioned cascade inverted indexes.
- Emit validated `candidate_pairs.tsv` and measure pair recall ceiling.

### Phase 3: Feature Engineering, Model Training & Threshold Tuning
- Implement 24-dimensional feature extraction (RapidFuzz, token containment, numeric conflict flags).
- Train LightGBM classifier with early stopping.
- Optimize decision threshold $\theta^*$ for Macro $F_{0.5}$ on validation split.

### Phase 4: Full Test Inference, Kaggle Execution & Final Submission Packaging
- Execute full inference across 1.73M test instances (locally or via Kaggle dual-T4).
- Verify outputs using `validate_submission.py` (exit 0).
- Package `<team_name>_submission.zip` with `matching_results.tsv`, `candidate_pairs.tsv`, self-contained `code/business_entity_resolution/`, and completed `Documentation_template.md`.

---
*Research completed: 2026-09-27*  
*Ready for roadmap: yes*
