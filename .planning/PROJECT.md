# Amazon ML Challenge 2026: Business Entity Resolution

## What This Is

An end-to-end Machine Learning solution for the Amazon ML Challenge 2026 Business Entity Resolution challenge. Given noisy, unstandardized business records across three independent data sources (`Source 1`, `Source 2`, `Source 3`), the system identifies all matching records from Sources 2 and 3 for every Source 1 entity, operating at high scale (>1.73M reference entities) and evaluating against the precision-heavy Macro $F_{0.5}$ metric.

## Core Value

Achieve high Macro $F_{0.5}$ score through a country-partitioned cascade blocking architecture and precision-optimized LightGBM classifier with threshold tuning, guaranteeing 100% compliance with submission format, fair-play constraints, and single zip packaging.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Data environment ingestion: verify train and test `.tsv` datasets (`student_resource/dataset/`) and submission validator (`student_resource/utils/validate_submission.py`).
- [ ] Text normalization pipeline: Unicode NFKD decomposition, multilingual diacritics stripping, legal entity suffix stripping, and address expansion.
- [ ] Country-partitioned cascade blocking engine: strict country partitioning (`US`, `India`, `France`), inverted index cascade keys (`by_both`, `by_name`, `by_addr`, `by_pref`), and `candidate_pairs.tsv` generation.
- [ ] 24-dimensional feature extraction: Jaro-Winkler, Levenshtein, token Jaccard & containment, 3-gram char Jaccard, numeric/street conflict detector, and missing address indicators.
- [ ] GBDT matching model & threshold tuning: LightGBM binary classifier trained on positive/hard-negative pairs, tuned to optimal decision threshold ($\theta^* \approx 0.70$) for Macro $F_{0.5}$.
- [ ] Dual compute execution: Local prototyping/validation on RTX 3050 and automated Kaggle dual-T4 GPU kernel execution for heavy test candidate blocking and inference.
- [ ] Dense bi-encoder representation exploration: evaluate adding semantic embedding cosine similarity features using compact open-source models ($\le 8\text{B}$ params, Apache 2.0/MIT).
- [ ] Submission packaging & validation: automated packaging of `<team_name>_submission.zip` with `output/matching_results.tsv`, `output/candidate_pairs.tsv`, runnable `code/business_entity_resolution/`, and completed `Documentation_template.md` passing `validate_submission.py` with exit 0.

### Out of Scope

- External web lookups or commercial ER APIs (strictly prohibited by competition fair-play rules; leads to disqualification).
- Geocoding APIs (e.g. Google Maps, OpenStreetMap APIs prohibited).
- Models exceeding 8B parameters or non-permissive/commercial licenses (disqualified under rule 5).
- All-pairs $O(N^2)$ brute-force comparison (computationally infeasible across 1.73M $\times$ 10.1M records).

## Context

- **Data Scale**: 1,732,544 test reference entities in Source 1; >5.09M in Source 2; >5.06M in Source 3. Total pairwise space exceeds 17 trillion pairs.
- **Evaluation**: Macro-averaged $F_{0.5}$ across all Source 1 entities. $F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$. Penalizes false merges $2\times$ more than missed links. Singletons with zero matches score 1.0 if predicted empty, and 0.0 if any match is predicted.
- **Country Set**: Training data contains `US` and `India`. Test data contains `US`, `India`, and `France` (zero France training records). Open-set handling required.
- **Hardware Profile**: Local laptop with NVIDIA RTX 3050 (4GB VRAM) for testing and rapid verification; remote Kaggle dual T4 GPUs (32GB VRAM total) with Kaggle CLI automated dispatch for full-scale test processing.

## Constraints

- **Evaluation Metric**: Macro $F_{0.5}$ heavily favors high precision ($\theta^* \approx 0.70$).
- **Fair Play**: Zero external data augmentation, external APIs, or web lookups.
- **Model Size & License**: Maximum 8 Billion parameters; permissive open-source licenses only (MIT, Apache 2.0).
- **Submission Format**: Strict `.tsv` format with required headers, one row per Source 1 entity, candidates as superset of matches. Must pass `validate_submission.py`.
- **Packaging Structure**: Single zip `<team_name>_submission.zip` containing `output/`, `code/business_entity_resolution/`, and `Documentation_template.md`.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Strict Country Partitioning | 0.00% cross-country matches empirically observed; reduces comparison space by >70% | — Pending |
| Multi-Key Inverted Index Cascade | Captures exact name+address, exact name, exact address, and token prefixes without $O(N^2)$ comparisons | — Pending |
| LightGBM with $\theta^* \approx 0.70$ | Optimized for Macro $F_{0.5}$ precision bias, suppressing noisy false merges and preserving singleton credit | — Pending |
| Dual-Engine Compute (Local + Kaggle) | Fast local iteration + Kaggle GPU acceleration for 1.73M test candidate scoring | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-27 after initialization*
