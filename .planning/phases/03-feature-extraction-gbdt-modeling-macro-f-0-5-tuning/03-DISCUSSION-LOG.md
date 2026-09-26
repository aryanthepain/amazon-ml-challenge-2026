# Phase 3: Feature Extraction, GBDT Modeling & Macro $F_{0.5}$ Tuning - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-27
**Phase:** 03-feature-extraction-gbdt-modeling-macro-f-0-5-tuning
**Areas discussed:** Direct Planning Handoff, Negative Sampling, Feature Vector Architecture, LightGBM Modeling, Macro F0.5 Calibration

---

## Area Selection & Planning Handoff

| Option | Description | Selected |
|--------|-------------|----------|
| Interactive Gray Area Discussion | Discuss Negative Sampling, Feature Vector, LightGBM Objective, Threshold Tuning in detail | |
| Direct Planning Handoff | Move directly to planning with standard ML engineering defaults ("just move to planning") | ✓ |

**User's choice:** "just move to planning"
**Notes:** The user elected to skip interactive gray area Q&A and proceed immediately to planning. All technical implementation choices were resolved under the agent's discretion, grounded strictly in REQUIREMENTS.md (FEAT-01..FEAT-04, MODEL-01..MODEL-04) and ROADMAP.md Phase 3 deliverables.

---

## the agent's Discretion

1. **Negative Sampling & Hard Negatives:** Adopted 1:4 positive-to-negative ratio. Hard negatives mined from candidate blocking results.
2. **Feature Vector Engineering:** Adopted 24-dimensional feature vector combining RapidFuzz SIMD metrics, char 3-grams, numeric street conflict flags, missing address indicators, and blocking rank/similarity scores.
3. **Model Selection & Loss:** LightGBM binary classifier (`objective="binary"`) with early stopping on validation ROC-AUC.
4. **Threshold Tuning:** Scalar grid search $\theta^* \in [0.40, 0.90]$ with step 0.02 evaluated against competition Macro $F_{0.5}$ with singleton protection (>95% singleton accuracy).
5. **Inference Assignment:** Argmax probability assignment for candidates $\ge \theta^*$; otherwise predicted as singleton.

## Deferred Ideas

- Cross-encoder / bi-encoder deep learning reranker (e.g. DeBERTa-v3) — deferred to Phase 4.
- Graph community detection / transitive closure resolution — deferred to Phase 5.
