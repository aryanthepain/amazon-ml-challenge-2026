---
phase: "03"
slug: "feature-extraction-gbdt-modeling-macro-f-0-5-tuning"
status: verified
threats_open: 0
asvs_level: 1
created: "2026-09-27"
---

# Phase 03 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| In-memory pair features → LightGBM | Extracted float feature vectors feeding tree learner | Feature matrices |
| Model weights → File system | Serialized model booster saved to and loaded from disk | Booster text file |
| Random sampling → Split assignment | Seed initialization controlling negative pair sampling | PRNG state |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-03-01 | Denial of Service | `src/features.py` | high | mitigate | Contiguous float32 numpy arrays and chunked batch vectorization to prevent RAM exhaustion during large-scale feature extraction. | closed |
| T-03-02 | Elevation of Privilege / Tampering | `src/model.py` | high | mitigate | Native LightGBM C++ text booster persistence format (`model.txt`) alongside joblib fallback, eliminating insecure pickle deserialization vectors. | closed |
| T-03-03 | Repudiation / Inconsistency | `src/dataset.py`, `src/model.py` | medium | mitigate | Enforce explicit `random_state=42` across LightGBM and dataset negative sampling to guarantee exact reproducibility across environments. | closed |

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-27 | 3 | 3 | 0 | Antigravity Security Auditor |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-09-27
