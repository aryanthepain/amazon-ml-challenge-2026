---
phase: "02"
slug: "text-preprocessing-cascade-blocking-engine"
status: verified
threats_open: 0
asvs_level: 1
created: "2026-09-27"
---

# Phase 02 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Raw strings → Preprocessing regexes | User and company names crossing into regex evaluation | Multilingual business strings |
| Entity pool → Sparse Matrix | Multi-million candidate records crossing into TF-IDF vectorizer | Sparse matrices & dot products |
| Blocker → TSV Exporter | In-memory candidate pairs exported to disk | Candidate pair TSV |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-02-01 | Denial of Service | `src/preprocessing.py` | high | mitigate | Pre-compile all regexes at module load with bounded word boundary anchors; use native unicodedata NFKD decomposition to prevent regex backtracking and memory leaks. | closed |
| T-02-02 | Denial of Service | `src/blocking.py` | high | mitigate | Enforce adaptive query batching (50-200) and direct CSR array slicing to prevent sparse matrix dot-product OOM spikes. | closed |
| T-02-03 | Tampering | `src/blocking.py` | medium | mitigate | Validate every generated candidate TSV against `student_resource/utils/validate_submission.py` to prevent formatting corruption. | closed |
| T-02-04 | Information Disclosure | `src/blocking.py` | critical | mitigate | Dynamic open-set country partitioning enforces zero cross-country match leakage. | closed |

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-27 | 4 | 4 | 0 | Antigravity Security Auditor |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-09-27
