---
phase: "01"
slug: "foundation-ingestion-compute-infrastructure"
status: verified
threats_open: 0
asvs_level: 1
created: "2026-09-27"
---

# Phase 01 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Raw TSV data → DataFrame | Untrusted text with malformed lines, carriage returns, or unescaped tabs crosses into memory | Unsanitized tabular strings |
| Prediction output → Validator | Output files parsed by official submission script | TSV prediction files |
| Local machine → Kaggle API | Local API tokens used to push code and fetch output artifacts | API credentials & secrets |
| Kaggle output → Local filesystem | Remote generated TSV files and model checkpoints downloaded locally | Remote artifacts |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-01-01 | Tampering | `src/data_loader.py` | high | mitigate | Use `csv.QUOTE_NONE`, explicit `sep="\t"`, strip `\r`/`\n` and fill NAs with `""` to prevent row corruption. | closed |
| T-01-02 | Information Disclosure | `scripts/split_dataset.py` | critical | mitigate | Enforce strict Source 1 entity ID disjointness assertion between train and validation splits to guarantee zero data leakage. | closed |
| T-01-03 | Denial of Service | `scripts/verify_submission.py` | medium | mitigate | Default to fast structural validation; require explicit `--full` flag for memory-intensive `--check-ids` verification. | closed |
| T-02-01 | Information Disclosure | `scripts/verify_kaggle_env.py` | high | mitigate | Never log or print raw API keys; only print presence/validity status and masked username. | closed |
| T-02-02 | Tampering | `scripts/kaggle_runner.py` | medium | mitigate | Validate downloaded files from Kaggle output with size and header checks before replacing local artifacts. | closed |

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-27 | 5 | 5 | 0 | Antigravity Security Auditor |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-09-27
