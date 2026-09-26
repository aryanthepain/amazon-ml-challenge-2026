# Pitfalls Research

**Domain:** Business Entity Resolution & Challenge Submission  
**Researched:** 2026-09-27  
**Confidence:** HIGH  

## Critical Pitfalls

### Pitfall 1: Disqualification via External API / Lookup
**What goes wrong:**
Using Google Maps geocoding API, OpenStreetMap, or corporate registration databases causes immediate competition disqualification.
**Why it happens:**
Teams believe address normalization is too hard without a geocoder.
**How to avoid:**
Use strict regex parsing, token normalization, and string similarity metrics natively on the provided data.
**Phase to address:** Phase 1 & 2.

---

### Pitfall 2: Memory Exhaustion (OOM) on 17.6 Trillion Pairs
**What goes wrong:**
Trying to join or score all pairs between 1.73M S1 records and 10.1M S2/S3 records crashes any machine with an OOM error.
**Why it happens:**
$1.73\text{M} \times 10.1\text{M} = 17,500,000,000,000$ pairs.
**How to avoid:**
Strict country partitioning (0% cross-country ground truth matches) + multi-key cascade inverted indexing pruning 99.98% of candidate space.
**Phase to address:** Phase 2.

---

### Pitfall 3: Submitting Low-Confidence Merges on Macro $F_{0.5}$
**What goes wrong:**
Standard classification thresholds ($\theta = 0.50$) produce many false merges, crashing the Macro $F_{0.5}$ score.
**Why it happens:**
$F_{0.5}$ penalizes false positives $2\times$ more than false negatives. Incorrectly merging a singleton yields 0.0 instead of 1.0.
**How to avoid:**
Calibrate decision threshold to $\theta^* \approx 0.70$ on validation set. When in doubt, predict singleton (empty string).
**Phase to address:** Phase 3.

---

### Pitfall 4: Validation Failure & Rejection on Format Rules
**What goes wrong:**
Submissions get status REJECTED on the leaderboard because:
- Missing Source 1 entities (must have all 1,732,544 rows).
- Duplicate IDs in comma-separated list.
- Quotes around values or space separators instead of tab (`\t`).
- Matches not appearing in `candidate_pairs.tsv`.
**Why it happens:**
Generating TSVs without running the official validator before uploading.
**How to avoid:**
Run `python student_resource/utils/validate_submission.py` locally on all emitted files before uploading. Exit code must be 0 (`PASS`).
**Phase to address:** Phase 1 & 5.

---

### Pitfall 5: Hardcoding Country to {US, India}
**What goes wrong:**
Test set introduces `France`, which does not appear in training data. Hardcoded country logic drops France, causing submission rejection for missing entities.
**Why it happens:**
Training set only contains US and India.
**How to avoid:**
Treat country as an open string set. Build indices dynamically per unique country present in the dataset.
**Phase to address:** Phase 2.

## "Looks Done But Isn't" Checklist

- [ ] **Test Output Row Count:** Must be exactly 1,732,544 rows in `matching_results.tsv` and `candidate_pairs.tsv`.
- [ ] **Superset Invariant:** Every ID in `matching_results.tsv` must exist in `candidate_pairs.tsv`.
- [ ] **Separator Invariant:** Must be single tab (`\t`), not comma or multiple spaces.
- [ ] **Singleton Representation:** S1 with no matches must have `source1_entity_id\t\n` (empty second column, no null string).
- [ ] **Local Validation:** `validate_submission.py` must print `PASS` and exit with 0.

---
*Pitfalls research for: Amazon ML Challenge 2026 Business Entity Resolution*  
*Researched: 2026-09-27*
