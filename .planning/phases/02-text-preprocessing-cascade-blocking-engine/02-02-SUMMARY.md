---
phase: 02-text-preprocessing-cascade-blocking-engine
plan: 02
status: complete
tasks_completed: 3
total_tasks: 3
files_created:
  - src/blocking.py
  - scripts/run_blocking.py
  - scripts/audit_blocking_recall.py
  - tests/test_phase2_blocking.py
files_modified:
  - scripts/verify.ps1
---

# Plan 02-02 Summary: Country-Partitioned Cascade Blocking Engine & Recall Ceiling Audit

## Objective
Build the country-partitioned cascade blocking engine, candidate pair generator, streaming TSV exporter, Parquet cache, and recall ceiling audit harness (`src/blocking.py`, `scripts/run_blocking.py`, `scripts/audit_blocking_recall.py`, `tests/test_phase2_blocking.py`) for Phase 2.

## Accomplishments
1. **Country-Partitioned Cascade Blocking Engine (`src/blocking.py`)**:
   - `InvertedIndexCascade`: Implemented 4-stage exact inverted index lookup tables:
     - Stage 1: `by_both` (`clean_name + "::" + clean_addr`, min length 8)
     - Stage 2: `by_name` (`clean_name`, min length 3)
     - Stage 3: `by_addr` (`clean_addr`, min length 10)
     - Stage 4: `by_pref` (first 2 tokens of `clean_name`, min length 4)
   - `TfidfSparseBlocker`: Subword character (3,4)-gram TF-IDF vectorization with `max_features=150000`, `min_df=2`, sublinear TF, and batched sparse matrix dot products (`s1_matrix.dot(target_matrix_t)`) with direct CSR array slicing.
   - `BlockingEngine`: Dynamic open-set country partitioning (`df['country'].dropna().unique()`) with zero cross-country match leakage (D-03). Merges $S_2 + S_3$ into a unified target pool (D-02), assigns exact matches first, and fills up to $K=25$ budget with TF-IDF fuzzy candidates meeting `min_similarity >= 0.18` (D-05, D-06, D-07).
   - Singleton compliance: Emits empty lists for entities without qualifying matches (D-08).

2. **Submission Exporter & Parquet Cache (`export_candidate_pairs_tsv`, `export_candidate_parquet_cache`)**:
   - `export_candidate_pairs_tsv`: Streams candidate pairs in chunked buffers to official `output/candidate_pairs.tsv` with header `source1_entity_id\tcandidate_entity_ids` and strict singleton representation `S1-ID\t\n` passing official `validate_submission.py`.
   - `export_candidate_parquet_cache`: Emits structured metadata tables (`['s1_id', 'candidate_id', 'blocking_stage', 'rank', 'cosine_sim']`) partitioned by country in `data/cache/candidates/country={country}/candidates.parquet` (D-14).

3. **Execution Runner CLI (`scripts/run_blocking.py`)**:
   - Command-line runner supporting `--split {train,val,test,dev}`, `--sample-size`, `--target-sample-size`, `--output-tsv`, and `--cache-parquet`.
   - Optimized with memory-bounded streaming and precomputed text normalization.
   - Verified on dev slice with 148 S1/s blocking throughput.

4. **Recall Ceiling & Space Reduction Audit Harness (`scripts/audit_blocking_recall.py`)**:
   - Audits generated candidate pairs against ground truth tables to evaluate empirical recall ceiling percentage, singleton accuracy, country breakdown (US vs India), and Cartesian comparison space reduction factor (D-15, BLOCK-04).
   - Emits structured JSON audit reports to `output/blocking_recall_audit.json`.

5. **Pytest Test Suite & Deterministic Verification Gate (`tests/test_phase2_blocking.py`, `scripts/verify.ps1`)**:
   - Built 6 unit and integration test functions:
     - `test_country_partition_zero_leakage`: Confirms 0 cross-country matches across US, India, France.
     - `test_inverted_index_cascade_keys`: Confirms exact stages and priority ranking.
     - `test_tfidf_sparse_matching`: Confirms subword char (3,4)-gram scoring and top-k truncation.
     - `test_singleton_empty_candidate`: Confirms singleton tab-delimited formatting.
     - `test_candidate_tsv_submission_validator`: Confirms generated candidates pass official `validate_submission.py` with exit code 0.
     - `test_audit_blocking_recall_ceiling`: Confirms mathematical recall ceiling calculation.
   - Updated `scripts/verify.ps1` to execute all 4 pytest suites + diagnostics with 100% pass rate.

## Verification
- `pytest tests/test_phase2_blocking.py -v`: 6/6 passed in 3.35s.
- `pwsh -File .\scripts\verify.ps1`: All 6 verification gates passed deterministically with exit code 0.
- `python scripts/run_blocking.py --help` & `python scripts/audit_blocking_recall.py --help`: Clean exit code 0.
