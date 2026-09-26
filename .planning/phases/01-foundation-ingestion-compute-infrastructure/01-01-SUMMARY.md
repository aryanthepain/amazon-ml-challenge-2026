---
phase: 01-foundation-ingestion-compute-infrastructure
plan: 01
status: complete
tasks_completed: 3
total_tasks: 3
files_created:
  - src/data_loader.py
  - src/__init__.py
  - evaluate.py
  - scripts/split_dataset.py
  - scripts/generate_dummy_submission.py
  - scripts/verify_submission.py
  - tests/test_phase1_foundation.py
  - scripts/verify.ps1
files_modified:
  - .gitignore
---

# Plan 01-01 Summary: Foundation Ingestion, Evaluation Harness & Mock Submission Pipeline

## Objective
Build the core ingestion engine, deterministic 80/20 holdout split generator, offline Macro $F_{0.5}$ evaluation harness, and mock submission validation pipeline for Phase 1.

## Accomplishments
1. **Dynamic Tabular Ingestion (`src/data_loader.py`)**:
   - Implemented `resolve_dataset_path()` scanning across multiple nested directory structures (`student_resource/dataset/`, `6ab10eb3b23ba_student_resource/...`, `/kaggle/input/...`, and custom directories).
   - Fast tabular loading with pyarrow fallback to C engine, `csv.QUOTE_NONE`, zero NaNs (empty string imputation), and newline/carriage-return sanitization.
   - Memory-efficient streaming generator via `DataLoader.stream_source()`.
   - Verified via `python src/data_loader.py --check` across all 7 dataset files (train source 1-3, ground truth, test source 1-3).

2. **Holdout Split Generator (`scripts/split_dataset.py`)**:
   - Deterministic 80/20 train/validation split on 2,206,821 Source 1 entity IDs with random seed 42.
   - Rigorously enforced zero entity ID overlap between train and validation sets.
   - Partitioned and saved `train_s1_ids.tsv`, `val_s1_ids.tsv`, `train_ground_truth.tsv`, and `val_ground_truth.tsv` (1,765,457 train entities, 441,364 validation entities).

3. **Offline Evaluation Harness (`evaluate.py`)**:
   - Exact implementation of competition Macro $F_{0.5}$ metric with proper singleton handling (score=1.0 when both empty, 0.0 on false merge) and harmonic mean calculation.
   - Supports `--sample-size` and `--json` diagnostic output.

4. **Mock Submission & Official Verification (`scripts/generate_dummy_submission.py`, `scripts/verify_submission.py`)**:
   - Schema-compliant mock submission generator outputting `matching_results.tsv` and `candidate_pairs.tsv`.
   - Verification runner wrapping official `validate_submission.py` utility.

5. **Deterministic Verification Gate (`tests/test_phase1_foundation.py`, `scripts/verify.ps1`)**:
   - Comprehensive pytest suite covering path resolution, split disjointness, metric mathematical correctness, and submission schema validation.
   - Orchestrated via `pwsh -File .\scripts\verify.ps1` with 100% pass rate.

## Verification
- `pwsh -File .\scripts\verify.ps1`: Exit code 0 (All 4 pytest tests pass, data loader diagnostic pass, evaluate.py CLI pass).
- Full 80/20 dataset split generated with zero entity overlap.
