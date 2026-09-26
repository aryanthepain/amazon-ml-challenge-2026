# Phase 1: Foundation, Ingestion & Compute Infrastructure - Technical Research

**Researched:** 2026-09-27
**Target Phase:** Phase 1: Foundation, Ingestion & Compute Infrastructure

---

## 1. Dataset Ingestion & Tabular Engine Analysis

### 1.1 Ingestion Specs & Scale
The dataset consists of tab-separated values (`.tsv`):
- `train_source1.tsv`: 210 MB (~2.2M records)
- `train_source2.tsv`: 489 MB (~5.0M records)
- `train_source3.tsv`: 504 MB (~5.3M records)
- `train_ground_truth.tsv`: 127 MB
- `test_source1.tsv`: 175 MB (1,732,544 reference entities)
- `test_source2.tsv`: 509 MB (~5.09M records)
- `test_source3.tsv`: 506 MB (~5.06M records)

### 1.2 Parsing Invariants & Edge Cases
1. **Explicit Tab Delimiter (`sep="\t"`)**:
   - Business addresses and names contain commas, quotes, and punctuation. Using `sep="\t"` with `quoting=csv.QUOTE_NONE` or standard Pandas defaults prevents accidental column splitting.
2. **Missing Address Fields**:
   - Source 2 and Source 3 have ~3.3% null `business_address` values. Missing fields must be filled with `""` (empty string) to prevent downstream string similarity functions (RapidFuzz, Levenshtein, Jaccard) from throwing `TypeError` on `float` (`np.nan`).
3. **Embedded Newlines / Whitespace**:
   - Rare linebreaks or carriage returns inside business addresses can corrupt row counts if not cleaned. Ingestion must strip `\r` and `\n` characters from string fields.
4. **Memory Footprint**:
   - Training files total ~1.3 GB raw text. In Pandas with default object strings, this expands to ~3.5–4.5 GB RAM. Using `pyarrow` string backend or PyArrow dictionary types reduces RAM consumption to <1.8 GB, safely fitting within laptop 8GB/16GB RAM constraints.

---

## 2. Train/Validation Split & Metric Formulation

### 2.1 Pure Random 80/20 Holdout Partition
- **Zero Leakage Invariant**: Partitioning must occur strictly on unique `entity_id` values of Source 1.
- Splitting $S_1$ into 80% train ($~1.76\text{M}$) and 20% validation ($~440\text{K}$) with `seed=42`.
- `train_ground_truth.tsv` and `val_ground_truth.tsv` are partitioned by filtering rows where `source1_entity_id` belongs to the respective split.
- Source 2 and Source 3 records are referenced by ground truth and stay in their global pools.

### 2.2 Macro $F_{0.5}$ Exact Implementation
Evaluation metric formula:
$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

For each validation Source 1 entity:
1. **Singleton Entity (0 ground truth matches)**:
   - If predicted matches is empty (`""` or empty set): Score = `1.0`.
   - If predicted matches is non-empty: Score = `0.0` (severe false-merge penalty).
2. **Non-Singleton Entity ($\ge 1$ ground truth matches)**:
   - If predicted matches is empty: Score = `0.0`.
   - If $\text{Precision} + \text{Recall} == 0$: Score = `0.0`.
   - Else: $F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$.
3. **Macro Average**:
   - $\text{Macro } F_{0.5} = \frac{1}{N_{S1}} \sum_{i=1}^{N_{S1}} F_{0.5}(S1_i)$.

---

## 3. Kaggle Dual-T4 Remote Automation Mechanics

### 3.1 Kaggle CLI Configuration
- Credentials loaded from `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`.
- API endpoints:
  - `kaggle competitions list` (diagnostic ping)
  - `kaggle datasets create -p <dir>` / `version`
  - `kaggle kernels push -p <dir>`
  - `kaggle kernels status <slug>`
  - `kaggle kernels output <slug> -p <output_dir>`

### 3.2 Kernel Specification (`kernel-metadata.json`)
```json
{
  "id": "aryanthepain/amazon-ml-challenge-2026-pipeline",
  "title": "Amazon ML Challenge 2026 Pipeline",
  "code_file": "kaggle_entrypoint.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "false",
  "dataset_sources": [
    "aryanthepain/amazon-ml-challenge-2026-dataset"
  ]
}
```

### 3.3 Hardware & Dual-GPU Verification
- Dual T4 GPUs (2x 16GB VRAM, 4 vCPUs, 30GB RAM).
- In Python:
  ```python
  import torch
  num_gpus = torch.cuda.device_count()
  print(f"GPUs available: {num_gpus}")
  for i in range(num_gpus):
      print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
  ```
- Smoke test requirement: A lightweight smoke kernel verifying Kaggle dataset access at `/kaggle/input/amazon-ml-challenge-2026-dataset/` and generating dummy outputs before dispatching heavy training or inference jobs.

---

## 4. Official Submission Validator & Packaging Standards

### 4.1 `validate_submission.py` Validation Protocol
- Invocation:
  ```bash
  python student_resource/utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir student_resource/dataset/test
  ```
- Constraints:
  - Exactly 1,732,544 rows in `matching_results.tsv` matching test Source 1 entities.
  - Header: `source1_entity_id\tmatched_entity_ids`.
  - Candidate pairs header: `source1_entity_id\tcandidate_entity_ids`.
  - Candidate set must be a superset of matched entities.
  - No duplicated IDs, no self-matches.
  - Exit code `0` on PASS.

### 4.2 Submission Archive Structure (`package_submission.py`)
Conforming to `jinay_sub/amazon_ml_challenge_project_bundle/package_submission.py`:
```
antigravity_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── preprocessing.py
│       │   ├── blocking.py
│       │   ├── similarity_metrics.py
│       │   ├── feature_extractor.py
│       │   ├── train_matcher.py
│       │   ├── tune_threshold.py
│       │   ├── evaluate_model.py
│       │   └── run_inference.py
│       ├── models/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

## 5. Verification Plan

1. **Unit Tests (`tests/test_phase1_foundation.py`)**:
   - `test_data_loader_integrity`: Verify path resolver and clean TSV loading with 0 NaN text fields.
   - `test_split_zero_leakage_and_reproducibility`: Verify 80/20 train/val Source 1 ID disjointness and determinism with seed 42.
   - `test_macro_f05_edge_cases`: Verify singleton score 1.0 on empty, 0.0 on false merge, and known non-singleton harmonic mean.
   - `test_dummy_submission_validation`: Verify `generate_dummy_submission.py` output passes `validate_submission.py` with exit 0.
2. **Kaggle CLI Diagnostic (`tests/test_kaggle_env.py`)**:
   - Verify `scripts/verify_kaggle_env.py` detects Kaggle credentials or reports clear diagnostic instructions.
3. **PowerShell Verification Gate (`scripts/verify.ps1`)**:
   - Runs `pytest tests/test_phase1_foundation.py -v`.
