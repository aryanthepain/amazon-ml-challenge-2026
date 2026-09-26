# Phase 1: Foundation, Ingestion & Compute Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-27
**Phase:** 1-Foundation, Ingestion & Compute Infrastructure
**Areas discussed:** DataFrame Ingestion & Memory Architecture, Train/Validation Holdout Split & Stratification, Kaggle Dual-T4 Remote Automation & Sync, Validation Harness & Mock Submission Protocol

---

## DataFrame Ingestion & Memory Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Polars | Rust-backed multi-threaded engine with lazy evaluation (scan_csv) | |
| Pandas with PyArrow engine | Standard DataFrame API with PyArrow backend | |
| DuckDB | SQL-native analytical engine on TSV files | |
| Keep similar to github repo | Standard Pandas with explicit tab separator pd.read_csv(sep='\t') as in official student_resource instructions | ✓ |

**User's choice:** "keep similar to the github repo"
**Notes:** Ingestion standardized on `pandas.read_csv(sep="\t")` to match `student_resource/README.md`, with PyArrow / memory-mapping acceleration where beneficial.

| Option | Description | Selected |
|--------|-------------|----------|
| Fill NA with empty strings ("") & strip newlines | Avoids NaN type errors and preserves 100% of rows | ✓ |
| Preserve nulls/NaNs with has_address flag | Lets models explicitly learn missingness | |
| You decide | Standardize on empty string representation with boolean missingness | |

**User's choice:** Fill NA with empty strings ("") and strip raw newlines/carriage returns
**Notes:** Ensures clean text distance calculations without NaN crashes while keeping row count exact.

| Option | Description | Selected |
|--------|-------------|----------|
| Dual-mode loader | Full in-memory load for training/validation + chunked generator for large test inference | ✓ |
| Pure in-memory loader | Always load full datasets into memory | |
| Always chunked / streaming | Enforce chunked streaming across train and test | |

**User's choice:** Dual-mode loader
**Notes:** Provides fast local development and safe execution on RAM-constrained machines.

| Option | Description | Selected |
|--------|-------------|----------|
| Dynamic Path Resolver | Checks relative paths across student_resource/dataset/, 6ab10eb3b23ba_student_resource/..., and --data-dir CLI flags | ✓ |
| Symlink / Standardize Directory | Create clean symlink or folder alias in project root | |
| Strict Config File | Define paths strictly in config.json | |

**User's choice:** Dynamic Path Resolver
**Notes:** Automatically locates TSVs across varying checkout directories.

---

## Train/Validation Holdout Split & Stratification

| Option | Description | Selected |
|--------|-------------|----------|
| Stratified by (Cardinality Bin × Country) | Balances singletons and US/India ratio with zero leakage | |
| Pure Random 80/20 Split on Source 1 IDs | Uniform random sampling of Source 1 IDs without grouping | ✓ |
| 5-Fold Stratified Split Generator | Pre-generates 5 reusable folds | |

**User's choice:** Pure Random 80/20 Split on Source 1 IDs — Uniform random sampling of Source 1 IDs without grouping
**Notes:** Simple, clean, unbundled random 80/20 partition partitioned strictly by Source 1 entity IDs (seed 42).

| Option | Description | Selected |
|--------|-------------|----------|
| Dedicated split files in data/splits/ | Pre-materializes val_s1_ids.tsv, train_s1_ids.tsv, and partitioned ground truth | ✓ |
| Single index mapping file | Single 2-column file mapping ID to split | |
| Deterministic hash partitioning | Partition dynamically using MurmurHash/MD5 | |

**User's choice:** Dedicated split files in `data/splits/`
**Notes:** Eliminates on-the-fly recomputation in Phases 2 and 3.

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone src/evaluation/metrics.py | Modular function computing exact Macro F0.5 | |
| Integrated validation script evaluate.py | Single unified validation CLI script evaluate.py | ✓ |

**User's choice:** Integrate directly into a single unified validation CLI script `evaluate.py`
**Notes:** Consolidates evaluation logic and metric calculation into one CLI tool.

| Option | Description | Selected |
|--------|-------------|----------|
| CLI flag --sample-size N / --dev mode | Dynamically extracts a mini consistent slice for instant sub-minute test cycles | ✓ |
| Permanent fixed mini dataset | Pre-generate fixed 2,000 entity folder | |
| Full dataset only | Always run on complete 80/20 split | |

**User's choice:** CLI flag `--sample-size N` / `--dev` mode
**Notes:** Enables fast sub-minute smoke tests and rapid feedback loops.

---

## Kaggle Dual-T4 Remote Automation & Sync

| Option | Description | Selected |
|--------|-------------|----------|
| Automated CLI packager (scripts/kaggle_runner.py) | Bundles modular pipeline code, pushes via CLI, monitors status, pulls outputs | ✓ |
| Kaggle GitHub clone script | Clones repo directly inside kernel | |
| Manual Notebook Workflow | Maintain dedicated notebook for web browser | |

**User's choice:** Automated CLI packager (`scripts/kaggle_runner.py`)
**Notes:** Provides programmatic execution without manual browser overhead.

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-uploaded Kaggle Dataset (/kaggle/input/...) | Mounts dataset instantly with zero transfer latency | ✓ |
| Direct GitHub release download | Kernel curls dataset from remote release tag | |

**User's choice:** Pre-uploaded Kaggle Dataset (`/kaggle/input/amazon-ml-challenge-2026-dataset/`)
**Notes:** Managed via a `scripts/kaggle_dataset.py` upload helper.

| Option | Description | Selected |
|--------|-------------|----------|
| Partition-based GPU Allocation | Assigns partitions/chunks across dual T4s with initial smoke Kaggle test | ✓ |
| Single-GPU baseline | Single GPU with multi-threaded CPU feature extraction | |

**User's choice:** Partition-based GPU Allocation, with an initial smoke Kaggle kernel run to verify dual GPUs, dataset access, and outputs before launching heavy jobs
**Notes:** Explicit user constraint to run and verify smoke Kaggle runs first.

| Option | Description | Selected |
|--------|-------------|----------|
| Automated Kaggle Pre-flight Doctor | Validates kaggle.json / env vars and runs lightweight ping with diagnostic tips | ✓ |
| Strict env variable validation | Python exceptions on missing env vars | |

**User's choice:** Automated Kaggle Pre-flight Doctor
**Notes:** Proactively diagnoses authentication and CLI configuration.

---

## Validation Harness & Mock Submission Protocol

| Option | Description | Selected |
|--------|-------------|----------|
| Full-schema Dummy Generator (scripts/generate_dummy_submission.py) | Emits compliant dummy TSVs proving exit 0 on validate_submission.py | ✓ |
| Exact-Name Baseline Submission | Naive exact-string baseline | |
| Synthetic Mock Test Fixture | 10-entity test fixture | |

**User's choice:** Full-schema Dummy Generator (`scripts/generate_dummy_submission.py`)
**Notes:** Proves offline validator compatibility on the full test set schema.

| Option | Description | Selected |
|--------|-------------|----------|
| Dual-mode Harness (verify_submission.py) | Fast structural validation by default (<5s) with --check-ids on --full | ✓ |
| Always Full Verification | Always pass --check-ids | |
| Fast Structural Only | Never pass --check-ids | |

**User's choice:** Dual-mode Harness (`verify_submission.py`)
**Notes:** Balances quick turnaround with comprehensive audit ability.

| Option | Description | Selected |
|--------|-------------|----------|
| Pytest Suite + verify.ps1 | tests/test_phase1_foundation.py wrapped by scripts/verify.ps1 | ✓ |
| Single Verification Script | python scripts/verify_phase1.py | |

**User's choice:** Pytest Suite + `verify.ps1`
**Notes:** Complies with Antigravity deterministic verification invariants.

| Option | Description | Selected |
|--------|-------------|----------|
| Reference Bundle Specification | Look in jinay sub folder: Antigravity AI / antigravity_submission.zip | ✓ |

**User's choice:** "look in the jinay sub folder"
**Notes:** Packaging aligns with `jinay_sub/amazon_ml_challenge_project_bundle/package_submission.py` and `jinay_sub/Documentation_template.md`.

---

## Antigravity's Discretion

- Standardized cache directories: `data/splits/`, `data/cache/`, `output/`.
- Dynamic path fallback chain for student resource datasets.

## Deferred Ideas

- None.
