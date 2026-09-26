# Phase 1: Foundation, Ingestion & Compute Infrastructure - Context

**Gathered:** 2026-09-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 1 establishes the verified data ingestion pipeline, local smoke and offline validation harness, pure random 80/20 holdout train/validation splits, and automated Kaggle dual-T4 GPU execution scripts. It delivers the foundational infrastructure required by text preprocessing, candidate blocking, and model training in subsequent phases, without implementing matching models or candidate generation itself.

</domain>

<decisions>
## Implementation Decisions

### DataFrame Ingestion & Memory Architecture
- **D-01:** Standardize on clean tabular ingestion mirroring the repository standards (`pandas` with explicit tab separator `sep="\t"` as in official instructions, leveraging pyarrow engine/memory mapping for high-throughput where beneficial). — **Reversibility:** costly — Ingestion signatures are consumed across all downstream feature and blocking stages.
- **D-02:** Handle missing text values by filling NAs with empty strings (`""`) and stripping embedded carriage returns/newlines (`\r`, `\n`) during loading to prevent line corruption and NaN type errors in string distance calculations.
- **D-03:** Provide a dual-mode data loader supporting both full in-memory loading (for fast local training and validation cycles) and chunked generator streaming (for memory-efficient full-scale test inference).
- **D-04:** Implement a dynamic dataset path resolver that transparently falls back across `student_resource/dataset/`, `6ab10eb3b23ba_student_resource/student_resource/dataset/`, and a `--data-dir` CLI override flag.

### Train/Validation Holdout Split & Stratification
- **D-05:** Construct a pure random 80/20 train/validation split strictly partitioned on unique Source 1 entity IDs (`random_state=42`), guaranteeing zero entity ID leakage between train and validation.
- **D-06:** Persist dedicated partition files into `data/splits/` (`train_s1_ids.tsv`, `val_s1_ids.tsv`, `train_ground_truth.tsv`, `val_ground_truth.tsv`) for zero-overhead, deterministic reuse across Phases 2 and 3.
- **D-07:** Implement the exact offline Macro $F_{0.5}$ metric (including singleton bonuses and false merge penalties) directly into a unified validation CLI script `evaluate.py`.
- **D-08:** Support fast prototyping and CI smoke testing via an optional `--sample-size N` / `--dev` CLI flag that extracts a small, self-contained slice of Source 1 entities and corresponding Source 2/3 ground truth records.

### Kaggle Dual-T4 Remote Automation & Sync
- **D-09:** Create an automated CLI runner/packager (`scripts/kaggle_runner.py`) that bundles modular Python code into a self-contained Kaggle kernel script, pushes via the Kaggle CLI API (`kaggle kernels push`), monitors remote execution status, and downloads output artifacts.
- **D-10:** Supply the ~2GB raw dataset to remote Kaggle kernels via a pre-uploaded Kaggle Dataset (`/kaggle/input/amazon-ml-challenge-2026-dataset/`) using a helper script (`scripts/kaggle_dataset.py`) for zero transfer latency at kernel boot.
- **D-11:** Structure Kaggle kernel compute to support dual-T4 GPU execution (country partitioning: US on GPU 0, India & France on GPU 1, or chunked data parallelism), enforcing an initial remote smoke test run to verify GPU detection (`torch.cuda.device_count() == 2`), dataset readability, and output generation before running heavy workloads.
- **D-12:** Build an automated pre-flight doctor script (`scripts/verify_kaggle_env.py`) that validates local Kaggle API credentials (`~/.kaggle/kaggle.json` or environment variables), CLI installation, and API connectivity with clear error messaging.

### Validation Harness & Mock Submission Protocol
- **D-13:** Develop a full-schema dummy submission generator (`scripts/generate_dummy_submission.py`) that emits structurally valid `matching_results.tsv` and `candidate_pairs.tsv` to verify exit code 0 (`PASS`) on the official `student_resource/utils/validate_submission.py`.
- **D-14:** Implement a dual-mode validation runner (`scripts/verify_submission.py`) that performs fast structural validation by default (<5s) and enables the memory-intensive `--check-ids` validation when `--full` is specified.
- **D-15:** Structure deterministic unit and regression testing with a Pytest suite (`tests/test_phase1_foundation.py`) verifying ingestion integrity, split reproducibility, Macro $F_{0.5}$ metric edge cases, and submission validator exit 0, orchestrated by `pwsh -File .\scripts\verify.ps1`.
- **D-16:** Standardize team naming and submission zip packaging around `antigravity` / `Antigravity AI` (`antigravity_submission.zip`), aligning strictly with the reference specification in `jinay_sub/amazon_ml_challenge_project_bundle/package_submission.py` and `jinay_sub/Documentation_template.md`. — **Reversibility:** costly — Regulates final zip structure and metadata required by portal scoring.

### Antigravity's Discretion
- Exact directory naming for intermediate cached files (standardized to `data/splits/`, `data/cache/`, and `output/`).
- Logging formatting and CLI progress bars (using `rich` or standard `logging`/`tqdm`).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Official Competition Specifications
- `6ab10eb3b23ba_student_resource/student_resource/README.md` — Official problem description, column definitions, noise patterns, and output TSV format rules.
- `6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py` — Official submission validation harness; defines exit code 0 constraints, header formatting, and `--check-ids` logic.
- `6ab10eb3b23ba_student_resource/student_resource/Documentation_template.md` — Official submission documentation requirements.

### Architecture & Reference Bundle
- `jinay_sub/amazon_ml_challenge_project_bundle/package_submission.py` — Reference packaging script defining `REQUIRED_FILES`, directory layout, and `antigravity_submission.zip` creation.
- `jinay_sub/Documentation_template.md` — Completed solution architecture template establishing Team Name: `Antigravity AI`, Macro $F_{0.5}$ metric formulation, and pipeline stages.
- `jinay_sub/PROJECT_OVERVIEW.md` — Problem scale analysis, country distribution observations (0.00% cross-country matches), and modular structure.
- `project_approach.md` — Baseline architecture specification, mathematical formulas for Macro $F_{0.5}$, and dual compute strategy.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `student_resource/utils/validate_submission.py`: Core validator checking TSV headers, row counts, candidate supersets, and ID sets.
- `jinay_sub/amazon_ml_challenge_project_bundle/package_submission.py`: Automated archive builder validating that all mandatory source and documentation files exist before compressing into `<team_name>_submission.zip`.
- `jinay_sub/Documentation_template.md`: Comprehensive technical writeup containing architectural diagrams, empirical findings, and metric definitions.

### Established Patterns
- TSV delimiter format: Strict tab (`\t`) separation across all inputs and outputs; commas used only inside ID lists without quoting.
- Entity ID prefix convention: `S1-` for Source 1 reference records, `S2-` for Source 2, `S3-` for Source 3.
- Country isolation: Strict preservation of country labels (`US`, `India`, `France`), maintaining open-set adaptability for France.

### Integration Points
- Raw input datasets: `student_resource/dataset/train/` and `student_resource/dataset/test/` (or via dynamic path resolver).
- Output artifacts: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- Split persistence: `data/splits/` storing train and validation ID subsets.

</code_context>

<specifics>
## Specific Ideas

- **Repository Alignment**: Ingestion follows standard `pandas.read_csv(..., sep="\t")` conventions demonstrated in `student_resource/README.md`, while remaining robust against newlines in text fields.
- **Split Choice**: Pure random 80/20 split on Source 1 IDs (seed 42) ensures clean separation without complex group dependencies.
- **Smoke Kaggle First**: Mandatory smoke Kaggle kernel run to verify dual-T4 GPU accessibility, dataset paths, and output synchronization prior to executing full test inference.
- **Jinay Bundle Reference**: Align packaging and team naming with `jinay_sub/amazon_ml_challenge_project_bundle/package_submission.py` and `jinay_sub/Documentation_template.md` (`Antigravity AI`).

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed strictly within the Phase 1 foundation, ingestion, and compute infrastructure scope.

</deferred>

---

*Phase: 01-Foundation, Ingestion & Compute Infrastructure*
*Context gathered: 2026-09-27*
