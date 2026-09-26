# Phase 2: Text Preprocessing & Cascade Blocking Engine - Context

**Gathered:** 2026-09-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 2 builds the scalable text normalization and country-partitioned cascade blocking engine that shrinks the 17.6-trillion pair space down to high-probability candidates. It delivers Unicode NFKD normalization, comprehensive 35+ corporate legal suffix stripping, canonical address word standardization, dynamic open-set country partitioning, 4-stage guarded inverted index keys, subword char (3,4)-gram TF-IDF sparse matrix candidate retrieval, unified target pooling (S2+S3), top-25 candidate budget with 0.18 min similarity, streaming chunked TSV exporter for `output/candidate_pairs.tsv`, partitioned Parquet candidate cache (`data/cache/candidates/`), empirical validation recall ceiling audit (`scripts/audit_blocking_recall.py` verifying $\ge 95\%$ recall ceiling), and deterministic Pytest test suite (`tests/test_phase2_blocking.py`).

</domain>

<decisions>
## Implementation Decisions

### Blocking Engine Architecture
- **D-01:** Hybrid Cascade Blocking — Inverted index exact keys (`by_both`, `by_name`, `by_addr`, `by_pref`) first, supplemented by char (3,4)-gram TF-IDF sparse matrix dot-product for fuzzy coverage. — **Reversibility:** costly — Downstream candidate generation and feature extraction depend on hybrid candidate ranking.
- **D-02:** Unified Target Pool ($S_2 + S_3$) — Index $S_2$ and $S_3$ records together in a single country index, enabling exact keys and TF-IDF dot-products to retrieve globally top-scoring candidates across both sources simultaneously. — **Reversibility:** costly — Determines candidate pool shape and index construction across all countries.
- **D-03:** Dynamic Open-Set Country Partitioning — Partition strictly by unique country values discovered at runtime (`df['country'].dropna().unique()`), building self-contained indexes per country without hardcoding `US` and `India`, ensuring native zero-shot support for `France` and open-set test data. — **Reversibility:** costly — Fundamental architectural invariant across train and test pipelines.
- **D-04:** 4-Stage Guarded Hierarchy for Exact Keys —
  1. `by_both`: exact `clean_name` + `clean_addr`
  2. `by_name`: exact `clean_name` ($\ge 3$ characters)
  3. `by_addr`: exact `clean_addr` ($\ge 10$ characters, excluding generic stopwords)
  4. `by_pref`: first 2 significant name tokens ($\ge 4$ characters total)
  — **Reversibility:** reversible — Stage keys can be adjusted with minor index dictionary tuning.

### Candidate Budget & Retention Cutoffs
- **D-05:** Top-25 Candidate Budget ($K=25$) — Enforce maximum 25 candidates per Source 1 entity, delivering $\ge 98\%$ validation recall ceiling while bounding Phase 3 feature engineering to $\le 43\text{M}$ candidate pairs across the 1.73M test set. — **Reversibility:** costly — Modifies candidate pair dataset volume and feature extraction runtimes.
- **D-06:** Minimum Similarity Cutoff $\ge 0.18$ — Discard TF-IDF character n-gram cosine similarities below $0.18$ to filter out random char noise while retaining distorted transliterations and spelling variants. — **Reversibility:** reversible — Threshold cutoff can be parameterized via config.
- **D-07:** Exact Priority First + TF-IDF Score Ranking — Candidate lists guarantee exact index matches first slots in order of discovery, filled out to top-$K$ with TF-IDF fuzzy candidates sorted descending by cosine similarity score, deduplicated by candidate entity ID. — **Reversibility:** reversible — Ranking policy is self-contained in candidate merging logic.
- **D-08:** Strict Official Spec Compliance for Singletons — Source 1 entities with zero candidates meeting cutoff are emitted as `S1-ID\t` (empty candidate string), strictly adhering to `validate_submission.py` expectations and protecting singleton macro $F_{0.5}$ precision. — **Reversibility:** one-way — Formatting rules enforced by competition validator.

### Multilingual Text Cleaning & Legal Suffix Scope
- **D-09:** Comprehensive Multilingual Corporate Suffix Regex (35+ Suffixes) — Explicitly strip legal designations covering US (`Inc`, `LLC`, `Corp`, `LLP`, `PC`, `Ltd`), India (`Pvt Ltd`, `Private Limited`, `Enterprises`), France (`SA`, `SAS`, `SARL`, `EURL`, `SNC`), and Europe (`GmbH`, `AG`, `BV`, `NV`, `SpA`) with word boundaries `\b` and trailing position priority. — **Reversibility:** costly — Changes canonical root business names across all subsequent tokenization, blocking, and feature similarity metrics.
- **D-10:** Selective Diacritic Stripping (NFKD + Filter Combining) — Apply `unicodedata.normalize('NFKD', text)` and strip combining marks (`not unicodedata.combining(c)`), decomposing French accents (`é` $\to$ `e`, `ç` $\to$ `c`) into base letters while preserving Unicode alphanumeric character integrity across scripts. — **Reversibility:** costly — Preprocessed strings are shared across blocking and feature extraction.
- **D-11:** Canonical Word Expansion for Addresses — Standardize street, unit, and state abbreviations to full lowercased words (`st` $\to$ `street`, `ave` $\to$ `avenue`, `ste` $\to$ `suite`, `ny` $\to$ `new york`, `mh` $\to$ `maharashtra`) using regexes, followed by stripping punctuation and collapsing whitespace. — **Reversibility:** costly — Normalizes addresses before exact and fuzzy indexing.
- **D-12:** Dual Representation with Name-Boosted Search Text — Maintain distinct `clean_name` and `clean_address` columns for exact index keys, and build a name-boosted composite string (`clean_name * 2 + ' ' + clean_address`) for TF-IDF subword character (3,4)-gram vectorization. — **Reversibility:** reversible — Search string generation logic can be tuned.

### Candidate Persistence & Cache Formats
- **D-13:** Streaming Chunked TSV Exporter — Stream candidate rows directly to `output/candidate_pairs.tsv` in buffered chunks (50,000 entities per batch) with immediate file flushing, ensuring constant memory footprint during 1.73M test inference and crash safety. — **Reversibility:** costly — Regulates disk export format consumed by official validation.
- **D-14:** Partitioned Parquet Cache (`data/cache/candidates/`) — Store candidate pairs with metadata (`s1_id`, `candidate_id`, `blocking_stage`, `rank`, `cosine_sim`) in country-partitioned Parquet files for high-throughput columnar reads during Phase 3 feature engineering. — **Reversibility:** costly — Consumed directly by Phase 3 pair dataset builder.
- **D-15:** Standalone Audit CLI & Structured JSON Report — Implement `scripts/audit_blocking_recall.py` evaluating blocking against `data/splits/val_ground_truth.tsv`, emitting `output/blocking_recall_audit.json` logging overall recall, country breakdown (US vs India), singleton preservation, and pair space reduction factor ($>99.9\%$). — **Reversibility:** reversible — Independent evaluation script.
- **D-16:** Deterministic Pytest Suite & Submission Validator Integration — Implement `tests/test_phase2_blocking.py` testing normalization edge cases, zero cross-country leakage, recall ceiling threshold, and official `validate_submission.py` exit 0 execution orchestrated via `pwsh -File .\scripts\verify.ps1`. — **Reversibility:** reversible — Follows established project CI invariants.

### Antigravity's Discretion
- Batch size for sparse matrix dot products (adaptive sizing 50–1000 depending on country target size to keep peak memory <4GB).
- Sublinear TF scaling (`sublinear_tf=True`) and character n-gram range tuning `(3, 4)` in `scikit-learn` `TfidfVectorizer`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Official Competition Specifications
- `student_resource/README.md` — Official problem description, column definitions, noise patterns, and output TSV format rules.
- `student_resource/utils/validate_submission.py` — Official submission validation harness; defines exit code 0 constraints, header formatting, and candidate pairs superset checks.
- `student_resource/Documentation_template.md` — Official submission documentation requirements.

### Architecture & Reference Bundle
- `jinay_sub/amazon_ml_challenge_project_bundle/code/business_entity_resolution/src/preprocessing.py` — Reference implementation of legal suffix regexes, address abbreviation expansions, and Unicode accent stripping.
- `jinay_sub/amazon_ml_challenge_project_bundle/code/business_entity_resolution/src/blocking.py` — Reference implementation of TF-IDF sparse matrix candidate generation, country partitioning, and batched CSR dot products.
- `project_approach.md` — Baseline architecture specification, cascade blocking inverted index keys, and candidate pruning parameters.
- `.planning/phases/01-foundation-ingestion-compute-infrastructure/01-CONTEXT.md` — Phase 1 foundational decisions, tabular loading standards, and train/val split locations (`data/splits/`).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/data_loader.py`: Ingestion module with path resolver (`find_dataset_file`), missing value sanitizer (`clean_text_series`), and streaming chunked loader.
- `scripts/split_dataset.py`: Reproducible stratified train/validation split generator providing `data/splits/train_s1_ids.tsv`, `data/splits/val_s1_ids.tsv`, and `data/splits/val_ground_truth.tsv`.
- `scripts/verify_submission.py`: Validation wrapper running official `validate_submission.py` with fast structural checks.
- `scripts/verify.ps1`: Deterministic verification pipeline running lint and test suites.

### Established Patterns
- TSV delimiter format: Strict tab (`\t`) separation across all inputs and outputs; commas used only inside ID lists without quoting.
- Country isolation: Strict preservation of country labels (`US`, `India`, `France`), maintaining open-set adaptability for France.
- Modular code architecture: Core logic in `src/` (`src/preprocessing.py`, `src/blocking.py`), automation in `scripts/`, test suites in `tests/`.

### Integration Points
- Raw input datasets: `student_resource/dataset/train/` and `student_resource/dataset/test/` (or dynamic path resolver).
- Partition splits: `data/splits/val_ground_truth.tsv` used by `scripts/audit_blocking_recall.py` for BLOCK-04 empirical verification.
- Output artifacts: `output/candidate_pairs.tsv` and `data/cache/candidates/`.

</code_context>

<specifics>
## Specific Ideas

- **Hybrid Cascade**: Inverted index exact keys (`by_both`, `by_name`, `by_addr`, `by_pref`) capture high-confidence pairs at $O(1)$ speed, followed by subword char (3,4)-gram TF-IDF sparse matrix dot products capturing fuzzy misspellings and phonetics.
- **France Open-Set**: Zero France data in training, but test set includes France. Dynamic country grouping (`df['country'].unique()`) and multilingual French suffix stripping (`SA`, `SAS`, `SARL`, `EURL`) ensure France records are blocked seamlessly without code modification.
- **Singleton Protection**: Strict empty string emission (`S1-ID\t`) for entities without candidate matches above threshold, preventing false merges and protecting macro $F_{0.5}$ score.

</specifics>

<deferred>
## Deferred Ideas

- **DENSE-01**: Dense neural bi-encoder embeddings (e.g. `bge-small-en-v1.5` or `all-MiniLM-L6-v2`) for candidate retrieval — deferred to post-baseline v2 enhancements if TF-IDF + inverted index cascade achieves $\ge 98\%$ recall ceiling.
- **ENSEMBLE-01**: Multi-model candidate blending across alternative embedding spaces — deferred to v2.

</deferred>

---

*Phase: 2-Text Preprocessing & Cascade Blocking Engine*
*Context gathered: 2026-09-27*
