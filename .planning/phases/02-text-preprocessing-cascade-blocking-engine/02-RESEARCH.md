# Phase 2: Text Preprocessing & Cascade Blocking Engine - Technical Research

**Researched:** 2026-09-27
**Target Phase:** Phase 2: Text Preprocessing & Cascade Blocking Engine

---

## 1. Text Normalization, Unicode De-biasing & Suffix Stripping

### 1.1 Unicode NFKD & Accent Handling (PREP-01)
- **Standard Library Primitives**: Use `unicodedata.normalize('NFKD', text)` combined with `unicodedata.combining(char)` filtering.
- **Transliteration & Diacritic Decomposition**:
  - French accents (`é`, `è`, `ê`, `ë`, `ç`, `ô`, `î`, `ï`, `à`, `ù`) decompose into base ASCII characters plus combining diacritical marks. Filtering marks (`[c for c in nfkd if not unicodedata.combining(c)]`) maps `café` $\to$ `cafe` and `société` $\to$ `societe` without external translation libraries.
  - Indic scripts and transliterations retain their base character code points, avoiding corrupted substitution tokens.
- **Sanitization Invariants**:
  - Missing strings (`None`, `""`, `"nan"`, `"null"`, `"none"`) must resolve to empty strings `""`.
  - Non-breaking spaces (`\u00a0`), tabs, and carriage returns are replaced with single spaces.

### 1.2 Corporate Legal Suffix Removal (PREP-02)
- **35+ Suffix Regex Suite**: Covers US, Indian, French, and European legal forms:
  ```python
  LEGAL_SUFFIXES = [
      # US & International
      r'\bprivate limited\b', r'\bpvt\s+ltd\b', r'\bpvt\b',
      r'\blimited liability company\b', r'\bllc\b',
      r'\blimited liability partnership\b', r'\bllp\b',
      r'\bincorporated\b', r'\bincorporation\b', r'\binc\b',
      r'\bcorporation\b', r'\bcorp\b',
      r'\blimited\b', r'\bltd\b',
      r'\bcompany\b', r'\bco\b',
      r'\bprofessional corporation\b', r'\bpc\b',
      # India
      r'\benterprises\b', r'\btrading company\b', r'\bindustries\b',
      r'\bsolutions\b', r'\bservices\b',
      # France
      r'\bsociete anonyme\b', r'\bsa\b',
      r'\bsociete par actions simplifiee\b', r'\bsas\b',
      r'\bsociete a responsabilite limitee\b', r'\bsarl\b',
      r'\bentreprise unipersonnelle a responsabilite limitee\b', r'\beurl\b',
      r'\bsociete en nom collectif\b', r'\bsnc\b',
      # Europe
      r'\bgmbh\b', r'\bag\b', r'\bbv\b', r'\bnv\b', r'\bspa\b'
  ]
  ```
- **Word Boundary & Trailing Rules**:
  - Compiled with `re.IGNORECASE` and word boundaries `\b` to avoid stripping substrings of real business names (e.g. `company` stripped only as a full token, not from `accompany`).
  - Applied prior to punctuation removal, followed by whitespace collapsing.

### 1.3 Address Standardization & Canonical Expansion (PREP-03)
- **Street & Unit Abbreviation Expansion**:
  - Common road types: `st` $\to$ `street`, `rd` $\to$ `road`, `ave` $\to$ `avenue`, `blvd` $\to$ `boulevard`, `dr` $\to$ `drive`, `hwy` $\to$ `highway`, `ln` $\to$ `lane`, `ct` $\to$ `court`, `pkwy` $\to$ `parkway`, `ter` $\to$ `terrace`, `no` $\to$ `number`.
  - Secondary units: `ste` $\to$ `suite`, `apt` $\to$ `apartment`, `fl` $\to$ `floor`.
- **State Standardization**:
  - US states: `ny` $\to$ `new york`, `ca` $\to$ `california`, `tx` $\to$ `texas`, `fl` $\to$ `florida`, `il` $\to$ `illinois`, `pa` $\to$ `pennsylvania`, etc.
  - Indian states: `up` $\to$ `uttar pradesh`, `mp` $\to$ `madhya pradesh`, `mh` $\to$ `maharashtra`, `ka` $\to$ `karnataka`, `dl` $\to$ `delhi`, `tn` $\to$ `tamil nadu`, etc.
- **Punctuation & Noise**:
  - Punctuation removed via unicode alphanumeric regex `r'[^\w\s]'`.
  - Digits and street numbers preserved verbatim for downstream numeric conflict detection.

---

## 2. Country-Partitioned Cascade Blocking Architecture

### 2.1 Zero Cross-Country Invariant (BLOCK-01)
- Empirical ground truth inspection confirms 0.00% cross-country matches.
- All blocking operations occur within isolated country partitions.
- Dynamic partition discovery:
  ```python
  unique_countries = s1_df['country'].dropna().unique()
  for country in unique_countries:
      s1_c = s1_df[s1_df['country'] == country]
      s2_c = s2_df[s2_df['country'] == country]
      s3_c = s3_df[s3_df['country'] == country]
  ```
- **France Zero-Shot Open-Set Compatibility**: Since training data has only `US` and `India` while test data adds `France`, dynamic country grouping ensures `France` receives an identical, self-contained indexing pipeline without hardcoded branch conditions.

### 2.2 Unified Target Pooling ($S_2 + S_3$)
- Source 2 and Source 3 are concatenated into a single target pool per country partition:
  $$\text{Target Pool} = S_2 \cup S_3$$
- Source 1 entities query this joint pool, allowing candidates from both sources to compete directly on exact index keys and TF-IDF similarity.

### 2.3 4-Stage Guarded Inverted Index Cascade (BLOCK-02)
Exact hash keys are constructed to rapidly resolve high-confidence pairs:
1. **Stage 1 (`by_both`)**: `clean_name + "::" + clean_addr` (requires `len(clean_addr) >= 8`).
2. **Stage 2 (`by_name`)**: `clean_name` (requires `len(clean_name) >= 3`).
3. **Stage 3 (`by_addr`)**: `clean_addr` (requires `len(clean_addr) >= 10`, excludes generic stop-phrases).
4. **Stage 4 (`by_pref`)**: First two significant tokens of `clean_name` (`token_0 + "_" + token_1`, requires `len >= 4`).

### 2.4 Subword Char (3,4)-Gram TF-IDF Sparse Dot-Product (BLOCK-02)
- **Vectorizer Configuration**:
  - `analyzer='char_wb'`, `ngram_range=(3, 4)`, `max_features=150000`, `sublinear_tf=True`, `dtype=np.float32`.
  - Search corpus: Name-boosted representation `clean_name + " " + clean_name + " " + clean_address`.
- **Memory & Throughput Optimization**:
  - Target matrix $M_T$ (shape: $N_{\text{targets}} \times D$) is fit and transformed once.
  - Transposed to Compressed Sparse Column format: $M_T^T = M_T.T.\text{tocsc}()$.
  - Source 1 queries transformed in batches (adaptive batch size: 50 for $>1\text{M}$ targets, 200 for $>200\text{K}$, 1000 otherwise).
  - Batched dot-product: $\text{SimBatch} = M_{S1\_batch} \cdot M_T^T$.
  - Direct CSR array slicing (`indptr`, `indices`, `data`) avoids dense numpy allocation.

### 2.5 Candidate Merging, Budget & Cutoffs (BLOCK-03)
- **Top-$K$ Budget**: Maximum $K = 25$ candidates per Source 1 entity.
- **Minimum Similarity**: Cosine score $\ge 0.18$.
- **Ranking Priority**:
  1. Exact inverted index matches placed first (ordered by discovery stage 1 through 4).
  2. Remaining slots up to $K=25$ filled with TF-IDF fuzzy candidates sorted descending by cosine similarity score.
  3. Strict deduplication on `candidate_id`.
- **Singleton Preservation**: If no candidate passes exact keys or $\ge 0.18$ cosine threshold, emit `S1-ID\t` (empty candidate string).

---

## 3. Data Persistence, Caching & Submission Integration

### 3.1 Streaming TSV Exporter
- Writes directly to `output/candidate_pairs.tsv`.
- Format: `source1_entity_id\tcandidate_ids` where candidate IDs are comma-separated and sorted.
- Buffering: Flushed in 50,000-entity chunks to keep memory usage flat (<2 GB RAM) across 1.73M records.

### 3.2 Intermediate Parquet Cache for Phase 3
- Location: `data/cache/candidates/` partitioned by `country`.
- Schema:
  | Column | Type | Description |
  |--------|------|-------------|
  | `s1_id` | string | Source 1 entity ID |
  | `candidate_id` | string | Source 2 or Source 3 candidate entity ID |
  | `blocking_stage` | string | `by_both`, `by_name`, `by_addr`, `by_pref`, or `tfidf` |
  | `rank` | int32 | Rank of candidate within this S1 entity (0 to 24) |
  | `cosine_sim` | float32 | TF-IDF cosine similarity score (or 1.0 for exact) |

### 3.3 Empirical Recall Ceiling Audit (BLOCK-04)
- Verification script: `scripts/audit_blocking_recall.py`.
- Ground truth reference: `data/splits/val_ground_truth.tsv` (generated in Phase 1).
- Metric:
  $$\text{Pair Recall Ceiling} = \frac{\sum_{(S_1, \text{Cand}) \in GT} \mathbb{I}(\text{Cand} \in \text{CandidateSet}(S_1))}{|GT_{\text{positive}}|}$$
- Output report: `output/blocking_recall_audit.json` capturing overall recall ($\ge 95\%$), recall by country (`US` vs `India`), average candidates per entity, and space reduction factor ($>99.9\%$).

---

## 4. Validation Architecture (Dimension 8)

### 4.1 Automated Test Suite (`tests/test_phase2_blocking.py`)
Deterministic test suite covering:
1. `test_unicode_normalization_accents`: Verifies French accent stripping (`café` $\to$ `cafe`, `société` $\to$ `societe`) and Indic transliteration safety.
2. `test_legal_suffix_stripping`: Verifies 35+ corporate suffixes stripped from tail positions while preserving legitimate internal tokens.
3. `test_address_abbreviation_expansion`: Verifies street and state abbreviations expand canonically.
4. `test_country_partition_isolation`: Validates zero cross-country candidate pairs on multi-country inputs.
5. `test_cascade_inverted_index`: Verifies exact matching stages (`by_both`, `by_name`, `by_addr`, `by_pref`).
6. `test_candidate_tsv_export_and_validation`: Generates sample candidate pairs and runs official `student_resource/utils/validate_submission.py` checking exit code 0.

### 4.2 Deterministic Verification Command
```powershell
pwsh -File .\scripts\verify.ps1
```
Runs pytest across Phase 1 and Phase 2 test suites and verifies clean exit 0.
