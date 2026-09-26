# Amazon ML Challenge 2026: Business Entity Resolution
## Comprehensive Technical Approach & System Specification

---

## 1. Challenge & Problem Definition

### 1.1 Objective
Build an end-to-end Machine Learning pipeline that, given business records from 3 independent data sources (`Source 1`, `Source 2`, `Source 3`) with noisy and inconsistent fields, determines which records across sources refer to the same real-world business entity.
- **Source 1 ($S_1$)**: The deduplicated reference source.
- **Source 2 ($S_2$)** & **Source 3 ($S_3$)**: Secondary and tertiary noisy sources.
- **Matching Cardinality**: Each Source 1 entity may match zero (singleton), one, or multiple records from Source 2 and Source 3.

### 1.2 Evaluation Metric: Macro $F_{0.5}$
The evaluation metric is the macro-averaged $F_{0.5}$ score across all Source 1 entities:
$$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

**Key Properties**:
- **Precision-heavy**: False merges (merging two distinct businesses) are penalized $2\times$ more severely than missed matches.
- **Singleton Handling**: Entities with no true matches score $1.0$ if predicted empty, and $0.0$ if any match is predicted. Accurate singleton identification is a major score driver.

### 1.3 Dataset Specifications
- **Format**: Strictly Tab-Separated Values (`.tsv`).
- **Data Location**: `6ab10eb3b23ba_student_resource/student_resource/dataset/`
  - Training:
    - `train_source1.tsv` (~210 MB)
    - `train_source2.tsv` (~489 MB)
    - `train_source3.tsv` (~504 MB)
    - `train_ground_truth.tsv` (~127 MB)
  - Testing:
    - `test_source1.tsv` (~175 MB, 1,732,544 records)
    - `test_source2.tsv` (~509 MB, ~5.09M records)
    - `test_source3.tsv` (~506 MB, ~5.06M records)
- **Columns**: `entity_id` (`S1-*`, `S2-*`, `S3-*`), `business_name`, `business_address`, `country`.
- **Country Distribution**:
  - Training: `US`, `India`.
  - Testing: `US`, `India`, and `France` (zero France training data).
  - Open label constraint: Never hardcode or filter to only `{US, India}`.

### 1.4 Hard Competition Constraints & Rules
1. **Academic Integrity & Fair Play**: STRICTLY ZERO external data lookups, commercial ER APIs, government registries, or geocoding APIs. All learning must be self-contained from training data.
2. **Model Bounds**: Maximum 8 Billion parameters; open-source permissive licenses only (MIT / Apache 2.0).
3. **Format Integrity**:
   - `matching_results.tsv`: `source1_entity_id\tmatched_entity_ids` (comma-separated, sorted, no duplicates).
   - `candidate_pairs.tsv`: `source1_entity_id\tcandidate_entity_ids` (must be superset of matches).
   - Validated locally via `student_resource/utils/validate_submission.py`.

---

## 2. Multi-Stage Pipeline Architecture

```
                  ┌────────────────────────────────────────┐
                  │ Raw Datasets (S1, S2, S3 TSV files)    │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ 1. Text Normalization & Cleaning       │
                  │    • Unicode NFKD transliteration      │
                  │    • Corporate suffix normalization    │
                  │    • Address & punctuation cleaning    │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ 2. Country-Partitioned Cascade Blocking│
                  │    • 100% strict country partition     │
                  │    • Multi-key inverted index:         │
                  │      - Exact (Name + Address)          │
                  │      - Exact Name                      │
                  │      - Normalized Address (>8 chars)   │
                  │      - Token Prefix keys               │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ 3. Candidate Pairs Generation          │
                  │    • candidate_pairs.tsv emitted       │
                  │    • High recall ceiling preservation  │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ 4. Feature Engineering (24-dim Vector) │
                  │    • Jaro-Winkler, Levenshtein ratio   │
                  │    • Token Jaccard, Containment        │
                  │    • 3-gram char Jaccard               │
                  │    • Street number conflict detector   │
                  │    • Missingness indicators            │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ 5. GBDT Matching & Macro F0.5 Tuning   │
                  │    • LightGBM Binary Classifier        │
                  │    • Threshold optimization (theta*~0.7)│
                  │    • Singleton threshold guard         │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ 6. Output & Verification               │
                  │    • matching_results.tsv              │
                  │    • Submission validator (exit 0)     │
                  │    • Code package & methodology zip    │
                  └────────────────────────────────────────┘
```

---

## 3. Detailed Component Engineering

### 3.1 Text Preprocessing & Cleaning (`src/preprocessing.py`)
- **Unicode NFKD Normalization**: Decomposes accents and diacritics (crucial for French business names and transliterated Indian business names).
- **Corporate Entity Stripping**: Removes suffixes like `LLC`, `INC`, `CORP`, `PVT LTD`, `LTD`, `SA`, `SAS`, `GMBH`, `CO`, `ENTERPRISES` to isolate root trade identities.
- **Address Standardization**: Expands abbreviations (`ST` $\to$ `STREET`, `RD` $\to$ `ROAD`, `AVE` $\to$ `AVENUE`, `BLVD` $\to$ `BOULEVARD`), strips non-alphanumeric noise, and standardizes whitespace.

### 3.2 Country-Partitioned Cascade Blocking (`src/blocking.py`)
- **Zero Cross-Country Invariant**: Ground truth analysis reveals 0.00% cross-country matches. Blocking strictly partitions records by country (`US`, `India`, `France`), shrinking comparison volume by $>70\%$ without recall loss.
- **Cascade Inverted Index Keys**:
  1. `by_both`: exact `clean_name` + exact `clean_addr`
  2. `by_name`: exact `clean_name`
  3. `by_addr`: exact `clean_addr` (length $>8$)
  4. `by_pref`: first two significant name tokens (handles prefix variations & typos)
- **Candidate Pruning**: Caps maximum candidates per Source 1 entity to control memory while retaining $>99\%$ recall ceiling.

### 3.3 Feature Extraction (`src/feature_extractor.py` & `src/similarity_metrics.py`)
For every candidate pair $(S_1, S_{2/3})$, extract a dense feature vector:
1. **Name Similarity**:
   - Jaro-Winkler score
   - Levenshtein ratio
   - Token Jaccard similarity
   - Token Containment (handling abbreviations and parent names)
   - Character 3-gram Jaccard similarity
   - Length ratio and length difference
2. **Address Similarity**:
   - Address Jaro-Winkler
   - Address Token Jaccard & Containment
   - Address Character 3-gram Jaccard
3. **Numeric & Postal Conflict Detector**:
   - Extract street numbers and digits via regex.
   - Calculate numeric token overlap.
   - Flag explicit numeric conflicts (e.g. `104 Broadway` vs `208 Broadway` $\to$ `has_number_conflict = 1.0`).
4. **Data Availability / Missingness Flags**:
   - `s1_has_addr`, `cand_has_addr`, `both_have_addr`.
5. **Cross-Domain Interaction**:
   - `name_sim * addr_sim` product features.

### 3.4 Classification & Threshold Tuning (`src/train_matcher.py`, `src/tune_threshold.py`)
- **Classifier**: High-throughput LightGBM binary classifier trained on sampled positive and hard-negative pairs.
- **Threshold Optimization**:
  - Rather than standard $0.50$, grid search threshold $\theta \in [0.40, 0.90]$ against validation Macro $F_{0.5}$.
  - Optimal threshold $\theta^* \approx 0.70$ aggressively suppresses false positives to safeguard precision.

---

## 4. Compute & Hybrid Execution Strategy (Kaggle + Local)

Following the proven architecture from `d:\projects\RL_project`:
1. **Dual Compute Tiers**:
   - **Local Compute**: Fast iterative prototyping, EDA, preprocessing validation, feature testing, and validation scoring on local CPU / NVIDIA GPU.
   - **Kaggle GPU Dispatch**: Automate remote Kaggle kernel packaging, dispatch, and artifact download via Kaggle API (`kaggle kernels push`, `kaggle kernels status`, `kaggle kernels output`) to parallelize full 1.73M test blocking, heavy feature generation, and model training.
2. **Deterministic Artifact Sync**:
   - Download trained model checkpoints and output files (`matching_results.tsv`, `candidate_pairs.tsv`) back into the local project workspace.
   - Run local offline submission validation (`validate_submission.py`).

---

## 5. Deliverables & Submission Packaging

A single automated script (`package_submission.py`) produces `<team_name>_submission.zip`:
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # Scored on live leaderboard
│   └── candidate_pairs.tsv         # Blocking candidate set audit
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # Complete, runnable pipeline code
│       ├── README.md               # End-to-end reproduction instructions
│       └── requirements.txt        # Pinned dependency environment
└── Documentation_template.md       # Complete methodology writeup
```

---

## 6. Implementation Milestones

- **Phase 1: Project Scaffolding & Data Environment Setup**: Data paths, verification scripts, git tracking, baseline validator checks.
- **Phase 2: Text Preprocessing & Strict Country-Cascade Blocking**: Inverted indexes, candidate generation, candidate_pairs exporter.
- **Phase 3: Feature Engineering & LightGBM Model Pipeline**: Feature extractors, dataset builder, classifier training, validation evaluation.
- **Phase 4: Remote Kaggle Dispatch & Acceleration**: Automated packaging and execution for heavy test inference.
- **Phase 5: Full 1.73M Test Inference, Packaging & Documentation**: Generating final outputs, passing `validate_submission.py` (exit 0), packaging submission zip, and completing `Documentation_template.md`.
