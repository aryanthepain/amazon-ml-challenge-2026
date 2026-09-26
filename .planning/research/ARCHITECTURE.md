# Architecture Research

**Domain:** Business Entity Resolution at Scale (Amazon ML Challenge 2026)  
**Researched:** 2026-09-27  
**Confidence:** HIGH  

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      1. Ingestion & Preprocessing                       │
│  ┌────────────────────────┐  ┌────────────────────────────────────────┐ │
│  │ Unicode NFKD Normalize │  │ Corporate Suffix & Address Clean       │ │
│  └───────────┬────────────┘  └───────────────────┬────────────────────┘ │
└──────────────┼───────────────────────────────────┼──────────────────────┘
               ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 2. Country-Partitioned Cascade Blocking                 │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Strict Country Partitioning (US, India, France)                   │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │ Inverted Index Cascade: by_both -> by_name -> by_addr -> by_pref  │  │
│  └───────────────────────────────────┬───────────────────────────────┘  │
└──────────────────────────────────────┼──────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      3. Candidate Generation Emit                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ candidate_pairs.tsv (High recall ceiling >= 96.6%)                │  │
│  └───────────────────────────────────┬───────────────────────────────┘  │
└──────────────────────────────────────┼──────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   4. Feature Extraction & Scoring                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ 24-dim Features: RapidFuzz metrics, Trigrams, Street Number flags │  │
│  │ Optional: Bi-Encoder Dense Semantic Embedding Cosine Similarities │  │
│  └───────────────────────────────────┬───────────────────────────────┘  │
└──────────────────────────────────────┼──────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               5. GBDT Matcher & Macro F0.5 Optimization                 │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ LightGBM Binary Classifier with optimal decision threshold theta* │  │
│  │ Precision Guard: theta* = 0.70 protects singletons & penalizes FP │  │
│  └───────────────────────────────────┬───────────────────────────────┘  │
└──────────────────────────────────────┼──────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   6. Submission Packaging & Auditing                    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ output/matching_results.tsv + output/candidate_pairs.tsv          │  │
│  │ validate_submission.py (stdlib check, exit 0)                     │  │
│  │ package_submission.py -> <team_name>_submission.zip              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `preprocessing.py` | Clean raw entity strings into canonical forms | Unicode NFKD, regex entity stripping, lowercase normalization |
| `blocking.py` | Generate candidate pairs from 17.6T search space | Multi-key inverted hash index partitioned by country |
| `feature_extractor.py` | Compute similarity and conflict signals | Vectorized string distance (RapidFuzz), numeric parsing |
| `train_matcher.py` | Fit binary classifier on pairs | LightGBM binary cross-entropy with early stopping |
| `tune_threshold.py` | Find threshold $\theta^*$ maximizing Macro $F_{0.5}$ | Grid search on validation set evaluating macro $F_{0.5}$ |
| `run_inference.py` | Full test prediction pipeline | Chunked streaming inference emitting TSV outputs |
| `package_submission.py` | Validate and bundle complete deliverable | Zip builder checking all required source files and documentation |

## Recommended Project Structure

```
d:/projects/amazon_ml_challenge/
├── .planning/                  # GSD planning documents (PROJECT, REQUIREMENTS, ROADMAP, STATE)
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
│       │   ├── run_inference.py
│       │   └── generate_full_submission.py
│       ├── models/             # Exported LightGBM model checkpoints
│       ├── README.md           # Self-contained reproduction instructions
│       └── requirements.txt    # Pinned dependencies
├── output/
│   ├── matching_results.tsv    # Scored leaderboard output
│   └── candidate_pairs.tsv     # Candidate set audit output
├── scripts/
│   ├── run_kaggle_kernel.py    # Remote Kaggle dual-T4 execution automation
│   └── verify.ps1              # Deterministic verification pipeline
├── student_resource/           # Local dataset & official validator (git-ignored)
│   ├── dataset/
│   └── utils/validate_submission.py
├── Documentation_template.md   # Official methodology report
├── package_submission.py       # Submission archive packager
└── project_approach.md         # Master technical specification
```

## Scaling Considerations

- **Memory Management**: Loading all test pairs into RAM at once requires >30 GB. Chunked stream processing (chunk size: 50,000 Source 1 entities) keeps RAM usage under 6 GB.
- **Kaggle Kernel Acceleration**: Packaging the modular `code/` folder into a single Kaggle notebook kernel allows offloading full test set candidate generation and feature extraction to Kaggle's dual Tesla T4 GPUs.

---
*Architecture research for: Amazon ML Challenge 2026 Business Entity Resolution*  
*Researched: 2026-09-27*
