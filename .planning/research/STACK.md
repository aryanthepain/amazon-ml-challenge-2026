# Stack Research

**Domain:** Business Entity Resolution & Record Linkage at Scale  
**Researched:** 2026-09-27  
**Confidence:** HIGH  

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.10+ / 3.11 | Core Runtime | Robust compatibility with LightGBM, PyTorch, rapidfuzz, and Kaggle environments |
| LightGBM | >=4.1.0 | Fast GBDT Matching Classifier | Ultra-fast histogram-based tree learning; excels at tabular similarity metrics with low memory footprint and high inference speed |
| Polars & Pandas | Polars >=0.20.0, Pandas >=2.0.0 | High-Throughput Ingestion & TSV Processing | Polars provides multi-threaded lazy execution for memory-efficient scanning of multi-GB TSV files; Pandas handles final formatting |
| RapidFuzz | >=3.6.0 | C++ Accelerated String Metrics | C++ vectorized Levenshtein, Jaro-Winkler, and token sorting metrics; 10x-50x faster than pure-Python fuzzywuzzy/Levenshtein |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-learn | >=1.4.0 | Precision-Recall & Threshold Tuning | Metric evaluation, train/val stratified splitting, ROC-AUC and PR-AUC computation |
| PyTorch | >=2.1.0 | Neural Inference Acceleration | Local GPU / Kaggle T4 inference if computing dense bi-encoder sentence representations |
| Sentence-Transformers | >=2.5.0 | Dense Semantic Embeddings | Supplementary semantic embeddings (e.g. `all-MiniLM-L6-v2`, `bge-small-en-v1.5`) within 8B parameter budget |
| kaggle | >=2.2.4 | Cloud Compute Automation | Automated remote kernel pushing, status polling, and artifact downloading to utilize 30 hrs/week dual T4 GPUs |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Git & GitHub CLI (`gh`) | Version Control & Issue Tracking | Phase-based branching (`feat/phase-<N>-*`) with linked issues |
| Kaggle CLI | GPU Execution Dispatch | Dual T4 GPU execution for full 1.73M test candidate generation and inference |
| PowerShell (`pwsh`) | Automation Scripts | Fast orchestration, verification pipelines, and completion alarms |

## Installation

```bash
# Core & Tabular ER Stack
pip install pandas polars rapidfuzz lightgbm scikit-learn pyyaml kaggle

# Optional Neural Embedding Stack (if evaluating bi-encoders)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install sentence-transformers
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| LightGBM | CatBoost / XGBoost | CatBoost handles high-cardinality categorical fields well, but is significantly slower to train and infer over millions of pairs than LightGBM. |
| RapidFuzz | FuzzyWuzzy / jellyfish | FuzzyWuzzy is pure Python with GPL licensing; RapidFuzz has MIT license and C++ SIMD optimizations. |
| Inverted Index Blocking | Dense Vector Index (FAISS) | Brute-force FAISS over 10.1M records requires massive VRAM and index build time; lexical inverted index with strict country partitioning is order-of-magnitude faster with ~97% recall ceiling. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| External geocoding / ER APIs (Google Maps, Nominatim) | Strictly prohibited by competition fair play rules; causes instant disqualification | In-memory regex parsing and street number conflict detection |
| Heavy LLMs > 8B parameters | Prohibited by rule 5 (max 8B parameter limit) | Compact bi-encoders (MiniLM, BGE) or GBDT with engineered features |
| Full $O(N^2)$ Pairwise Cartesian Joins | $1.73\text{M} \times 10.1\text{M} \approx 17.6 \times 10^{12}$ comparisons (OOM crash) | Multi-key inverted index cascade blocking with country partitioning |

---
*Stack research for: Amazon ML Challenge 2026 Business Entity Resolution*  
*Researched: 2026-09-27*
