# Phase 2: Text Preprocessing & Cascade Blocking Engine - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-27
**Phase:** 2-Text Preprocessing & Cascade Blocking Engine
**Areas discussed:** Blocking Engine Architecture, Candidate Budget & Retention Cutoffs, Multilingual Text Cleaning & Legal Suffix Scope, Candidate Persistence & Cache Formats

---

## Blocking Engine Architecture

### Q1: Candidate Retrieval Strategy Within Country Partitions
| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid Cascade | Inverted index exact keys (name+addr, name, addr, 2-token prefix) first, supplemented by char (3,4)-gram TF-IDF sparse matrix dot-product for fuzzy coverage | ✓ |
| Pure Char (3,4)-gram TF-IDF | Batched sparse matrix multiplication over char n-gram TF-IDF vectors (baseline approach) | |
| Pure Inverted Index Dictionary | Multi-key hash cascade only (fastest O(N) lookup, but misses spelling variants and minor typos) | |
| You decide | Antigravity picks the most performant and recall-optimal design | |

**User's choice:** Hybrid Cascade
**Notes:** Provides fast $O(1)$ discovery for exact/prefix matches while catching misspelled and noisy records via subword TF-IDF.

### Q2: Target Record Pooling (S2 + S3)
| Option | Description | Selected |
|--------|-------------|----------|
| Unified Target Pool (S2 + S3) | Index S2 and S3 together in a single country index, letting exact keys and TF-IDF dot-products retrieve the globally most relevant candidates across both sources | ✓ |
| Separate Source Indexes | Index S2 and S3 independently, allocate dedicated candidate quota (e.g., top-12 each), and union the results | |
| Source Priority Staging | Query S2 first, then query S3 only for remaining quota or missing entity coverage | |
| You decide | Antigravity picks based on indexing throughput and recall ceiling | |

**User's choice:** Unified Target Pool (S2 + S3)
**Notes:** Joint indexing avoids redundant vectorizer passes and allows true global similarity competition between S2 and S3 records.

### Q3: Open-Set Country Partitioning (France & Dynamic Sets)
| Option | Description | Selected |
|--------|-------------|----------|
| Dynamic Open-Set Partitioning | Partition strictly by unique country values discovered at runtime (`df['country'].unique()`), building self-contained indexes per country without hardcoding countries | ✓ |
| Static Country Handlers + Fallback | Dedicated tuned configurations for known countries (US, India) with a generalized fallback handler for France | |
| You decide | Antigravity implements the cleanest, most general open-set partitioner | |

**User's choice:** Dynamic Open-Set Partitioning
**Notes:** Guarantees zero code changes required when evaluating test data containing France.

### Q4: Exact Inverted Index Key Hierarchy & Guards
| Option | Description | Selected |
|--------|-------------|----------|
| 4-Stage Guarded Hierarchy | 1) `by_both` (clean_name + clean_addr), 2) `by_name` (clean_name >= 3 chars), 3) `by_addr` (clean_addr >= 10 chars, excluding generic stopwords), 4) `by_pref` (first 2 significant name tokens) | ✓ |
| Name-Centric Exact Indexing | Exact clean_name and 2-token prefix keys only; delegate all address similarity to TF-IDF sparse matrix dot-products | |
| Locality Sensitive Hashing (MinHash LSH) | Character n-gram MinHash indexing for approximate nearest neighbors instead of token hash tables | |
| You decide | Antigravity tunes the inverted index key guards for optimal precision and memory footprint | |

**User's choice:** 4-Stage Guarded Hierarchy
**Notes:** Length safeguards prevent massive bucket explosion on short/generic addresses while securing high-precision exact matches.

---

## Candidate Budget & Retention Cutoffs

### Q1: Maximum Candidate Budget (K) per S1 Entity
| Option | Description | Selected |
|--------|-------------|----------|
| Top-25 Candidates (K=25) | Delivers proven >=98% recall ceiling while capping downstream Phase 3 pair evaluation at <= 43M total pairs | ✓ |
| Tight Top-15 Candidates (K=15) | Cuts feature extraction time and storage by 40%, optimizing for fast test inference speed | |
| Broad Top-40 Candidates (K=40) | Ultra-conservative recall ceiling protection, accepting higher downstream feature computation | |
| You decide | Antigravity benchmarks recall vs pair volume to lock the optimal K | |

**User's choice:** Top-25 Candidates (K=25)
**Notes:** Preserves recall headroom while maintaining feasible runtime for 24-feature extraction in Phase 3.

### Q2: Minimum Cosine Similarity Threshold for Fuzzy Matches
| Option | Description | Selected |
|--------|-------------|----------|
| Minimum Similarity 0.18 | Proven sweet spot in baseline; eliminates random char noise while capturing distorted transliterations and typos | ✓ |
| Conservative 0.25 | Stricter cutoff; keeps candidate sets small (5-10 per entity) and prunes marginal matches early | |
| Permissive 0.12 | Looser cutoff; surfaces borderline candidates to provide rich hard-negative training examples for Phase 3 LightGBM | |
| You decide | Antigravity calibrates the cutoff on validation split ground truth | |

**User's choice:** Minimum Similarity 0.18
**Notes:** Re-confirmed baseline threshold value for effective noise suppression.

### Q3: Candidate Merging and Ranking Across Stages
| Option | Description | Selected |
|--------|-------------|----------|
| Exact Priority First + TF-IDF Score Rank | Guarantee exact matches (both, name, addr) top slots in candidate list, filled out to top-K with TF-IDF fuzzy candidates sorted by cosine score | ✓ |
| Pure Cosine Similarity Re-scoring | Re-score all candidates (exact + fuzzy) via TF-IDF dot product and rank purely by numerical similarity score | |
| Reciprocal Rank Fusion (RRF) | Combine ordinal ranks from the 4 inverted index keys and TF-IDF into an RRF composite score | |
| You decide | Antigravity picks the ranking policy that best separates true positives in top-K | |

**User's choice:** Exact Priority First + TF-IDF Score Rank
**Notes:** Exact matches are guaranteed preservation without being crowded out by high TF-IDF scores on common tokens.

### Q4: Singleton Representation in Candidate TSV
| Option | Description | Selected |
|--------|-------------|----------|
| Strict Official Spec Compliance | Emit `S1-ID\t` (empty candidate string) when no matches pass cutoff, perfectly mirroring validate_submission.py requirements and protecting singletons | ✓ |
| Force Minimum 1 Nearest Neighbor | Always assign at least 1 candidate (the highest dot-product score) even if below 0.18, letting Phase 3 model reject it | |
| You decide | Antigravity adheres strictly to the official evaluation harness behavior | |

**User's choice:** Strict Official Spec Compliance
**Notes:** Prevents false matches from leaking into candidate sets for true singleton entities.

---

## Multilingual Text Cleaning & Legal Suffix Scope

### Q1: Corporate Legal Suffix Regex Coverage
| Option | Description | Selected |
|--------|-------------|----------|
| Comprehensive Multilingual Regex (35+ Suffixes) | Explicitly cover US (Inc, LLC, Corp, LLP), India (Pvt Ltd, Private Limited), France (SA, SAS, SARL, EURL), and European (GmbH, BV, NV) with strict word boundaries and trailing priority | ✓ |
| Core US & India Only (15 Suffixes) | Focus regex only on high-frequency US/India terms, relying on char n-grams for French legal variations | |
| Corpus-Driven Suffix Pruning | Statistically identify top-frequency tail tokens across the corpus to automatically prune common business legal designations | |
| You decide | Antigravity implements the robust multilingual regex suite | |

**User's choice:** Comprehensive Multilingual Regex (35+ Suffixes)
**Notes:** Broad international legal coverage prepares the pipeline for multilingual test evaluations.

### Q2: Unicode Diacritic and Accent Normalization
| Option | Description | Selected |
|--------|-------------|----------|
| Selective Diacritic Stripping (NFKD + Filter Combining) | Decomposes French accents (é -> e, ç -> c) into base letters while preserving Unicode alphanumeric character integrity across scripts | ✓ |
| Aggressive 7-bit ASCII Coercion | Forcibly transliterates all non-ASCII glyphs to closest Latin characters or drops unmapped glyphs | |
| You decide | Antigravity picks the cleanest Unicode handling that avoids data loss | |

**User's choice:** Selective Diacritic Stripping (NFKD + Filter Combining)
**Notes:** Prevents character loss for Indic transliterated words while eliminating French accent mismatches.

### Q3: Address Standardization Strategy
| Option | Description | Selected |
|--------|-------------|----------|
| Canonical Word Expansion | Expand street, unit, and state abbreviations to full lowercased words (st -> street, ave -> avenue, ste -> suite, ny -> new york, mh -> maharashtra) with word boundary regexes | ✓ |
| Short Abbreviation Contraction | Contract all words to short standard tokens (street -> st, avenue -> ave, suite -> ste) to minimize string length | |
| Minimal Punctuation & Whitespace Only | Strip punctuation and extra spaces, but keep raw address words intact without dictionary expansion | |
| You decide | Antigravity optimizes abbreviation rules to maximize token overlap across sources | |

**User's choice:** Canonical Word Expansion
**Notes:** Maximizes token overlap between sources using full words and canonical state names.

### Q4: Text Representation for Inverted Indexing vs TF-IDF
| Option | Description | Selected |
|--------|-------------|----------|
| Dual Representation with Name-Boosted Search Text | Maintain distinct clean_name and clean_address columns for exact index keys, and build a name-boosted composite string (clean_name * 2 + ' ' + clean_address) for TF-IDF matching | ✓ |
| Separate Dual TF-IDF Vectorizers | Train separate TF-IDF models for name and address, running two independent matrix multiplications and blending similarity scores | |
| Single Unweighted Flat String | Concatenate clean_name and clean_address 1:1 for both exact hashing and TF-IDF vectorization | |
| You decide | Antigravity benchmarks name vs address weighting for optimal candidate retrieval | |

**User's choice:** Dual Representation with Name-Boosted Search Text
**Notes:** Name carries higher entity discrimination than address, so 2x weighting in search text yields superior ranking.

---

## Candidate Persistence & Cache Formats

### Q1: Large-Scale TSV Export Mechanism
| Option | Description | Selected |
|--------|-------------|----------|
| Streaming Chunked Exporter | Stream rows directly to output/candidate_pairs.tsv in buffered chunks (50,000 entities/batch) with explicit flushing, keeping memory usage constant and protecting against OOM crashes | ✓ |
| In-Memory Aggregation & Export | Accumulate full results in memory dictionary and export via pandas.to_csv(sep='\t') at completion | |
| You decide | Antigravity picks the most robust, low-memory exporter | |

**User's choice:** Streaming Chunked Exporter
**Notes:** Memory footprint stays flat (<2GB) regardless of 1.73M dataset scale.

### Q2: Intermediate Candidate Storage for Phase 3 Feature Extraction
| Option | Description | Selected |
|--------|-------------|----------|
| Partitioned Parquet Cache (data/cache/candidates/) | Store pairs with metadata (s1_id, cand_id, stage, rank, cosine_sim) in country-partitioned Parquet files for ultra-fast columnar reads during Phase 3 feature engineering | ✓ |
| Direct TSV Parsing in Phase 3 | Do not write binary cache; let Phase 3 parse output/candidate_pairs.tsv on-the-fly | |
| Serialized Joblib / Pickle Dictionaries | Dump Python candidate dicts to disk using joblib for exact in-memory reproduction | |
| You decide | Antigravity implements the fastest data interchange format for Phase 3 | |

**User's choice:** Partitioned Parquet Cache (data/cache/candidates/)
**Notes:** Country partitioning enables parallel chunked loading in Phase 3 without reparsing raw text strings.

### Q3: Recall Ceiling Verification Report (BLOCK-04)
| Option | Description | Selected |
|--------|-------------|----------|
| Standalone Audit CLI & Structured JSON | Evaluate against val_ground_truth.tsv, computing overall pair recall, country breakdown, singleton preservation, and space reduction factor (>99.9%) | ✓ |
| Stdout Console Logging Only | Print recall and reduction metrics to the console during blocking execution without writing persistent JSON audit files | |
| You decide | Antigravity implements the audit tooling conforming to verification standards | |

**User's choice:** Standalone Audit CLI & Structured JSON (output/blocking_recall_audit.json)
**Notes:** Generates reproducible, auditable proof that the >=95% recall ceiling requirement is satisfied.

### Q4: Verification Test Suite Integration
| Option | Description | Selected |
|--------|-------------|----------|
| Deterministic Pytest Suite + Validator Harness | tests/test_phase2_blocking.py testing normalization edge cases, zero cross-country leakage, recall ceiling threshold, and validate_submission.py exit 0 via scripts/verify.ps1 | ✓ |
| Basic Functional Script Check | Smoke test the main blocking script on a dev sample without formal unit test assertions | |
| You decide | Antigravity implements full test coverage adhering to project invariants | |

**User's choice:** Deterministic Pytest Suite + Validator Harness
**Notes:** Ensures all phase outputs pass automated local regression and submission validator gates.

---

## Antigravity's Discretion

- Adaptive batch size for sparse matrix multiplications (50 to 1000 records) to keep peak RAM usage under 4GB across all country partitions.
- Sublinear term frequency scaling (`sublinear_tf=True`) for char n-gram TF-IDF vectorization.

## Deferred Ideas

- None — discussion remained tightly focused on Phase 2 scope. Neural embeddings (DENSE-01) and model blending (ENSEMBLE-01) remain deferred to v2.
