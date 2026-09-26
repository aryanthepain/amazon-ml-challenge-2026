"""
Scalable Country-Partitioned Hybrid Cascade Blocking and Candidate Generation Engine (Phase 2).

Combines a 4-stage exact inverted index hierarchy (name+address, name, address, 2-token prefix)
with subword char (3,4)-gram TF-IDF sparse matrix dot-products across a unified target pool (S2+S3).
Strictly partitions by country, enforces top-25 candidate budget (K=25) with min_similarity >= 0.18,
and outputs official submission-compliant candidate TSVs and partitioned Parquet caches.
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from src.preprocessing import (
        create_search_text,
        normalize_business_address,
        normalize_business_name,
    )
except ModuleNotFoundError:
    from preprocessing import (
        create_search_text,
        normalize_business_address,
        normalize_business_name,
    )


class InvertedIndexCascade:
    """
    4-Stage Guarded Inverted Index Hierarchy (D-01, D-04).

    Builds fast exact lookup dictionaries over target entities:
    - Stage 1: by_both (clean_name + "::" + clean_addr, min addr length 8)
    - Stage 2: by_name (clean_name, min length 3)
    - Stage 3: by_addr (clean_addr, min length 10)
    - Stage 4: by_pref (first 2 tokens of clean_name, min prefix length 4)
    """

    def __init__(self):
        self.by_both: Dict[str, List[str]] = defaultdict(list)
        self.by_name: Dict[str, List[str]] = defaultdict(list)
        self.by_addr: Dict[str, List[str]] = defaultdict(list)
        self.by_pref: Dict[str, List[str]] = defaultdict(list)
        self.target_count = 0

    def build(self, target_df: pd.DataFrame, clean_names: List[str], clean_addrs: List[str]):
        """Populate inverted index tables for the unified target pool."""
        target_ids = target_df['entity_id'].values
        self.target_count = len(target_ids)

        for eid, name, addr in zip(target_ids, clean_names, clean_addrs):
            # Stage 1: by_both (name + addr)
            if name and addr and len(addr) >= 8:
                key_both = f"{name}::{addr}"
                self.by_both[key_both].append(eid)

            # Stage 2: by_name
            if name and len(name) >= 3:
                self.by_name[name].append(eid)

            # Stage 3: by_addr
            if addr and len(addr) >= 10:
                self.by_addr[addr].append(eid)

            # Stage 4: by_pref (first 2 significant tokens)
            if name:
                tokens = [t for t in name.split() if len(t) >= 2]
                if len(tokens) >= 2:
                    pref = f"{tokens[0]}_{tokens[1]}"
                    if len(pref) >= 4:
                        self.by_pref[pref].append(eid)

    def query_exact(self, s1_name: str, s1_addr: str) -> List[Tuple[str, str, float]]:
        """
        Query exact stages in priority order: by_both -> by_name -> by_addr -> by_pref.
        Returns deduplicated [(candidate_id, stage_name, default_score), ...].
        """
        matches: List[Tuple[str, str, float]] = []
        seen: Set[str] = set()

        # 1. by_both
        if s1_name and s1_addr and len(s1_addr) >= 8:
            key_both = f"{s1_name}::{s1_addr}"
            for cid in self.by_both.get(key_both, []):
                if cid not in seen:
                    seen.add(cid)
                    matches.append((cid, "exact_both", 1.0))

        # 2. by_name
        if s1_name and len(s1_name) >= 3:
            for cid in self.by_name.get(s1_name, []):
                if cid not in seen:
                    seen.add(cid)
                    matches.append((cid, "exact_name", 0.95))

        # 3. by_addr
        if s1_addr and len(s1_addr) >= 10:
            for cid in self.by_addr.get(s1_addr, []):
                if cid not in seen:
                    seen.add(cid)
                    matches.append((cid, "exact_addr", 0.90))

        # 4. by_pref
        if s1_name:
            tokens = [t for t in s1_name.split() if len(t) >= 2]
            if len(tokens) >= 2:
                pref = f"{tokens[0]}_{tokens[1]}"
                if len(pref) >= 4:
                    for cid in self.by_pref.get(pref, []):
                        if cid not in seen:
                            seen.add(cid)
                            matches.append((cid, "exact_pref", 0.85))

        return matches


class TfidfSparseBlocker:
    """
    Subword Character n-gram TF-IDF Sparse Matrix Dot-Product Blocker (D-01, D-02).
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 4),
        max_features: int = 150000,
        min_df: int = 2,
        min_similarity: float = 0.18,
        batch_size: int = 1000,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.min_similarity = min_similarity
        self.batch_size = batch_size

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.target_matrix_t: Optional[csr_matrix] = None
        self.target_ids: np.ndarray = np.array([])

    def fit_target(self, target_ids: np.ndarray, target_corpus: List[str]):
        """Fit vectorizer and index target pool matrix."""
        self.target_ids = target_ids
        num_targets = len(target_ids)

        if num_targets == 0:
            return

        effective_min_df = self.min_df if num_targets >= 100 else 1
        effective_max_df = 0.30 if num_targets >= 1000 else 1.0

        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=effective_min_df,
            max_df=effective_max_df,
            dtype=np.float32,
            sublinear_tf=True
        )

        try:
            target_matrix = self.vectorizer.fit_transform(target_corpus)
        except ValueError:
            # Fallback if too few features or strict df thresholds
            self.vectorizer.min_df = 1
            self.vectorizer.max_df = 1.0
            target_matrix = self.vectorizer.fit_transform(target_corpus)

        # Transpose target matrix once to CSC format for fast column slicing / dot product
        self.target_matrix_t = target_matrix.T.tocsc()

    def query_s1(
        self,
        s1_ids: np.ndarray,
        s1_corpus: List[str],
        top_k: int = 25
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Query S1 corpus against target pool in memory-bounded batches."""
        if self.target_matrix_t is None or len(self.target_ids) == 0 or len(s1_ids) == 0:
            return {eid: [] for eid in s1_ids}

        s1_matrix = self.vectorizer.transform(s1_corpus).tocsr()
        num_targets = len(self.target_ids)
        num_s1 = len(s1_ids)

        # Adaptive batch sizing to prevent sparse dot-product memory spikes
        if num_targets > 1_000_000:
            batch_size = 50
        elif num_targets > 200_000:
            batch_size = 200
        else:
            batch_size = min(self.batch_size, 1000)

        results: Dict[str, List[Tuple[str, float]]] = {}

        for start_idx in range(0, num_s1, batch_size):
            end_idx = min(start_idx + batch_size, num_s1)
            batch_s1_matrix = s1_matrix[start_idx:end_idx]

            # Sparse dot product: (batch_size, num_targets)
            sim_batch = batch_s1_matrix.dot(self.target_matrix_t)

            indptr = sim_batch.indptr
            indices = sim_batch.indices
            data = sim_batch.data

            for i in range(end_idx - start_idx):
                global_s1_idx = start_idx + i
                s1_id = s1_ids[global_s1_idx]

                start_p = indptr[i]
                end_p = indptr[i + 1]
                if start_p == end_p:
                    results[s1_id] = []
                    continue

                col_indices = indices[start_p:end_p]
                scores = data[start_p:end_p]

                valid_mask = scores >= self.min_similarity
                if not np.any(valid_mask):
                    results[s1_id] = []
                    continue

                filtered_cols = col_indices[valid_mask]
                filtered_scores = scores[valid_mask]

                if len(filtered_scores) > top_k:
                    top_indices = np.argpartition(-filtered_scores, top_k)[:top_k]
                    top_indices = top_indices[np.argsort(-filtered_scores[top_indices])]
                    final_cols = filtered_cols[top_indices]
                    final_scores = filtered_scores[top_indices]
                else:
                    sorted_order = np.argsort(-filtered_scores)
                    final_cols = filtered_cols[sorted_order]
                    final_scores = filtered_scores[sorted_order]

                candidates = [
                    (self.target_ids[col_idx], float(sc))
                    for col_idx, sc in zip(final_cols, final_scores)
                ]
                results[s1_id] = candidates

        return results


class BlockingEngine:
    """
    Country-Partitioned Cascade Blocking Engine (BLOCK-01 through BLOCK-04).
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 4),
        max_features: int = 150000,
        min_df: int = 2,
        top_k: int = 25,
        min_similarity: float = 0.18,
        batch_size: int = 1000,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.batch_size = batch_size

    def _prepare_corpus(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """Extract and normalize name, address, and create composite search text."""
        raw_names = df['business_name'].fillna('')
        raw_addrs = df['business_address'].fillna('')

        clean_names = [normalize_business_name(n) for n in raw_names]
        clean_addrs = [normalize_business_address(a) for a in raw_addrs]
        search_texts = [
            f"{n} {n} {a}".strip() if a else f"{n} {n}".strip()
            for n, a in zip(clean_names, clean_addrs)
        ]
        return clean_names, clean_addrs, search_texts

    def generate_candidates_for_country(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
    ) -> Tuple[Dict[str, List[str]], List[dict]]:
        """
        Generate candidate pairs for all S1 entities within a single country partition.

        Returns:
            candidates_dict: {s1_id: [candidate_id_1, candidate_id_2, ...]}
            cache_records: list of dicts with keys [s1_id, candidate_id, blocking_stage, rank, cosine_sim]
        """
        if len(s1_df) == 0:
            return {}, []

        # Combine S2 and S3 into unified target pool (D-02)
        target_dfs = []
        if len(s2_df) > 0:
            target_dfs.append(s2_df[['entity_id', 'business_name', 'business_address']])
        if len(s3_df) > 0:
            target_dfs.append(s3_df[['entity_id', 'business_name', 'business_address']])

        s1_ids = s1_df['entity_id'].values
        if not target_dfs:
            return {eid: [] for eid in s1_ids}, []

        target_df = pd.concat(target_dfs, ignore_index=True)
        target_ids = target_df['entity_id'].values

        # Prepare normalized corpora
        target_names, target_addrs, target_search = self._prepare_corpus(target_df)
        s1_names, s1_addrs, s1_search = self._prepare_corpus(s1_df)

        # 1. Build and query Inverted Index Exact Cascade (D-01, D-04)
        inv_index = InvertedIndexCascade()
        inv_index.build(target_df, target_names, target_addrs)

        # 2. Build and query TF-IDF Sparse Blocker (D-01, D-02)
        tfidf_blocker = TfidfSparseBlocker(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            min_similarity=self.min_similarity,
            batch_size=self.batch_size,
        )
        tfidf_blocker.fit_target(target_ids, target_search)
        tfidf_results = tfidf_blocker.query_s1(s1_ids, s1_search, top_k=self.top_k)

        # 3. Hybrid Combination (D-05, D-06, D-07, D-08)
        candidates_dict: Dict[str, List[str]] = {}
        cache_records: List[dict] = []

        for s1_id, name, addr in zip(s1_ids, s1_names, s1_addrs):
            exact_matches = inv_index.query_exact(name, addr)
            fuzzy_matches = tfidf_results.get(s1_id, [])

            selected_cands: List[str] = []
            seen_cands: Set[str] = set()

            # Add exact matches first
            for cid, stage, score in exact_matches:
                if cid not in seen_cands and len(selected_cands) < self.top_k:
                    seen_cands.add(cid)
                    selected_cands.append(cid)
                    cache_records.append({
                        's1_id': s1_id,
                        'candidate_id': cid,
                        'blocking_stage': stage,
                        'rank': len(selected_cands),
                        'cosine_sim': score
                    })

            # Fill remaining budget with TF-IDF fuzzy candidates
            for cid, sim in fuzzy_matches:
                if cid not in seen_cands and len(selected_cands) < self.top_k:
                    seen_cands.add(cid)
                    selected_cands.append(cid)
                    cache_records.append({
                        's1_id': s1_id,
                        'candidate_id': cid,
                        'blocking_stage': 'tfidf_fuzzy',
                        'rank': len(selected_cands),
                        'cosine_sim': sim
                    })

            candidates_dict[s1_id] = selected_cands

        return candidates_dict, cache_records

    def block_datasets(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        cache_dir: Optional[str] = None
    ) -> Tuple[Dict[str, List[str]], List[dict]]:
        """
        Execute blocking partitioned strictly by country (D-03).
        """
        start_time = time.time()
        all_candidates: Dict[str, List[str]] = {}
        all_records: List[dict] = []

        # Open-set dynamic country partitioning
        unique_countries = s1_df['country'].dropna().unique()
        print(f"=== Starting Blocking across {len(unique_countries)} countries: {list(unique_countries)} ===")

        for country in unique_countries:
            t0 = time.time()
            s1_country = s1_df[s1_df['country'] == country].reset_index(drop=True)
            s2_country = (
                s2_df[s2_df['country'] == country].reset_index(drop=True)
                if ('country' in s2_df.columns and len(s2_df) > 0)
                else pd.DataFrame(columns=['entity_id', 'business_name', 'business_address', 'country'])
            )
            s3_country = (
                s3_df[s3_df['country'] == country].reset_index(drop=True)
                if ('country' in s3_df.columns and len(s3_df) > 0)
                else pd.DataFrame(columns=['entity_id', 'business_name', 'business_address', 'country'])
            )

            print(f"\nProcessing Country: '{country}' (S1: {len(s1_country):,}, S2: {len(s2_country):,}, S3: {len(s3_country):,})")
            country_candidates, country_records = self.generate_candidates_for_country(
                s1_country, s2_country, s3_country
            )
            all_candidates.update(country_candidates)
            all_records.extend(country_records)

            if cache_dir and country_records:
                export_candidate_parquet_cache(country_records, cache_dir, country=str(country))

            print(f"  Finished '{country}' in {time.time() - t0:.2f}s.")

        # Ensure all S1 entities have an entry even if country was missing or novel
        all_s1_ids = s1_df['entity_id'].values
        for eid in all_s1_ids:
            if eid not in all_candidates:
                all_candidates[eid] = []

        print(f"\n=== Blocking complete in {time.time() - start_time:.2f}s. Total S1 entities: {len(all_candidates):,} ===")
        return all_candidates, all_records


def export_candidate_pairs_tsv(
    candidates_dict: Dict[str, List[str]],
    output_path: str,
    chunk_size: int = 50000
):
    """
    Stream candidate pairs to official submission TSV format (D-08, D-13).

    Output format:
    source1_entity_id\tcandidate_entity_ids
    S1-001\tS2-001,S3-001
    S1-002\t
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Official header (validate_submission.py CANDIDATE_HEADER)
        f.write("source1_entity_id\tcandidate_entity_ids\n")

        buffer: List[str] = []
        for s1_id, cand_list in candidates_dict.items():
            if cand_list:
                # Comma-separated sorted IDs without intra-list duplicates
                deduped_sorted = sorted(list(dict.fromkeys(cand_list)))
                buffer.append(f"{s1_id}\t{','.join(deduped_sorted)}\n")
            else:
                # Strict official singleton format: empty string after tab
                buffer.append(f"{s1_id}\t\n")

            if len(buffer) >= chunk_size:
                f.writelines(buffer)
                buffer.clear()

        if buffer:
            f.writelines(buffer)
            buffer.clear()


def export_candidate_parquet_cache(
    candidate_records: List[dict],
    cache_dir: str = "data/cache/candidates",
    country: str = "all"
):
    """
    Persist candidate pairs with metadata into partitioned Parquet files (D-14).
    """
    if not candidate_records:
        return

    out_dir = Path(cache_dir) / f"country={country}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "candidates.parquet"

    df = pd.DataFrame(candidate_records)
    # Ensure expected schema
    expected_cols = ['s1_id', 'candidate_id', 'blocking_stage', 'rank', 'cosine_sim']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df[expected_cols].to_parquet(out_file, index=False)


if __name__ == "__main__":
    # Test with synthetic cross-country data
    s1_test = pd.DataFrame([
        {'entity_id': 'S1-001', 'business_name': 'Payne Enterprises LLC', 'business_address': '3315 Fremont St, Peoria, IL', 'country': 'US'},
        {'entity_id': 'S1-002', 'business_name': 'Ss Food Private Limited', 'business_address': 'Af-684, Ghaziabad, UP', 'country': 'India'},
        {'entity_id': 'S1-003', 'business_name': 'Carrefour SAS', 'business_address': '12 Rue de la Paix, Paris', 'country': 'France'},
        {'entity_id': 'S1-004', 'business_name': 'Zzz Unique Corp', 'business_address': '999 Nowhere Way, Anchorage, AK', 'country': 'US'},
    ])

    s2_test = pd.DataFrame([
        {'entity_id': 'S2-001', 'business_name': 'Payne Enterprises', 'business_address': '3315 Fremont St, Peoria, Illinois', 'country': 'US'},
        {'entity_id': 'S2-002', 'business_name': 'Ss Food', 'business_address': 'Ghaziabad, Uttar Pradesh', 'country': 'India'},
        {'entity_id': 'S2-003', 'business_name': 'Carrefour Market', 'business_address': 'Rue de la Paix, Paris', 'country': 'France'},
        {'entity_id': 'S2-004', 'business_name': 'Indian Business', 'business_address': 'Delhi, DL', 'country': 'India'},
    ])

    s3_test = pd.DataFrame([
        {'entity_id': 'S3-001', 'business_name': 'Payne Enterprises Inc', 'business_address': 'Fremont Saint, Peoria, IL', 'country': 'US'},
        {'entity_id': 'S3-002', 'business_name': 'French Cafe', 'business_address': 'Paris, France', 'country': 'France'},
    ])

    engine = BlockingEngine(top_k=5, min_similarity=0.15)
    candidates, records = engine.block_datasets(s1_test, s2_test, s3_test)

    print("\nCandidates:")
    for s1_id, cands in candidates.items():
        print(f"  {s1_id}: {cands}")

    assert 'S2-001' in candidates['S1-001']
    assert 'S2-002' in candidates['S1-002']
    assert 'S2-003' in candidates['S1-003']
    assert candidates['S1-004'] == []  # Singleton

    # Assert 0 cross country matches
    assert 'S2-004' not in candidates['S1-001']
    assert 'S2-001' not in candidates['S1-002']

    test_out = "output/test_candidate_pairs.tsv"
    export_candidate_pairs_tsv(candidates, test_out)
    assert os.path.exists(test_out)
    with open(test_out, encoding="utf-8") as f:
        lines = f.readlines()
        assert lines[0] == "source1_entity_id\tcandidate_entity_ids\n"
        assert any(l.startswith("S1-004\t\n") for l in lines)
    if os.path.exists(test_out):
        os.remove(test_out)

    print("\nAll blocking unit tests passed successfully!")
