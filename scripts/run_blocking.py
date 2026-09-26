#!/usr/bin/env python3
"""
CLI Runner for Scalable Blocking and Candidate Generation (Phase 2).

Loads datasets using DataLoader, filters Source 1 by split (train, val, test, dev),
executes the hybrid cascade BlockingEngine, and outputs official candidate TSVs
and partitioned Parquet candidate caches.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from src.blocking import BlockingEngine, export_candidate_pairs_tsv, export_candidate_parquet_cache
from src.data_loader import DataLoader, load_tsv, resolve_dataset_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Country-Partitioned Cascade Blocking on Amazon ML Challenge Datasets"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "dev"],
        default="val",
        help="Dataset split to block for S1 (train, val, test, or small dev slice; default: val)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit number of S1 entities to process (useful for rapid dev benchmarks)",
    )
    parser.add_argument(
        "--target-sample-size",
        type=int,
        default=None,
        help="Limit number of target entities (S2+S3) for rapid dev testing (default: None, full target pool)",
    )
    parser.add_argument(
        "--output-tsv",
        type=str,
        default="output/candidate_pairs.tsv",
        help="Path for exported candidate pairs TSV (default: output/candidate_pairs.tsv)",
    )
    parser.add_argument(
        "--cache-parquet",
        type=str,
        default=None,
        help="Directory to save partitioned Parquet cache (e.g. data/cache/candidates)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Custom root directory for dataset TSVs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Maximum candidates per Source 1 entity (default: 25)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.18,
        help="Minimum TF-IDF cosine similarity threshold (default: 0.18)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Base batch size for sparse dot product (default: 1000)",
    )
    return parser.parse_args()


def load_split_s1_ids(split: str) -> set:
    """Load pre-computed split IDs from data/splits/ if available."""
    split_file = REPO_ROOT / "data" / "splits" / f"{split}_s1_ids.tsv"
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file {split_file} not found. Run scripts/split_dataset.py first.")
    
    df = pd.read_csv(split_file, sep="\t", dtype=str)
    return set(df["entity_id"].dropna().values)


def main():
    args = parse_args()
    loader = DataLoader(data_dir=args.data_dir)

    print("=======================================================", flush=True)
    print("  AMAZON ML CHALLENGE 2026: CASCADE BLOCKING RUNNER   ", flush=True)
    print("=======================================================", flush=True)
    print(f"Split:          {args.split}", flush=True)
    print(f"Sample Size:    {args.sample_size or 'Full Split'}", flush=True)
    print(f"Target Subsample:{args.target_sample_size or 'Full Targets'}", flush=True)
    print(f"Top-K Budget:   {args.top_k}", flush=True)
    print(f"Min Similarity: {args.min_similarity}", flush=True)
    print(f"Output TSV:     {args.output_tsv}", flush=True)
    print(f"Parquet Cache:  {args.cache_parquet or 'Disabled'}", flush=True)
    print("-------------------------------------------------------", flush=True)

    is_test = args.split == "test"
    dataset_split = "test" if is_test else "train"

    # 1. Load Source 2 and Source 3 target pool
    print(f"\n[1/3] Loading target pools (Source 2 and Source 3, split={dataset_split})...", flush=True)
    t0 = time.time()
    target_sub = args.target_sample_size
    if args.split == "dev" and target_sub is None:
        target_sub = 5000

    s2_df = loader.load_source(2, split=dataset_split, nrows=target_sub)
    s3_df = loader.load_source(3, split=dataset_split, nrows=target_sub)
    print(f"Loaded {len(s2_df):,} S2 and {len(s3_df):,} S3 entities in {time.time() - t0:.2f}s.", flush=True)

    # 2. Load Source 1 query entities
    print(f"\n[2/3] Loading Source 1 queries (split={dataset_split})...", flush=True)
    t0 = time.time()
    s1_df = loader.load_source(1, split=dataset_split)

    if args.split in ("train", "val"):
        print(f"Filtering S1 by {args.split} split entity IDs...", flush=True)
        split_ids = load_split_s1_ids(args.split)
        s1_df = s1_df[s1_df["entity_id"].isin(split_ids)].reset_index(drop=True)
        print(f"Filtered to {len(s1_df):,} S1 entities for {args.split} split.", flush=True)
    elif args.split == "dev":
        dev_size = args.sample_size or 500
        print(f"Creating dev slice of {dev_size:,} S1 entities...", flush=True)
        s1_df = s1_df.head(dev_size).reset_index(drop=True)

    if args.sample_size is not None and args.split != "dev":
        print(f"Subsampling S1 to first {args.sample_size:,} entities...", flush=True)
        s1_df = s1_df.head(args.sample_size).reset_index(drop=True)

    print(f"Total S1 entities to process: {len(s1_df):,}", flush=True)

    # 3. Execute Blocking Engine
    print("\n[3/3] Initializing and running Blocking Engine...", flush=True)
    engine = BlockingEngine(
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        batch_size=args.batch_size,
    )

    t_block = time.time()
    candidates, records = engine.block_datasets(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        cache_dir=args.cache_parquet
    )
    block_duration = time.time() - t_block

    # 4. Export Candidate TSV
    print(f"\nExporting candidate pairs to {args.output_tsv}...", flush=True)
    t_exp = time.time()
    export_candidate_pairs_tsv(candidates, args.output_tsv)
    print(f"Candidate TSV exported in {time.time() - t_exp:.2f}s.", flush=True)

    # Compute quick stats
    total_s1 = len(candidates)
    total_pairs = sum(len(c) for c in candidates.values())
    singletons = sum(1 for c in candidates.values() if len(c) == 0)
    avg_cands = total_pairs / total_s1 if total_s1 > 0 else 0

    print("\n=======================================================", flush=True)
    print("  BLOCKING RUN COMPLETE                                ", flush=True)
    print("=======================================================", flush=True)
    print(f"Total S1 Processed:   {total_s1:,}", flush=True)
    print(f"Total Candidate Pairs:{total_pairs:,}", flush=True)
    print(f"Singletons (0 cands): {singletons:,} ({singletons / total_s1 * 100:.1f}%)", flush=True)
    print(f"Avg Candidates / S1:  {avg_cands:.2f}", flush=True)
    print(f"Blocking Wall Time:   {block_duration:.2f}s ({total_s1 / max(0.001, block_duration):.0f} S1/s)", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    main()
