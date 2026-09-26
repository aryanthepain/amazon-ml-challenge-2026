#!/usr/bin/env python3
"""
Deterministic Train / Validation Split Generator for Amazon ML Challenge 2026.

Creates a pure random 80/20 split based on Source 1 entity IDs with zero leakage.
Saves split partitions and partitioned ground truth tables.
Supports --sample-size N / --dev mode for rapid local iteration.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import DataLoader


def create_splits(
    seed: int = 42,
    val_ratio: float = 0.20,
    sample_size: int = None,
    output_dir: str = "data/splits",
    data_dir: str = None
):
    print(f"=== Amazon ML Challenge 2026: Dataset Split Generator ===")
    print(f"Random Seed: {seed} | Validation Ratio: {val_ratio}")
    if sample_size:
        print(f"Sample / Dev Mode Active: Sample Size = {sample_size}")

    loader = DataLoader(data_dir=data_dir)
    
    # 1. Load train source1 and ground truth
    print("Loading train_source1.tsv and train_ground_truth.tsv...")
    s1_df = loader.load_source(source_id=1, split="train", nrows=sample_size if sample_size else None)
    gt_df = loader.load_ground_truth(split="train")
    
    # Ensure entity IDs are sorted for cross-platform determinism before shuffle
    unique_s1_ids = sorted(list(s1_df["entity_id"].unique()))
    total_entities = len(unique_s1_ids)
    print(f"Total Source 1 entities loaded: {total_entities}")
    
    # 2. Shuffle deterministically
    rng = np.random.RandomState(seed)
    shuffled_ids = np.array(unique_s1_ids)
    rng.shuffle(shuffled_ids)
    
    # 3. Partition
    n_val = int(round(total_entities * val_ratio))
    n_train = total_entities - n_val
    
    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:]
    
    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    
    # 4. Invariant check: zero overlap
    overlap = train_id_set.intersection(val_id_set)
    assert len(overlap) == 0, f"FATAL: Source 1 entity overlap detected! Overlap size: {len(overlap)}"
    
    # 5. Filter ground truth
    train_gt = gt_df[gt_df["source1_entity_id"].isin(train_id_set)].copy()
    val_gt = gt_df[gt_df["source1_entity_id"].isin(val_id_set)].copy()
    
    # Count singletons (empty matched_entity_ids)
    train_singletons = (train_gt["matched_entity_ids"] == "").sum()
    val_singletons = (val_gt["matched_entity_ids"] == "").sum()
    
    print("\n--- Split Summary ---")
    print(f"Train Entities: {len(train_ids)} ({len(train_ids)/total_entities:.1%}) | "
          f"Ground Truth Rows: {len(train_gt)} | Singletons: {train_singletons}")
    print(f"Val Entities:   {len(val_ids)} ({len(val_ids)/total_entities:.1%}) | "
          f"Ground Truth Rows: {len(val_gt)} | Singletons: {val_singletons}")
    
    # 6. Save partitions
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame({"entity_id": train_ids}).to_csv(
        out_path / "train_s1_ids.tsv", sep="\t", index=False, header=True
    )
    pd.DataFrame({"entity_id": val_ids}).to_csv(
        out_path / "val_s1_ids.tsv", sep="\t", index=False, header=True
    )
    train_gt.to_csv(
        out_path / "train_ground_truth.tsv", sep="\t", index=False, header=True
    )
    val_gt.to_csv(
        out_path / "val_ground_truth.tsv", sep="\t", index=False, header=True
    )
    
    print(f"\nSuccessfully wrote split files to: {out_path.resolve()}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Deterministic Train/Val Dataset Split Generator")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting (default: 42)")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation fraction (default: 0.20)")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size for quick prototyping")
    parser.add_argument("--dev", action="store_true", help="Quick dev mode (uses sample-size 1000 and data/splits/dev)")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Output directory for split tables")
    parser.add_argument("--data-dir", type=str, default=None, help="Custom data directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    sample_size = args.sample_size
    if args.dev:
        if sample_size is None:
            sample_size = 1000
        if output_dir == "data/splits":
            output_dir = "data/splits/dev"

    create_splits(
        seed=args.seed,
        val_ratio=args.val_ratio,
        sample_size=sample_size,
        output_dir=output_dir,
        data_dir=args.data_dir
    )


if __name__ == "__main__":
    main()
