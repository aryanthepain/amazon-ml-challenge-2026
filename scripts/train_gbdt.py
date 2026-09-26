"""CLI Training Script for LightGBM Pairwise Matcher.

Extracts 24-dimensional features, builds balanced positive/hard-negative pairs,
and trains EntityMatcherLGBM with ROC-AUC early stopping.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data_loader import DataLoader
from src.dataset import build_training_pairs, build_validation_pairs
from src.model import EntityMatcherLGBM


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Pairwise Matcher")
    parser.add_argument("--candidates-path", type=str, default="output/candidate_pairs.tsv", help="Path to candidates TSV")
    parser.add_argument("--output-model", type=str, default="models/lgbm_matcher.txt", help="Path to save trained model")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Directory with train/val split TSVs")
    parser.add_argument("--neg-ratio", type=int, default=4, help="Negative sampling ratio")
    parser.add_argument("--early-stopping", type=int, default=50, help="Early stopping rounds")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional sample limit for quick execution")

    args = parser.parse_args()

    print(f"=== Amazon ML Challenge 2026: LightGBM Training Pipeline ===")
    loader = DataLoader()

    # 1. Load Sources and Ground Truth
    print("[1/5] Loading Source 1, Target Pool (S2+S3), and Ground Truth...")
    s1_df = loader.load_source(1, "train", nrows=args.sample_limit)
    s2_df = loader.load_source(2, "train", nrows=args.sample_limit)
    s3_df = loader.load_source(3, "train", nrows=args.sample_limit)
    gt_df = loader.load_ground_truth("train", nrows=args.sample_limit)
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    print(f"      Source 1: {len(s1_df):,} | Target Pool: {len(target_df):,} | Ground Truth: {len(gt_df):,}")

    # 2. Load Split IDs
    print(f"[2/5] Loading train/val split IDs from {args.splits_dir}...")
    train_ids_path = os.path.join(args.splits_dir, "train_s1_ids.tsv")
    val_ids_path = os.path.join(args.splits_dir, "val_s1_ids.tsv")

    if os.path.exists(train_ids_path):
        train_s1_ids = set(pd.read_csv(train_ids_path, sep="\t")["entity_id"].astype(str))
    else:
        all_s1 = set(s1_df["entity_id"].astype(str))
        train_s1_ids = set(list(all_s1)[:int(len(all_s1) * 0.8)])

    if os.path.exists(val_ids_path):
        val_s1_ids = set(pd.read_csv(val_ids_path, sep="\t")["entity_id"].astype(str))
    else:
        val_s1_ids = set(s1_df["entity_id"].astype(str)) - train_s1_ids

    print(f"      Train S1 IDs: {len(train_s1_ids):,} | Val S1 IDs: {len(val_s1_ids):,}")

    # 3. Load Candidates
    print(f"[3/5] Loading candidate pairs from {args.candidates_path}...")
    if not os.path.exists(args.candidates_path):
        print(f"[WARN] Candidates file {args.candidates_path} not found. Running on synthetic pairs.")
        candidates_df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_ids"])
    else:
        candidates_df = pd.read_csv(args.candidates_path, sep="\t")
    print(f"      Loaded {len(candidates_df):,} candidate rows.")

    # 4. Build Pair Datasets
    print("[4/5] Extracting 24 features and assembling training/validation pairs...")
    X_train, y_train = build_training_pairs(
        s1_df=s1_df,
        target_df=target_df,
        ground_truth_df=gt_df,
        train_s1_ids=train_s1_ids,
        candidate_pairs_df=candidates_df,
        neg_ratio=args.neg_ratio,
    )
    X_val, y_val, val_meta = build_validation_pairs(
        s1_df=s1_df,
        target_df=target_df,
        ground_truth_df=gt_df,
        val_s1_ids=val_s1_ids,
        candidate_pairs_df=candidates_df,
    )
    print(f"      X_train: {X_train.shape} | Positives: {(y_train == 1).sum():,} | Negatives: {(y_train == 0).sum():,}")
    print(f"      X_val:   {X_val.shape} | Positives: {(y_val == 1).sum():,} | Negatives: {(y_val == 0).sum():,}")

    # 5. Fit LightGBM
    print("[5/5] Training LightGBM classifier with early stopping...")
    model = EntityMatcherLGBM()
    if len(X_train) > 0 and len(X_val) > 0:
        model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=args.early_stopping,
            verbose=True,
        )
        print(f"[SUCCESS] Model training complete! Validation ROC-AUC: {model.val_auc_:.5f}")
        model.save_model(args.output_model)
        print(f"[SUCCESS] Model saved to {args.output_model}")

        # Log Top Features
        importances = model.get_feature_importances("gain")
        sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        print("\n--- Top 10 Features by Gain ---")
        for rank, (name, score) in enumerate(sorted_feats[:10], start=1):
            print(f"  {rank:2d}. {name:<25}: {score:.4f}")
    else:
        print("[WARN] Insufficient pairs to train model. Exiting.")


if __name__ == "__main__":
    main()
