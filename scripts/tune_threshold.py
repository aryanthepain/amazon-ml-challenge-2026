"""CLI Threshold Calibration Script for Macro F0.5 Optimization.

Evaluates decision thresholds theta in [0.40, 0.90] to maximize competition
Macro F0.5 while ensuring singleton accuracy >= 95%.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data_loader import DataLoader
from src.dataset import build_validation_pairs, parse_ground_truth
from src.model import EntityMatcherLGBM
from src.tuning import grid_search_threshold


def main():
    parser = argparse.ArgumentParser(description="Tune Decision Threshold for Macro F0.5")
    parser.add_argument("--model-path", type=str, default="models/lgbm_matcher.txt", help="Path to trained model")
    parser.add_argument("--candidates-path", type=str, default="output/candidate_pairs.tsv", help="Path to candidates TSV")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Directory with train/val split TSVs")
    parser.add_argument("--output-report", type=str, default="output/threshold_tuning_report.csv", help="Path to save tuning curve CSV")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional sample limit for quick execution")

    args = parser.parse_args()

    print(f"=== Amazon ML Challenge 2026: Macro F0.5 Threshold Calibration ===")
    loader = DataLoader()

    # 1. Load Sources and Ground Truth
    print("[1/4] Loading validation split and ground truth...")
    s1_df = loader.load_source(1, "train", nrows=args.sample_limit)
    s2_df = loader.load_source(2, "train", nrows=args.sample_limit)
    s3_df = loader.load_source(3, "train", nrows=args.sample_limit)
    gt_df = loader.load_ground_truth("train", nrows=args.sample_limit)
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)

    val_ids_path = os.path.join(args.splits_dir, "val_s1_ids.tsv")
    if os.path.exists(val_ids_path):
        val_s1_ids = set(pd.read_csv(val_ids_path, sep="\t")["entity_id"].astype(str))
    else:
        all_s1 = set(s1_df["entity_id"].astype(str))
        val_s1_ids = set(list(all_s1)[int(len(all_s1) * 0.8):])
    print(f"      Validation S1 entities: {len(val_s1_ids):,}")

    # 2. Load Model and Candidates
    print(f"[2/4] Loading trained model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model file {args.model_path} does not exist. Train model first.")
        return

    model = EntityMatcherLGBM.load_model(args.model_path)
    candidates_df = pd.read_csv(args.candidates_path, sep="\t") if os.path.exists(args.candidates_path) else pd.DataFrame()

    # 3. Predict Validation Probabilities
    print("[3/4] Extracting validation features and computing match probabilities...")
    X_val, y_val, val_meta = build_validation_pairs(
        s1_df=s1_df,
        target_df=target_df,
        ground_truth_df=gt_df,
        val_s1_ids=val_s1_ids,
        candidate_pairs_df=candidates_df,
    )

    if len(X_val) == 0:
        print("[WARN] No validation candidate pairs found.")
        return

    probs = model.predict_proba(X_val)
    val_meta["pred_prob"] = probs

    # Build candidate preds dict: s1_id -> [(cand_id, prob)]
    cand_preds_dict = {}
    for s1_id, grp in val_meta.groupby("source1_entity_id"):
        cand_preds_dict[s1_id] = list(zip(grp["candidate_id"], grp["pred_prob"]))

    gt_dict = parse_ground_truth(gt_df)

    # 4. Grid Search Threshold
    print("[4/4] Searching threshold grid theta in [0.40, 0.90] with step 0.02...")
    best_theta, best_f05, report_df = grid_search_threshold(
        val_s1_ids=val_s1_ids,
        ground_truth_dict=gt_dict,
        candidate_preds_dict=cand_preds_dict,
        theta_min=0.40,
        theta_max=0.90,
        step=0.02,
        min_singleton_acc=0.95,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
    report_df.to_csv(args.output_report, index=False)
    print(f"[SUCCESS] Tuning report saved to {args.output_report}")
    print(f"\n=======================================================")
    print(f" [RESULT] Optimal Threshold θ*: {best_theta:.2f}")
    print(f"          Max Macro F0.5 Score: {best_f05:.5f}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
