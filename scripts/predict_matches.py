"""CLI Match Prediction & Submission Generator.

Generates final predictions using the trained LightGBM matcher and calibrated
threshold theta*, producing competition-compliant TSV submissions.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data_loader import DataLoader
from src.dataset import build_entity_lookups, parse_candidate_pairs
from src.features import extract_features_batch
from src.matcher import SingleMatchResolver
from src.model import EntityMatcherLGBM


def main():
    parser = argparse.ArgumentParser(description="Predict Entity Matches for Submission")
    parser.add_argument("--model-path", type=str, default="models/lgbm_matcher.txt", help="Path to trained model")
    parser.add_argument("--candidates-path", type=str, default="output/candidate_pairs.tsv", help="Path to candidates TSV")
    parser.add_argument("--theta", type=float, default=0.70, help="Decision threshold theta*")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate")
    parser.add_argument("--output-submission", type=str, default="output/submission.tsv", help="Output path for submission TSV")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional sample limit for quick execution")

    args = parser.parse_args()

    print(f"=== Amazon ML Challenge 2026: Match Prediction Pipeline ===")
    loader = DataLoader()

    # 1. Load Model
    print(f"[1/4] Loading LightGBM model from {args.model_path} with theta={args.theta:.2f}...")
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model file {args.model_path} does not exist.")
        sys.exit(1)

    model = EntityMatcherLGBM.load_model(args.model_path)
    resolver = SingleMatchResolver(model=model, theta=args.theta)

    # 2. Load Source 1 and Target Records
    print(f"[2/4] Loading {args.split} split records...")
    s1_df = loader.load_source(1, args.split, nrows=args.sample_limit)
    s2_df = loader.load_source(2, args.split, nrows=args.sample_limit)
    s3_df = loader.load_source(3, args.split, nrows=args.sample_limit)
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    print(f"      Source 1: {len(s1_df):,} | Target Pool: {len(target_df):,}")

    # 3. Load Candidates
    print(f"[3/4] Loading candidate pairs from {args.candidates_path}...")
    if not os.path.exists(args.candidates_path):
        print(f"[ERROR] Candidates file {args.candidates_path} does not exist.")
        sys.exit(1)

    candidates_df = pd.read_csv(args.candidates_path, sep="\t")
    cand_map = parse_candidate_pairs(candidates_df)
    s1_lookup, target_lookup, _ = build_entity_lookups(s1_df, target_df)

    # 4. Score Candidate Pairs and Resolve Single Matches
    print("[4/4] Scoring candidate pairs and resolving single matches...")
    all_s1_ids = list(s1_lookup.keys())
    cand_preds: dict = {}

    for s1_id in all_s1_ids:
        s1_name, s1_addr, _ = s1_lookup[s1_id]
        cands = cand_map.get(s1_id, [])
        if not cands:
            cand_preds[s1_id] = []
            continue

        valid_cands = [c for c in cands if c[0] in target_lookup]
        if not valid_cands:
            cand_preds[s1_id] = []
            continue

        records = [
            {
                "s1_name": s1_name,
                "s1_addr": s1_addr,
                "cand_name": target_lookup[cid][0],
                "cand_addr": target_lookup[cid][1],
                "blocking_sim": sim,
                "blocking_rank": rank,
            }
            for cid, rank, sim in valid_cands
        ]
        feats = extract_features_batch(records)
        probs = model.predict_proba(feats)
        cand_preds[s1_id] = list(zip([c[0] for c in valid_cands], probs))

    submission_df = resolver.resolve_from_pair_scores(cand_preds, all_s1_ids)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_submission)), exist_ok=True)
    submission_df.to_csv(args.output_submission, sep="\t", index=False)

    num_matches = (submission_df["matched_id"] != "").sum()
    num_singletons = (submission_df["matched_id"] == "").sum()
    print(f"[SUCCESS] Submission written to {args.output_submission}")
    print(f"          Total Source 1 Records: {len(submission_df):,}")
    print(f"          Assigned Matches:       {num_matches:,} ({num_matches / len(submission_df) * 100:.2f}%)")
    print(f"          Predicted Singletons:   {num_singletons:,} ({num_singletons / len(submission_df) * 100:.2f}%)")


if __name__ == "__main__":
    main()
