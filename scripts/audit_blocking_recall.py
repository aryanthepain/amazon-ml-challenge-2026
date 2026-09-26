#!/usr/bin/env python3
"""
Recall Ceiling and Space Reduction Audit CLI for Cascade Blocking Engine (Phase 2, D-15, BLOCK-04).

Evaluates candidate pair sets against ground truth annotations to measure:
- Positive pair recall ceiling (Target >= 95%)
- Country-level recall breakdown (US vs India)
- Singleton handling accuracy
- Search space reduction factor (Target > 99.9%)
Emits structured audit results to output/blocking_recall_audit.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from src.data_loader import DataLoader, resolve_dataset_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit Blocking Recall Ceiling & Space Reduction Factor"
    )
    parser.add_argument(
        "--candidates",
        "-c",
        type=str,
        default="output/candidate_pairs.tsv",
        help="Path to generated candidate pairs TSV (default: output/candidate_pairs.tsv)",
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        type=str,
        default=None,
        help="Path to validation ground truth TSV (default: data/splits/val_ground_truth.tsv)",
    )
    parser.add_argument(
        "--s1-source",
        type=str,
        default=None,
        help="Path to train_source1.tsv to load country annotations for breakdown",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        type=str,
        default="output/blocking_recall_audit.json",
        help="Path to write structured audit report (default: output/blocking_recall_audit.json)",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.95,
        help="Minimum required recall ceiling (0.0 to 1.0, default: 0.95)",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Do not return exit code 1 if recall is below threshold (diagnostic mode)",
    )
    return parser.parse_args()


def load_candidate_pairs(filepath: str) -> Dict[str, Set[str]]:
    """Load candidate pairs TSV into {s1_id: set(candidate_ids)}."""
    candidates: Dict[str, Set[str]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            s1_id = parts[0].strip()
            cands_str = parts[1].strip() if len(parts) > 1 else ""
            cands_set = set(cands_str.split(",")) if cands_str else set()
            candidates[s1_id] = cands_set
    return candidates


def load_ground_truth(filepath: str) -> Dict[str, Set[str]]:
    """Load ground truth TSV into {s1_id: set(matched_ids)}."""
    gt: Dict[str, Set[str]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            s1_id = parts[0].strip()
            matches_str = parts[1].strip() if len(parts) > 1 else ""
            matches_set = set(matches_str.split(",")) if matches_str else set()
            gt[s1_id] = matches_set
    return gt


def evaluate_recall(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    s1_countries: Optional[Dict[str, str]] = None,
    total_target_pool: int = 1545672,  # Total N_s2 + N_s3 in full training set
) -> dict:
    """Compute empirical recall ceiling, space reduction, and country breakdown."""
    evaluated_s1_ids = [s1 for s1 in candidates if s1 in ground_truth]
    if not evaluated_s1_ids:
        # If candidates has S1 IDs not in provided ground truth, intersect
        evaluated_s1_ids = list(set(candidates.keys()) & set(ground_truth.keys()))

    total_evaluated_s1 = len(evaluated_s1_ids)
    if total_evaluated_s1 == 0:
        raise ValueError("Zero overlapping S1 IDs between candidates and ground truth.")

    total_positive_pairs = 0
    captured_positive_pairs = 0
    true_singletons = 0
    accurate_singletons = 0
    total_candidates_emitted = 0

    country_stats: Dict[str, Dict[str, int]] = {}

    for s1 in evaluated_s1_ids:
        true_matches = ground_truth[s1]
        cands = candidates.get(s1, set())
        total_candidates_emitted += len(cands)

        country = s1_countries.get(s1, "Unknown") if s1_countries else "All"
        if country not in country_stats:
            country_stats[country] = {
                "s1_count": 0,
                "total_positives": 0,
                "captured_positives": 0,
                "singletons": 0,
                "accurate_singletons": 0,
            }

        c_stat = country_stats[country]
        c_stat["s1_count"] += 1

        if not true_matches:
            true_singletons += 1
            c_stat["singletons"] += 1
            if len(cands) == 0:
                accurate_singletons += 1
                c_stat["accurate_singletons"] += 1
        else:
            for mid in true_matches:
                total_positive_pairs += 1
                c_stat["total_positives"] += 1
                if mid in cands:
                    captured_positive_pairs += 1
                    c_stat["captured_positives"] += 1

    overall_recall = (
        (captured_positive_pairs / total_positive_pairs * 100.0)
        if total_positive_pairs > 0
        else 100.0
    )
    singleton_accuracy = (
        (accurate_singletons / true_singletons * 100.0)
        if true_singletons > 0
        else 100.0
    )

    # Search space reduction factor
    cartesian_space = total_evaluated_s1 * total_target_pool
    space_reduction = (
        1.0 - (total_candidates_emitted / cartesian_space)
        if cartesian_space > 0
        else 1.0
    )

    # Format country breakdown
    country_breakdown = {}
    for c_name, st in country_stats.items():
        c_recall = (
            (st["captured_positives"] / st["total_positives"] * 100.0)
            if st["total_positives"] > 0
            else 100.0
        )
        c_sing_acc = (
            (st["accurate_singletons"] / st["singletons"] * 100.0)
            if st["singletons"] > 0
            else 100.0
        )
        country_breakdown[c_name] = {
            "s1_count": st["s1_count"],
            "total_positive_pairs": st["total_positives"],
            "captured_positive_pairs": st["captured_positives"],
            "pair_recall_ceiling_pct": round(c_recall, 2),
            "singleton_count": st["singletons"],
            "accurate_singletons": st["accurate_singletons"],
            "singleton_accuracy_pct": round(c_sing_acc, 2),
        }

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_s1_evaluated": total_evaluated_s1,
        "total_candidate_pairs_emitted": total_candidates_emitted,
        "avg_candidates_per_s1": round(total_candidates_emitted / total_evaluated_s1, 2),
        "total_positive_pairs": total_positive_pairs,
        "captured_positive_pairs": captured_positive_pairs,
        "pair_recall_ceiling_pct": round(overall_recall, 2),
        "true_singletons": true_singletons,
        "accurate_singletons": accurate_singletons,
        "singleton_accuracy_pct": round(singleton_accuracy, 2),
        "search_space_reduction_factor": round(space_reduction, 6),
        "search_space_reduction_pct": round(space_reduction * 100.0, 4),
        "country_breakdown": country_breakdown,
    }


def main():
    args = parse_args()

    # Resolve ground truth path
    gt_path = args.ground_truth
    if not gt_path:
        gt_candidates = [
            REPO_ROOT / "data" / "splits" / "val_ground_truth.tsv",
            REPO_ROOT / "data" / "splits" / "train_ground_truth.tsv",
        ]
        for c in gt_candidates:
            if c.is_file():
                gt_path = str(c)
                break
        if not gt_path:
            try:
                gt_path = resolve_dataset_path("train_ground_truth.tsv")
            except FileNotFoundError:
                raise FileNotFoundError("Could not find validation or training ground truth file.")

    if not os.path.isfile(args.candidates):
        raise FileNotFoundError(f"Candidates file not found: {args.candidates}")

    print("=======================================================")
    print("  AMAZON ML CHALLENGE 2026: BLOCKING RECALL AUDIT     ")
    print("=======================================================")
    print(f"Candidates File:   {args.candidates}")
    print(f"Ground Truth File: {gt_path}")
    print(f"Target Min Recall: {args.min_recall * 100:.1f}%")
    print(f"Output JSON:       {args.output_json}")
    print("-------------------------------------------------------")

    t0 = time.time()
    print("Loading candidate pairs and ground truth...")
    candidates = load_candidate_pairs(args.candidates)
    ground_truth = load_ground_truth(gt_path)

    # Optional country metadata
    s1_countries: Dict[str, str] = {}
    try:
        s1_file = args.s1_source or resolve_dataset_path("train_source1.tsv")
        print(f"Loading country annotations from {s1_file}...")
        df_s1 = pd.read_csv(s1_file, sep="\t", usecols=["entity_id", "country"], dtype=str)
        s1_countries = dict(zip(df_s1["entity_id"], df_s1["country"].fillna("Unknown")))
    except Exception as e:
        print(f"Note: Could not load country metadata ({e}); country breakdown will be aggregated.")

    print("\nAuditing recall ceiling and space reduction...")
    audit_results = evaluate_recall(
        candidates=candidates,
        ground_truth=ground_truth,
        s1_countries=s1_countries,
    )
    print(f"Audit completed in {time.time() - t0:.2f}s.")

    # Write JSON report
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"Structured audit report written to: {args.output_json}")

    # Display results
    print("\n=======================================================")
    print("  AUDIT METRICS & FINDINGS                             ")
    print("=======================================================")
    print(f"Total S1 Evaluated:     {audit_results['total_s1_evaluated']:,}")
    print(f"Total True Positives:   {audit_results['total_positive_pairs']:,}")
    print(f"Captured True Positives:{audit_results['captured_positive_pairs']:,}")
    print(f"Pair Recall Ceiling:    {audit_results['pair_recall_ceiling_pct']:.2f}% (Target: >={args.min_recall * 100:.1f}%)")
    print(f"True Singletons:        {audit_results['true_singletons']:,}")
    print(f"Singleton Accuracy:     {audit_results['singleton_accuracy_pct']:.2f}%")
    print(f"Space Reduction:        {audit_results['search_space_reduction_pct']:.4f}% (Factor: {audit_results['search_space_reduction_factor']:.6f})")

    if audit_results["country_breakdown"]:
        print("\nCountry Breakdown:")
        for c_name, st in audit_results["country_breakdown"].items():
            print(f"  [{c_name}] S1: {st['s1_count']:,} | Recall: {st['pair_recall_ceiling_pct']:.2f}% | Singletons: {st['singleton_accuracy_pct']:.2f}%")

    passed = audit_results["pair_recall_ceiling_pct"] >= (args.min_recall * 100.0)
    print("=======================================================")
    if passed:
        print(" [PASS] Blocking recall ceiling meets competition requirements! ")
        print("=======================================================")
        sys.exit(0)
    else:
        print(" [FAIL] Blocking recall ceiling is BELOW threshold!              ")
        print("=======================================================")
        if not args.no_fail:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
