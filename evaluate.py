#!/usr/bin/env python3
"""
Official Offline Evaluation Harness for Amazon ML Challenge 2026.

Computes exact Macro F0.5 score across all Source 1 entities with proper
singleton handling (score=1.0 if both empty, 0.0 if false merge) and
harmonic mean calculation for matched entities.
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, Set, Tuple
import pandas as pd


def parse_entity_mapping(filepath: str, id_col: str, list_col: str) -> Dict[str, Set[str]]:
    """
    Parse a TSV mapping file into a dictionary of {s1_id: set(target_ids)}.
    Empty values or missing values are treated as empty sets.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    mapping: Dict[str, Set[str]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return mapping
        parts = [p.strip() for p in header.rstrip("\r\n").split("\t")]
        try:
            id_idx = parts.index(id_col)
            list_idx = parts.index(list_col)
        except ValueError:
            # Fallback to column index 0 and 1
            id_idx, list_idx = 0, 1

        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            cols = line.split("\t")
            s1_id = cols[id_idx].strip()
            if not s1_id:
                continue
            if len(cols) > list_idx and cols[list_idx].strip():
                targets = {t.strip() for t in cols[list_idx].split(",") if t.strip()}
            else:
                targets = set()
            mapping[s1_id] = targets

    return mapping


def compute_entity_f05(pred_set: Set[str], gt_set: Set[str]) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F0.5 for a single Source 1 entity.
    Returns (precision, recall, f05).
    """
    # Case 1: Ground truth is a singleton (no matches)
    if len(gt_set) == 0:
        if len(pred_set) == 0:
            # Correctly predicted singleton
            return 1.0, 1.0, 1.0
        else:
            # False merge on a singleton
            return 0.0, 0.0, 0.0

    # Case 2: Ground truth has matches
    if len(pred_set) == 0:
        # Missed all matches
        return 0.0, 0.0, 0.0

    tp = len(pred_set.intersection(gt_set))
    if tp == 0:
        return 0.0, 0.0, 0.0

    prec = tp / len(pred_set)
    rec = tp / len(gt_set)

    # Beta = 0.5 -> beta^2 = 0.25
    # F0.5 = (1.25 * P * R) / (0.25 * P + R)
    denom = 0.25 * prec + rec
    f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0

    return prec, rec, f05


def evaluate(
    predictions_path: str,
    ground_truth_path: str,
    pred_id_col: str = "source1_entity_id",
    pred_list_col: str = "matched_entity_ids",
    gt_id_col: str = "source1_entity_id",
    gt_list_col: str = "matched_entity_ids",
    sample_size: int = None
) -> dict:
    """
    Evaluate predictions against ground truth and return metrics dict.
    """
    preds = parse_entity_mapping(predictions_path, pred_id_col, pred_list_col)
    gt = parse_entity_mapping(ground_truth_path, gt_id_col, gt_list_col)

    if not gt:
        raise ValueError(f"Ground truth table is empty: {ground_truth_path}")

    eval_ids = sorted(list(gt.keys()))
    if sample_size and sample_size < len(eval_ids):
        eval_ids = eval_ids[:sample_size]

    total_entities = len(eval_ids)
    prec_sum = 0.0
    rec_sum = 0.0
    f05_sum = 0.0

    gt_singletons = 0
    correct_singletons = 0
    non_singleton_entities = 0
    non_singleton_f05_sum = 0.0

    missing_in_preds = 0

    for s1_id in eval_ids:
        gt_set = gt.get(s1_id, set())
        pred_set = preds.get(s1_id, set())

        if s1_id not in preds:
            missing_in_preds += 1

        is_gt_singleton = (len(gt_set) == 0)
        if is_gt_singleton:
            gt_singletons += 1
            if len(pred_set) == 0:
                correct_singletons += 1
        else:
            non_singleton_entities += 1

        p, r, f = compute_entity_f05(pred_set, gt_set)
        prec_sum += p
        rec_sum += r
        f05_sum += f

        if not is_gt_singleton:
            non_singleton_f05_sum += f

    macro_prec = prec_sum / total_entities
    macro_rec = rec_sum / total_entities
    macro_f05 = f05_sum / total_entities
    singleton_acc = (correct_singletons / gt_singletons) if gt_singletons > 0 else 0.0
    non_singleton_macro_f05 = (non_singleton_f05_sum / non_singleton_entities) if non_singleton_entities > 0 else 0.0

    results = {
        "macro_f05": round(macro_f05, 5),
        "macro_precision": round(macro_prec, 5),
        "macro_recall": round(macro_rec, 5),
        "total_evaluated_entities": total_entities,
        "missing_in_predictions": missing_in_preds,
        "ground_truth_singletons": gt_singletons,
        "singleton_percentage": round((gt_singletons / total_entities) * 100, 2),
        "singleton_accuracy": round(singleton_acc * 100, 2),
        "non_singleton_entities": non_singleton_entities,
        "non_singleton_macro_f05": round(non_singleton_macro_f05, 5),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Macro F0.5 Evaluator")
    parser.add_argument("--predictions", "-p", required=True, help="Path to predictions TSV (matching_results.tsv)")
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth TSV")
    parser.add_argument("--sample-size", type=int, default=None, help="Evaluate on first N entities")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    results = evaluate(
        predictions_path=args.predictions,
        ground_truth_path=args.ground_truth,
        sample_size=args.sample_size
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n=======================================================")
        print("     AMAZON ML CHALLENGE 2026: EVALUATION REPORT       ")
        print("=======================================================")
        print(f" Macro F0.5 Score:          {results['macro_f05']:.5f}")
        print(f" Macro Precision:           {results['macro_precision']:.5f}")
        print(f" Macro Recall:              {results['macro_recall']:.5f}")
        print("-------------------------------------------------------")
        print(f" Total Evaluated Entities:  {results['total_evaluated_entities']}")
        print(f" Missing in Predictions:    {results['missing_in_predictions']}")
        print(f" GT Singletons:             {results['ground_truth_singletons']} ({results['singleton_percentage']}%)")
        print(f" Singleton Accuracy:        {results['singleton_accuracy']}%")
        print(f" Non-Singleton F0.5:        {results['non_singleton_macro_f05']:.5f}")
        print("=======================================================\n")


if __name__ == "__main__":
    main()
