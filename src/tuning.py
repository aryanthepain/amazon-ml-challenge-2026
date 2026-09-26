"""Macro F0.5 Metric Computation and Grid Search Threshold Calibration.

Optimizes decision threshold theta* in [0.40, 0.90] to balance precision
2x over recall (beta=0.5) and safeguard singleton entity accuracy (>95%).
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


def compute_macro_f05(precision: float, recall: float) -> float:
    """Compute Macro F_0.5 score weighting precision 2x over recall."""
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = 0.25  # 0.5^2
    denom = beta_sq * precision + recall
    if denom <= 0.0:
        return 0.0
    return float((1.0 + beta_sq) * (precision * recall) / denom)


def evaluate_predictions_at_threshold(
    val_s1_ids: Set[str],
    ground_truth_dict: Dict[str, Optional[Set[str]]],
    candidate_preds_dict: Dict[str, List[Tuple[str, float]]],
    theta: float,
) -> Dict[str, float]:
    """Evaluate entity resolution predictions at a specific decision threshold theta.

    Singletons are represented as ground truth having empty set or None.
    Prediction logic:
        Among candidates with P(match) >= theta, assign argmax P(match).
        If all candidates < theta, predict singleton (None).
    """
    tp = 0
    fp = 0
    fn = 0
    tn_singletons = 0
    total_singletons = 0
    total_matches = 0

    for s1_id in val_s1_ids:
        true_targets = ground_truth_dict.get(s1_id)
        is_singleton = not bool(true_targets)

        if is_singleton:
            total_singletons += 1
        else:
            total_matches += 1

        # Retrieve candidates and filter by threshold
        cands = candidate_preds_dict.get(s1_id, [])
        valid_cands = [(cid, prob) for cid, prob in cands if prob >= theta]

        predicted_target: Optional[str] = None
        if valid_cands:
            # Pick argmax probability
            predicted_target = max(valid_cands, key=lambda x: x[1])[0]

        if predicted_target is not None:
            if not is_singleton and true_targets and predicted_target in true_targets:
                tp += 1
            else:
                fp += 1
                if not is_singleton:
                    fn += 1
        else:
            # Predicted singleton (None)
            if is_singleton:
                tn_singletons += 1
            else:
                fn += 1

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f05 = compute_macro_f05(precision, recall)
    singleton_acc = float(tn_singletons / total_singletons) if total_singletons > 0 else 1.0

    return {
        "theta": float(theta),
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "singleton_accuracy": singleton_acc,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn_singletons": float(tn_singletons),
        "total_singletons": float(total_singletons),
        "total_matches": float(total_matches),
    }


def grid_search_threshold(
    val_s1_ids: Set[str],
    ground_truth_dict: Dict[str, Optional[Set[str]]],
    candidate_preds_dict: Dict[str, List[Tuple[str, float]]],
    theta_min: float = 0.40,
    theta_max: float = 0.90,
    step: float = 0.02,
    min_singleton_acc: float = 0.90,
) -> Tuple[float, float, pd.DataFrame]:
    """Search for theta* in [theta_min, theta_max] maximizing Macro F0.5.

    Returns:
        best_theta: optimal decision threshold float
        best_f05: highest achieved Macro F0.5 score
        report_df: full evaluation curve DataFrame across all candidate thresholds
    """
    rows: List[Dict[str, float]] = []
    thetas = np.arange(theta_min, theta_max + 1e-5, step)

    best_theta = 0.70
    best_f05 = -1.0

    for th in thetas:
        res = evaluate_predictions_at_threshold(
            val_s1_ids=val_s1_ids,
            ground_truth_dict=ground_truth_dict,
            candidate_preds_dict=candidate_preds_dict,
            theta=float(th),
        )
        rows.append(res)

        # Optimize F0.5 with priority on singleton accuracy constraint
        if res["singleton_accuracy"] >= min_singleton_acc and res["f05"] > best_f05:
            best_f05 = res["f05"]
            best_theta = float(th)

    # Fallback to pure argmax F0.5 if constraint could not be met
    if best_f05 < 0.0 and rows:
        best_res = max(rows, key=lambda x: x["f05"])
        best_theta = best_res["theta"]
        best_f05 = best_res["f05"]

    report_df = pd.DataFrame(rows)
    return best_theta, best_f05, report_df
