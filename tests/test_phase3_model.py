"""Unit tests for Phase 3 LightGBM Modeling, Macro F0.5 Tuning, and Inference Resolver."""

import os
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_NAMES
from src.matcher import SingleMatchResolver
from src.model import EntityMatcherLGBM
from src.tuning import (
    compute_macro_f05,
    evaluate_predictions_at_threshold,
    grid_search_threshold,
)


def test_compute_macro_f05():
    """Verify Macro F0.5 formula weighting precision 2x over recall (beta=0.5)."""
    # 1. Edge cases
    assert compute_macro_f05(0.0, 0.0) == 0.0
    assert compute_macro_f05(1.0, 0.0) == 0.0
    assert compute_macro_f05(0.0, 1.0) == 0.0
    assert pytest.approx(1.0, abs=1e-5) == compute_macro_f05(1.0, 1.0)

    # 2. Precision weighted 2x over recall:
    # High precision (1.0) + low recall (0.5) -> F0.5 = 1.25 * 0.5 / (0.25 + 0.5) = 0.625 / 0.75 = 0.8333
    f_high_p = compute_macro_f05(1.0, 0.5)
    assert pytest.approx(0.8333, abs=1e-3) == f_high_p

    # Low precision (0.5) + high recall (1.0) -> F0.5 = 1.25 * 0.5 / (0.25*0.5 + 1.0) = 0.625 / 1.125 = 0.5556
    f_low_p = compute_macro_f05(0.5, 1.0)
    assert pytest.approx(0.5556, abs=1e-3) == f_low_p

    # Confirm precision is valued strictly higher than recall
    assert f_high_p > f_low_p


def test_lgbm_matcher_fit_predict_and_persistence(tmp_path):
    """Verify LightGBM model fits, predicts probabilities, and serializes/deserializes (MODEL-02)."""
    rng = np.random.default_rng(42)
    # Synthetic dataset with strong signal in feature 0 (name_jw) and feature 15 (exact street number)
    X_train = rng.uniform(0.0, 1.0, size=(200, 24)).astype(np.float32)
    y_train = ((X_train[:, 0] > 0.6) & (X_train[:, 15] > 0.5)).astype(np.int32)
    # Ensure both classes exist
    y_train[0] = 1
    y_train[1] = 0

    X_val = rng.uniform(0.0, 1.0, size=(50, 24)).astype(np.float32)
    y_val = ((X_val[:, 0] > 0.6) & (X_val[:, 15] > 0.5)).astype(np.int32)
    y_val[0] = 1
    y_val[1] = 0

    model = EntityMatcherLGBM(n_estimators=30, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=10)

    # 1. Predictions
    probs = model.predict_proba(X_val)
    assert probs.shape == (50,)
    assert 0.0 <= probs.min() <= probs.max() <= 1.0
    assert model.val_auc_ > 0.80

    # 2. Feature Importances
    importances = model.get_feature_importances("gain")
    assert len(importances) == 24
    for fname in FEATURE_NAMES:
        assert fname in importances
        assert importances[fname] >= 0.0

    # 3. Persistence
    model_file = str(tmp_path / "test_matcher.txt")
    model.save_model(model_file)
    assert os.path.exists(model_file)

    loaded_model = EntityMatcherLGBM.load_model(model_file)
    loaded_probs = loaded_model.predict_proba(X_val)
    assert np.allclose(probs, loaded_probs, atol=1e-5)


def test_evaluate_predictions_at_threshold():
    """Verify threshold evaluation correctly credits singletons and computes precision/recall (MODEL-04)."""
    val_s1_ids = {"s1_match", "s1_wrong", "s1_singleton_true", "s1_singleton_false_pos"}
    gt = {
        "s1_match": {"target_1"},
        "s1_wrong": {"target_2"},
        "s1_singleton_true": set(),
        "s1_singleton_false_pos": set(),
    }
    candidate_preds = {
        "s1_match": [("target_1", 0.90)],                  # TP
        "s1_wrong": [("target_9", 0.85)],                  # FP + FN
        "s1_singleton_true": [("target_5", 0.30)],         # True Negative Singleton
        "s1_singleton_false_pos": [("target_6", 0.75)],    # False Positive on Singleton
    }

    # At theta = 0.70:
    # s1_match: pred target_1 (TP)
    # s1_wrong: pred target_9 (FP, FN)
    # s1_singleton_true: cand 0.30 < 0.70 -> pred None (TN singleton)
    # s1_singleton_false_pos: cand 0.75 >= 0.70 -> pred target_6 (FP)
    res = evaluate_predictions_at_threshold(val_s1_ids, gt, candidate_preds, theta=0.70)
    assert res["tp"] == 1
    assert res["fp"] == 2  # s1_wrong + s1_singleton_false_pos
    assert res["fn"] == 1  # s1_wrong
    assert res["tn_singletons"] == 1
    assert res["singleton_accuracy"] == 0.5  # 1/2


def test_grid_search_threshold():
    """Verify threshold calibration identifies theta* maximizing Macro F0.5 (MODEL-03)."""
    val_s1_ids = {f"s1_{i}" for i in range(10)}
    # 5 matching entities, 5 singletons
    gt = {f"s1_{i}": {f"t_{i}"} if i < 5 else set() for i in range(10)}

    # High quality predictions: true matches have prob ~0.85, non-matches have prob ~0.50
    candidate_preds = {}
    for i in range(5):
        candidate_preds[f"s1_{i}"] = [(f"t_{i}", 0.85), (f"t_wrong_{i}", 0.40)]
    for i in range(5, 10):
        candidate_preds[f"s1_{i}"] = [(f"t_random_{i}", 0.55)]

    best_theta, best_f05, report_df = grid_search_threshold(
        val_s1_ids=val_s1_ids,
        ground_truth_dict=gt,
        candidate_preds_dict=candidate_preds,
        theta_min=0.40,
        theta_max=0.90,
        step=0.05,
        min_singleton_acc=0.80,
    )

    # Optimal threshold should be between 0.60 and 0.80 (rejects 0.55 false positives, accepts 0.85 true positives)
    assert 0.60 <= best_theta <= 0.80
    assert best_f05 > 0.80
    assert len(report_df) > 5


def test_single_match_resolver_behavior(tmp_path):
    """Verify SingleMatchResolver selects argmax above threshold and outputs submission format (D-10)."""
    # Create simple fitted model
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(100, 24)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.int32)
    y[0], y[1] = 1, 0
    model = EntityMatcherLGBM(n_estimators=10)
    model.fit(X, y, X[:20], y[:20])

    resolver = SingleMatchResolver(model=model, theta=0.70)

    # 1. Prediction with pair scores
    candidate_preds = {
        "S1-1": [("T-1", 0.65), ("T-2", 0.85)],  # T-2 exceeds 0.70
        "S1-2": [("T-3", 0.50), ("T-4", 0.60)],  # None exceed 0.70 -> Singleton
        "S1-3": [],                               # Singleton
    }
    all_s1 = ["S1-1", "S1-2", "S1-3"]
    sub_df = resolver.resolve_from_pair_scores(candidate_preds, all_s1)

    assert list(sub_df.columns) == ["source_1_id", "matched_id"]
    assert len(sub_df) == 3

    row1 = sub_df[sub_df["source_1_id"] == "S1-1"].iloc[0]
    assert row1["matched_id"] == "T-2"

    row2 = sub_df[sub_df["source_1_id"] == "S1-2"].iloc[0]
    assert row2["matched_id"] == ""

    row3 = sub_df[sub_df["source_1_id"] == "S1-3"].iloc[0]
    assert row3["matched_id"] == ""
