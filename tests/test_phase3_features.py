"""Unit tests for Phase 3 24-dimensional feature extraction and dataset builder."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_NAMES,
    char_3gram_jaccard,
    extract_24_features,
    extract_address_features,
    extract_composite_features,
    extract_features_batch,
    extract_name_features,
    extract_numeric_features,
)
from src.dataset import (
    build_training_pairs,
    build_validation_pairs,
    parse_candidate_pairs,
    parse_ground_truth,
)


def test_feature_names_count():
    """Assert exactly 24 features defined."""
    assert len(FEATURE_NAMES) == 24
    assert "has_number_conflict" in FEATURE_NAMES
    assert "has_exact_number_match" in FEATURE_NAMES
    assert "blocking_sim" in FEATURE_NAMES


def test_extract_name_features():
    """Verify name features return 5 bounded floats."""
    feats = extract_name_features("Amazon Web Services Inc", "Amazon Web Services LLC")
    assert len(feats) == 5
    for val in feats:
        assert 0.0 <= val <= 1.0
    assert feats[0] > 0.85  # Jaro-Winkler high for similar names

    # Empty/None safety
    empty_feats = extract_name_features(None, "")
    assert empty_feats == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_extract_address_features():
    """Verify address features return 4 bounded floats."""
    feats = extract_address_features("410 Terry Ave N, Seattle, WA", "410 Terry Avenue North, Seattle, WA")
    assert len(feats) == 4
    for val in feats:
        assert 0.0 <= val <= 1.0
    assert feats[0] > 0.80

    empty_feats = extract_address_features("", None)
    assert empty_feats == [0.0, 0.0, 0.0, 0.0]


def test_char_3gram_jaccard():
    """Verify subword char 3-gram calculation."""
    assert char_3gram_jaccard("apple", "apples") > 0.5
    assert char_3gram_jaccard("microsoft", "micro hard") > 0.0
    assert char_3gram_jaccard("", "apple") == 0.0
    assert char_3gram_jaccard(None, None) == 0.0


def test_street_number_conflict_detection():
    """Verify street number match and conflict indicator flags (FEAT-03)."""
    # 1. Clear conflict: 102 vs 108 Main St
    num_jaccard, exact_match, conflict = extract_numeric_features("102 Main Street", "108 Main Street")
    assert conflict == 1.0
    assert exact_match == 0.0

    # 2. Exact match: 102 vs 102 Main St
    num_jaccard, exact_match, conflict = extract_numeric_features("102 Main Street", "102 Main Street")
    assert conflict == 0.0
    assert exact_match == 1.0
    assert num_jaccard == 1.0

    # 3. Missing numbers in one or both
    num_jaccard, exact_match, conflict = extract_numeric_features("Main Street", "102 Main Street")
    assert conflict == 0.0
    assert exact_match == 0.0


def test_extract_24_features_full():
    """Verify full 24-dim feature vector schema and bounds."""
    vec = extract_24_features(
        s1_name="Target Store",
        s1_addr="100 Broadway, New York, NY",
        cand_name="Target",
        cand_addr="200 Broadway, New York, NY",
        blocking_sim=0.75,
        blocking_rank=3,
    )
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (24,)
    assert vec.dtype == np.float32
    assert 0.0 <= vec.min() <= vec.max() <= 1.0
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()

    # Index 15 is has_exact_number_match (should be 0.0 for 100 vs 200)
    assert vec[15] == 0.0
    # Index 16 is has_number_conflict (should be 1.0 for 100 vs 200)
    assert vec[16] == 1.0
    # Index 17, 18, 19: address indicators
    assert vec[17] == 1.0
    assert vec[18] == 1.0
    assert vec[19] == 1.0
    # Index 22: blocking_sim
    assert vec[22] == pytest.approx(0.75, abs=1e-4)


def test_missing_address_indicator():
    """Verify missing address doesn't crash and flags indicators properly (FEAT-04)."""
    vec = extract_24_features("Acme Corp", "", "Acme Corporation", "123 Industrial Way")
    assert vec.shape == (24,)
    assert vec[17] == 0.0  # s1_has_addr
    assert vec[18] == 1.0  # cand_has_addr
    assert vec[19] == 0.0  # both_have_addr
    assert vec[0] > 0.8    # name_jw is still high


def test_batch_feature_extractor():
    """Verify batch extraction across multiple pairs."""
    records = [
        {"s1_name": "A", "s1_addr": "1 St", "cand_name": "A", "cand_addr": "1 St", "blocking_sim": 0.9, "blocking_rank": 1},
        {"s1_name": "B", "s1_addr": "2 St", "cand_name": "C", "cand_addr": "3 St", "blocking_sim": 0.2, "blocking_rank": 5},
    ]
    batch = extract_features_batch(records)
    assert batch.shape == (2, 24)
    assert not np.isnan(batch).any()


def test_dataset_pair_builder_and_negative_mining():
    """Verify training and validation pair dataset builder (MODEL-01)."""
    s1_df = pd.DataFrame([
        {"entity_id": "S1-1", "name": "Google", "address": "1600 Amphitheatre Pkwy", "country": "US"},
        {"entity_id": "S1-2", "name": "Apple", "address": "1 Infinite Loop", "country": "US"},
        {"entity_id": "S1-3", "name": "Microsoft", "address": "1 Microsoft Way", "country": "US"},
    ])
    target_df = pd.DataFrame([
        {"entity_id": "T-1", "name": "Google LLC", "address": "1600 Amphitheatre Parkway", "country": "US"},
        {"entity_id": "T-2", "name": "Apple Inc", "address": "1 Apple Park Way", "country": "US"},
        {"entity_id": "T-3", "name": "Microsoft Corp", "address": "1 Microsoft Way", "country": "US"},
        {"entity_id": "T-4", "name": "Alphabet Inc", "address": "1600 Amphitheatre Pkwy", "country": "US"},
        {"entity_id": "T-5", "name": "Fruit Store", "address": "2 Infinite Loop", "country": "US"},
        {"entity_id": "T-6", "name": "Windows PC", "address": "9 Redmond Way", "country": "US"},
        {"entity_id": "T-7", "name": "General Tech", "address": "100 Main St", "country": "US"},
    ])
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "T-1"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": "T-2"},
        {"source1_entity_id": "S1-3", "matched_entity_ids": "T-3"},
    ])
    candidates_df = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_ids": "T-1,T-4,T-7"},
        {"source1_entity_id": "S1-2", "candidate_entity_ids": "T-2,T-5,T-7"},
        {"source1_entity_id": "S1-3", "candidate_entity_ids": "T-3,T-6,T-7"},
    ])

    train_ids = {"S1-1", "S1-2"}
    val_ids = {"S1-3"}

    # Train pairs with 1:2 neg ratio for small test
    X_train, y_train = build_training_pairs(
        s1_df=s1_df,
        target_df=target_df,
        ground_truth_df=gt_df,
        train_s1_ids=train_ids,
        candidate_pairs_df=candidates_df,
        neg_ratio=2,
    )
    assert X_train.shape[1] == 24
    assert len(y_train) == 6  # 2 positives * (1 pos + 2 negs) = 6
    assert (y_train == 1).sum() == 2
    assert (y_train == 0).sum() == 4

    # Validation pairs without downsampling
    X_val, y_val, val_meta = build_validation_pairs(
        s1_df=s1_df,
        target_df=target_df,
        ground_truth_df=gt_df,
        val_s1_ids=val_ids,
        candidate_pairs_df=candidates_df,
    )
    assert X_val.shape[1] == 24
    assert len(val_meta) == 3  # S1-3 has 3 candidates
    assert (y_val == 1).sum() == 1  # T-3 is positive
    assert (y_val == 0).sum() == 2
