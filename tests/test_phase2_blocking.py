"""
Pytest unit & integration test suite for Phase 2 blocking engine (BLOCK-01 through BLOCK-04).

Verifies:
1. Country partition zero leakage across US, India, France.
2. 4-Stage Inverted Index Cascade hierarchy (both, name, addr, pref).
3. TF-IDF character (3,4)-gram sparse matching and top-k budget truncation.
4. Strict singleton emission format (empty string after tab).
5. Official submission validator compliance with exit code 0.
6. Recall ceiling and space reduction mathematical evaluation.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.blocking import (
    BlockingEngine,
    InvertedIndexCascade,
    TfidfSparseBlocker,
    export_candidate_pairs_tsv,
    export_candidate_parquet_cache,
)
from scripts.audit_blocking_recall import evaluate_recall
from scripts.verify_submission import find_validator_script


def test_country_partition_zero_leakage():
    """Verify zero cross-country match candidate leakage (D-03, BLOCK-01)."""
    s1_df = pd.DataFrame([
        {'entity_id': 'S1-US', 'business_name': 'Global Tech LLC', 'business_address': '100 Main St, NY', 'country': 'US'},
        {'entity_id': 'S1-IN', 'business_name': 'Global Tech Pvt Ltd', 'business_address': 'MG Road, Bangalore, KA', 'country': 'India'},
        {'entity_id': 'S1-FR', 'business_name': 'Global Tech SAS', 'business_address': '10 Rue de la Paix, Paris', 'country': 'France'},
    ])

    s2_df = pd.DataFrame([
        {'entity_id': 'S2-US', 'business_name': 'Global Tech', 'business_address': '100 Main St, New York', 'country': 'US'},
        {'entity_id': 'S2-IN', 'business_name': 'Global Tech', 'business_address': 'MG Road, Bangalore, Karnataka', 'country': 'India'},
        {'entity_id': 'S2-FR', 'business_name': 'Global Tech', 'business_address': '10 Rue de la Paix, Paris', 'country': 'France'},
    ])

    s3_df = pd.DataFrame([
        {'entity_id': 'S3-US', 'business_name': 'Global Tech Inc', 'business_address': '100 Main St, NY', 'country': 'US'},
        {'entity_id': 'S3-IN', 'business_name': 'Global Tech', 'business_address': 'Bangalore, India', 'country': 'India'},
        {'entity_id': 'S3-FR', 'business_name': 'Global Tech France', 'business_address': 'Paris, France', 'country': 'France'},
    ])

    engine = BlockingEngine(top_k=10, min_similarity=0.18)
    candidates, records = engine.block_datasets(s1_df, s2_df, s3_df)

    # Assert US only matches US
    assert 'S2-US' in candidates['S1-US']
    assert 'S3-US' in candidates['S1-US']
    assert 'S2-IN' not in candidates['S1-US']
    assert 'S2-FR' not in candidates['S1-US']

    # Assert India only matches India
    assert 'S2-IN' in candidates['S1-IN']
    assert 'S2-US' not in candidates['S1-IN']
    assert 'S2-FR' not in candidates['S1-IN']

    # Assert France only matches France
    assert 'S2-FR' in candidates['S1-FR']
    assert 'S2-US' not in candidates['S1-FR']
    assert 'S2-IN' not in candidates['S1-FR']


def test_inverted_index_cascade_keys():
    """Verify 4-stage exact inverted index hierarchy (D-04, BLOCK-02)."""
    target_df = pd.DataFrame([
        {'entity_id': 'S2-001', 'business_name': 'Acme Supply Corp', 'business_address': '500 Commerce Way, Dallas, TX'},
        {'entity_id': 'S2-002', 'business_name': 'Acme Logistics Inc', 'business_address': '999 Industrial Pkwy, Chicago, IL'},
        {'entity_id': 'S2-003', 'business_name': 'Beta Holdings LLC', 'business_address': '500 Commerce Way, Dallas, TX'},
    ])
    clean_names = ['acme supply', 'acme logistics', 'beta holdings']
    clean_addrs = ['500 commerce way dallas texas', '999 industrial parkway chicago illinois', '500 commerce way dallas texas']

    index = InvertedIndexCascade()
    index.build(target_df, clean_names, clean_addrs)

    # 1. Test by_both exact match
    matches_both = index.query_exact('acme supply', '500 commerce way dallas texas')
    assert any(cid == 'S2-001' and stage == 'exact_both' for cid, stage, _ in matches_both)

    # 2. Test by_name exact match
    matches_name = index.query_exact('acme supply', 'unknown address somewhere')
    assert any(cid == 'S2-001' and stage == 'exact_name' for cid, stage, _ in matches_name)

    # 3. Test by_addr exact match
    matches_addr = index.query_exact('unknown company', '500 commerce way dallas texas')
    matched_ids = [cid for cid, stage, _ in matches_addr if stage == 'exact_addr']
    assert 'S2-001' in matched_ids
    assert 'S2-003' in matched_ids

    # 4. Test by_pref prefix match
    matches_pref = index.query_exact('acme logistics global', 'other address')
    assert any(cid == 'S2-002' and stage == 'exact_pref' for cid, stage, _ in matches_pref)


def test_tfidf_sparse_matching():
    """Verify subword char (3,4)-gram TF-IDF matching and top-k budget (D-01, D-05, D-06)."""
    target_ids = np.array([f"S2-{i:03d}" for i in range(30)])
    target_corpus = [f"business enterprise number {i} road ave {i}" for i in range(30)]

    blocker = TfidfSparseBlocker(
        ngram_range=(3, 4),
        max_features=10000,
        min_df=1,
        min_similarity=0.10,
    )
    blocker.fit_target(target_ids, target_corpus)

    s1_ids = np.array(['S1-Q1'])
    s1_corpus = ["business enterprise number 0 road ave 0"]

    results = blocker.query_s1(s1_ids, s1_corpus, top_k=5)
    cands = results['S1-Q1']

    # Must be capped at top_k=5
    assert len(cands) <= 5
    # S2-000 should be the top match with high similarity
    top_cand, top_score = cands[0]
    assert top_cand == 'S2-000'
    assert top_score >= 0.90


def test_singleton_empty_candidate():
    """Verify strict singleton empty string emission (D-08, D-13, BLOCK-03)."""
    candidates = {
        'S1-MATCH': ['S2-001', 'S3-002'],
        'S1-SINGLETON': [],
    }

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".tsv") as tmp:
        tmp_path = tmp.name

    try:
        export_candidate_pairs_tsv(candidates, tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert lines[0] == "source1_entity_id\tcandidate_entity_ids\n"
        assert "S1-MATCH\tS2-001,S3-002\n" in lines
        assert "S1-SINGLETON\t\n" in lines
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_candidate_tsv_submission_validator():
    """Verify generated candidate pairs pass official validate_submission.py with exit code 0."""
    validator = find_validator_script()
    assert os.path.isfile(validator), f"Validator not found at {validator}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()

        # Create dummy test sources
        pd.DataFrame([
            {'entity_id': 'S1-001', 'business_name': 'Test A', 'business_address': 'Addr A', 'country': 'US'},
            {'entity_id': 'S1-002', 'business_name': 'Test B', 'business_address': 'Addr B', 'country': 'US'},
        ]).to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)

        pd.DataFrame([
            {'entity_id': 'S2-001', 'business_name': 'Test A', 'business_address': 'Addr A', 'country': 'US'},
        ]).to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)

        pd.DataFrame([
            {'entity_id': 'S3-001', 'business_name': 'Test B', 'business_address': 'Addr B', 'country': 'US'},
        ]).to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

        candidate_file = tmp_path / "candidate_pairs.tsv"
        candidates = {
            'S1-001': ['S2-001'],
            'S1-002': ['S3-001'],
        }
        export_candidate_pairs_tsv(candidates, str(candidate_file))

        matching_file = tmp_path / "matching_results.tsv"
        with open(matching_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-001\tS2-001\n")
            f.write("S1-002\tS3-001\n")

        # Run official validator
        cmd = [
            sys.executable,
            validator,
            "--matching", str(matching_file),
            "--candidate", str(candidate_file),
            "--test-dir", str(test_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Validator failed: {res.stdout}\n{res.stderr}"


def test_audit_blocking_recall_ceiling():
    """Verify recall ceiling math and space reduction evaluation (D-15, BLOCK-04)."""
    candidates = {
        'S1-001': {'S2-001', 'S3-001'},
        'S1-002': {'S2-002'},
        'S1-003': set(),  # missed positive
        'S1-004': set(),  # true singleton
    }
    ground_truth = {
        'S1-001': {'S2-001', 'S3-001'},
        'S1-002': {'S2-002'},
        'S1-003': {'S2-003'},
        'S1-004': set(),
    }
    s1_countries = {
        'S1-001': 'US',
        'S1-002': 'India',
        'S1-003': 'US',
        'S1-004': 'US',
    }

    audit = evaluate_recall(
        candidates=candidates,
        ground_truth=ground_truth,
        s1_countries=s1_countries,
        total_target_pool=1000,
    )

    # Total positive pairs = 2 (S1-001) + 1 (S1-002) + 1 (S1-003) = 4
    # Captured positives = 2 (S1-001) + 1 (S1-002) = 3
    # Recall = 3 / 4 = 75.0%
    assert audit['total_positive_pairs'] == 4
    assert audit['captured_positive_pairs'] == 3
    assert audit['pair_recall_ceiling_pct'] == 75.0
    assert audit['true_singletons'] == 1
    assert audit['accurate_singletons'] == 1
    assert audit['singleton_accuracy_pct'] == 100.0
    assert audit['search_space_reduction_pct'] > 99.0
