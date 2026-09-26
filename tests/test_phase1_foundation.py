"""
Pytest Suite for Phase 1 Foundation & Evaluation Infrastructure.

Verifies:
1. Dynamic dataset path resolution and non-NaN sanitization.
2. Train/validation split disjointness and zero leakage.
3. Offline Macro F0.5 evaluation metric accuracy and edge cases.
4. Dummy submission schema compliance passing official validate_submission.py with exit code 0.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import DataLoader, resolve_dataset_path
from evaluate import compute_entity_f05, evaluate
from scripts.split_dataset import create_splits
from scripts.generate_dummy_submission import generate_dummy_submission
from scripts.verify_submission import run_verification, find_validator_script, find_test_dir


def test_data_loader_paths():
    """Verify dynamic path resolution finds dataset files and loads without NaNs."""
    loader = DataLoader()
    
    # Verify train source1 resolves and has required columns
    s1_path = loader.resolve("train_source1.tsv")
    assert os.path.exists(s1_path)
    
    df = loader.load_source(1, split="train", nrows=20)
    assert len(df) == 20
    assert "entity_id" in df.columns
    assert "business_name" in df.columns
    assert "business_address" in df.columns
    assert "country" in df.columns
    
    # Verify 0 NaNs
    assert df.isna().sum().sum() == 0


def test_split_disjointness():
    """Verify 80/20 train/validation split has zero Source 1 entity ID overlap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = create_splits(
            seed=42,
            val_ratio=0.20,
            sample_size=500,
            output_dir=tmpdir
        )
        
        train_s1 = pd.read_csv(Path(out_dir) / "train_s1_ids.tsv", sep="\t", dtype=str)
        val_s1 = pd.read_csv(Path(out_dir) / "val_s1_ids.tsv", sep="\t", dtype=str)
        
        train_set = set(train_s1["entity_id"])
        val_set = set(val_s1["entity_id"])
        
        # Zero overlap
        assert len(train_set.intersection(val_set)) == 0
        assert len(train_set) == 400
        assert len(val_set) == 100
        
        # Ground truth files present and disjoint
        train_gt = pd.read_csv(Path(out_dir) / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
        val_gt = pd.read_csv(Path(out_dir) / "val_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
        
        assert set(train_gt["source1_entity_id"]).issubset(train_set)
        assert set(val_gt["source1_entity_id"]).issubset(val_set)


def test_macro_f05_math():
    """Verify precision, recall, and F0.5 calculation for singletons and non-singletons."""
    # 1. Correct singleton: GT is empty, pred is empty -> F0.5 = 1.0
    p, r, f = compute_entity_f05(pred_set=set(), gt_set=set())
    assert (p, r, f) == (1.0, 1.0, 1.0)
    
    # 2. False merge on singleton: GT empty, pred has entity -> F0.5 = 0.0
    p, r, f = compute_entity_f05(pred_set={"S2-999"}, gt_set=set())
    assert (p, r, f) == (0.0, 0.0, 0.0)
    
    # 3. Missed matches on non-singleton: GT has entity, pred empty -> F0.5 = 0.0
    p, r, f = compute_entity_f05(pred_set=set(), gt_set={"S2-111"})
    assert (p, r, f) == (0.0, 0.0, 0.0)
    
    # 4. Perfect match: GT={'S2-1'}, pred={'S2-1'} -> F0.5 = 1.0
    p, r, f = compute_entity_f05(pred_set={"S2-1"}, gt_set={"S2-1"})
    assert (p, r, f) == (1.0, 1.0, 1.0)
    
    # 5. High precision (P=1.0, R=0.5): GT={'S2-1', 'S2-2'}, pred={'S2-1'}
    # F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 = ~0.83333
    p, r, f_high_prec = compute_entity_f05(pred_set={"S2-1"}, gt_set={"S2-1", "S2-2"})
    assert pytest.approx(f_high_prec, 0.001) == 5.0 / 6.0
    
    # 6. High recall (P=0.5, R=1.0): GT={'S2-1'}, pred={'S2-1', 'S2-2'}
    # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 = ~0.55556
    p, r, f_high_rec = compute_entity_f05(pred_set={"S2-1", "S2-2"}, gt_set={"S2-1"})
    assert pytest.approx(f_high_rec, 0.001) == 5.0 / 9.0
    
    # F0.5 weights precision more heavily than recall:
    assert f_high_prec > f_high_rec


def test_dummy_submission_exit_zero():
    """Verify that generated mock submission passes the official submission validator with exit code 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate mock files for all required test S1 entities
        matching_file, candidate_file = generate_dummy_submission(output_dir=tmpdir)
        
        assert os.path.exists(matching_file)
        assert os.path.exists(candidate_file)
        
        # Run official validator
        exit_code = run_verification(
            matching_path=str(matching_file),
            candidate_path=str(candidate_file),
            full=False  # Fast structural verification
        )
        assert exit_code == 0
