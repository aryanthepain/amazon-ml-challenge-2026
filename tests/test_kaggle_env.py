"""
Unit Test Suite for Kaggle Environment Automation & Configuration.

Verifies:
1. Valid JSON schema, GPU enablement, and dataset linkages in kernel-metadata.json.
2. Automated bundling and packaging logic in kaggle_runner.py.
3. Pre-flight doctor credentials checking and string masking.
4. Dataset metadata generator in kaggle_dataset.py.
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_kaggle_env import mask_string, check_kaggle_credentials, run_doctor
from scripts.kaggle_dataset import prepare_dataset_metadata, DATASET_SLUG
from scripts.kaggle_runner import get_kernel_slug, prepare_bundle, push_kernel


def test_kernel_metadata_validity():
    """Verify kernel-metadata.json exists, has valid syntax, and requests GPU compute."""
    meta_path = REPO_ROOT / "kaggle" / "kernel-metadata.json"
    assert meta_path.is_file(), f"Missing metadata file: {meta_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "id" in data
    assert "title" in data
    assert data.get("language") == "python"
    assert data.get("kernel_type") == "script"
    
    # GPU must be enabled (either string "true" or boolean True)
    assert data.get("enable_gpu") in [True, "true"]

    # Dataset sources linked
    assert "dataset_sources" in data
    assert isinstance(data["dataset_sources"], list)
    assert len(data["dataset_sources"]) > 0


def test_string_masking():
    """Verify credentials masking prevents secret leakage in logs."""
    assert mask_string("secrettoken", 3) == "sec********"
    assert mask_string("abc", 3) == "***"
    assert mask_string("", 3) == "Not Set"
    assert mask_string(None, 3) == "Not Set"


def test_kaggle_credentials_checker():
    """Verify credentials checker returns structured dictionary."""
    cred = check_kaggle_credentials()
    assert isinstance(cred, dict)
    assert "has_credentials_file" in cred
    assert "has_env_credentials" in cred
    assert "username" in cred
    assert "masked_username" in cred


def test_dataset_metadata_generator():
    """Verify dataset packager creates valid metadata file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_file = prepare_dataset_metadata(tmpdir, title="test-dataset")
        assert os.path.exists(meta_file)

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data.get("title") == "test-dataset"
        assert DATASET_SLUG in data.get("id")
        assert "licenses" in data


def test_kaggle_runner_bundle():
    """Verify runner packaging synchronizes src directory into target folder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_dir = Path(tmpdir)
        prepare_bundle(target_dir)

        # src/ should now exist in target
        assert (target_dir / "src").is_dir()
        assert (target_dir / "src" / "data_loader.py").is_file()


def test_kaggle_runner_dry_run_push():
    """Verify push_kernel handles dry_run flag without calling external CLI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_dir = Path(tmpdir)
        # Create dummy kernel-metadata.json and entrypoint
        with open(target_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
            json.dump({"id": "test/test", "code_file": "kaggle_entrypoint.py"}, f)
        with open(target_dir / "kaggle_entrypoint.py", "w", encoding="utf-8") as f:
            f.write("print('hello')\n")

        ret = push_kernel(kaggle_dir=target_dir, dry_run=True)
        assert ret == 0


def test_doctor_check_only_mode():
    """Verify pre-flight doctor check-only returns exit code 0 when credentials exist."""
    ret = run_doctor(check_only=True, json_mode=True)
    assert ret == 0
