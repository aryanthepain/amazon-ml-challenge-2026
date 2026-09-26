#!/usr/bin/env python3
"""
Submission Verification Harness for Amazon ML Challenge 2026.

Locates the official `validate_submission.py` utility, resolves the test
dataset directory, and runs structural or full ID verification.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import resolve_dataset_path


def find_validator_script() -> str:
    candidates = [
        "student_resource/utils/validate_submission.py",
        "6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py",
        "utils/validate_submission.py",
    ]
    for c in candidates:
        full = REPO_ROOT / c
        if full.is_file():
            return str(full)
    raise FileNotFoundError("Could not find official validate_submission.py script.")


def find_test_dir() -> str:
    try:
        s1_path = resolve_dataset_path("test_source1.tsv")
        return str(Path(s1_path).parent)
    except FileNotFoundError:
        # Fallback to candidate locations
        candidates = [
            "student_resource/dataset/test",
            "6ab10eb3b23ba_student_resource/student_resource/dataset/test",
            "dataset/test"
        ]
        for c in candidates:
            full = REPO_ROOT / c
            if full.is_dir():
                return str(full)
        raise FileNotFoundError("Could not find test dataset directory containing test_source1.tsv.")


def run_verification(
    matching_path: str = "output/matching_results.tsv",
    candidate_path: str = "output/candidate_pairs.tsv",
    test_dir: str = None,
    full: bool = False
) -> int:
    validator_path = find_validator_script()
    if not test_dir:
        test_dir = find_test_dir()

    print("=== Amazon ML Challenge 2026: Submission Verification ===")
    print(f"Validator: {validator_path}")
    print(f"Test Dir:  {test_dir}")
    print(f"Matching:  {matching_path}")
    print(f"Candidate: {candidate_path}")
    print(f"Mode:      {'Full ID check (--check-ids)' if full else 'Fast structural check'}")

    cmd = [
        sys.executable,
        validator_path,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir
    ]
    if full:
        cmd.append("--check-ids")

    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Official Submission Validation")
    parser.add_argument("--matching", "-m", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", "-c", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", "-t", default=None, help="Directory containing test_source1/2/3.tsv")
    parser.add_argument("--full", action="store_true", help="Run full ID existence checks (--check-ids)")
    args = parser.parse_args()

    exit_code = run_verification(
        matching_path=args.matching,
        candidate_path=args.candidate,
        test_dir=args.test_dir,
        full=args.full
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
