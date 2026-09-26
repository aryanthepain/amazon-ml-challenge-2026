#!/usr/bin/env python3
"""
Mock Submission Generator for Amazon ML Challenge 2026.

Generates strictly schema-compliant `matching_results.tsv` and `candidate_pairs.tsv`
for test Source 1 entities. Guarantees that matches are a subset of candidates
and conforms to all official validator formatting requirements.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import DataLoader, resolve_dataset_path


def generate_dummy_submission(
    output_dir: str = "output",
    test_s1_path: str = None,
    sample_size: int = None
):
    print("=== Amazon ML Challenge 2026: Mock Submission Generator ===")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not test_s1_path:
        test_s1_path = resolve_dataset_path("test_source1.tsv")
    print(f"Reading test Source 1 IDs from: {test_s1_path}")

    # Read S1 IDs cleanly
    s1_ids = []
    with open(test_s1_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            entity_id = line.split("\t", 1)[0].strip()
            if entity_id:
                s1_ids.append(entity_id)

    if sample_size and sample_size < len(s1_ids):
        print(f"Subsampling to first {sample_size} IDs for testing...")
        s1_ids = s1_ids[:sample_size]

    total_s1 = len(s1_ids)
    print(f"Total Source 1 entities for submission: {total_s1}")

    matching_file = out_dir / "matching_results.tsv"
    candidate_file = out_dir / "candidate_pairs.tsv"

    # Write matching_results.tsv: header source1_entity_id\tmatched_entity_ids
    print(f"Writing {matching_file}...")
    with open(matching_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_ids:
            f.write(f"{s1}\t\n")

    # Write candidate_pairs.tsv: header source1_entity_id\tcandidate_entity_ids
    print(f"Writing {candidate_file}...")
    with open(candidate_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_ids:
            f.write(f"{s1}\t\n")

    print("\nMock submission generation complete.")
    print(f"  Matching Results: {matching_file.resolve()}")
    print(f"  Candidate Pairs:  {candidate_file.resolve()}")
    return matching_file, candidate_file


def main():
    parser = argparse.ArgumentParser(description="Generate Mock Submission Files")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory for generated TSVs")
    parser.add_argument("--test-s1", default=None, help="Custom path to test_source1.tsv")
    parser.add_argument("--sample-size", type=int, default=None, help="Limit number of entities")
    args = parser.parse_args()

    generate_dummy_submission(
        output_dir=args.output_dir,
        test_s1_path=args.test_s1,
        sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()
