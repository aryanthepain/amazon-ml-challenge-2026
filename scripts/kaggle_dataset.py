#!/usr/bin/env python3
"""
Kaggle Dataset Packager & Sync Utility for Amazon ML Challenge 2026.

Prepares dataset-metadata.json and synchronizes dataset files to Kaggle
via the Kaggle CLI (datasets create / datasets version).
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_kaggle_env import check_kaggle_credentials


DATASET_SLUG = "amazon-ml-challenge-2026-dataset"


def prepare_dataset_metadata(target_dir: str, title: str = "amazon-ml-challenge-2026-dataset") -> str:
    """Generate or update dataset-metadata.json in target directory."""
    p = Path(target_dir)
    p.mkdir(parents=True, exist_ok=True)
    meta_path = p / "dataset-metadata.json"

    cred = check_kaggle_credentials()
    username = cred.get("username") or "aryanguptaji"

    metadata = {
        "title": title,
        "id": f"{username}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}]
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[METADATA] Generated dataset metadata at: {meta_path.resolve()}")
    return str(meta_path)


def resolve_local_dataset_root() -> Path:
    """Find the root directory containing train and test dataset folders."""
    candidates = [
        REPO_ROOT / "student_resource" / "dataset",
        REPO_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset",
        REPO_ROOT / "dataset"
    ]
    for c in candidates:
        if (c / "train").is_dir() and (c / "test").is_dir():
            return c
    raise FileNotFoundError("Could not find dataset folder containing both train and test subfolders.")


def main():
    parser = argparse.ArgumentParser(description="Kaggle Dataset Management Utility")
    parser.add_argument("--action", choices=["metadata", "create", "version"], default="metadata",
                        help="Action to perform: metadata, create, or version")
    parser.add_argument("--dir", default=None, help="Directory containing dataset files")
    parser.add_argument("--message", "-m", default="Initial upload of competition dataset",
                        help="Version message (used with --action version)")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing Kaggle CLI")
    args = parser.parse_args()

    target_dir = Path(args.dir) if args.dir else resolve_local_dataset_root()
    print(f"Target Dataset Directory: {target_dir.resolve()}")

    meta_file = prepare_dataset_metadata(str(target_dir))

    if args.action == "metadata":
        print("[DONE] Metadata prepared successfully.")
        return 0

    cmd = ["kaggle", "datasets"]
    if args.action == "create":
        cmd.extend(["create", "-p", str(target_dir), "-u"])
    elif args.action == "version":
        cmd.extend(["version", "-p", str(target_dir), "-m", args.message])

    print(f"[CMD] Executing: {' '.join(cmd)}")
    if args.dry_run:
        print("[DRY-RUN] Command skipped.")
        return 0

    res = subprocess.run(cmd)
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
