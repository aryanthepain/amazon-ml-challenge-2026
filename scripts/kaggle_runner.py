#!/usr/bin/env python3
"""
Automated Kaggle CLI Kernel Dispatcher & Artifact Synchronizer.

Provides push, status, logs, pull, and run-smoke commands for executing
remote GPU workloads on Kaggle and pulling results back to the local repository.
"""

import os
import sys
import shutil
import time
import json
import argparse
import subprocess
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_kaggle_env import check_kaggle_credentials, run_doctor


KERNEL_NAME = "amazon-ml-challenge-2026-smoke"


def get_kernel_slug(username: str = None) -> str:
    if not username:
        cred = check_kaggle_credentials()
        username = cred.get("username") or "aryanguptaji"
    return f"{username}/{KERNEL_NAME}"


def prepare_bundle(kaggle_dir: Path) -> None:
    """Bundle local src/ package into kaggle execution folder."""
    src_dir = REPO_ROOT / "src"
    target_src = kaggle_dir / "src"
    if src_dir.is_dir():
        if target_src.exists():
            shutil.rmtree(target_src)
        shutil.copytree(src_dir, target_src)
        print(f"[BUNDLE] Synchronized {src_dir} -> {target_src}")


def push_kernel(kaggle_dir: Path = None, dry_run: bool = False) -> int:
    """Prepare bundle and dispatch kernel via kaggle CLI."""
    if not kaggle_dir:
        kaggle_dir = REPO_ROOT / "kaggle"

    print("=== Amazon ML Challenge 2026: Push Kaggle Kernel ===")
    prepare_bundle(kaggle_dir)

    cmd = ["kaggle", "kernels", "push", "-p", str(kaggle_dir)]
    print(f"[CMD] Executing: {' '.join(cmd)}")
    if dry_run:
        print("[DRY-RUN] Push skipped.")
        return 0

    res = subprocess.run(cmd)
    return res.returncode


def get_kernel_status(slug: str = None) -> str:
    """Query execution status of remote kernel."""
    if not slug:
        slug = get_kernel_slug()
    cmd = ["kaggle", "kernels", "status", slug]
    print(f"[CMD] Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    status_str = res.stdout.strip()
    print(status_str)
    return status_str


def pull_outputs(slug: str = None, output_dir: str = "output/kaggle", dry_run: bool = False) -> int:
    """Download output artifacts from Kaggle kernel and validate integrity."""
    if not slug:
        slug = get_kernel_slug()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = ["kaggle", "kernels", "output", slug, "-p", str(out_path)]
    print(f"[CMD] Executing: {' '.join(cmd)}")
    if dry_run:
        print("[DRY-RUN] Download skipped.")
        return 0

    res = subprocess.run(cmd)
    if res.returncode != 0:
        return res.returncode

    # Validate downloaded TSV files (STRIDE T-02-02)
    print("\n[VALIDATE] Validating downloaded artifacts...")
    for tsv_file in out_path.glob("*.tsv"):
        size = tsv_file.stat().st_size
        print(f"  Artifact: {tsv_file.name} ({size} bytes)")
        if size == 0:
            print(f"  [WARN] Downloaded file {tsv_file.name} is empty.")
    return 0


def run_smoke(kaggle_dir: Path = None, timeout_seconds: int = 600) -> int:
    """Run complete pre-flight check, push smoke kernel, and monitor."""
    print("=== Step 1: Pre-Flight Kaggle Environment Verification ===")
    doctor_code = run_doctor(check_only=False)
    if doctor_code != 0:
        print("[FAIL] Pre-flight doctor failed. Fix credentials before dispatching.")
        return doctor_code

    print("\n=== Step 2: Dispatch Smoke Test Kernel ===")
    push_code = push_kernel(kaggle_dir=kaggle_dir)
    if push_code != 0:
        print("[FAIL] Kernel push failed.")
        return push_code

    print("\n=== Step 3: Monitor Execution Status ===")
    slug = get_kernel_slug()
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        status = get_kernel_status(slug)
        if "complete" in status.lower():
            print("[SUCCESS] Kernel execution completed successfully.")
            print("\n=== Step 4: Synchronize Outputs ===")
            return pull_outputs(slug)
        elif "error" in status.lower() or "failed" in status.lower():
            print(f"[FAIL] Kernel reported error state: {status}")
            return 1
        
        print("Waiting 15 seconds for next status check (Ctrl+C to abort monitoring)...")
        time.sleep(15)

    print("[TIMEOUT] Monitoring timed out. Check Kaggle web UI directly.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Kaggle Remote Kernel Runner & Monitor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # push
    p_push = subparsers.add_parser("push", help="Bundle modules and push kernel to Kaggle")
    p_push.add_argument("--dry-run", action="store_true", help="Prepare bundle without pushing")
    p_push.add_argument("--dir", default=None, help="Custom kaggle directory")

    # status
    p_status = subparsers.add_parser("status", help="Check remote kernel status")
    p_status.add_argument("--slug", default=None, help="Kernel slug (default: username/amazon-ml-challenge-2026-smoke)")

    # pull
    p_pull = subparsers.add_parser("pull", help="Download output files from kernel")
    p_pull.add_argument("--slug", default=None, help="Kernel slug")
    p_pull.add_argument("--output-dir", "-o", default="output/kaggle", help="Local download directory")
    p_pull.add_argument("--dry-run", action="store_true")

    # run-smoke
    p_smoke = subparsers.add_parser("run-smoke", help="Verify, push smoke test, and monitor")
    p_smoke.add_argument("--timeout", type=int, default=600, help="Polling timeout in seconds")

    args = parser.parse_args()

    if args.command == "push":
        kaggle_dir = Path(args.dir) if args.dir else REPO_ROOT / "kaggle"
        sys.exit(push_kernel(kaggle_dir=kaggle_dir, dry_run=args.dry_run))
    elif args.command == "status":
        get_kernel_status(args.slug)
    elif args.command == "pull":
        sys.exit(pull_outputs(slug=args.slug, output_dir=args.output_dir, dry_run=args.dry_run))
    elif args.command == "run-smoke":
        sys.exit(run_smoke(timeout_seconds=args.timeout))


if __name__ == "__main__":
    main()
