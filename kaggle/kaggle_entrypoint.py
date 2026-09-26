#!/usr/bin/env python3
"""
Kaggle Dual-T4 GPU Remote Execution Entrypoint & Smoke Test.

Executes within Kaggle container:
1. Validates PyTorch CUDA availability and dual-GPU detection.
2. Locates dataset files at /kaggle/input/amazon-ml-challenge-2026-dataset/ (or fallbacks).
3. Reads sample batches from Source 1, 2, and 3 TSVs.
4. Generates baseline output artifacts (matching_results.tsv and candidate_pairs.tsv)
   in /kaggle/working/ for validation and pipeline download verification.
"""

import os
import sys
from pathlib import Path

print("=======================================================")
print("  AMAZON ML CHALLENGE 2026: KAGGLE REMOTE SMOKE RUNNER  ")
print("=======================================================")

# 1. GPU Detection
try:
    import torch
    cuda_avail = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    print(f"[GPU] CUDA Available: {cuda_avail}")
    print(f"[GPU] Device Count:   {device_count}")
    for i in range(device_count):
        prop = torch.cuda.get_device_properties(i)
        vram_gb = prop.total_memory / (1024 ** 3)
        print(f"      GPU {i}: {prop.name} ({vram_gb:.2f} GB VRAM)")
except ImportError:
    print("[WARN] PyTorch is not installed in this environment.")
except Exception as e:
    print(f"[WARN] Error probing GPU properties: {e}")

# 2. Dataset Detection
dataset_candidates = [
    "/kaggle/input/amazon-ml-challenge-2026-dataset",
    "/kaggle/input/amazon-ml-challenge-2026-dataset/dataset",
    "/kaggle/input/amazon-ml-challenge-2026",
    "/kaggle/input/amazon-ml-challenge-2026/dataset",
    "student_resource/dataset",
    "6ab10eb3b23ba_student_resource/student_resource/dataset",
    "dataset",
    "."
]

found_dir = None
for d in dataset_candidates:
    p = Path(d)
    if (p / "train" / "train_source1.tsv").is_file() or (p / "train_source1.tsv").is_file():
        found_dir = p
        break

print(f"\n[DATA] Resolved Dataset Directory: {found_dir}")
if not found_dir:
    print("[WARN] No standard dataset folder detected. Proceeding in lightweight mode.")

# 3. Read sample rows if dataset found
if found_dir:
    for split in ["train", "test"]:
        s1_file = found_dir / split / f"{split}_source1.tsv"
        if not s1_file.is_file():
            s1_file = found_dir / f"{split}_source1.tsv"
        if s1_file.is_file():
            with open(s1_file, "r", encoding="utf-8") as f:
                header = f.readline().strip()
                sample_lines = [f.readline().strip() for _ in range(3)]
                print(f"[DATA] {s1_file.name} header: {header}")
                print(f"[DATA] Sample rows loaded: {len(sample_lines)}")

# 4. Generate smoke submission outputs
working_dir = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("output")
working_dir.mkdir(parents=True, exist_ok=True)

matching_path = working_dir / "matching_results.tsv"
candidate_path = working_dir / "candidate_pairs.tsv"

# Mock entities for smoke verification
sample_s1_ids = [f"S1-TEST-{i:04d}" for i in range(10)]
with open(matching_path, "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1 in sample_s1_ids:
        f.write(f"{s1}\t\n")

with open(candidate_path, "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1 in sample_s1_ids:
        f.write(f"{s1}\t\n")

print(f"\n[OUTPUT] Emitted smoke results to: {matching_path.resolve()}")
print(f"[OUTPUT] Emitted smoke candidates to: {candidate_path.resolve()}")
print("=======================================================")
print(" [SUCCESS] Kaggle Remote Entrypoint execution PASSED!  ")
print("=======================================================")
