"""
Tabular Data Ingestion Module for Amazon ML Challenge 2026.

Provides dynamic dataset path resolution, memory-efficient loading with pyarrow fallback,
string sanitization (no row corruption from newlines/carriage returns, zero NaNs),
and chunked streaming for large test sources.
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Generator, Optional, Union, List
import pandas as pd


AMAZON_ML_DATA_DIR = os.getenv("AMAZON_ML_DATA_DIR", "")


def resolve_dataset_path(filename: str, custom_dir: Optional[str] = None) -> str:
    """
    Resolve the absolute or relative path to a dataset file.
    
    Searches across:
    1. custom_dir (if provided)
    2. AMAZON_ML_DATA_DIR environment variable
    3. student_resource/dataset/ (with train/ and test/ subdirectories)
    4. 6ab10eb3b23ba_student_resource/student_resource/dataset/
    5. /kaggle/input/amazon-ml-challenge-2026/
    6. Local dataset/ folder
    7. Current working directory
    """
    candidate_bases = []
    if custom_dir:
        candidate_bases.append(custom_dir)
    if AMAZON_ML_DATA_DIR:
        candidate_bases.append(AMAZON_ML_DATA_DIR)
        
    candidate_bases.extend([
        "student_resource/dataset",
        "6ab10eb3b23ba_student_resource/student_resource/dataset",
        "/kaggle/input/amazon-ml-challenge-2026/dataset",
        "/kaggle/input/amazon-ml-challenge-2026",
        "dataset",
        "."
    ])
    
    # Infer split subfolder if applicable
    target_subdirs = []
    if "train" in filename.lower():
        target_subdirs = ["train", ""]
    elif "test" in filename.lower():
        target_subdirs = ["test", ""]
    else:
        target_subdirs = ["", "train", "test"]

    tried = []
    for base in candidate_bases:
        for sub in target_subdirs:
            full_path = os.path.normpath(os.path.join(base, sub, filename))
            tried.append(full_path)
            if os.path.isfile(full_path):
                return full_path

    raise FileNotFoundError(
        f"Could not resolve dataset file '{filename}'. Searched paths:\n  " +
        "\n  ".join(tried[:10]) + ("\n  ... and more" if len(tried) > 10 else "")
    )


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip carriage returns, newlines, and leading/trailing whitespace from string columns.
    Ensures zero NaN values.
    """
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.name.startswith("string"):
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.replace("\r", " ", regex=False)
                .str.replace("\n", " ", regex=False)
                .str.strip()
            )
        else:
            df[col] = df[col].fillna("")
    return df


def load_tsv(
    filepath: str,
    chunksize: Optional[int] = None,
    nrows: Optional[int] = None,
    use_pyarrow: bool = True
) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
    """
    Load a tab-separated file with strict non-quoting and empty string NA handling.
    Tries pyarrow engine first, falling back to 'c' engine if necessary.
    """
    engine = "pyarrow" if use_pyarrow else "c"
    
    # Common arguments for reading raw TSVs safely
    read_kwargs = dict(
        sep="\t",
        quoting=csv.QUOTE_NONE,
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )

    try:
        if chunksize is not None:
            # pyarrow doesn't support chunksize in read_csv directly on all pandas versions,
            # so we use engine="c" for chunked streaming
            reader = pd.read_csv(filepath, chunksize=chunksize, engine="c", **read_kwargs)
            def _chunk_gen():
                for chunk in reader:
                    yield sanitize_dataframe(chunk)
            return _chunk_gen()
        else:
            df = pd.read_csv(filepath, engine=engine, **read_kwargs)
            return sanitize_dataframe(df)
    except Exception as e:
        if use_pyarrow:
            # Fallback to C engine
            return load_tsv(filepath, chunksize=chunksize, nrows=nrows, use_pyarrow=False)
        raise e


class DataLoader:
    """Unified tabular loader for Amazon ML Challenge 2026 data."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir

    def resolve(self, filename: str) -> str:
        return resolve_dataset_path(filename, custom_dir=self.data_dir)

    def load_source(
        self,
        source_id: Union[int, str],
        split: str = "train",
        nrows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load source1, source2, or source3 for the given split (train/test).
        source_id can be 1, 2, 3 or 'source1', 'train_source1.tsv'.
        """
        sid = str(source_id).lower()
        if not sid.startswith("source") and not sid.startswith(split):
            sid = f"source{sid}"
        if not sid.endswith(".tsv"):
            filename = f"{split}_{sid}.tsv"
        else:
            filename = sid
            
        path = self.resolve(filename)
        return load_tsv(path, nrows=nrows)

    def load_ground_truth(
        self,
        split: str = "train",
        nrows: Optional[int] = None
    ) -> pd.DataFrame:
        """Load ground truth mapping (source1_entity_id -> matched_entity_ids)."""
        filename = f"{split}_ground_truth.tsv"
        path = self.resolve(filename)
        return load_tsv(path, nrows=nrows)

    def stream_source(
        self,
        source_id: Union[int, str],
        split: str = "test",
        chunk_size: int = 50000
    ) -> Generator[pd.DataFrame, None, None]:
        """Stream large source tables in chunks."""
        sid = str(source_id).lower()
        if not sid.startswith("source") and not sid.startswith(split):
            sid = f"source{sid}"
        if not sid.endswith(".tsv"):
            filename = f"{split}_{sid}.tsv"
        else:
            filename = sid
            
        path = self.resolve(filename)
        return load_tsv(path, chunksize=chunk_size)


def load_dataset(
    source_id: Union[int, str],
    split: str = "train",
    nrows: Optional[int] = None,
    data_dir: Optional[str] = None
) -> pd.DataFrame:
    """Convenience functional loader matching legacy scripts."""
    loader = DataLoader(data_dir=data_dir)
    return loader.load_source(source_id=source_id, split=split, nrows=nrows)


def check_pipeline() -> bool:
    """Self-test and verification utility."""
    print("=== Amazon ML Challenge 2026 Data Loader Diagnostic ===")
    loader = DataLoader()
    
    files_to_check = [
        ("train", 1, "train_source1.tsv"),
        ("train", 2, "train_source2.tsv"),
        ("train", 3, "train_source3.tsv"),
        ("train", "gt", "train_ground_truth.tsv"),
        ("test", 1, "test_source1.tsv"),
        ("test", 2, "test_source2.tsv"),
        ("test", 3, "test_source3.tsv"),
    ]
    
    all_ok = True
    for split, sid, fname in files_to_check:
        try:
            path = loader.resolve(fname)
            if sid == "gt":
                df = loader.load_ground_truth(split=split, nrows=5)
            else:
                df = loader.load_source(source_id=sid, split=split, nrows=5)
            n_nan = df.isna().sum().sum()
            print(f"[OK] {fname}")
            print(f"     Path: {path}")
            print(f"     Cols: {list(df.columns)} | Sample shape: {df.shape} | NaNs: {n_nan}")
        except Exception as e:
            print(f"[FAIL] {fname}: {e}")
            all_ok = False
            
    # Test streaming
    try:
        stream = loader.stream_source(source_id=1, split="test", chunk_size=100)
        chunk = next(stream)
        print(f"[OK] Stream test chunk shape: {chunk.shape}")
    except Exception as e:
        print(f"[FAIL] Stream test: {e}")
        all_ok = False

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Loader Inspection")
    parser.add_argument("--check", action="store_true", help="Run comprehensive path & loading diagnostics")
    args = parser.parse_args()
    
    success = check_pipeline()
    if not success:
        exit(1)
    print("\nAll dataset ingestion checks PASSED successfully.")
