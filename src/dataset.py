"""Pairwise Dataset Builder with Hard Negative Mining.

Constructs balanced training pairs (1:4 positive to hard-negative ratio)
mined directly from Phase 2 cascade candidate blocking output, and builds
full validation candidate pair sets for threshold tuning and evaluation.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.features import extract_features_batch
from src.preprocessing import normalize_business_name, normalize_business_address


def parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Parse ground truth dataframe into a dict of s1_id -> set(target_ids)."""
    gt_map: Dict[str, Set[str]] = {}
    s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else "source_1_id"
    match_col = "matched_entity_ids" if "matched_entity_ids" in gt_df.columns else "target_id"

    for _, row in gt_df.iterrows():
        s1_id = str(row[s1_col]).strip()
        val = row[match_col]
        if pd.isna(val) or val is None:
            gt_map[s1_id] = set()
        elif isinstance(val, str):
            val_clean = val.strip()
            if val_clean:
                gt_map[s1_id] = {t.strip() for t in val_clean.split(",") if t.strip()}
            else:
                gt_map[s1_id] = set()
        else:
            gt_map[s1_id] = {str(val).strip()}

    return gt_map


def parse_candidate_pairs(candidates_df: pd.DataFrame) -> Dict[str, List[Tuple[str, int, float]]]:
    """Parse candidate pairs dataframe into dict: s1_id -> [(cand_id, rank, sim_score), ...]."""
    cand_map: Dict[str, List[Tuple[str, int, float]]] = {}
    s1_col = "source1_entity_id" if "source1_entity_id" in candidates_df.columns else "source_1_id"

    if "candidate_entity_ids" in candidates_df.columns:
        for _, row in candidates_df.iterrows():
            s1_id = str(row[s1_col]).strip()
            cands_str = str(row["candidate_entity_ids"]) if pd.notna(row["candidate_entity_ids"]) else ""
            if not cands_str:
                cand_map[s1_id] = []
                continue
            cand_ids = [c.strip() for c in cands_str.split(",") if c.strip()]
            records = []
            for rank_idx, cid in enumerate(cand_ids, start=1):
                # Approximation for TF-IDF cosine score if not explicitly recorded
                est_sim = max(0.18, 1.0 - (rank_idx - 1) * 0.03)
                records.append((cid, rank_idx, float(est_sim)))
            cand_map[s1_id] = records
    elif "target_id" in candidates_df.columns or "candidate_id" in candidates_df.columns:
        cand_col = "candidate_id" if "candidate_id" in candidates_df.columns else "target_id"
        sim_col = "similarity_score" if "similarity_score" in candidates_df.columns else "score"
        rank_col = "rank" if "rank" in candidates_df.columns else None

        for _, row in candidates_df.iterrows():
            s1_id = str(row[s1_col]).strip()
            cid = str(row[cand_col]).strip()
            sim = float(row[sim_col]) if sim_col in candidates_df.columns and pd.notna(row[sim_col]) else 0.5
            rank = int(row[rank_col]) if rank_col and pd.notna(row[rank_col]) else 1
            cand_map.setdefault(s1_id, []).append((cid, rank, sim))

    return cand_map


def build_entity_lookups(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    preprocess: bool = True,
) -> Tuple[Dict[str, Tuple[str, str, str]], Dict[str, Tuple[str, str, str]], Dict[str, List[str]]]:
    """Create fast tuple lookups (clean_name, clean_address, country) for S1 and Target pool."""
    s1_lookup: Dict[str, Tuple[str, str, str]] = {}
    target_lookup: Dict[str, Tuple[str, str, str]] = {}
    country_targets: Dict[str, List[str]] = {}

    s1_id_col = "entity_id" if "entity_id" in s1_df.columns else "source_1_id"
    for _, row in s1_df.iterrows():
        eid = str(row[s1_id_col]).strip()
        raw_name = str(row["name"]) if pd.notna(row.get("name")) else ""
        raw_addr = str(row["address"]) if pd.notna(row.get("address")) else ""
        country = str(row["country"]).strip().upper() if pd.notna(row.get("country")) else ""
        c_name = normalize_business_name(raw_name) if preprocess else raw_name
        c_addr = normalize_business_address(raw_addr) if preprocess else raw_addr
        s1_lookup[eid] = (c_name, c_addr, country)

    target_id_col = "entity_id" if "entity_id" in target_df.columns else "target_id"
    for _, row in target_df.iterrows():
        eid = str(row[target_id_col]).strip()
        raw_name = str(row["name"]) if pd.notna(row.get("name")) else ""
        raw_addr = str(row["address"]) if pd.notna(row.get("address")) else ""
        country = str(row["country"]).strip().upper() if pd.notna(row.get("country")) else ""
        c_name = normalize_business_name(raw_name) if preprocess else raw_name
        c_addr = normalize_business_address(raw_addr) if preprocess else raw_addr
        target_lookup[eid] = (c_name, c_addr, country)
        if country:
            country_targets.setdefault(country, []).append(eid)

    return s1_lookup, target_lookup, country_targets


def build_training_pairs(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    train_s1_ids: Set[str],
    candidate_pairs_df: pd.DataFrame,
    neg_ratio: int = 4,
    random_state: int = 42,
    preprocess: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble balanced training set pairs with hard negative mining.

    Returns:
        X_train: np.ndarray of shape (N, 24) float32
        y_train: np.ndarray of shape (N,) int32
    """
    rng = np.random.default_rng(random_state)
    gt_map = parse_ground_truth(ground_truth_df)
    cand_map = parse_candidate_pairs(candidate_pairs_df)
    s1_lookup, target_lookup, country_targets = build_entity_lookups(s1_df, target_df, preprocess=preprocess)

    pair_records: List[Dict[str, Any]] = []
    labels: List[int] = []

    for s1_id in train_s1_ids:
        if s1_id not in s1_lookup:
            continue
        s1_name, s1_addr, s1_country = s1_lookup[s1_id]
        true_targets = gt_map.get(s1_id, set())

        # 1. Positive pairs
        pos_count = 0
        for tid in true_targets:
            if tid in target_lookup:
                t_name, t_addr, _ = target_lookup[tid]
                pair_records.append({
                    "s1_name": s1_name,
                    "s1_addr": s1_addr,
                    "cand_name": t_name,
                    "cand_addr": t_addr,
                    "blocking_sim": 1.0,
                    "blocking_rank": 1,
                })
                labels.append(1)
                pos_count += 1

        if pos_count == 0:
            continue

        # 2. Hard negative mining from candidate blocking output
        cands = cand_map.get(s1_id, [])
        neg_cands = [c for c in cands if c[0] not in true_targets and c[0] in target_lookup]
        target_neg_count = pos_count * neg_ratio

        chosen_negs = neg_cands[:target_neg_count]
        for cid, rank, sim in chosen_negs:
            t_name, t_addr, _ = target_lookup[cid]
            pair_records.append({
                "s1_name": s1_name,
                "s1_addr": s1_addr,
                "cand_name": t_name,
                "cand_addr": t_addr,
                "blocking_sim": sim,
                "blocking_rank": rank,
            })
            labels.append(0)

        # Backfill random negatives if candidates were fewer than target_neg_count
        shortfall = target_neg_count - len(chosen_negs)
        if shortfall > 0:
            pool = country_targets.get(s1_country, [])
            valid_pool = [t for t in pool if t not in true_targets]
            if valid_pool:
                replace = len(valid_pool) < shortfall
                sampled = rng.choice(valid_pool, size=shortfall, replace=replace)
                for tid in sampled:
                    t_name, t_addr, _ = target_lookup[tid]
                    pair_records.append({
                        "s1_name": s1_name,
                        "s1_addr": s1_addr,
                        "cand_name": t_name,
                        "cand_addr": t_addr,
                        "blocking_sim": 0.1,
                        "blocking_rank": 25,
                    })
                    labels.append(0)

    if not pair_records:
        return np.empty((0, 24), dtype=np.float32), np.empty((0,), dtype=np.int32)

    X_train = extract_features_batch(pair_records)
    y_train = np.array(labels, dtype=np.int32)
    return X_train, y_train


def build_validation_pairs(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    val_s1_ids: Set[str],
    candidate_pairs_df: pd.DataFrame,
    preprocess: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Assemble validation candidate pairs without negative downsampling.

    Returns:
        X_val: np.ndarray of shape (N, 24) float32
        y_val: np.ndarray of shape (N,) int32
        val_meta_df: DataFrame preserving [source1_entity_id, candidate_id, label, blocking_sim, blocking_rank]
    """
    gt_map = parse_ground_truth(ground_truth_df)
    cand_map = parse_candidate_pairs(candidate_pairs_df)
    s1_lookup, target_lookup, _ = build_entity_lookups(s1_df, target_df, preprocess=preprocess)

    pair_records: List[Dict[str, Any]] = []
    labels: List[int] = []
    meta_records: List[Dict[str, Any]] = []

    for s1_id in val_s1_ids:
        if s1_id not in s1_lookup:
            continue
        s1_name, s1_addr, _ = s1_lookup[s1_id]
        true_targets = gt_map.get(s1_id, set())
        cands = cand_map.get(s1_id, [])

        for cid, rank, sim in cands:
            if cid not in target_lookup:
                continue
            t_name, t_addr, _ = target_lookup[cid]
            lbl = 1 if cid in true_targets else 0

            pair_records.append({
                "s1_name": s1_name,
                "s1_addr": s1_addr,
                "cand_name": t_name,
                "cand_addr": t_addr,
                "blocking_sim": sim,
                "blocking_rank": rank,
            })
            labels.append(lbl)
            meta_records.append({
                "source1_entity_id": s1_id,
                "candidate_id": cid,
                "label": lbl,
                "blocking_sim": sim,
                "blocking_rank": rank,
            })

    if not pair_records:
        return (
            np.empty((0, 24), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            pd.DataFrame(columns=["source1_entity_id", "candidate_id", "label", "blocking_sim", "blocking_rank"]),
        )

    X_val = extract_features_batch(pair_records)
    y_val = np.array(labels, dtype=np.int32)
    meta_df = pd.DataFrame(meta_records)
    return X_val, y_val, meta_df
