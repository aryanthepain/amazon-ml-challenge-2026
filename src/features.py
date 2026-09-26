"""24-Dimensional Pairwise Feature Extraction Engine.

Implements high-speed RapidFuzz C++ SIMD string distance metrics, character
3-gram Jaccard overlap, street number conflict detection, missing address
indicators, and candidate blocking priors for LightGBM pairwise classification.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import re
import numpy as np
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz import fuzz

RE_DIGITS = re.compile(r'\b\d+\b')

FEATURE_NAMES: Tuple[str, ...] = (
    "name_jw",
    "name_lev",
    "name_token_sort",
    "name_token_set",
    "name_containment",
    "addr_jw",
    "addr_lev",
    "addr_token_sort",
    "addr_token_set",
    "combo_jw",
    "combo_lev",
    "combo_token_jaccard",
    "name_char3gram",
    "addr_char3gram",
    "num_jaccard",
    "has_exact_number_match",
    "has_number_conflict",
    "s1_has_addr",
    "cand_has_addr",
    "both_have_addr",
    "name_len_diff_ratio",
    "addr_len_diff_ratio",
    "blocking_sim",
    "blocking_rank",
)


def char_3gram_jaccard(text1: Optional[str], text2: Optional[str]) -> float:
    """Compute character 3-gram Jaccard similarity after removing spaces."""
    if not text1 or not text2:
        return 0.0
    s1 = text1.replace(" ", "")
    s2 = text2.replace(" ", "")
    if len(s1) < 3 or len(s2) < 3:
        return 1.0 if s1 == s2 and len(s1) > 0 else 0.0
    g1: Set[str] = {s1[i : i + 3] for i in range(len(s1) - 2)}
    g2: Set[str] = {s2[i : i + 3] for i in range(len(s2) - 2)}
    inter = len(g1 & g2)
    union = len(g1 | g2)
    return float(inter / union) if union > 0 else 0.0


def extract_name_features(s1_name: Optional[str], cand_name: Optional[str]) -> List[float]:
    """Compute 5 RapidFuzz normalized name similarity features."""
    s1 = (s1_name or "").strip()
    s2 = (cand_name or "").strip()
    if not s1 or not s2:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    jw = float(JaroWinkler.similarity(s1, s2))
    lev = float(Levenshtein.normalized_similarity(s1, s2))
    tsort = float(fuzz.token_sort_ratio(s1, s2)) / 100.0
    tset = float(fuzz.token_set_ratio(s1, s2)) / 100.0

    t1 = set(s1.split())
    t2 = set(s2.split())
    shorter = min(len(t1), len(t2))
    containment = float(len(t1 & t2) / shorter) if shorter > 0 else 0.0

    return [jw, lev, tsort, tset, containment]


def extract_address_features(s1_addr: Optional[str], cand_addr: Optional[str]) -> List[float]:
    """Compute 4 RapidFuzz normalized address similarity features."""
    a1 = (s1_addr or "").strip()
    a2 = (cand_addr or "").strip()
    if not a1 or not a2:
        return [0.0, 0.0, 0.0, 0.0]

    jw = float(JaroWinkler.similarity(a1, a2))
    lev = float(Levenshtein.normalized_similarity(a1, a2))
    tsort = float(fuzz.token_sort_ratio(a1, a2)) / 100.0
    tset = float(fuzz.token_set_ratio(a1, a2)) / 100.0

    return [jw, lev, tsort, tset]


def extract_composite_features(
    s1_name: Optional[str],
    s1_addr: Optional[str],
    cand_name: Optional[str],
    cand_addr: Optional[str],
) -> List[float]:
    """Compute 3 composite name+address similarity features."""
    c1 = f"{s1_name or ''} {s1_addr or ''}".strip()
    c2 = f"{cand_name or ''} {cand_addr or ''}".strip()
    if not c1 or not c2:
        return [0.0, 0.0, 0.0]

    jw = float(JaroWinkler.similarity(c1, c2))
    lev = float(Levenshtein.normalized_similarity(c1, c2))
    toks1 = set(c1.split())
    toks2 = set(c2.split())
    union = len(toks1 | toks2)
    jaccard = float(len(toks1 & toks2) / union) if union > 0 else 0.0

    return [jw, lev, jaccard]


def extract_numeric_features(s1_addr: Optional[str], cand_addr: Optional[str]) -> List[float]:
    """Compute 3 street number and numeric conflict features."""
    a1 = s1_addr or ""
    a2 = cand_addr or ""
    n1 = set(RE_DIGITS.findall(a1))
    n2 = set(RE_DIGITS.findall(a2))

    if n1 and n2:
        union = len(n1 | n2)
        num_jaccard = float(len(n1 & n2) / union) if union > 0 else 0.0
        exact_match = 1.0 if n1 == n2 else 0.0
        conflict = 1.0 if n1 != n2 else 0.0
    else:
        num_jaccard = 0.0
        exact_match = 0.0
        conflict = 0.0

    return [num_jaccard, exact_match, conflict]


def extract_24_features(
    s1_name: Optional[str],
    s1_addr: Optional[str],
    cand_name: Optional[str],
    cand_addr: Optional[str],
    blocking_sim: float = 0.0,
    blocking_rank: int = 1,
) -> np.ndarray:
    """Extract full normalized 24-dimensional feature vector for a candidate pair.

    Returns:
        np.ndarray of shape (24,) and dtype np.float32.
    """
    n1 = (s1_name or "").strip()
    n2 = (cand_name or "").strip()
    a1 = (s1_addr or "").strip()
    a2 = (cand_addr or "").strip()

    # 1. Name features (5)
    name_feats = extract_name_features(n1, n2)

    # 2. Address features (4)
    addr_feats = extract_address_features(a1, a2)

    # 3. Composite features (3)
    combo_feats = extract_composite_features(n1, a1, n2, a2)

    # 4. Character 3-gram features (2)
    name_char3 = char_3gram_jaccard(n1, n2)
    addr_char3 = char_3gram_jaccard(a1, a2)

    # 5. Street conflict features (3)
    num_feats = extract_numeric_features(a1, a2)

    # 6. Missing metadata & length ratios (5)
    s1_has_addr = 1.0 if a1 else 0.0
    cand_has_addr = 1.0 if a2 else 0.0
    both_have_addr = 1.0 if (a1 and a2) else 0.0

    max_n = max(len(n1), len(n2), 1)
    name_len_diff = float(abs(len(n1) - len(n2)) / max_n)

    max_a = max(len(a1), len(a2), 1)
    addr_len_diff = float(abs(len(a1) - len(a2)) / max_a)

    # 7. Blocking priors (2)
    sim_clipped = float(np.clip(blocking_sim, 0.0, 1.0))
    rank_norm = float(max(0.0, 1.0 - (float(blocking_rank) - 1.0) / 25.0))

    vec = [
        *name_feats,
        *addr_feats,
        *combo_feats,
        name_char3,
        addr_char3,
        *num_feats,
        s1_has_addr,
        cand_has_addr,
        both_have_addr,
        name_len_diff,
        addr_len_diff,
        sim_clipped,
        rank_norm,
    ]

    return np.array(vec, dtype=np.float32)


def extract_features_batch(pair_records: List[Dict[str, Any]]) -> np.ndarray:
    """Vectorize 24-feature extraction across a batch of candidate pair dicts.

    Each dict must provide:
        s1_name, s1_addr, cand_name, cand_addr,
        optional: blocking_sim (default 0.0), blocking_rank (default 1)
    """
    if not pair_records:
        return np.empty((0, 24), dtype=np.float32)

    out = np.zeros((len(pair_records), 24), dtype=np.float32)
    for idx, rec in enumerate(pair_records):
        out[idx] = extract_24_features(
            s1_name=rec.get("s1_name"),
            s1_addr=rec.get("s1_addr"),
            cand_name=rec.get("cand_name"),
            cand_addr=rec.get("cand_addr"),
            blocking_sim=float(rec.get("blocking_sim", 0.0)),
            blocking_rank=int(rec.get("blocking_rank", 1)),
        )
    return out
