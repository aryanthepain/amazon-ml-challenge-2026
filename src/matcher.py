"""Single-Match Inference Resolver for Entity Resolution.

Applies calibrated threshold theta* and argmax probability candidate assignment
to generate competition-compliant predictions (with singletons left unmatched).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.features import extract_24_features, extract_features_batch
from src.model import EntityMatcherLGBM


class SingleMatchResolver:
    """Inference Match Resolver applying decision threshold theta*."""

    def __init__(self, model: EntityMatcherLGBM, theta: float = 0.70):
        self.model = model
        self.theta = float(theta)

    def resolve_candidate_list(
        self,
        s1_name: str,
        s1_addr: str,
        candidates: List[Tuple[str, str, str, float, int]],  # [(cand_id, cand_name, cand_addr, sim, rank)]
    ) -> Optional[str]:
        """Resolve candidates for a single S1 record.

        Returns:
            Matched candidate ID if max P(match) >= theta, else None (singleton).
        """
        if not candidates:
            return None

        # Build feature records
        records = [
            {
                "s1_name": s1_name,
                "s1_addr": s1_addr,
                "cand_name": cand_name,
                "cand_addr": cand_addr,
                "blocking_sim": sim,
                "blocking_rank": rank,
            }
            for _, cand_name, cand_addr, sim, rank in candidates
        ]

        feats = extract_features_batch(records)
        probs = self.model.predict_proba(feats)

        # Filter by threshold
        valid_indices = np.where(probs >= self.theta)[0]
        if len(valid_indices) == 0:
            return None

        best_idx = valid_indices[np.argmax(probs[valid_indices])]
        return candidates[best_idx][0]

    def resolve_from_pair_scores(
        self,
        candidate_preds: Dict[str, List[Tuple[str, float]]],
        all_s1_ids: List[str],
    ) -> pd.DataFrame:
        """Resolve candidate matches from pre-computed candidate probabilities.

        Returns:
            DataFrame with columns: ['source_1_id', 'matched_id']
            where singletons have matched_id as empty string "".
        """
        rows: List[Dict[str, str]] = []
        for s1_id in all_s1_ids:
            cands = candidate_preds.get(s1_id, [])
            valid_cands = [(cid, p) for cid, p in cands if p >= self.theta]

            if valid_cands:
                best_match = max(valid_cands, key=lambda x: x[1])[0]
                rows.append({"source_1_id": s1_id, "matched_id": best_match})
            else:
                rows.append({"source_1_id": s1_id, "matched_id": ""})

        return pd.DataFrame(rows)
