"""Pure fusion math for the signal ensemble.

The ensemble score is a weight-normalised sum of per-voter scores,
clipped to [0, 1], then scaled to 0..100. The regime bonus is a
separate multiplier applied at the signal-build loop, not here.

All formulas are locked by product memory (Step 2 §1.12) and must not
drift without an explicit decision + outcome-model retrain.
"""
from __future__ import annotations

from typing import List, Optional

from .types import EnsembleVoter


def compute_ensemble_score(voters: List[EnsembleVoter]) -> float:
    """Weighted average over voters, returning 0..100.

    No-fallbacks contract: every voter passed in here represents a
    real loaded model. If a required model wasn't loaded,
    ``SignalGenerator.__init__`` would have raised long before we reach
    this fn. Therefore there is no ``available`` check and no weight
    renormalization — the weight table is the score table.

    Bear regime confidence × 0.6 is applied *after* this fn in the
    signal-build loop (Step 2 §1.12 size gate). Sentiment is applied
    post-dedup to allow batched scoring.
    """
    if not voters:
        raise ValueError("ensemble has no voters — at least one required")
    total_w = sum(v.weight for v in voters)
    if total_w <= 0:
        raise ValueError("ensemble voter weights sum to <= 0")
    weighted = sum(v.weight * max(0.0, min(1.0, v.score)) for v in voters)
    return 100.0 * weighted / total_w


def regime_bonus(regime_id: Optional[int]) -> float:
    """Bull=1.0, sideways=0.5, bear=0.0. Defaults to 0.5 when unknown."""
    return {0: 1.0, 1: 0.5, 2: 0.0}.get(regime_id if regime_id is not None else -1, 0.5)
