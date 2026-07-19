"""Builder functions for ensemble voters.

Each builder consumes one model's raw output and produces a normalised
``EnsembleVoter``. Centralising these here makes the ensemble weights
visible in one place and lets the orchestrator stay focused on flow
rather than weight bookkeeping.

The pipeline is BUY-only — every voter's ``direction_agrees`` checks
whether that model corroborates a long thesis.

Weights are locked in product memory (Step 2 §1.12) and must not drift
without an explicit decision + outcome-model retrain.
"""
from __future__ import annotations

from typing import Optional

from .ensemble import regime_bonus
from .types import EnsembleVoter

# Locked voter weights — DO NOT EDIT without a memory update.
# Sentiment ("Mood"/finbert_india) removed from the ensemble 2026-06-06 — it was
# a tiny 0.10 voter (an LLM at runtime, not real FinBERT) that added noise, not
# edge. compute_ensemble_score normalises by the sum of weights, so the four
# remaining voters re-weight proportionally with no manual rebalance. Mood is
# now a standalone on-demand engine (SentimentEngine.score_symbol), not a voter.
WEIGHTS = {
    "lgbm_signal_gate": 0.30,
    "tft_swing": 0.30,
    "qlib_alpha158": 0.20,
    "hmm_regime": 0.10,
}


def make_lgbm_voter(buy_prob: float, lgbm_direction: str) -> EnsembleVoter:
    """LGBMGate BUY probability. Direction agrees iff the gate says BUY."""
    return EnsembleVoter(
        name="lgbm_signal_gate",
        weight=WEIGHTS["lgbm_signal_gate"],
        score=buy_prob,
        direction_agrees=(lgbm_direction == "BUY"),
    )


def make_tft_voter(tft_score: float, tft_direction: str) -> EnsembleVoter:
    """TFT swing forecast. Direction agrees iff forecast is bullish."""
    return EnsembleVoter(
        name="tft_swing",
        weight=WEIGHTS["tft_swing"],
        score=tft_score,
        direction_agrees=(tft_direction == "bullish"),
    )


def make_qlib_voter(qlib_score: float) -> EnsembleVoter:
    """Qlib Alpha158 cross-sectional rank, 0..1.

    Long bias above 0.5.
    """
    return EnsembleVoter(
        name="qlib_alpha158",
        weight=WEIGHTS["qlib_alpha158"],
        score=qlib_score,
        direction_agrees=(qlib_score >= 0.5),
    )


def make_regime_voter(regime_id: Optional[int], bear_active: bool) -> EnsembleVoter:
    """HMM regime detector. Score follows regime_bonus
    (bull=1.0 / sideways=0.5 / bear=0.0). Direction agrees while NOT in bear.
    """
    return EnsembleVoter(
        name="hmm_regime",
        weight=WEIGHTS["hmm_regime"],
        score=regime_bonus(regime_id),
        direction_agrees=(not bear_active),
    )
