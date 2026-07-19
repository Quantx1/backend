"""
Multi-model sentiment ensemble — cross-check the primary LLM call against
additional FREE/open sentiment models and report AGREEMENT.

Models combined per headline:
  * llm      — the primary LLM classifier result (passed in; real, free model)
  * finbert  — FinBERT-India (real trained model) when its weights are loaded
  * lexicon  — the finance polarity lexicon (transparent, instant, offline)

This is corroboration, not substitution: the displayed Mood stays the LLM-
derived materiality-weighted score; the ensemble only adds a "N of M models
agree" confidence signal so users can see when the read is robust vs contested.
Honest-empty per model (FinBERT absent → it simply doesn't vote). Pure aside
from the optional FinBERT model call.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from .finance_lexicon import lexicon_score

logger = logging.getLogger(__name__)


def _sign(x: Optional[float], eps: float = 0.05) -> int:
    if x is None:
        return 0
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _finbert_scores(titles: Sequence[str]) -> Optional[List[float]]:
    """FinBERT-India scores when the model is loaded, else None (no vote)."""
    try:
        from .finbert_india import get_finbert
        fb = get_finbert()
        if not getattr(fb, "ready", False):
            return None
        rows = fb.classify_batch(list(titles))
        if not rows:
            return None
        return [float(r.get("score", 0.0)) for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.debug("finbert ensemble vote unavailable: %s", exc)
        return None


def cross_check(titles: Sequence[str], llm_scores: Sequence[float]) -> List[Dict[str, Any]]:
    """Per-title agreement across the available sentiment models.

    Returns one dict per title:
      {llm, lexicon, finbert|None, models_total, models_agree, consensus}
    where consensus ∈ {positive,negative,neutral} is the majority sign among
    NON-abstaining models, and models_agree counts models matching it.
    """
    titles = list(titles)
    lex = [lexicon_score(t) for t in titles]
    fb = _finbert_scores(titles)

    out: List[Dict[str, Any]] = []
    for i, _t in enumerate(titles):
        llm = float(llm_scores[i]) if i < len(llm_scores) else 0.0
        votes: Dict[str, Optional[float]] = {"llm": llm, "lexicon": lex[i]}
        if fb is not None and i < len(fb):
            votes["finbert"] = fb[i]
        signs = [(_sign(v)) for v in votes.values() if v is not None]
        non_abstain = [s for s in signs if s != 0]
        pos = non_abstain.count(1)
        neg = non_abstain.count(-1)
        if pos == 0 and neg == 0:
            consensus, agree = "neutral", 0
        elif pos >= neg:
            consensus, agree = "positive", pos
        else:
            consensus, agree = "negative", neg
        out.append({
            "llm": round(llm, 4),
            "lexicon": round(lex[i], 4),
            "finbert": round(fb[i], 4) if (fb is not None and i < len(fb)) else None,
            "models_total": len(non_abstain),
            "models_agree": agree,
            "consensus": consensus,
        })
    return out


def models_available() -> List[str]:
    """Which sentiment models can vote right now (for UI disclosure)."""
    models = ["llm", "lexicon"]
    try:
        from .finbert_india import get_finbert
        if getattr(get_finbert(), "ready", False):
            models.append("finbert")
    except Exception:  # noqa: BLE001
        pass
    return models


__all__ = ["cross_check", "models_available"]
