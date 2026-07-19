"""AI Options Copilot — the conversational wrapper over the rule-based F&O
suggester ("What is the best NIFTY trade today?").

This invents NOTHING. It assembles REAL option-chain facts from the existing
`fetch_index_snapshot` (PCR / max-pain / ATM IV / IV-rank / regime bias / DTE),
runs the existing rule-based `suggest_strategies` to get ranked candidates, and
returns the candidates deterministically (ALWAYS — 0 LLM tokens). Only when the
user explicitly clicks (`use_llm=True`) does it call the grounded reasoner to
narrate which candidate is the best trade right now and its risk/reward, cached
per symbol/day, free-first model. Honest-empty when the chain is unavailable.

Per the locked decisions: LLMs do NOT generate option signals — the substance
is the rule-based suggester; this is purely a narration/ranking layer.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

from .snapshot import IndexSnapshot, fetch_index_snapshot
from .strategies import classify_vix_regime, suggest_strategies

logger = logging.getLogger(__name__)

# The leading element of `suggest_strategies` is a descriptive VIX-regime note,
# not an actionable trade — its name is prefixed this way. We surface the regime
# separately in `facts` and keep only real trade candidates in `strategies`.
_REGIME_NOTE_PREFIX = "VIX Regime:"

# How many ranked trade candidates the copilot surfaces. The suggester already
# sorts by confidence then name; the best is the head of that list.
_TOP_N = 3


def _india_vix() -> Optional[float]:
    """Best-effort India VIX from the shared market provider. Degrades to None
    (the suggester + regime classifier both handle a missing VIX)."""
    try:
        from ...data.market import get_market_data_provider
        mp = get_market_data_provider()
        q = mp.get_quote("VIX")
        if q and getattr(q, "ltp", None):
            return float(q.ltp)
    except Exception as e:
        logger.debug("options_copilot: VIX fetch failed: %s", e)
    return None


def _iv_rank(symbol: str, atm_iv: Optional[float]) -> Optional[float]:
    """Best-effort IV Rank (records today's ATM IV + reads trailing window).
    None until enough history exists — the suggester treats None as 'unknown'."""
    try:
        from .iv_store import iv_rank_percentile
        return iv_rank_percentile(symbol, atm_iv).get("iv_rank")
    except Exception as e:
        logger.debug("options_copilot: iv_rank failed for %s: %s", symbol, e)
        return None


def _bias_from_pcr(pcr_tag: Optional[str]) -> str:
    """Map the snapshot's PCR classification to a plain directional bias."""
    if pcr_tag in ("extreme_bullish", "bullish"):
        return "bullish"
    if pcr_tag in ("extreme_bearish", "bearish"):
        return "bearish"
    return "neutral"


def assemble_facts(
    snap: IndexSnapshot,
    *,
    vix: Optional[float],
    iv_rank: Optional[float],
) -> Dict[str, Any]:
    """Tight, REAL fact dict for the copilot — the same numbers the snapshot
    already computed, reshaped for narration. Zero tokens, zero invention."""
    return {
        "symbol": snap.symbol,
        "spot": round(snap.spot, 2) if snap.spot else None,
        "pcr_oi": round(snap.pcr_oi, 3) if snap.pcr_oi is not None else None,
        "pcr_tag": snap.pcr_tag,
        "bias": _bias_from_pcr(snap.pcr_tag),
        "max_pain": round(snap.max_pain, 2) if snap.max_pain is not None else None,
        "max_pain_distance_pct": (
            round(snap.max_pain_distance_pct, 2)
            if snap.max_pain_distance_pct is not None else None
        ),
        "pull_to_max_pain_signal": snap.pull_to_max_pain_signal,
        "atm_iv": round(snap.iv_atm, 4) if snap.iv_atm is not None else None,
        "iv_rank": iv_rank,
        "india_vix": round(vix, 2) if vix is not None else None,
        "vix_regime": classify_vix_regime(vix),
        "days_to_expiry": snap.days_to_expiry,
    }


def _trade_candidates(suggestions: List[Any]) -> List[Dict[str, Any]]:
    """Drop the leading descriptive VIX-regime note (it's not a trade) and keep
    the top-N ranked, actionable strategy candidates as plain dicts."""
    out: List[Dict[str, Any]] = []
    for s in suggestions:
        if str(getattr(s, "name", "")).startswith(_REGIME_NOTE_PREFIX):
            continue
        out.append(s.to_dict() if hasattr(s, "to_dict") else dict(s))
    return out[:_TOP_N]


def best_trade(symbol: str = "NIFTY", *, use_llm: bool = False) -> Dict[str, Any]:
    """{symbol, facts, strategies, narrative}.

    `strategies` is the deterministic, ranked rule-based candidate list and is
    ALWAYS returned (0 tokens) — the best trade is `strategies[0]`. `narrative`
    is the grounded reasoner's pick + risk/reward, cached per symbol/day, ONLY
    when `use_llm`. Returns honest-empty (`strategies=[]`, `facts=None`,
    `narrative=None`) when the option chain is unavailable.
    """
    sym = symbol.strip().upper()
    snap = fetch_index_snapshot(sym)
    if snap is None:
        return {"symbol": sym, "facts": None, "strategies": [], "narrative": None}

    vix = _india_vix()
    iv_rank = _iv_rank(sym, snap.iv_atm)
    facts = assemble_facts(snap, vix=vix, iv_rank=iv_rank)
    strategies = _trade_candidates(suggest_strategies(snap, vix=vix, iv_rank=iv_rank))

    narrative: Optional[str] = None
    if use_llm and strategies:
        from ...ai.agents.grounded import grounded_reason
        narrative = grounded_reason(
            {**facts, "strategies": strategies},
            "Given ONLY these option-chain stats and these rule-suggested "
            "strategies, explain in 3-4 sentences which is the best trade right "
            "now and the risk/reward — do not invent any new strategy.",
            cache_key=f"optcopilot:{sym}:{date.today().isoformat()}",
            role="responder",
        )

    return {
        "symbol": sym,
        "facts": facts,
        "strategies": strategies,
        "narrative": narrative,
    }
