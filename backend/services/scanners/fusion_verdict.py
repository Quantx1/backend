"""
Fusion Verdict — the single, explainable per-symbol setup verdict.

The audit's headline gap: the app has every input (regime, smart-money,
volume, mood, ML alpha) but nothing FUSES them into one ranked, actionable
verdict like the "RELIANCE setup → high-quality" example. This service is
that fusion layer.

Design (locked invariants):
  * Deterministic weighted fusion — the LLM only NARRATES, never decides.
  * Honest-empty per factor: a factor with no data is omitted and its
    weight is redistributed; <2 factors → "Insufficient data".
  * Event-risk is a GATE, not a vote: an imminent earnings window caps the
    verdict to "Hold off" regardless of how bullish the factors look.
  * Reuses existing services (stock_scores / volume_intelligence /
    options_flow / regime / event_risk) — no new data access duplicated.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Factor weights (renormalized over whichever factors actually have data).
_WEIGHTS = {
    "alpha": 0.28,        # cross-sectional ML rank (Qlib alpha158)
    "trend": 0.18,        # momentum + trend percentile blend
    "smart_money": 0.20,  # options OI lean (writing balance / PCR)
    "volume": 0.14,       # accumulation vs churn
    "mood": 0.12,         # news sentiment
    "regime": 0.08,       # market-context tilt
}


def _lean(score: Optional[float]) -> str:
    if score is None:
        return "neutral"
    if score > 0.12:
        return "bullish"
    if score < -0.12:
        return "bearish"
    return "neutral"


def _pct_to_score(pct: Optional[float]) -> Optional[float]:
    """Map a 0..100 percentile to a -1..1 score (50 → 0)."""
    if pct is None:
        return None
    return max(-1.0, min(1.0, (float(pct) - 50.0) / 50.0))


def fuse(factors: List[Dict[str, Any]], *, gated: bool) -> Dict[str, Any]:
    """Pure fusion core. ``factors`` carry score (-1..1 or None) + weight.

    Returns {verdict, composite (0..100|None), direction, gated, factors}.
    Unit-tested without any I/O.
    """
    scored = [f for f in factors if f.get("score") is not None]
    if len(scored) < 2:
        return {
            "verdict": "Insufficient data",
            "composite": None,
            "direction": "neutral",
            "gated": gated,
            "factors": factors,
        }

    wsum = sum(f["weight"] for f in scored) or 1.0
    avg = sum(f["score"] * f["weight"] for f in scored) / wsum  # -1..1
    composite = round((avg + 1) / 2 * 100)
    direction = _lean(avg)

    if gated:
        verdict = "Hold off — event risk"
    elif composite >= 72:
        verdict = "Strong setup"
    elif composite >= 58:
        verdict = "Constructive"
    elif composite >= 42:
        verdict = "Mixed"
    elif composite >= 30:
        verdict = "Weak"
    else:
        verdict = "Avoid"

    return {
        "verdict": verdict,
        "composite": composite,
        "direction": direction,
        "gated": gated,
        "factors": factors,
    }


# ─────────────────────────────── readers ──────────────────────────────────


def _regime_score() -> Optional[Dict[str, Any]]:
    try:
        from ...core.database import get_supabase_admin
        rows = (
            get_supabase_admin()
            .table("regime_history")
            .select("regime, prob_bull, prob_bear, detected_at")
            .order("detected_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        if not rows:
            return None
        r = rows[0]
        name = (r.get("regime") or "").lower()
        score = {"bull": 0.7, "sideways": 0.0, "bear": -0.7}.get(name)
        return {"name": name, "score": score, "detail": f"market regime: {name or 'unknown'}"}
    except Exception as exc:  # noqa: BLE001
        logger.debug("fusion regime read failed: %s", exc)
        return None


def _smart_money_score(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        from ..fno_scanner.options_flow import options_flow
        flow = options_flow(symbol)
    except Exception as exc:  # noqa: BLE001
        logger.debug("fusion options_flow failed for %s: %s", symbol, exc)
        return None
    if not flow:
        return None
    lean = (flow.get("lean") or "neutral").lower()
    score = {"bullish": 0.6, "bearish": -0.6, "neutral": 0.0}.get(lean, 0.0)
    pcr = flow.get("pcr")
    detail = f"options OI lean {lean}"
    if pcr is not None:
        detail += f" (PCR {pcr})"
    return {"score": score, "detail": detail, "lean": lean}


def _volume_score(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        from ..market.volume_intelligence import volume_intel
        vi = volume_intel(symbol)
    except Exception as exc:  # noqa: BLE001
        logger.debug("fusion volume_intel failed for %s: %s", symbol, exc)
        return None
    sig = (vi.get("signal") or "").lower()
    # signal vocabulary: accumulation / distribution / churn / quiet
    score = {"accumulation": 0.5, "distribution": -0.5, "churn": -0.1}.get(sig)
    if score is None:
        return None
    return {"score": score, "detail": vi.get("drivers", [None])[0] or f"volume: {sig}"}


def build_verdict(symbol: str, *, use_llm: bool = False, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Assemble the fused verdict for one symbol from real sources.

    Honest-empty everywhere; deterministic; optional grounded narration.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return {"symbol": "", "verdict": "Insufficient data", "composite": None,
                "direction": "neutral", "gated": False, "factors": []}

    factors: List[Dict[str, Any]] = []

    # 1. Alpha + trend/momentum + mood — from the unified Scores block.
    try:
        from .stock_scores import scores as _scores
        sc = _scores(sym)
        by_key = {e["key"]: e for e in (sc.get("scores") or [])}
    except Exception as exc:  # noqa: BLE001
        logger.debug("fusion scores failed for %s: %s", sym, exc)
        by_key = {}

    if "alpha" in by_key:
        s = _pct_to_score(by_key["alpha"].get("pct"))
        factors.append({"key": "alpha", "label": "Alpha rank", "weight": _WEIGHTS["alpha"],
                        "score": s, "lean": _lean(s), "detail": by_key["alpha"].get("note")})

    trend_pcts = [by_key[k].get("pct") for k in ("momentum", "trend") if k in by_key]
    trend_pcts = [p for p in trend_pcts if p is not None]
    if trend_pcts:
        s = _pct_to_score(sum(trend_pcts) / len(trend_pcts))
        factors.append({"key": "trend", "label": "Trend & momentum", "weight": _WEIGHTS["trend"],
                        "score": s, "lean": _lean(s), "detail": "momentum/trend percentile vs universe"})

    if "mood" in by_key:
        mv = by_key["mood"].get("value")
        s = max(-1.0, min(1.0, float(mv))) if mv is not None else None
        factors.append({"key": "mood", "label": "News mood", "weight": _WEIGHTS["mood"],
                        "score": s, "lean": _lean(s), "detail": by_key["mood"].get("note")})

    # 2. Smart-money (options OI) + 3. Volume + 4. Regime.
    sm = _smart_money_score(sym)
    if sm:
        factors.append({"key": "smart_money", "label": "Smart money (OI)", "weight": _WEIGHTS["smart_money"],
                        "score": sm["score"], "lean": _lean(sm["score"]), "detail": sm["detail"]})
    vol = _volume_score(sym)
    if vol:
        factors.append({"key": "volume", "label": "Volume", "weight": _WEIGHTS["volume"],
                        "score": vol["score"], "lean": _lean(vol["score"]), "detail": vol["detail"]})
    reg = _regime_score()
    if reg and reg.get("score") is not None:
        factors.append({"key": "regime", "label": "Market regime", "weight": _WEIGHTS["regime"],
                        "score": reg["score"], "lean": _lean(reg["score"]), "detail": reg["detail"]})

    # 5. Event-risk GATE (not a vote).
    gated = False
    try:
        from .event_risk import symbols_in_event_window
        gated = bool(symbols_in_event_window([sym]))
    except Exception:
        gated = False
    if gated:
        factors.append({"key": "event_risk", "label": "Event risk", "weight": 0.0,
                        "score": None, "lean": "blocked",
                        "detail": "earnings inside the blackout window — entries suppressed"})

    out = fuse(factors, gated=gated)
    out["symbol"] = sym
    out["as_of"] = date.today().isoformat()

    # Optional grounded narration (cached per symbol/day) — narrate only.
    narrative = None
    if use_llm and out["composite"] is not None:
        try:
            from ...ai.agents.grounded import grounded_reason
            facts = {
                "symbol": sym, "verdict": out["verdict"], "composite": out["composite"],
                "direction": out["direction"],
                "factors": [{"label": f["label"], "lean": f["lean"], "detail": f.get("detail")}
                            for f in factors],
            }
            narrative = grounded_reason(
                facts,
                f"In 2-3 sentences, summarise the fused trade verdict for {sym} from these factors. "
                f"Do not invent any number not present in the facts.",
                cache_key=f"fusionverdict:{sym}:{date.today().isoformat()}",
                user_id=user_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("fusion narration failed for %s: %s", sym, exc)
    out["narrative"] = narrative
    return out


__all__ = ["build_verdict", "fuse"]
