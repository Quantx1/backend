"""
================================================================================
AI DOSSIER ROUTES — /stock/[symbol] per-stock consolidated output (PR 33)
================================================================================
Merges every engine's read on a single symbol into one payload:

    Archimedes       — latest swing forecast (TFT p10/p50/p90)
    Eratosthenes     — latest cross-sectional rank
    Apollonius       — latest long-horizon trajectory
    Thales           — latest news sentiment (14d window)
    Pythagoras       — current market regime (shared)
    Hipparchus       — latest intraday forecast, if any
    Hypatia          — upcoming earnings + beat probability (if any)

Tier split:
    Free   → bare payload (engine names + directional tag only)
    Pro    → scores + quantile bands + thesis text
    Elite  → full numeric detail + debate excerpt hook

**Naming rule:** response uses Greek engine names as top-level keys.
Internal column names (``tft_p50``, ``qlib_score``, etc.) NEVER leak to
the payload — they are translated via ``backend/core/public_models``.
================================================================================
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from ..core.database import get_supabase_admin
from ..core.public_models import PUBLIC_MODELS, public_label
from ..core.tiers import Tier, UserTier, tier_rank
from ..middleware.tier_gate import current_user_tier
from ..data.market_calendar import IST

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dossier", tags=["dossier"])


# ============================================================================
# Helpers
# ============================================================================


def _canon_symbol(s: str) -> str:
    s = (s or "").upper().strip()
    if s.endswith(".NS"):
        s = s[:-3]
    return s


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bull_bear_tag(up_prob: Optional[float]) -> str:
    if up_prob is None:
        return "neutral"
    if up_prob >= 0.60:
        return "bullish"
    if up_prob <= 0.40:
        return "bearish"
    return "neutral"


def _from_quantiles(p10: Optional[float], p50: Optional[float], p90: Optional[float]) -> str:
    """Derive a direction tag from the forecast quantile band."""
    if p50 is None:
        return "neutral"
    if p10 is not None and p10 > 0:
        return "bullish"
    if p90 is not None and p90 < 0:
        return "bearish"
    if p50 > 0:
        return "bullish_tilt"
    if p50 < 0:
        return "bearish_tilt"
    return "neutral"


def _latest_signal_row(sb, symbol: str) -> Dict[str, Any]:
    try:
        rows = (
            sb.table("signals")
            .select(
                "id, symbol, direction, entry_price, stop_loss, target, confidence, "
                "tft_p10, tft_p50, tft_p90, lgbm_buy_prob, qlib_score, qlib_rank, "
                "timesfm_p50, chronos_p50, hgnc_up_prob, finbert_sentiment, "
                "regime_at_signal, explanation_text, signal_type, created_at"
            )
            .eq("symbol", symbol)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return (rows.data or [None])[0] or {}
    except Exception as exc:
        logger.debug("dossier signal lookup failed %s: %s", symbol, exc)
        return {}


def _latest_regime(sb) -> Dict[str, Any]:
    try:
        rows = (
            sb.table("regime_history")
            .select("regime, prob_bull, prob_sideways, prob_bear, vix, detected_at")
            .order("detected_at", desc=True)
            .limit(1)
            .execute()
        )
        return (rows.data or [None])[0] or {}
    except Exception as exc:
        logger.debug("dossier regime lookup failed: %s", exc)
        return {}


def _latest_sentiment(sb, symbol: str) -> Dict[str, Any]:
    """Mean sentiment over last 14 trade dates with data."""
    try:
        cutoff = (date.today() - timedelta(days=14)).isoformat()
        rows = (
            sb.table("news_sentiment")
            .select("mean_score, headline_count, trade_date")
            .eq("symbol", symbol)
            .gte("trade_date", cutoff)
            .order("trade_date", desc=True)
            .limit(14)
            .execute()
        )
        data = rows.data or []
        if not data:
            return {}
        total = sum(int(r.get("headline_count") or 0) for r in data)
        if total <= 0:
            mean = sum(float(r.get("mean_score") or 0) for r in data) / len(data)
        else:
            mean = sum(
                float(r.get("mean_score") or 0) * int(r.get("headline_count") or 0)
                for r in data
            ) / total
        return {
            "mean_score": round(max(-1.0, min(1.0, mean)), 3),
            "headline_count": total,
            "trade_dates": len(data),
            "latest_date": str(data[0].get("trade_date")),
        }
    except Exception as exc:
        logger.debug("dossier sentiment lookup failed %s: %s", symbol, exc)
        return {}


def _latest_alpha_rank(sb, symbol: str) -> Dict[str, Any]:
    try:
        rows = (
            sb.table("alpha_scores")
            .select("symbol, qlib_rank, qlib_score_raw, trade_date")
            .eq("symbol", symbol)
            .order("trade_date", desc=True)
            .limit(1)
            .execute()
        )
        return (rows.data or [None])[0] or {}
    except Exception as exc:
        logger.debug("dossier alpha_scores lookup failed %s: %s", symbol, exc)
        return {}


def _upcoming_earnings(sb, symbol: str) -> Dict[str, Any]:
    try:
        today = date.today().isoformat()
        horizon = (date.today() + timedelta(days=60)).isoformat()
        rows = (
            sb.table("earnings_predictions")
            .select("symbol, announce_date, beat_prob, confidence")
            .eq("symbol", symbol)
            .gte("announce_date", today)
            .lte("announce_date", horizon)
            .order("announce_date")
            .limit(1)
            .execute()
        )
        return (rows.data or [None])[0] or {}
    except Exception as exc:
        logger.debug("dossier earnings lookup failed %s: %s", symbol, exc)
        return {}


def _spot(symbol: str) -> Optional[float]:
    try:
        from ..data.market import MarketData
        q = MarketData().get_quote(symbol)
        if q and q.ltp:
            return float(q.ltp)
    except Exception as exc:
        logger.debug("dossier spot lookup failed %s: %s", symbol, exc)
    return None


def _is_pro(tier: Tier) -> bool:
    return tier_rank(tier) >= tier_rank(Tier.PRO)


def _is_elite(tier: Tier) -> bool:
    return tier_rank(tier) >= tier_rank(Tier.ELITE)


# ============================================================================
# Engine blocks — each returns the Greek-branded payload for one engine.
# Internal model names never appear in output dicts.
# ============================================================================


def _block_swing(signal_row: Dict[str, Any], is_pro: bool) -> Dict[str, Any]:
    p10 = _safe_float(signal_row.get("tft_p10"))
    p50 = _safe_float(signal_row.get("tft_p50"))
    p90 = _safe_float(signal_row.get("tft_p90"))
    tag = _from_quantiles(p10, p50, p90)
    out: Dict[str, Any] = {
        "engine": public_label("swing_forecast"),
        "role": PUBLIC_MODELS["swing_forecast"].role,
        "available": p50 is not None,
        "direction": tag,
    }
    if is_pro and p50 is not None:
        out["p10"] = round(p10, 4) if p10 is not None else None
        out["p50"] = round(p50, 4)
        out["p90"] = round(p90, 4) if p90 is not None else None
    return out


def _block_rank(alpha_row: Dict[str, Any], is_pro: bool) -> Dict[str, Any]:
    rank = alpha_row.get("qlib_rank")
    out: Dict[str, Any] = {
        "engine": public_label("cross_sectional_ranker"),
        "role": PUBLIC_MODELS["cross_sectional_ranker"].role,
        "available": rank is not None,
    }
    if rank is not None:
        out["rank"] = int(rank)
        out["date"] = str(alpha_row.get("trade_date"))
        if is_pro:
            raw = _safe_float(alpha_row.get("qlib_score_raw"))
            if raw is not None:
                out["score"] = round(raw, 4)
    return out


def _block_sentiment(sent: Dict[str, Any], is_pro: bool) -> Dict[str, Any]:
    score = _safe_float(sent.get("mean_score"))
    out: Dict[str, Any] = {
        "engine": public_label("sentiment_engine"),
        "role": PUBLIC_MODELS["sentiment_engine"].role,
        "available": score is not None,
    }
    if score is not None:
        out["direction"] = (
            "bullish" if score > 0.15
            else "bearish" if score < -0.15
            else "neutral"
        )
        if is_pro:
            out["score"] = score
            out["headline_count"] = int(sent.get("headline_count") or 0)
            out["trade_dates"] = int(sent.get("trade_dates") or 0)
    return out


def _block_regime(regime_row: Dict[str, Any], is_pro: bool) -> Dict[str, Any]:
    name = regime_row.get("regime")
    out: Dict[str, Any] = {
        "engine": public_label("regime_detector"),
        "role": PUBLIC_MODELS["regime_detector"].role,
        "available": name is not None,
    }
    if name:
        out["regime"] = name
        if is_pro:
            out["prob_bull"] = _safe_float(regime_row.get("prob_bull"))
            out["prob_sideways"] = _safe_float(regime_row.get("prob_sideways"))
            out["prob_bear"] = _safe_float(regime_row.get("prob_bear"))
            out["vix"] = _safe_float(regime_row.get("vix"))
    return out


def _block_intraday(signal_row: Dict[str, Any], is_pro: bool) -> Dict[str, Any]:
    # Use lgbm_buy_prob from signals when signal_type == intraday.
    up = _safe_float(signal_row.get("lgbm_buy_prob"))
    is_intraday = (signal_row.get("signal_type") == "intraday")
    out: Dict[str, Any] = {
        "engine": public_label("intraday_forecast"),
        "role": PUBLIC_MODELS["intraday_forecast"].role,
        "available": is_intraday and up is not None,
    }
    if is_intraday and up is not None:
        out["direction"] = _bull_bear_tag(up)
        if is_pro:
            out["up_prob"] = round(up, 3)
    return out


def _block_earnings(er: Dict[str, Any], is_pro: bool) -> Dict[str, Any]:
    bp = _safe_float(er.get("beat_prob"))
    out: Dict[str, Any] = {
        "engine": public_label("earnings_predictor"),
        "role": PUBLIC_MODELS["earnings_predictor"].role,
        "available": bp is not None,
    }
    if bp is not None:
        out["announce_date"] = str(er.get("announce_date"))
        out["direction"] = "bullish" if bp >= 0.70 else "bearish" if bp <= 0.30 else "non_directional"
        if is_pro:
            out["beat_prob"] = bp
            out["confidence"] = er.get("confidence")
    return out


def _overall_tag(engines: List[Dict[str, Any]]) -> str:
    """Consensus direction over the available engines."""
    bull = sum(1 for e in engines if e.get("direction") in {"bullish", "bullish_tilt"})
    bear = sum(1 for e in engines if e.get("direction") in {"bearish", "bearish_tilt"})
    tot = sum(1 for e in engines if e.get("direction"))
    if tot == 0:
        return "neutral"
    if bull / tot >= 0.6:
        return "bullish"
    if bear / tot >= 0.6:
        return "bearish"
    return "mixed"


# ============================================================================
# Route
# ============================================================================


@router.get("/{symbol}")
async def get_dossier(
    symbol: str,
    user: UserTier = Depends(current_user_tier),
) -> Dict[str, Any]:
    """Consolidated per-stock engine dossier.

    Free tier: directional tags only.
    Pro+:     full numeric scores + quantile bands + regime probabilities.
    Elite:    reserved for on-demand Socrates debate trigger + extended metrics.

    The response never mentions TFT, Qlib, LightGBM, FinBERT, Chronos,
    TimesFM, HMM, or any other internal architecture name — only the
    public Greek engine names.
    """
    canon = _canon_symbol(symbol)
    if not canon:
        raise HTTPException(status_code=400, detail="invalid_symbol")

    is_pro = _is_pro(user.tier) or user.is_admin
    is_elite = _is_elite(user.tier) or user.is_admin

    sb = get_supabase_admin()
    signal_row = _latest_signal_row(sb, canon)
    alpha_row = _latest_alpha_rank(sb, canon)
    sent = _latest_sentiment(sb, canon)
    regime_row = _latest_regime(sb)
    earnings = _upcoming_earnings(sb, canon)
    spot = _spot(canon)

    # Alpha Factory — unified per-stock Scores block (best-effort like every
    # other block; deterministic math only, honest-empty when sources are thin).
    scores_block = None
    try:
        from ..services.scanners.stock_scores import scores as stock_scores
        s = stock_scores(canon)
        if s.get("scores"):
            scores_block = {
                "symbol": s.get("symbol", canon),
                "scores": s["scores"],
                "composite": s.get("composite"),
            }
    except Exception as exc:
        logger.debug("dossier scores block failed %s: %s", canon, exc)

    engines = [
        _block_swing(signal_row, is_pro),
        _block_rank(alpha_row, is_pro),
        _block_sentiment(sent, is_pro),
        _block_intraday(signal_row, is_pro),
        _block_earnings(earnings, is_pro),
        _block_regime(regime_row, is_pro),
    ]
    consensus = _overall_tag(engines)

    return {
        "symbol": canon,
        "as_of": datetime.now(IST).isoformat(),
        "spot": round(spot, 2) if spot else None,
        "tier": user.tier.value,
        "consensus": consensus,
        "engines": engines,
        "debate_available": is_elite,
        "scores": scores_block,
        "latest_signal": {
            "id": signal_row.get("id"),
            "direction": signal_row.get("direction"),
            "entry_price": _safe_float(signal_row.get("entry_price")),
            "stop_loss": _safe_float(signal_row.get("stop_loss")),
            "target": _safe_float(signal_row.get("target")),
            "created_at": signal_row.get("created_at"),
            "explanation_text": signal_row.get("explanation_text") if is_pro else None,
        } if signal_row else None,
    }


__all__ = ["router"]
