"""
================================================================================
EARNINGS ROUTES — F9 Earnings calendar (PR 31)
================================================================================
HTTP surface for ``/earnings`` — upcoming announcements per symbol, read
from the cached ``earnings_predictions`` table hydrated by the calendar
scan. The ML surprise-predictor (EarningsScout) has been removed; rows
carry calendar dates and any cached ``beat_prob`` (typically null).

Tier split (Step 1 §C4):
    * ``earnings_basic``    (Pro)   — calendar + cached prediction row

Endpoints:
    GET  /api/earnings/upcoming?days=14         — calendar list (Pro)
    GET  /api/earnings/symbol/{symbol}          — per-symbol detail (Pro)
================================================================================
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from ..core.database import get_supabase_admin
from ..core.tiers import UserTier
from ..middleware.tier_gate import RequireFeature
from ..ai.earnings import fetch_upcoming_earnings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/earnings", tags=["earnings"])


# ============================================================================
# Pydantic
# ============================================================================


class UpcomingRow(BaseModel):
    symbol: str
    announce_date: str
    beat_prob: Optional[float]
    confidence: Optional[str]
    thesis: Optional[str]          # one-line direction label (Pro)
    direction: Optional[str]       # 'bullish' / 'bearish' / 'non_directional'
    evidence: Dict[str, Any] = {}


# ============================================================================
# Helpers
# ============================================================================


def _direction_and_thesis(beat_prob: Optional[float]) -> tuple[Optional[str], Optional[str]]:
    if beat_prob is None:
        return None, None
    if beat_prob >= 0.70:
        return "bullish", f"{round(beat_prob * 100)}% beat probability — directional long bias"
    if beat_prob <= 0.30:
        return "bearish", f"{round(beat_prob * 100)}% beat probability — directional short bias"
    return (
        "non_directional",
        f"{round(beat_prob * 100)}% beat — uncertain, volatility-expansion setup",
    )


# ============================================================================
# Routes
# ============================================================================


@router.get("/upcoming", response_model=List[UpcomingRow])
async def get_upcoming(
    days: int = Query(14, ge=1, le=60),
    user: UserTier = Depends(RequireFeature("earnings_basic")),
) -> List[UpcomingRow]:
    """Upcoming earnings in the next ``days`` window (Pro).

    Reads ``earnings_predictions`` table — the daily 17:00 IST scheduler
    job hydrates it. When the table is fresh, returns rows ordered by
    announce_date ascending.
    """
    rows = fetch_upcoming_earnings(days=days)
    out: List[UpcomingRow] = []
    for r in rows:
        direction, thesis = _direction_and_thesis(r.beat_prob)
        out.append(UpcomingRow(
            symbol=r.symbol,
            announce_date=r.announce_date,
            beat_prob=r.beat_prob,
            confidence=r.confidence,
            direction=direction,
            thesis=thesis,
            evidence=r.evidence,
        ))
    return out


@router.get("/symbol/{symbol}")
async def get_symbol_detail(
    symbol: str,
    user: UserTier = Depends(RequireFeature("earnings_basic")),
) -> Dict[str, Any]:
    """Per-symbol detail — cached ``earnings_predictions`` row, if any.

    Returns the calendar row (announce_date + any cached beat_prob, which
    is typically null since the ML surprise predictor was removed). When
    no upcoming row exists, returns a minimal shape with a null beat_prob
    rather than 404, so the UI can render the calendar state cleanly.
    """
    sb = get_supabase_admin()
    sym = symbol.upper()
    today = date.today()
    horizon = (today + timedelta(days=60)).isoformat()
    try:
        rows = (
            sb.table("earnings_predictions")
            .select("symbol, announce_date, beat_prob, confidence, evidence, computed_at")
            .eq("symbol", sym)
            .gte("announce_date", today.isoformat())
            .lte("announce_date", horizon)
            .order("announce_date", desc=False)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.warning("earnings detail DB read failed %s: %s", sym, exc)
        rows = None

    row = (rows.data or [None])[0] if rows else None
    if row is None:
        return {
            "symbol": sym,
            "announce_date": None,
            "beat_prob": None,
            "direction": None,
            "thesis": None,
        }

    direction, thesis = _direction_and_thesis(
        float(row["beat_prob"]) if row.get("beat_prob") is not None else None,
    )
    return {
        **row,
        "direction": direction,
        "thesis": thesis,
    }


__all__ = ["router"]
