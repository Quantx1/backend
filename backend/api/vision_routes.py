"""
================================================================================
VISION ROUTES — B2 chart-vision analysis (PR 46)
================================================================================
Endpoints:

    POST /api/ai/vision/analyze/{symbol}    — Pro on signal stocks
    POST /api/ai/vision/analyze/any/{symbol} — Elite on any symbol

Both routes gate via feature keys:
    ``finagent_vision`` (Pro)             — limited to stocks appearing in
                                             the user's active signals or
                                             watchlist. Enforced here.
    ``finagent_vision_anywhere`` (Elite)  — no symbol restriction.

Naming note (moat): the feature key retains ``finagent_*`` for backwards
compatibility with the tier matrix; users see "chart vision" in the UI.
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Path

from ..core.database import get_supabase_admin
from ..core.tiers import UserTier
from ..middleware.llm_caps import enforce_llm_cap
from ..middleware.tier_gate import RequireFeature
from ..ai.vision import analyze_chart

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/vision", tags=["ai-vision"])


def _user_touched_symbol(user_id: str, symbol: str) -> bool:
    """Pro gate: symbol must appear in user's active signals (last 30d)
    or watchlist. Admins and Elite bypass this at the router level."""
    sym = symbol.upper().replace(".NS", "")
    sb = get_supabase_admin()
    # Watchlist.
    try:
        wl = (
            sb.table("watchlist")
            .select("id")
            .eq("user_id", user_id)
            .eq("symbol", sym)
            .limit(1)
            .execute()
        )
        if wl.data:
            return True
    except Exception:
        pass
    # Active signals in last 30d.
    try:
        cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
        sig = (
            sb.table("signals")
            .select("id")
            .eq("symbol", sym)
            .gte("created_at", cutoff)
            .limit(1)
            .execute()
        )
        if sig.data:
            return True
    except Exception:
        pass
    return False


@router.post("/analyze/{symbol}")
async def analyze_on_signal_stock(
    symbol: str = Path(..., min_length=1, max_length=32),
    user: UserTier = Depends(RequireFeature("finagent_vision")),
    _cap: UserTier = Depends(enforce_llm_cap("chart_vision")),
) -> Dict[str, Any]:
    """Pro: analyze chart for a stock that's in the user's signal flow
    or watchlist. Non-matching symbols return 403 with an upgrade hint."""
    sym = symbol.upper().replace(".NS", "")
    if not user.is_admin:
        from ..core.tiers import Tier, tier_rank
        if tier_rank(user.tier) < tier_rank(Tier.ELITE):
            if not _user_touched_symbol(user.user_id, sym):
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "vision_symbol_restricted",
                        "reason": "Pro tier supports chart vision only for symbols in your signals or watchlist.",
                        "upgrade_url": "/pricing",
                    },
                )

    analysis = await analyze_chart(sym)
    return asdict(analysis)


@router.post("/analyze/any/{symbol}")
async def analyze_any_stock(
    symbol: str = Path(..., min_length=1, max_length=32),
    user: UserTier = Depends(RequireFeature("finagent_vision_anywhere")),
    _cap: UserTier = Depends(enforce_llm_cap("chart_vision")),
) -> Dict[str, Any]:
    """Elite: analyze chart for any symbol — no restriction beyond a
    valid NSE ticker. Unknown tickers return ``available=false``."""
    sym = symbol.upper().replace(".NS", "")
    analysis = await analyze_chart(sym)
    return asdict(analysis)


__all__ = ["router"]
