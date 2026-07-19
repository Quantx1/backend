"""
IPO routes (Phase 4, 2026-07-12).

    GET /api/ipo/calendar   — open + upcoming IPOs (NSE primary-market feed)

Public data (no auth) — the primary-market calendar is public NSE information.
Honest-empty when NSE is unreachable. No GMP (grey-market premium) — unofficial
data we do not source (see services/ipo/ipo_calendar).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ipo", tags=["ipo"])


@router.get("/calendar")
async def ipo_calendar():
    """Open + upcoming IPOs with price band, dates, status and (for open
    issues) the live subscription multiple. Honest-empty when NSE is down."""
    import asyncio

    from ..services.ipo.ipo_calendar import fetch_ipo_calendar

    data = await asyncio.to_thread(fetch_ipo_calendar)
    return {"success": True, **data}
