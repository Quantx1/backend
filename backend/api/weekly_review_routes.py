"""
================================================================================
WEEKLY REVIEW ROUTES — N10 (PR 38)
================================================================================
Pro+ users get a Sunday-generated personal review. This router exposes:

    GET  /api/weekly-review/latest      — this week's (or most recent) review
    GET  /api/weekly-review/history     — last N reviews (default 8)
    POST /api/weekly-review/generate    — on-demand regenerate (admin / self)

Scheduler: ``weekly_review_generate`` runs every Sunday at 08:00 IST and
fans out the generator for every Pro+ user with
``onboarding_completed=true``. On-demand POST is tier-gated the same way.
================================================================================
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..core.database import get_supabase_admin
from ..core.tiers import UserTier
from ..middleware.tier_gate import RequireFeature
from ..data.market_calendar import IST

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/weekly-review", tags=["weekly-review"])


# ============================================================================
# Pydantic
# ============================================================================


class ReviewRow(BaseModel):
    week_of: str
    content_markdown: str
    week_return_pct: Optional[float]
    nifty_return_pct: Optional[float]
    generated_at: str


# ============================================================================
# Helpers
# ============================================================================


def _monday_of_week(today: Optional[date] = None) -> date:
    today = today or datetime.now(IST).date()
    return today - timedelta(days=today.weekday())


# ============================================================================
# Routes
# ============================================================================


@router.get("/latest", response_model=ReviewRow)
async def get_latest(
    user: UserTier = Depends(RequireFeature("weekly_review")),
) -> ReviewRow:
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("user_weekly_reviews")
            .select("week_of, content_markdown, week_return_pct, nifty_return_pct, generated_at")
            .eq("user_id", user.user_id)
            .order("week_of", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.error("weekly review lookup failed: %s", exc)
        raise HTTPException(status_code=500, detail="lookup_failed")
    row = (rows.data or [None])[0]
    if row is None:
        raise HTTPException(status_code=404, detail="no_review_yet")
    return ReviewRow(
        week_of=str(row["week_of"]),
        content_markdown=str(row["content_markdown"] or ""),
        week_return_pct=row.get("week_return_pct"),
        nifty_return_pct=row.get("nifty_return_pct"),
        generated_at=str(row["generated_at"]),
    )


@router.get("/history", response_model=List[ReviewRow])
async def get_history(
    limit: int = 8,
    user: UserTier = Depends(RequireFeature("weekly_review")),
) -> List[ReviewRow]:
    limit = max(1, min(52, int(limit)))
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("user_weekly_reviews")
            .select("week_of, content_markdown, week_return_pct, nifty_return_pct, generated_at")
            .eq("user_id", user.user_id)
            .order("week_of", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception:
        return []
    return [
        ReviewRow(
            week_of=str(r["week_of"]),
            content_markdown=str(r.get("content_markdown") or ""),
            week_return_pct=r.get("week_return_pct"),
            nifty_return_pct=r.get("nifty_return_pct"),
            generated_at=str(r["generated_at"]),
        )
        for r in rows.data or []
    ]


@router.post("/generate", response_model=ReviewRow)
async def generate_now(
    user: UserTier = Depends(RequireFeature("weekly_review")),
) -> ReviewRow:
    """On-demand regenerate this week's review for the caller. Useful
    mid-week (before the Sunday scheduler fires) or as an admin fallback."""
    try:
        from ..ai.weekly_review import generate_review_for_user
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"generator_unavailable: {exc}")

    sb = get_supabase_admin()
    review = await generate_review_for_user(
        user_id=user.user_id,
        supabase_client=sb,
        week_of=_monday_of_week(),
        persist=True,
    )
    return ReviewRow(
        week_of=review.week_of,
        content_markdown=review.content_markdown,
        week_return_pct=review.week_return_pct,
        nifty_return_pct=review.nifty_return_pct,
        generated_at=datetime.utcnow().isoformat(),
    )


__all__ = ["router"]
