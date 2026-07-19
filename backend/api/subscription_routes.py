"""
Subscription API routes — public plan catalog.

The plan catalog is unauthenticated so the marketing pricing page can
render without a session. Per-user subscription status lives in
``payment_routes.py`` (``/api/payments/subscription-status``).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Subscription"])


def _get_supabase_admin():
    from .app import get_supabase_admin
    return get_supabase_admin()


@router.get("/api/plans")
async def get_plans():
    """Get all active subscription plans, ordered by ``sort_order``."""
    try:
        supabase = _get_supabase_admin()
        result = (
            supabase.table("subscription_plans")
            .select("*")
            .eq("is_active", True)
            .order("sort_order")
            .execute()
        )
        return {"plans": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
