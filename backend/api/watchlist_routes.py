"""
Watchlist API routes — list, add, remove, alert thresholds.

CRUD over the ``watchlist`` table. The alert-threshold endpoint
re-arms the price-alert debounce columns whenever a threshold
changes, so a user adjusting their alert level gets a fresh
notification on the next crossing.

The live-price evaluator that consumes these thresholds lives in
``watchlist_live_routes.py`` — keep them separate.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from ..core.tiers import Tier, UserTier
from ..middleware.tier_gate import current_user_tier
from ..schemas import WatchlistAdd, WatchlistUpdate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Watchlist"])

# Free-tier watchlist size cap. Mirrors FREE_SYMBOL_CAP in
# watchlist_live_routes.py — keep the constants in sync (or extract to
# core.tiers if a third callsite shows up).
FREE_WATCHLIST_CAP: int = 5


def _get_supabase_admin():
    from .app import get_supabase_admin
    return get_supabase_admin()


def _get_current_user_dep():
    from .app import get_current_user
    return get_current_user


@router.get("/api/watchlist")
async def get_watchlist(user=Depends(_get_current_user_dep())):
    """Get the current user's watchlist, newest entries first."""
    try:
        supabase = _get_supabase_admin()
        result = (
            supabase.table("watchlist")
            .select("*")
            .eq("user_id", user.id)
            .order("added_at", desc=True)
            .limit(100)
            .execute()
        )
        return {"watchlist": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/watchlist")
async def add_to_watchlist(
    data: WatchlistAdd,
    user=Depends(_get_current_user_dep()),
    tier: UserTier = Depends(current_user_tier),
):
    """Add a stock to the watchlist. Returns ``success: false`` (not
    a 4xx) on duplicate so the frontend can render a soft toast.

    PR-U 2026-05-28: Free users are capped at ``FREE_WATCHLIST_CAP``
    symbols. Hitting the cap returns 402 with a structured payload so
    the frontend can render an upgrade CTA. Admins bypass.
    """
    supabase = _get_supabase_admin()

    # Free-tier cap check. Count BEFORE inserting so we get an exact
    # number rather than relying on the insert to fail.
    if tier.tier == Tier.FREE and not tier.is_admin:
        existing = (
            supabase.table("watchlist")
            .select("id", count="exact")
            .eq("user_id", user.id)
            .execute()
        )
        current = int(getattr(existing, "count", 0) or 0)
        if current >= FREE_WATCHLIST_CAP:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "watchlist_cap_reached",
                    "message": f"Free watchlist is limited to {FREE_WATCHLIST_CAP} symbols. Upgrade to Pro for unlimited.",
                    "current_tier": tier.tier.value,
                    "required_tier": Tier.PRO.value,
                    "feature": "watchlist_unlimited",
                    "current_count": current,
                    "cap": FREE_WATCHLIST_CAP,
                    "upgrade_url": "/pricing",
                },
            )

    try:
        supabase.table("watchlist").insert({
            "user_id": user.id,
            "symbol": data.symbol.upper(),
            "segment": data.segment.value,
            "alert_price_above": data.alert_price_above,
            "alert_price_below": data.alert_price_below,
            "alert_enabled": data.alert_price_above is not None or data.alert_price_below is not None,
        }).execute()
        return {"success": True}
    except Exception as e:
        logger.warning(f"Watchlist add failed: {e}")
        return {"success": False, "message": "Already in watchlist"}


@router.delete("/api/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str, user=Depends(_get_current_user_dep())):
    """Remove a stock from the watchlist by symbol."""
    try:
        supabase = _get_supabase_admin()
        (
            supabase.table("watchlist")
            .delete()
            .eq("user_id", user.id)
            .eq("symbol", symbol.upper())
            .execute()
        )
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/watchlist/{symbol}/alerts")
async def update_watchlist_alerts(
    symbol: str,
    data: WatchlistUpdate,
    user=Depends(_get_current_user_dep()),
):
    """Partial update of a watchlist row's alert thresholds.

    Resets the debounce columns whenever a threshold value changes so
    the price-alert scanner re-arms and fires fresh on the next
    crossing. Matches by ``(user_id, symbol)`` — no id roundtrip.
    """
    sym = symbol.upper()
    sb = _get_supabase_admin()

    try:
        existing = (
            sb.table("watchlist")
            .select("alert_price_above, alert_price_below, alert_enabled")
            .eq("user_id", user.id)
            .eq("symbol", sym)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.warning("watchlist alerts lookup failed sym=%s: %s", sym, exc)
        raise HTTPException(status_code=500, detail="lookup_failed")

    row = (existing.data or [None])[0]
    if row is None:
        raise HTTPException(status_code=404, detail="not_in_watchlist")

    update: Dict[str, Any] = {}
    threshold_changed = False
    if data.alert_price_above is not None or "alert_price_above" in data.model_fields_set:
        new_above = data.alert_price_above
        if new_above != row.get("alert_price_above"):
            threshold_changed = True
        update["alert_price_above"] = new_above
    if data.alert_price_below is not None or "alert_price_below" in data.model_fields_set:
        new_below = data.alert_price_below
        if new_below != row.get("alert_price_below"):
            threshold_changed = True
        update["alert_price_below"] = new_below
    if data.alert_enabled is not None:
        update["alert_enabled"] = bool(data.alert_enabled)
    if data.notes is not None:
        update["notes"] = data.notes[:500]

    if not update:
        return {"success": True, "updated": False}

    if threshold_changed:
        update["alert_last_fired_at"] = None
        update["alert_last_fired_direction"] = None

    try:
        sb.table("watchlist").update(update).eq("user_id", user.id).eq("symbol", sym).execute()
    except Exception as exc:
        logger.error("watchlist alerts update failed sym=%s: %s", sym, exc)
        raise HTTPException(status_code=500, detail="update_failed")

    return {"success": True, "updated": True, "rearmed": threshold_changed}
