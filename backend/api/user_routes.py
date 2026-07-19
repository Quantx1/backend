"""
User profile API routes — read + update profile, tier resolution,
UI preferences, trading stats rollup.

Pure CRUD over ``user_profiles`` plus a couple of derived rollups
(stats, tier-feature map). Auth via ``Depends(get_current_user)`` is
imported lazily from ``app.py`` to avoid circular dep at module load.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import ProfileUpdate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["User"])


def _get_supabase_admin():
    """Lazy import of the Supabase service-role client."""
    from .app import get_supabase_admin
    return get_supabase_admin()


def _get_current_user_dep():
    """Dependency factory — defers ``get_current_user`` import."""
    from .app import get_current_user
    return get_current_user


def _get_user_profile_dep():
    """Dependency factory — defers ``get_user_profile`` import."""
    from .app import get_user_profile
    return get_user_profile


@router.get("/api/user/profile")
async def get_profile(profile=Depends(_get_user_profile_dep())):
    """Get current user profile."""
    return profile


@router.get("/api/user/tier")
async def get_user_tier(user=Depends(_get_current_user_dep())):
    """Return the user's tier + per-feature access map + Copilot credit cap.

    The frontend consumes this once per session to pre-paint tier-gated
    UI (locks on Pro/Elite features, Upgrade CTAs, Copilot credit budget).
    """
    from ..core.tiers import feature_access_map, resolve_user_tier
    from ..middleware.tier_gate import copilot_daily_cap

    ut = resolve_user_tier(str(user.id))
    return {
        "user_id": ut.user_id,
        "tier": ut.tier.value,
        "is_admin": ut.is_admin,
        "features": feature_access_map(ut.tier),
        "copilot_daily_cap": copilot_daily_cap(ut.tier),
    }


@router.put("/api/user/profile")
async def update_profile(data: ProfileUpdate, user=Depends(_get_current_user_dep())):
    """Update user profile."""
    try:
        supabase = _get_supabase_admin()
        update_data = {k: v for k, v in data.model_dump().items() if v is not None}
        update_data["updated_at"] = datetime.utcnow().isoformat()
        result = (
            supabase.table("user_profiles")
            .update(update_data)
            .eq("id", user.id)
            .execute()
        )
        return {"success": True, "data": result.data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/user/ui-preferences")
async def get_ui_preferences(user=Depends(_get_current_user_dep())):
    """PR 123 — read user's cross-device UI preferences.

    First consumer is watchlist alert preset pins. Returns ``{}`` for
    rows with no prefs set (default JSONB value); the frontend merges
    against its own session defaults.
    """
    sb = _get_supabase_admin()
    rows = (
        sb.table("user_profiles")
        .select("ui_preferences")
        .eq("id", user.id)
        .limit(1)
        .execute()
    )
    if not rows.data:
        return {"ui_preferences": {}}
    return {"ui_preferences": rows.data[0].get("ui_preferences") or {}}


@router.put("/api/user/ui-preferences")
async def update_ui_preferences(payload: dict, user=Depends(_get_current_user_dep())):
    """PR 123 — replace the user's UI preferences blob.

    Whole-document write keeps the surface trivial; the blob is small
    (≤4KB realistically) and we don't need server-side merge semantics.
    Validate the top-level shape so a client bug can't pollute storage
    with arbitrary keys.
    """
    ALLOWED_KEYS = {"watchlist_preset_pins"}
    # Dual-mode 2026-06-12 — `ui_mode` is the user's experience mode:
    # "managed" (beginner: AI runs the account, simple shell) or "pro"
    # (full trading terminal). String-valued, unlike the dict-valued keys.
    STRING_KEYS = {"ui_mode": {"managed", "pro"}}
    prefs = payload.get("ui_preferences")
    if not isinstance(prefs, dict):
        raise HTTPException(status_code=422, detail="ui_preferences must be an object")
    cleaned: Dict[str, Any] = {
        k: v for k, v in prefs.items() if k in ALLOWED_KEYS and isinstance(v, dict)
    }
    for key, valid in STRING_KEYS.items():
        if key in prefs and isinstance(prefs[key], str) and prefs[key] in valid:
            cleaned[key] = prefs[key]
    # PR 123 — watchlist_preset_pins values must be a known preset id.
    if "watchlist_preset_pins" in cleaned:
        valid_ids = {"pct5", "pct10", "pct5_breakout", "pct5_drop", "atr1", "atr2"}
        cleaned["watchlist_preset_pins"] = {
            sym.upper(): pid
            for sym, pid in cleaned["watchlist_preset_pins"].items()
            if isinstance(sym, str) and isinstance(pid, str) and pid in valid_ids
        }
    sb = _get_supabase_admin()
    sb.table("user_profiles").update({
        "ui_preferences": cleaned,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", user.id).execute()
    return {"success": True, "ui_preferences": cleaned}


@router.get("/api/user/stats")
async def get_user_stats(user=Depends(_get_current_user_dep())):
    """Get user trading statistics — capital, P&L breakdown, win-rate, open positions."""
    try:
        supabase = _get_supabase_admin()

        profile = (
            supabase.table("user_profiles")
            .select("*")
            .eq("id", user.id)
            .single()
            .execute()
        )
        positions = (
            supabase.table("positions")
            .select("*")
            .eq("user_id", user.id)
            .eq("is_active", True)
            .limit(100)
            .execute()
        )

        today = date.today().isoformat()
        today_trades = (
            supabase.table("trades")
            .select("net_pnl")
            .eq("user_id", user.id)
            .eq("status", "closed")
            .gte("closed_at", today)
            .limit(200)
            .execute()
        )

        week_start = (date.today() - timedelta(days=date.today().weekday())).isoformat()
        week_trades = (
            supabase.table("trades")
            .select("net_pnl, status")
            .eq("user_id", user.id)
            .gte("created_at", week_start)
            .limit(500)
            .execute()
        )

        p = profile.data
        pos = positions.data or []

        unrealized_pnl = sum(float(pos_item.get("unrealized_pnl", 0) or 0) for pos_item in pos)
        today_pnl = sum(float(t.get("net_pnl", 0) or 0) for t in today_trades.data or [])
        week_pnl = sum(
            float(t.get("net_pnl", 0) or 0)
            for t in week_trades.data or []
            if t.get("status") == "closed"
        )

        win_rate = (p["winning_trades"] / p["total_trades"] * 100) if p["total_trades"] > 0 else 0

        return {
            "capital": p["capital"],
            "total_pnl": p["total_pnl"],
            "total_trades": p["total_trades"],
            "winning_trades": p["winning_trades"],
            "win_rate": round(win_rate, 2),
            "open_positions": len(pos),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "today_pnl": round(today_pnl, 2),
            "week_pnl": round(week_pnl, 2),
            "subscription_status": p["subscription_status"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
