"""
Managed-mode routes — the beginner ("AI runs my account") surface.

    GET /api/managed/overview — single aggregate behind the managed Home:
        health score, money, risk level, AutoPilot state + plain-English
        activity, regime, drawdown. Deterministic, zero LLM, honest-null.

NOT tier-gated: Free/Pro users get ``autopilot.available = false`` so the
UI shows an honest upsell instead of a 403. The auto-trader's own control
routes stay Elite-gated.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/managed", tags=["Managed"])


def _get_current_user_dep():
    """Lazy import to avoid circular dep at module load (same pattern as
    user_routes)."""
    from .app import get_current_user
    return get_current_user


@router.get("/overview")
async def get_managed_overview(
    user=Depends(_get_current_user_dep()),
) -> Dict[str, Any]:
    from ..services.portfolio.managed_overview import build_overview
    uid = getattr(user, "id", None) or (
        user.get("id") if isinstance(user, dict) else None
    )
    return await asyncio.to_thread(build_overview, str(uid))


class PaperAutopilotToggle(BaseModel):
    enabled: bool


@router.post("/paper-autopilot")
async def toggle_paper_autopilot(
    body: PaperAutopilotToggle,
    user=Depends(_get_current_user_dep()),
) -> Dict[str, Any]:
    """Pricing v2 (2026-06-12) — Paper AutoPilot opt-in, ANY tier.

    Turns the daily AutoPilot run on with ``auto_trader_config.mode='paper'``:
    the AI trades virtual positions only (no broker, no suitability gate).
    Free's whole AutoPilot experience; Pro/Elite can use it to trial before
    going live. The 'paper' mode flag persists across tier upgrades, so
    upgrading never silently flips a user to real money — going live is an
    explicit action on /autopilot.
    """
    uid = str(getattr(user, "id", None) or (
        user.get("id") if isinstance(user, dict) else ""
    ))

    def _write() -> Dict[str, Any]:
        from ..core.database import get_supabase_admin
        sb = get_supabase_admin()
        rows = (
            sb.table("user_profiles")
            .select("auto_trader_config")
            .eq("id", uid)
            .limit(1)
            .execute()
            .data
            or []
        )
        cfg = (rows[0].get("auto_trader_config") if rows else None) or {}
        update: Dict[str, Any] = {
            "auto_trader_enabled": bool(body.enabled),
            "auto_trader_config": {**cfg, "mode": "paper"},
        }
        sb.table("user_profiles").update(update).eq("id", uid).execute()
        return {"enabled": bool(body.enabled), "mode": "paper", "ok": True}

    return await asyncio.to_thread(_write)


__all__ = ["router"]
