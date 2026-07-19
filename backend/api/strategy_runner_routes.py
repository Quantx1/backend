"""StrategyRunner API endpoints — PR-FAN.

  GET  /api/strategies/{id}/signals       per-strategy signal feed
  GET  /api/strategies/{id}/positions     open + closed positions
  POST /api/strategies/runner/run-now     manual trigger for the caller's strategies
  GET  /api/strategies/runner/status      admin observability (last 20 ticks)

  GET  /api/strategies/ai-overlay         caller's overlay settings
  PATCH /api/strategies/ai-overlay        update overlay settings
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.database import get_supabase_admin
from ..services.strategy_runner import StrategyRunner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/strategies", tags=["Strategy Runner"])


def _get_user_profile_dep():
    from .app import get_user_profile
    return get_user_profile


# ─────────────────────────────────────────────────────────────────────
# Per-strategy signal feed
# ─────────────────────────────────────────────────────────────────────


@router.get("/{strategy_id}/signals")
async def get_strategy_signals(
    strategy_id: str,
    limit: int = 50,
    profile=Depends(_get_user_profile_dep()),
) -> Dict[str, Any]:
    """Recent signals fired by this strategy for the caller."""
    sb = get_supabase_admin()
    # Verify ownership
    own = (
        sb.table("user_strategies")
        .select("id")
        .eq("id", strategy_id)
        .eq("user_id", profile["id"])
        .limit(1)
        .execute()
    )
    if not own.data:
        raise HTTPException(status_code=404, detail="strategy not found")

    rows = (
        sb.table("signals")
        .select("id, symbol, action, entry_price, stop_loss, target_1, "
                "confidence, status, created_at, market_context")
        .eq("user_id", profile["id"])
        .eq("strategy_id", strategy_id)
        .order("created_at", desc=True)
        .limit(max(1, min(limit, 200)))
        .execute()
    )
    return {"signals": rows.data or [], "count": len(rows.data or [])}


@router.get("/{strategy_id}/positions")
async def get_strategy_positions(
    strategy_id: str,
    status: Optional[str] = None,
    profile=Depends(_get_user_profile_dep()),
) -> Dict[str, Any]:
    """Open + closed positions for this strategy (RLS owner-scoped)."""
    sb = get_supabase_admin()
    own = (
        sb.table("user_strategies")
        .select("id")
        .eq("id", strategy_id)
        .eq("user_id", profile["id"])
        .limit(1)
        .execute()
    )
    if not own.data:
        raise HTTPException(status_code=404, detail="strategy not found")

    q = (
        sb.table("strategy_positions")
        .select("*")
        .eq("user_id", profile["id"])
        .eq("strategy_id", strategy_id)
        .order("entry_at", desc=True)
        .limit(200)
    )
    if status:
        if status not in ("open", "closing", "closed"):
            raise HTTPException(status_code=400, detail="invalid status filter")
        q = q.eq("status", status)
    rows = q.execute()
    return {"positions": rows.data or [], "count": len(rows.data or [])}


# ─────────────────────────────────────────────────────────────────────
# Manual runner trigger
# ─────────────────────────────────────────────────────────────────────


@router.post("/runner/run-now")
async def run_strategies_now(
    profile=Depends(_get_user_profile_dep()),
) -> Dict[str, Any]:
    """Evaluate all of the caller's live strategies right now. Useful for
    the 'run my strategies' button on the dashboard."""
    sb = get_supabase_admin()
    runner = StrategyRunner(sb)
    report = await runner.run_for_user(profile["id"])
    return report.to_dict()


# ─────────────────────────────────────────────────────────────────────
# Admin runner status
# ─────────────────────────────────────────────────────────────────────


@router.get("/runner/status")
async def runner_status(
    limit: int = 20,
    profile=Depends(_get_user_profile_dep()),
) -> Dict[str, Any]:
    """Last N runner ticks. Free for any auth'd user — useful for both
    admins (system health) and end users (when did my last tick run?)."""
    sb = get_supabase_admin()
    rows = (
        sb.table("strategy_runner_runs")
        .select("*")
        .order("tick_at", desc=True)
        .limit(max(1, min(limit, 100)))
        .execute()
    )
    return {"runs": rows.data or [], "count": len(rows.data or [])}


# ─────────────────────────────────────────────────────────────────────
# AI overlay settings
# ─────────────────────────────────────────────────────────────────────


class AIOverlayUpdate(BaseModel):
    regime_gate_enabled: Optional[bool] = None
    blocked_regimes: Optional[List[str]] = Field(default=None, max_length=3)
    vix_overlay_enabled: Optional[bool] = None
    vix_hard_block_threshold: Optional[float] = Field(default=None, ge=10, le=80)
    alpha_rank_filter_enabled: Optional[bool] = None
    alpha_top_k: Optional[int] = Field(default=None, ge=1, le=50)
    max_gross_exposure_pct: Optional[float] = Field(default=None, ge=10, le=100)
    max_per_stock_pct: Optional[float] = Field(default=None, ge=1, le=25)


@router.get("/ai-overlay")
async def get_overlay(
    profile=Depends(_get_user_profile_dep()),
) -> Dict[str, Any]:
    sb = get_supabase_admin()
    rows = (
        sb.table("user_ai_overlay_settings")
        .select("*")
        .eq("user_id", profile["id"])
        .limit(1)
        .execute()
    )
    row = (rows.data or [None])[0]
    if not row:
        # Return defaults if user has never customised
        from ..services.strategy_runner.ai_overlay import AIOverlaySettings
        settings = AIOverlaySettings()
        return {
            "settings": {
                "regime_gate_enabled": settings.regime_gate_enabled,
                "blocked_regimes": settings.blocked_regimes,
                "vix_overlay_enabled": settings.vix_overlay_enabled,
                "vix_hard_block_threshold": settings.vix_hard_block_threshold,
                "alpha_rank_filter_enabled": settings.alpha_rank_filter_enabled,
                "alpha_top_k": settings.alpha_top_k,
                "max_gross_exposure_pct": settings.max_gross_exposure_pct,
                "max_per_stock_pct": settings.max_per_stock_pct,
            },
            "is_default": True,
        }
    return {"settings": row, "is_default": False}


@router.patch("/ai-overlay")
async def update_overlay(
    body: AIOverlayUpdate,
    profile=Depends(_get_user_profile_dep()),
) -> Dict[str, Any]:
    """Update caller's AI overlay settings. Any field omitted preserves
    current value."""
    sb = get_supabase_admin()
    user_id = profile["id"]

    # Load current to preserve unchanged fields
    rows = (
        sb.table("user_ai_overlay_settings")
        .select("*")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    current = (rows.data or [None])[0] or {}
    payload: Dict[str, Any] = {"user_id": user_id}
    update_dict = body.model_dump(exclude_none=True)
    if not update_dict:
        raise HTTPException(status_code=400, detail="no fields to update")
    payload.update(current)
    payload.update(update_dict)
    # blocked_regimes validation
    if "blocked_regimes" in update_dict:
        for r in update_dict["blocked_regimes"]:
            if r not in ("bull", "sideways", "bear"):
                raise HTTPException(
                    status_code=422,
                    detail={"error": "invalid_regime", "got": r,
                            "valid": ["bull", "sideways", "bear"]},
                )

    sb.table("user_ai_overlay_settings").upsert(
        payload, on_conflict="user_id",
    ).execute()
    return {"settings": payload, "updated_fields": list(update_dict.keys())}
