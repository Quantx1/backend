"""AutoPilot stream toggle endpoints — PR-AS.

  GET   /api/autopilot/streams                    list current state
  PATCH /api/autopilot/streams/{stream}           toggle + set allocation

Pro/Elite tier required (Free tier can't AutoPilot anyway).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.database import get_supabase_admin
from ..services.autopilot.streams import (
    BUILTIN_STREAMS,
    is_builtin,
    list_streams_for_user,
    total_allocated_pct,
    upsert_stream,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/autopilot", tags=["AutoPilot Streams"])


def _get_user_profile_dep():
    from .app import get_user_profile
    return get_user_profile


_ALLOWED_TIERS = {"pro", "elite"}


def _user_tier(profile: Dict[str, Any]) -> str:
    return str(profile.get("subscription_tier") or profile.get("tier") or "free").lower()


class StreamToggleRequest(BaseModel):
    enabled: bool
    allocated_capital_pct: float = Field(ge=0, le=100)
    # Only for stream="user_strategy"
    user_strategy_id: Optional[str] = Field(default=None, max_length=64)


@router.get("/streams")
async def list_streams(profile=Depends(_get_user_profile_dep())) -> Dict[str, Any]:
    """Return every stream's current state + total allocation."""
    tier = _user_tier(profile)
    sb = get_supabase_admin()
    states = list_streams_for_user(sb, profile["id"])
    return {
        "streams": [
            {
                "stream": s.stream,
                "user_strategy_id": s.user_strategy_id,
                "enabled": s.enabled,
                "allocated_capital_pct": s.allocated_capital_pct,
                "is_prod": s.is_prod,
                "last_enabled_at": s.last_enabled_at,
                "last_disabled_at": s.last_disabled_at,
            }
            for s in states
        ],
        "total_allocated_pct": total_allocated_pct(states),
        "tier": tier,
        "tier_allows_autopilot": tier in _ALLOWED_TIERS,
        "builtin_streams": list(BUILTIN_STREAMS),
    }


@router.patch("/streams/{stream}")
async def update_stream(
    stream: str,
    body: StreamToggleRequest,
    profile=Depends(_get_user_profile_dep()),
) -> Dict[str, Any]:
    """Toggle a stream + set its capital allocation. Validates the
    cross-stream sum stays ≤ 100% of the user's capital."""
    tier = _user_tier(profile)
    if body.enabled and tier not in _ALLOWED_TIERS:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "tier_required",
                "message": "AutoPilot requires Pro or Elite tier.",
                "your_tier": tier,
            },
        )

    # Sanity: stream name must be one of the built-ins OR "user_strategy"
    if stream != "user_strategy" and not is_builtin(stream):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unknown_stream",
                "got": stream,
                "valid": list(BUILTIN_STREAMS) + ["user_strategy"],
            },
        )
    if stream == "user_strategy" and not body.user_strategy_id:
        raise HTTPException(
            status_code=400,
            detail={"error": "user_strategy_id_required",
                    "message": "stream='user_strategy' requires user_strategy_id in body"},
        )

    sb = get_supabase_admin()
    try:
        state = upsert_stream(
            sb,
            user_id=profile["id"],
            stream=stream,
            user_strategy_id=body.user_strategy_id,
            enabled=body.enabled,
            allocated_capital_pct=body.allocated_capital_pct,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_allocation",
                                                     "message": str(exc)})

    # Return the full set so the frontend can re-render totals in one shot
    states = list_streams_for_user(sb, profile["id"])
    return {
        "updated_stream": {
            "stream": state.stream,
            "user_strategy_id": state.user_strategy_id,
            "enabled": state.enabled,
            "allocated_capital_pct": state.allocated_capital_pct,
        },
        "streams": [
            {
                "stream": s.stream,
                "user_strategy_id": s.user_strategy_id,
                "enabled": s.enabled,
                "allocated_capital_pct": s.allocated_capital_pct,
                "is_prod": s.is_prod,
            }
            for s in states
        ],
        "total_allocated_pct": total_allocated_pct(states),
    }
