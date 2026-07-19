"""
Positions API routes — list, get, update SL/target, close.

The ``close`` endpoint delegates to ``trades_routes.close_trade_record``
since the trade row is the source of truth for P&L. The position is
deactivated as a side effect of closing the trade.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import CloseTrade, PositionUpdate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Positions"])


def _get_supabase_admin():
    from .app import get_supabase_admin
    return get_supabase_admin()


def _get_current_user_dep():
    from .app import get_current_user
    return get_current_user


@router.get("/api/positions/open")
async def get_open_positions(user=Depends(_get_current_user_dep())):
    """Get active positions. The canonical endpoint — frontend uses this
    exclusively (see frontend/lib/api.ts:396)."""
    try:
        supabase = _get_supabase_admin()
        result = (
            supabase.table("positions")
            .select("*")
            .eq("user_id", user.id)
            .eq("is_active", True)
            .limit(100)
            .execute()
        )
        return {"positions": result.data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Pre-consolidation alias. Frontend uses /api/positions/open exclusively;
# this stays functional for any external consumer but is hidden from OpenAPI
# so new clients pick the canonical path. P1-4 consolidation 2026-05-08.
@router.get("/api/positions", include_in_schema=False)
async def get_positions(user=Depends(_get_current_user_dep())):
    """[DEPRECATED] Use /api/positions/open. Kept for back-compat."""
    return await get_open_positions(user)


@router.get("/api/positions/{position_id}")
async def get_position(position_id: str, user=Depends(_get_current_user_dep())):
    """Get a single position."""
    try:
        supabase = _get_supabase_admin()
        result = (
            supabase.table("positions")
            .select("*")
            .eq("id", position_id)
            .eq("user_id", user.id)
            .single()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Position not found")
        return result.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/positions/{position_id}")
async def update_position(
    position_id: str,
    data: PositionUpdate,
    user=Depends(_get_current_user_dep()),
):
    """Update position stop_loss / target. Mirrors the change to the
    parent trade row so the SL/TP stay consistent across views."""
    try:
        supabase = _get_supabase_admin()

        update_data = {}
        if data.stop_loss:
            update_data["stop_loss"] = data.stop_loss
        if data.target:
            update_data["target"] = data.target

        if update_data:
            (
                supabase.table("positions")
                .update(update_data)
                .eq("id", position_id)
                .eq("user_id", user.id)
                .execute()
            )
            # Also update the parent trade row.
            position = (
                supabase.table("positions")
                .select("trade_id")
                .eq("id", position_id)
                .eq("user_id", user.id)
                .single()
                .execute()
            )
            trade_id = position.data.get("trade_id") if position.data else None
            if trade_id:
                (
                    supabase.table("trades")
                    .update(update_data)
                    .eq("id", trade_id)
                    .eq("user_id", user.id)
                    .execute()
                )

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/positions/{position_id}/close")
async def close_position(
    position_id: str,
    data: CloseTrade = CloseTrade(),
    user=Depends(_get_current_user_dep()),
):
    """Close an open position by position id.

    Delegates to ``trades_routes.close_trade_record`` so the close
    pipeline (P&L calc, broker exec for live, observability hooks) is
    identical whether the user closes via the trades view or the
    positions view.
    """
    try:
        supabase = _get_supabase_admin()
        position = (
            supabase.table("positions")
            .select("trade_id")
            .eq("id", position_id)
            .eq("user_id", user.id)
            .single()
            .execute()
        )
        if not position.data or not position.data.get("trade_id"):
            raise HTTPException(status_code=404, detail="Position not found")

        trade_id = position.data["trade_id"]
        from .trades_routes import close_trade_record
        return await close_trade_record(trade_id, data, user.id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
