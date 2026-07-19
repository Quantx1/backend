"""
Notifications API routes — list, mark-read, mark-all-read.

Read/write surface over the ``notifications`` table. Each row is
user-scoped, so every endpoint filters by ``user_id`` from the JWT.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Notifications"])


def _get_supabase_admin():
    from .app import get_supabase_admin
    return get_supabase_admin()


def _get_current_user_dep():
    from .app import get_current_user
    return get_current_user


@router.get("/api/notifications")
async def get_notifications(
    unread_only: bool = False,
    limit: int = Query(default=50, ge=1, le=200),
    user=Depends(_get_current_user_dep()),
):
    """Get user notifications, newest first."""
    try:
        supabase = _get_supabase_admin()
        query = supabase.table("notifications").select("*").eq("user_id", user.id)
        if unread_only:
            query = query.eq("is_read", False)
        result = query.order("created_at", desc=True).limit(limit).execute()
        return {"notifications": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    user=Depends(_get_current_user_dep()),
):
    """Mark a single notification as read."""
    try:
        supabase = _get_supabase_admin()
        (
            supabase.table("notifications")
            .update({"is_read": True, "read_at": datetime.utcnow().isoformat()})
            .eq("id", notification_id)
            .eq("user_id", user.id)
            .execute()
        )
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/notifications/read-all")
async def mark_all_notifications_read(user=Depends(_get_current_user_dep())):
    """Mark all unread notifications as read for the current user."""
    try:
        supabase = _get_supabase_admin()
        (
            supabase.table("notifications")
            .update({"is_read": True, "read_at": datetime.utcnow().isoformat()})
            .eq("user_id", user.id)
            .eq("is_read", False)
            .execute()
        )
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
