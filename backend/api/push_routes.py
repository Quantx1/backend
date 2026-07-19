"""
Web Push subscription API routes — VAPID key, subscribe, unsubscribe.

Persists browser push subscriptions to ``push_subscriptions``. The
worker that fans out notifications reads from this table; expired
endpoints get pruned by ``PushService`` on 410 responses.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Push"])


def _get_supabase_admin():
    from .app import get_supabase_admin
    return get_supabase_admin()


def _get_current_user_dep():
    from .app import get_current_user
    return get_current_user


def _get_settings():
    from ..core.config import settings
    return settings


@router.get("/api/push/vapid-key")
async def get_vapid_key():
    """Return the VAPID public key for the frontend's `pushManager.subscribe`."""
    settings = _get_settings()
    if not settings.VAPID_PUBLIC_KEY:
        raise HTTPException(status_code=503, detail="Web Push not configured")
    return {"public_key": settings.VAPID_PUBLIC_KEY}


@router.post("/api/push/subscribe")
async def push_subscribe(request: Request, user=Depends(_get_current_user_dep())):
    """Save a push subscription for the current user. Idempotent on
    ``(user_id, endpoint)`` so repeat saves from the same browser
    update keys/user-agent rather than duplicating.
    """
    try:
        data = await request.json()
        endpoint = data.get("endpoint")
        keys = data.get("keys", {})
        p256dh = keys.get("p256dh")
        auth = keys.get("auth")

        if not endpoint or not p256dh or not auth:
            raise HTTPException(status_code=400, detail="Missing subscription fields")

        supabase = _get_supabase_admin()
        (
            supabase.table("push_subscriptions")
            .upsert(
                {
                    "user_id": user.id,
                    "endpoint": endpoint,
                    "p256dh": p256dh,
                    "auth": auth,
                    "user_agent": request.headers.get("user-agent", ""),
                },
                on_conflict="user_id,endpoint",
            )
            .execute()
        )
        return {"success": True, "message": "Push subscription saved"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/push/unsubscribe")
async def push_unsubscribe(request: Request, user=Depends(_get_current_user_dep())):
    """Remove a push subscription. Matches by ``(user_id, endpoint)`` —
    we never delete by id since the client doesn't know it.
    """
    try:
        data = await request.json()
        endpoint = data.get("endpoint")
        if not endpoint:
            raise HTTPException(status_code=400, detail="Missing endpoint")

        supabase = _get_supabase_admin()
        (
            supabase.table("push_subscriptions")
            .delete()
            .eq("user_id", user.id)
            .eq("endpoint", endpoint)
            .execute()
        )
        return {"success": True, "message": "Push subscription removed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
