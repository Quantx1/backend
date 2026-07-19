"""
admin_audit — lightweight helper for writing one row per admin mutation.

Callers pass the FastAPI ``Request`` when available so we can denormalize
``ip_address`` + ``user_agent`` into the row. Fully fire-and-forget: any
write failure is logged at ``WARNING`` but never raised back to the
caller, because we must not fail the admin action just because the audit
insert hiccupped.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


VALID_TARGET_TYPES = {
    "user", "tier", "ml_model", "scheduler_job",
    "system_flag", "payment", "signal", "other",
}


def _extract_client_info(request: Optional[Any]) -> Dict[str, Optional[str]]:
    if request is None:
        return {"ip_address": None, "user_agent": None}
    try:
        headers = getattr(request, "headers", {}) or {}
        client = getattr(request, "client", None)
        # Honor X-Forwarded-For when behind a proxy (Railway/Vercel).
        forwarded = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
        ip = (forwarded.split(",")[0].strip() if forwarded else None) or (
            getattr(client, "host", None) if client else None
        )
        ua = headers.get("user-agent") or headers.get("User-Agent")
        return {"ip_address": ip, "user_agent": ua}
    except Exception:
        return {"ip_address": None, "user_agent": None}


def log_admin_action(
    *,
    actor_id: Optional[str],
    actor_email: Optional[str],
    action: str,
    target_type: str = "other",
    target_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    request: Optional[Any] = None,
    supabase_client: Any = None,
) -> None:
    """Insert one row into ``admin_audit_log``. Never raises."""
    if target_type not in VALID_TARGET_TYPES:
        target_type = "other"
    if supabase_client is None:
        try:
            from ..core.database import get_supabase_admin
            supabase_client = get_supabase_admin()
        except Exception as exc:
            logger.debug("admin_audit: supabase client unavailable: %s", exc)
            return

    client_info = _extract_client_info(request)

    row = {
        "actor_id": str(actor_id) if actor_id else None,
        "actor_email": actor_email,
        "action": str(action)[:80],
        "target_type": target_type,
        "target_id": str(target_id)[:120] if target_id else None,
        "payload": payload or {},
        "ip_address": client_info.get("ip_address"),
        "user_agent": client_info.get("user_agent"),
        "created_at": datetime.utcnow().isoformat(),
    }
    try:
        supabase_client.table("admin_audit_log").insert(row).execute()
    except Exception as exc:
        # Never fail the admin action over the audit write.
        logger.warning("admin_audit log write failed (%s): %s", action, exc)


__all__ = ["log_admin_action", "VALID_TARGET_TYPES"]
