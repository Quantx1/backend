"""
System status route — admin-only configuration health check.

Returns the per-subsystem ``get_startup_status()`` snapshot used by the
admin command center to surface missing env vars and degraded clients.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from ..core.config import settings, get_startup_status

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin"])


def _get_current_user_dep():
    from .app import get_current_user
    return get_current_user


@router.get("/api/system/status")
async def system_status(user=Depends(_get_current_user_dep())):
    """Return configuration status of all subsystems (admin only)."""
    if user.email not in settings.admin_emails_list:
        raise HTTPException(status_code=403, detail="Admin access required")

    status = get_startup_status()
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "config_status": status,
        "timestamp": datetime.utcnow().isoformat(),
    }
