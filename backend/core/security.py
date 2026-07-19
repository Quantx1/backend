"""
Security utilities and authentication
"""
import logging
from types import SimpleNamespace
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Any, Optional

from .config import settings
from .database import supabase

logger = logging.getLogger(__name__)
# auto_error=False so the dev bypass can run without an Authorization header;
# the missing-token case is handled explicitly below (401) in normal mode.
security = HTTPBearer(auto_error=False)

# DEV-ONLY mock user (SimpleNamespace mirrors the Supabase user's .id/.email).
_DEV_USER = SimpleNamespace(id="00000000-0000-0000-0000-000000000000", email="dev@local.test")


def _dev_auth_enabled() -> bool:
    """Auth bypass is permitted ONLY outside production AND when explicitly
    flagged. Belt-and-suspenders: production can never enable it."""
    return settings.APP_ENV != "production" and getattr(settings, "DEV_AUTH_BYPASS", False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Any:
    """
    Get current authenticated user from the JWT token.

    Raises HTTPException(401) if the token is missing/invalid. In local dev with
    DEV_AUTH_BYPASS=true (never in production), returns a mock user so gated/agent
    endpoints can be exercised without a Supabase login.
    """
    if _dev_auth_enabled():
        logger.warning("DEV_AUTH_BYPASS active — returning mock dev user (NEVER use in production)")
        return _DEV_USER

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    try:
        token = credentials.credentials
        user_response = supabase.auth.get_user(token)

        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )

        return user_response.user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


# NOTE: get_user_profile + verify_subscription_access removed 2026-06-06 —
# they were dead (every caller imports get_user_profile from api.app, and the
# tier system uses core/tiers.py + middleware/tier_gate.py). The old
# verify_subscription_access also carried a stale free/starter/pro hierarchy
# (starter retired, no elite). get_current_user above is the live auth leaf.
