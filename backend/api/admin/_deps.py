"""
Shared dependencies for the admin API package.

Holds the auth gate (``get_admin_user``), shared role enum, and the
Pydantic schemas used by multiple admin sub-routers. Everything here
is also re-exported from ``admin_routes`` for backwards-compatible
imports (e.g. ``payment_routes`` does
``from .admin_routes import get_admin_user`` via a lazy import wrapper).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from ...core.config import settings

logger = logging.getLogger(__name__)

_security = HTTPBearer()


# ============================================================================
# ROLE ENUM + USER SCHEMA
# ============================================================================


class AdminRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    SUPPORT_ADMIN = "support_admin"
    READ_ONLY = "read_only"


class AdminUser(BaseModel):
    id: str
    email: str
    role: AdminRole


# ============================================================================
# DOMAIN SCHEMAS — user management
# ============================================================================


class UserListItem(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    phone: Optional[str]
    capital: float
    trading_mode: str
    subscription_status: str
    subscription_plan: Optional[str]
    broker_connected: bool
    broker_name: Optional[str]
    total_trades: int
    winning_trades: int
    total_pnl: float
    created_at: str
    last_login: Optional[str]
    last_active: Optional[str]
    is_suspended: bool = False
    is_banned: bool = False


class UserDetailResponse(BaseModel):
    user: UserListItem
    trading_settings: Dict[str, Any]
    recent_activity: List[Dict[str, Any]]
    payment_history: List[Dict[str, Any]]
    positions: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]


class UserListResponse(BaseModel):
    users: List[UserListItem]
    total: int
    page: int
    page_size: int
    total_pages: int


class UserActionRequest(BaseModel):
    reason: Optional[str] = None


class SubscriptionResetRequest(BaseModel):
    new_plan_id: Optional[str] = None
    new_status: str = "free"
    reason: str


# ============================================================================
# AUTH DEPENDENCY
# ============================================================================


# In-memory admin override (left in place for dev scripting). Production
# admin status comes from ``user_profiles.is_admin``.
ADMIN_USERS: Dict[str, AdminRole] = {}


def get_supabase_admin():
    """Get Supabase admin client — imported from app.py at call time
    to avoid a startup-time circular dep with the FastAPI app object."""
    from ..app import get_supabase_admin as _get_admin
    return _get_admin()


async def get_admin_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> AdminUser:
    """
    Verify user has admin access.

    Authority order (first match wins):
      1. user_profiles.is_admin = true  → SUPER_ADMIN (primary path, post-PR 1)
      2. email in ADMIN_EMAILS env var  → SUPER_ADMIN (bootstrap fallback for
         first-time deploys before anyone has is_admin=true in the DB)
      3. anything else                  → 403

    With JWT signature verification enabled (PR 1), the JWT email claim
    is cryptographically trusted, so ADMIN_EMAILS bootstrap is safe.
    """
    from ..app import get_current_user

    try:
        user = await get_current_user(credentials)
        user_id = user.id
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin auth error: {e}")
        raise HTTPException(status_code=401, detail="Authentication required")

    supabase = get_supabase_admin()

    try:
        result = (
            supabase.table("user_profiles")
            .select("id, email, is_admin")
            .eq("id", user_id)
            .single()
            .execute()
        )
    except Exception:
        # Column may not exist on first deploy; retry without it so we
        # can still honor the ADMIN_EMAILS bootstrap path.
        result = (
            supabase.table("user_profiles")
            .select("id, email")
            .eq("id", user_id)
            .single()
            .execute()
        )

    if not result.data:
        raise HTTPException(status_code=403, detail="Admin access required")

    email = result.data.get("email")
    is_admin_flag = bool(result.data.get("is_admin", False))

    admin_role: Optional[AdminRole] = None
    if is_admin_flag:
        admin_role = AdminRole.SUPER_ADMIN
    if not admin_role and email and email in settings.admin_emails_list:
        admin_role = AdminRole.SUPER_ADMIN
    if not admin_role:
        admin_role = ADMIN_USERS.get(user_id)
    if not admin_role:
        raise HTTPException(status_code=403, detail="Admin access required")

    return AdminUser(id=user_id, email=email, role=admin_role)


def require_role(*allowed_roles: AdminRole):
    """Dependency factory — gates an endpoint on a specific admin role."""

    async def role_checker(admin: AdminUser = Depends(get_admin_user)):
        if admin.role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Required role: {', '.join([r.value for r in allowed_roles])}",
            )
        return admin

    return role_checker
