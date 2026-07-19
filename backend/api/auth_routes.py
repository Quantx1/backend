"""
Auth API routes — signup, login, refresh, logout, forgot-password, me.

Thin wrappers over Supabase auth. The full JWT verification logic
lives in ``app.py::get_current_user`` (referenced here via lazy import
to avoid the circular dependency app ↔ auth_routes at module load).
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from ..core.config import settings
from ..schemas import UserLogin, UserSignup

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Auth"])


def _mask_email(e: str) -> str:
    """Mask an email for logging — keep only the first char of the local
    part so logs never carry full PII (SEBI/security hygiene)."""
    if not e or "@" not in e:
        return "<redacted>"
    name, _, dom = e.partition("@")
    return (name[:1] or "?") + "***@" + dom


def _get_supabase():
    """Lazy import — app.py wires the Supabase client at startup."""
    from .app import get_supabase
    return get_supabase()


def _get_supabase_admin():
    """Lazy import — admin client (service-role) used for last_login update."""
    from .app import get_supabase_admin
    return get_supabase_admin()


def _get_current_user_dep():
    """Dependency factory — defers the get_current_user import until
    request time so this module doesn't import app.py at load."""
    from .app import get_current_user
    return get_current_user


@router.post("/api/auth/signup")
async def signup(data: UserSignup):
    """Create new user account.

    Beta posture (PR-Y 2026-05-29): we use the **admin** client with
    ``email_confirm=True`` so new beta users can login immediately
    without clicking a confirmation email. When the Supabase project
    has confirmation enabled in production we revisit this; for now
    the friction of email-click is worse than the friction of a
    typo'd email address.

    Also creates the matching ``user_profiles`` row so the platform
    layout's onboarding gate works on the first navigation.
    """
    try:
        admin = _get_supabase_admin()
        # admin.create_user with email_confirm=True bypasses the
        # confirmation email flow.
        res = admin.auth.admin.create_user({
            "email": data.email,
            "password": data.password,
            "email_confirm": True,
            "user_metadata": {
                "full_name": data.full_name,
                "phone": data.phone,
            },
        })
        if not res or not res.user:
            raise HTTPException(status_code=400, detail="Signup failed")

        # Seed the user_profiles row. PR-Y default: tier=free,
        # onboarding NOT completed (so the user gets the risk-quiz
        # bounce on first dashboard hit).
        try:
            admin.table("user_profiles").upsert({
                "id": res.user.id,
                "email": data.email,
                "full_name": data.full_name,
                "phone": data.phone,
                "tier": "free",
                "onboarding_completed": False,
            }, on_conflict="id").execute()
        except Exception as exc:
            logger.warning("user_profiles upsert failed for %s: %s", _mask_email(data.email), exc)

        return {
            "success": True,
            "message": "Account created.",
            "user_id": res.user.id,
        }
    except HTTPException:
        raise
    except Exception as e:
        # Supabase returns 422 with "User already registered" — surface
        # that nicely so the frontend can route to /login.
        msg = str(e)
        if "already" in msg.lower() or "registered" in msg.lower():
            raise HTTPException(status_code=409, detail="Email already registered")
        logger.error("signup failed for %s: %s", _mask_email(data.email), e)
        raise HTTPException(status_code=400, detail=msg)


@router.post("/api/auth/login")
async def login(data: UserLogin):
    """Login with email and password."""
    try:
        supabase = _get_supabase()
        response = supabase.auth.sign_in_with_password({
            "email": data.email, "password": data.password,
        })
        if response.user and response.session:
            # Update last login + last_active timestamps.
            supabase_admin = _get_supabase_admin()
            supabase_admin.table("user_profiles").update({
                "last_login": datetime.utcnow().isoformat(),
                "last_active": datetime.utcnow().isoformat(),
            }).eq("id", response.user.id).execute()

            return {
                "success": True,
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token,
                "expires_at": response.session.expires_at,
                "user": {"id": response.user.id, "email": response.user.email},
            }
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/api/auth/refresh")
async def refresh(refresh_token: str):
    """Refresh access token."""
    try:
        supabase = _get_supabase()
        response = supabase.auth.refresh_session(refresh_token)
        if response.session:
            return {
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token,
            }
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/api/auth/logout")
async def logout(user=Depends(_get_current_user_dep())):
    """Logout user. The frontend clears the local session; this
    endpoint is a placeholder so the client can fire a beacon."""
    return {"success": True}


@router.post("/api/auth/forgot-password")
async def forgot_password(email: str):
    """Send password reset email."""
    try:
        supabase = _get_supabase()
        supabase.auth.reset_password_email(email, {
            "redirect_to": f"{settings.FRONTEND_URL}/reset-password",
        })
        return {"success": True, "message": "Password reset email sent"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/auth/me")
async def get_current_user_info(user=Depends(_get_current_user_dep())):
    """Get current authenticated user info."""
    return {"user_id": user.id, "email": user.email}
