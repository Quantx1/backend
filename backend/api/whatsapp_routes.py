"""
================================================================================
WHATSAPP ROUTES — F12 Pro digest channel (PR 60)
================================================================================
Phone opt-in + OTP verification + digest toggle. Four endpoints:

    POST /api/whatsapp/link/start       — authed, issues OTP to phone
    POST /api/whatsapp/link/verify      — authed, validates OTP
    POST /api/whatsapp/link/disconnect  — authed, clears verified state
    POST /api/whatsapp/digest/toggle    — authed, opt-in/out flag

Flow:
    1. Frontend hits ``/link/start`` with an E.164 phone. We generate a
       6-digit OTP, stash it on ``user_profiles.whatsapp_otp`` +
       ``whatsapp_otp_expires_at`` (10 min TTL), and call the
       ``whatsapp_service`` to send the code. If provider isn't
       configured the OTP gets stored anyway — useful for staged
       onboarding where ops hands out codes manually.
    2. User types the code; frontend hits ``/link/verify`` with the code.
       We compare ``hmac.compare_digest``, enforce an attempt cap, and
       on success flip ``whatsapp_verified=true`` + null out OTP fields.
    3. ``/digest/toggle`` flips the ``whatsapp_digest_enabled`` flag —
       separate from verified because verification is mechanical, opt-in
       is marketing consent.
    4. ``/link/disconnect`` nulls every WhatsApp field.

All endpoints are behind RequireFeature("whatsapp_digest") (Pro gate).
No webhook here — message-delivery status tracking via the BSP
dashboard is enough for v1.
================================================================================
"""

from __future__ import annotations

import hmac
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, validator

from ..core.database import get_supabase_admin
from ..core.tiers import UserTier
from ..middleware.tier_gate import RequireFeature
from ..platform import whatsapp as whatsapp_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])


# ============================================================================
# Models
# ============================================================================


class StartRequest(BaseModel):
    phone: str = Field(..., min_length=8, max_length=20,
                       description="E.164 phone; country code required (e.g. +919876543210)")

    @validator("phone")
    def _phone(cls, v: str) -> str:
        digits = "".join(ch for ch in v if ch.isdigit())
        if len(digits) < 8 or len(digits) > 15:
            raise ValueError("phone must be 8-15 digits (E.164)")
        # Store with leading + so downstream code can distinguish formats.
        return f"+{digits}"


class VerifyRequest(BaseModel):
    code: str = Field(..., min_length=4, max_length=8)

    @validator("code")
    def _code_digits(cls, v: str) -> str:
        if not v.isdigit():
            raise ValueError("code must be numeric")
        return v


class ToggleRequest(BaseModel):
    enabled: bool


class StartResponse(BaseModel):
    phone: str
    expires_at: str
    provider_configured: bool
    delivered: bool


class VerifyResponse(BaseModel):
    verified: bool


class StatusResponse(BaseModel):
    phone: Optional[str] = None
    verified: bool = False
    digest_enabled: bool = False
    provider_configured: bool = False


# ============================================================================
# Helpers
# ============================================================================


_OTP_TTL_MINUTES = 10
_MAX_ATTEMPTS = 5


def _six_digit_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


# ============================================================================
# Routes
# ============================================================================


@router.get("/link/status", response_model=StatusResponse)
async def link_status(
    user: UserTier = Depends(RequireFeature("whatsapp_digest")),
) -> StatusResponse:
    """Used by the settings / onboarding page to render the right
    state (phone masked, verified badge, digest toggle)."""
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("user_profiles")
            .select("whatsapp_phone, whatsapp_verified, whatsapp_digest_enabled")
            .eq("id", user.user_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.error("whatsapp status lookup failed: %s", exc)
        raise HTTPException(status_code=500, detail="lookup_failed")

    row = (rows.data or [None])[0] or {}
    phone = row.get("whatsapp_phone")
    masked = None
    if phone:
        # Mask all but last 4 — never echo the full stored number back
        # to the client after capture.
        tail = phone[-4:]
        masked = f"{phone[:3]}••••{tail}" if len(phone) > 7 else phone

    return StatusResponse(
        phone=masked,
        verified=bool(row.get("whatsapp_verified")),
        digest_enabled=bool(row.get("whatsapp_digest_enabled")),
        provider_configured=whatsapp_service.is_configured(),
    )


@router.post("/link/start", response_model=StartResponse)
async def link_start(
    body: StartRequest,
    user: UserTier = Depends(RequireFeature("whatsapp_digest")),
) -> StartResponse:
    """Mint a 6-digit OTP, persist it on the user profile with a 10-min
    TTL, and attempt delivery via the configured provider. If the
    provider isn't configured we still persist the OTP so ops can
    validate end-to-end manually."""
    otp = _six_digit_otp()
    expires = datetime.now(timezone.utc) + timedelta(minutes=_OTP_TTL_MINUTES)
    sb = get_supabase_admin()
    try:
        sb.table("user_profiles").update({
            "whatsapp_phone": body.phone,
            "whatsapp_verified": False,  # re-verify on number change
            "whatsapp_otp": otp,
            "whatsapp_otp_expires_at": expires.isoformat(),
            "whatsapp_otp_attempts": 0,
        }).eq("id", user.user_id).execute()
    except Exception as exc:
        logger.error("whatsapp link start persist failed: %s", exc)
        raise HTTPException(status_code=500, detail="persist_failed")

    delivered = False
    provider_ok = whatsapp_service.is_configured()
    if provider_ok:
        try:
            delivered = await whatsapp_service.send_otp(body.phone, otp, _OTP_TTL_MINUTES)
        except Exception as exc:
            logger.warning("whatsapp otp send raised: %s", exc)

    return StartResponse(
        phone=body.phone,
        expires_at=expires.isoformat(),
        provider_configured=provider_ok,
        delivered=delivered,
    )


@router.post("/link/verify", response_model=VerifyResponse)
async def link_verify(
    body: VerifyRequest,
    user: UserTier = Depends(RequireFeature("whatsapp_digest")),
) -> VerifyResponse:
    """Compare the submitted code against the stored OTP. Enforces TTL
    + attempt cap. Uses ``hmac.compare_digest`` for the comparison to
    prevent timing side-channels."""
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("user_profiles")
            .select("whatsapp_otp, whatsapp_otp_expires_at, whatsapp_otp_attempts")
            .eq("id", user.user_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.error("whatsapp verify lookup failed: %s", exc)
        raise HTTPException(status_code=500, detail="lookup_failed")

    row = (rows.data or [None])[0]
    if not row or not row.get("whatsapp_otp"):
        raise HTTPException(status_code=400, detail="no_pending_verification")

    attempts = int(row.get("whatsapp_otp_attempts") or 0)
    if attempts >= _MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="too_many_attempts")

    expires_at = row.get("whatsapp_otp_expires_at")
    if expires_at:
        try:
            exp_dt = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            if exp_dt < datetime.now(timezone.utc):
                raise HTTPException(status_code=410, detail="otp_expired")
        except HTTPException:
            raise
        except Exception:
            pass

    stored = str(row["whatsapp_otp"])
    if not hmac.compare_digest(stored, body.code):
        # Bump attempt counter — best-effort, keep serving the response
        try:
            sb.table("user_profiles").update({
                "whatsapp_otp_attempts": attempts + 1,
            }).eq("id", user.user_id).execute()
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="invalid_code")

    # Success — flip verified, clear OTP fields, leave digest opt-in OFF
    # (separate consent).
    try:
        sb.table("user_profiles").update({
            "whatsapp_verified": True,
            "whatsapp_otp": None,
            "whatsapp_otp_expires_at": None,
            "whatsapp_otp_attempts": 0,
        }).eq("id", user.user_id).execute()
    except Exception as exc:
        logger.error("whatsapp verify persist failed: %s", exc)
        raise HTTPException(status_code=500, detail="persist_failed")

    return VerifyResponse(verified=True)


@router.post("/link/disconnect", response_model=StatusResponse)
async def link_disconnect(
    user: UserTier = Depends(RequireFeature("whatsapp_digest")),
) -> StatusResponse:
    sb = get_supabase_admin()
    try:
        sb.table("user_profiles").update({
            "whatsapp_phone": None,
            "whatsapp_verified": False,
            "whatsapp_otp": None,
            "whatsapp_otp_expires_at": None,
            "whatsapp_otp_attempts": 0,
            "whatsapp_digest_enabled": False,
        }).eq("id", user.user_id).execute()
    except Exception as exc:
        logger.error("whatsapp disconnect failed: %s", exc)
        raise HTTPException(status_code=500, detail="persist_failed")
    return StatusResponse(provider_configured=whatsapp_service.is_configured())


@router.post("/digest/toggle", response_model=StatusResponse)
async def digest_toggle(
    body: ToggleRequest,
    user: UserTier = Depends(RequireFeature("whatsapp_digest")),
) -> StatusResponse:
    """Flip the digest opt-in. Refuses when the user hasn't verified
    a number — the UI shouldn't allow reaching this state, but enforce
    server-side too."""
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("user_profiles")
            .select("whatsapp_phone, whatsapp_verified, whatsapp_digest_enabled")
            .eq("id", user.user_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.error("whatsapp toggle lookup failed: %s", exc)
        raise HTTPException(status_code=500, detail="lookup_failed")

    row = (rows.data or [None])[0] or {}
    if body.enabled and not (row.get("whatsapp_verified") and row.get("whatsapp_phone")):
        raise HTTPException(status_code=400, detail="not_verified")

    try:
        sb.table("user_profiles").update({
            "whatsapp_digest_enabled": bool(body.enabled),
        }).eq("id", user.user_id).execute()
    except Exception as exc:
        logger.error("whatsapp toggle persist failed: %s", exc)
        raise HTTPException(status_code=500, detail="persist_failed")

    phone = row.get("whatsapp_phone")
    masked = None
    if phone:
        tail = phone[-4:]
        masked = f"{phone[:3]}••••{tail}" if len(phone) > 7 else phone

    return StatusResponse(
        phone=masked,
        verified=bool(row.get("whatsapp_verified")),
        digest_enabled=bool(body.enabled),
        provider_configured=whatsapp_service.is_configured(),
    )


__all__ = ["router"]
