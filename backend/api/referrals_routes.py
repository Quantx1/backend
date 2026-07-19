"""
================================================================================
REFERRALS ROUTES — N12 virality loop (PR 42)
================================================================================
Endpoints:

    GET  /api/referrals/status          — my referral code + stats + recent list
    POST /api/referrals/rotate-code     — issue a fresh code (admin or abuse recovery)
    POST /api/referrals/attribute       — public: record the ``ref=`` param at signup
    GET  /api/referrals/resolve/{code}  — public: code → referrer_id for signup hook

Reward flow:
    1. Anyone (authed or not) shares their signup link with ?ref={code}
    2. New signup lands → POST /api/referrals/attribute sets referred_by
       + inserts a pending row in user_referrals
    3. When the referred user completes their first paid upgrade, the
       payment webhook calls ``credit_referral_on_first_paid(user_id)``
       (backend/services/referrals.py) which flips status=rewarded
       and bumps ``referral_credit_months`` on BOTH profiles.

Gate: ``RequireFeature("referrals")`` = Free — everyone can refer.
================================================================================
"""

from __future__ import annotations

import logging
import secrets
import string
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.database import get_supabase_admin
from ..core.tiers import UserTier
from ..middleware.tier_gate import RequireFeature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/referrals", tags=["referrals"])

CODE_ALPHABET = string.ascii_uppercase + string.digits
CODE_LENGTH = 8


# ============================================================================
# Pydantic
# ============================================================================


class ReferralStats(BaseModel):
    invited: int
    signed_up: int
    rewarded: int
    pending: int
    credit_months: int      # current user's accumulated reward in months


class ReferralRow(BaseModel):
    id: str
    referred_email: Optional[str]
    referred_user_id: Optional[str]
    status: str
    created_at: str
    signed_up_at: Optional[str]
    rewarded_at: Optional[str]


class ReferralStatus(BaseModel):
    code: str
    share_url: str
    stats: ReferralStats
    recent: List[ReferralRow]


class AttributeRequest(BaseModel):
    referred_user_id: str = Field(..., min_length=1)
    code: str = Field(..., min_length=4, max_length=16)
    referred_email: Optional[str] = None


class ResolveResponse(BaseModel):
    valid: bool
    referrer_id: Optional[str] = None


# ============================================================================
# Helpers
# ============================================================================


def _new_code(sb) -> str:
    """Generate an unused 8-char base32-ish code."""
    for _ in range(10):
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
        try:
            rows = (
                sb.table("user_profiles")
                .select("id")
                .eq("referral_code", code)
                .limit(1)
                .execute()
            )
            if not rows.data:
                return code
        except Exception:
            # If the lookup itself fails, the uniqueness constraint will catch
            # a collision on insert — extremely unlikely at 8 chars × 36^8.
            return code
    raise HTTPException(status_code=500, detail="code_generation_failed")


def _ensure_code(sb, user_id: str) -> str:
    """Return the user's referral code, minting one the first time."""
    try:
        rows = (
            sb.table("user_profiles")
            .select("referral_code")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.error("referral code lookup failed: %s", exc)
        raise HTTPException(status_code=500, detail="lookup_failed")
    row = (rows.data or [None])[0]
    if row and row.get("referral_code"):
        return row["referral_code"]

    code = _new_code(sb)
    try:
        sb.table("user_profiles").update({"referral_code": code}).eq("id", user_id).execute()
    except Exception as exc:
        logger.error("referral code mint failed: %s", exc)
        raise HTTPException(status_code=500, detail="mint_failed")
    return code


def _share_url(code: str) -> str:
    """Public signup URL with ref query. The FE origin is not reliably
    known server-side (multi-domain preview) — we return the path and
    let the client prefix ``location.origin``."""
    return f"/signup?ref={code}"


def _stats(sb, user_id: str) -> ReferralStats:
    invited = signed_up = rewarded = pending = 0
    try:
        rows = (
            sb.table("user_referrals")
            .select("status")
            .eq("referrer_id", user_id)
            .limit(500)
            .execute()
        )
        for r in rows.data or []:
            invited += 1
            s = r.get("status")
            if s == "signed_up":
                signed_up += 1
            elif s == "rewarded":
                signed_up += 1
                rewarded += 1
            elif s == "pending":
                pending += 1
    except Exception as exc:
        logger.debug("referral stats failed: %s", exc)

    credit = 0
    try:
        rows = (
            sb.table("user_profiles")
            .select("referral_credit_months")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        if rows.data:
            credit = int(rows.data[0].get("referral_credit_months") or 0)
    except Exception:
        pass

    return ReferralStats(
        invited=invited,
        signed_up=signed_up,
        rewarded=rewarded,
        pending=pending,
        credit_months=credit,
    )


def _recent(sb, user_id: str, limit: int = 20) -> List[ReferralRow]:
    try:
        rows = (
            sb.table("user_referrals")
            .select("id, referred_email, referred_user_id, status, created_at, signed_up_at, rewarded_at")
            .eq("referrer_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception:
        return []
    out: List[ReferralRow] = []
    for r in rows.data or []:
        out.append(ReferralRow(
            id=str(r["id"]),
            referred_email=r.get("referred_email"),
            referred_user_id=str(r["referred_user_id"]) if r.get("referred_user_id") else None,
            status=str(r.get("status") or "pending"),
            created_at=str(r["created_at"]),
            signed_up_at=str(r["signed_up_at"]) if r.get("signed_up_at") else None,
            rewarded_at=str(r["rewarded_at"]) if r.get("rewarded_at") else None,
        ))
    return out


# ============================================================================
# Routes
# ============================================================================


@router.get("/status", response_model=ReferralStatus)
async def get_status(
    user: UserTier = Depends(RequireFeature("referrals")),
) -> ReferralStatus:
    sb = get_supabase_admin()
    code = _ensure_code(sb, user.user_id)
    return ReferralStatus(
        code=code,
        share_url=_share_url(code),
        stats=_stats(sb, user.user_id),
        recent=_recent(sb, user.user_id),
    )


@router.post("/rotate-code", response_model=ReferralStatus)
async def rotate_code(
    user: UserTier = Depends(RequireFeature("referrals")),
) -> ReferralStatus:
    sb = get_supabase_admin()
    code = _new_code(sb)
    try:
        sb.table("user_profiles").update({"referral_code": code}).eq("id", user.user_id).execute()
    except Exception as exc:
        logger.error("rotate-code failed: %s", exc)
        raise HTTPException(status_code=500, detail="rotate_failed")
    return ReferralStatus(
        code=code,
        share_url=_share_url(code),
        stats=_stats(sb, user.user_id),
        recent=_recent(sb, user.user_id),
    )


@router.get("/resolve/{code}", response_model=ResolveResponse)
async def resolve_code(code: str) -> ResolveResponse:
    """Public lookup used by the signup page to validate a ref code
    before user registration. Returns the referrer_id on hit."""
    code = (code or "").strip().upper()
    if not code or len(code) < 4:
        return ResolveResponse(valid=False)
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("user_profiles")
            .select("id")
            .eq("referral_code", code)
            .limit(1)
            .execute()
        )
        row = (rows.data or [None])[0]
        if not row:
            return ResolveResponse(valid=False)
        return ResolveResponse(valid=True, referrer_id=str(row["id"]))
    except Exception as exc:
        logger.debug("resolve failed: %s", exc)
        return ResolveResponse(valid=False)


@router.post("/attribute")
async def attribute_signup(body: AttributeRequest) -> Dict[str, Any]:
    """Public — called by the signup page right after auth.user creation.

    Sets ``referred_by`` on the new profile and inserts a pending row in
    ``user_referrals``. Idempotent: repeat calls for the same
    (referrer, referred) pair are no-ops.
    """
    code = body.code.strip().upper()
    sb = get_supabase_admin()

    # Resolve code → referrer.
    try:
        rref = (
            sb.table("user_profiles")
            .select("id")
            .eq("referral_code", code)
            .limit(1)
            .execute()
        )
        ref_row = (rref.data or [None])[0]
        if not ref_row:
            raise HTTPException(status_code=404, detail="invalid_code")
        referrer_id = str(ref_row["id"])
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("attribute resolve failed: %s", exc)
        raise HTTPException(status_code=500, detail="resolve_failed")

    if referrer_id == body.referred_user_id:
        raise HTTPException(status_code=400, detail="self_referral_not_allowed")

    # Guard: don't re-attribute if already referred.
    try:
        existing = (
            sb.table("user_profiles")
            .select("referred_by")
            .eq("id", body.referred_user_id)
            .limit(1)
            .execute()
        )
        row = (existing.data or [None])[0]
        if row and row.get("referred_by"):
            return {"attributed": False, "reason": "already_attributed"}
    except Exception:
        pass

    try:
        sb.table("user_profiles").update({
            "referred_by": referrer_id,
        }).eq("id", body.referred_user_id).execute()
    except Exception as exc:
        logger.error("attribute update failed: %s", exc)
        raise HTTPException(status_code=500, detail="attribute_failed")

    # Insert referrals row — status flips to signed_up immediately since
    # the user just finished signup.
    try:
        sb.table("user_referrals").insert({
            "referrer_id": referrer_id,
            "referred_user_id": body.referred_user_id,
            "referred_email": body.referred_email,
            "status": "signed_up",
            "signed_up_at": datetime.utcnow().isoformat(),
        }).execute()
    except Exception as exc:
        logger.warning("user_referrals insert failed: %s", exc)

    try:
        from ..observability import EventName, track
        track(EventName.SIGNUP_COMPLETED, body.referred_user_id, {
            "referrer_id": referrer_id,
            "via_referral": True,
        })
    except Exception:
        pass

    return {"attributed": True, "referrer_id": referrer_id}


__all__ = ["router"]
