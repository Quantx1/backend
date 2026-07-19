"""
Referral reward resolver.

Called from the payment webhook when a user completes their first paid
upgrade. Bumps ``referral_credit_months`` by 1 for both sides and flips
the ``user_referrals`` row to status=rewarded. Safe to call repeatedly —
each referral pair can only be rewarded once (status-guard).

The actual billing credit (extending paid subscription period) is the
payment-flow's responsibility; this module just accumulates the entitlement.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

CREDIT_MONTHS_PER_REWARD = 1


def credit_referral_on_first_paid(
    user_id: str,
    *,
    supabase_client=None,
) -> Optional[dict]:
    """Resolve + apply reward for the user's *first* paid upgrade.

    Returns a summary dict ``{referrer_id, months_credited}`` when a
    reward fires, or ``None`` when there is nothing to do (no referrer,
    already rewarded, or lookup failed).
    """
    if supabase_client is None:
        from ..core.database import get_supabase_admin
        supabase_client = get_supabase_admin()

    # 1. Who referred this user?
    try:
        rows = (
            supabase_client.table("user_profiles")
            .select("id, referred_by, referral_credit_months")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        row = (rows.data or [None])[0]
        if not row or not row.get("referred_by"):
            return None
        referrer_id = str(row["referred_by"])
    except Exception as exc:
        logger.warning("referral reward: profile lookup failed %s: %s", user_id, exc)
        return None

    # 2. Guard against double-reward (referrals row check).
    try:
        existing = (
            supabase_client.table("user_referrals")
            .select("id, status")
            .eq("referrer_id", referrer_id)
            .eq("referred_user_id", user_id)
            .limit(1)
            .execute()
        )
        row = (existing.data or [None])[0]
        if row and row.get("status") == "rewarded":
            return None
        referrals_id = row["id"] if row else None
    except Exception as exc:
        logger.debug("referral reward: pair lookup skipped: %s", exc)
        referrals_id = None

    # 3. Bump referral_credit_months on both sides atomically-ish (two
    # updates; we accept non-atomicity here since we're guarded by the
    # referrals row state flip at the end).
    for uid in (referrer_id, user_id):
        try:
            cur = (
                supabase_client.table("user_profiles")
                .select("referral_credit_months")
                .eq("id", uid)
                .limit(1)
                .execute()
            )
            prev = int((cur.data or [{}])[0].get("referral_credit_months") or 0)
            supabase_client.table("user_profiles").update({
                "referral_credit_months": prev + CREDIT_MONTHS_PER_REWARD,
            }).eq("id", uid).execute()
        except Exception as exc:
            logger.error("referral credit bump failed for %s: %s", uid, exc)

    # 4. Flip the referrals row to rewarded.
    try:
        if referrals_id:
            supabase_client.table("user_referrals").update({
                "status": "rewarded",
                "rewarded_at": datetime.utcnow().isoformat(),
            }).eq("id", referrals_id).execute()
        else:
            # Defensive insert — pair row missing but profile claims referred_by.
            supabase_client.table("user_referrals").insert({
                "referrer_id": referrer_id,
                "referred_user_id": user_id,
                "status": "rewarded",
                "signed_up_at": datetime.utcnow().isoformat(),
                "rewarded_at": datetime.utcnow().isoformat(),
            }).execute()
    except Exception as exc:
        logger.warning("referral reward flip failed: %s", exc)

    logger.info(
        "Referral reward fired: referrer=%s referred=%s months=%d",
        referrer_id, user_id, CREDIT_MONTHS_PER_REWARD,
    )
    return {
        "referrer_id": referrer_id,
        "referred_user_id": user_id,
        "months_credited": CREDIT_MONTHS_PER_REWARD,
    }


def consume_referral_credit(
    user_id: str,
    *,
    supabase_client=None,
) -> int:
    """Atomically (best-effort) claim the user's accumulated referral
    credit. Returns the number of months consumed — the payment flow
    extends ``subscription_end`` by this many months.

    Safe to call from the successful-payment hook: zeros out the column
    after reading, so retries of the same payment won't double-apply.
    """
    if supabase_client is None:
        from ..core.database import get_supabase_admin
        supabase_client = get_supabase_admin()

    try:
        rows = (
            supabase_client.table("user_profiles")
            .select("referral_credit_months")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        row = (rows.data or [None])[0]
        if not row:
            return 0
        months = int(row.get("referral_credit_months") or 0)
        if months <= 0:
            return 0
    except Exception as exc:
        logger.warning("referral credit read failed %s: %s", user_id, exc)
        return 0

    # Zero it out. If the write fails we just return 0 to be safe — the
    # user can re-consume on a later renewal, which is fine.
    try:
        supabase_client.table("user_profiles").update({
            "referral_credit_months": 0,
        }).eq("id", user_id).execute()
    except Exception as exc:
        logger.error("referral credit zero-out failed %s: %s", user_id, exc)
        return 0

    logger.info("Referral credit consumed: user=%s months=%d", user_id, months)
    return months


def expire_pending_referrals(
    *,
    supabase_client=None,
    days: int = 90,
) -> int:
    """Soft-expire ``user_referrals`` rows that have been ``pending``
    for more than ``days`` days. Returns count expired. Called daily by
    the scheduler (``referral_expire_pending``)."""
    if supabase_client is None:
        from ..core.database import get_supabase_admin
        supabase_client = get_supabase_admin()

    from datetime import timedelta
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    try:
        rows = (
            supabase_client.table("user_referrals")
            .select("id")
            .eq("status", "pending")
            .lt("created_at", cutoff)
            .limit(1000)
            .execute()
        )
        ids = [str(r["id"]) for r in (rows.data or [])]
        if not ids:
            return 0
    except Exception as exc:
        logger.debug("expire lookup failed: %s", exc)
        return 0

    try:
        supabase_client.table("user_referrals").update({
            "status": "expired",
        }).in_("id", ids).execute()
    except Exception as exc:
        logger.warning("expire update failed: %s", exc)
        return 0

    logger.info("Expired %d pending referrals older than %d days", len(ids), days)
    return len(ids)


__all__ = [
    "credit_referral_on_first_paid",
    "consume_referral_credit",
    "expire_pending_referrals",
    "CREDIT_MONTHS_PER_REWARD",
]
