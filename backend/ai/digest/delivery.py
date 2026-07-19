"""
Digest fan-out.

Pulls every eligible user, builds their payload via the generator, and
sends via each channel they've opted into. Two entry points match the
scheduler jobs:

    deliver_morning_all(supabase_client)
    deliver_evening_all(supabase_client)

Channel eligibility:
    * Telegram — any tier, ``telegram_connected=true`` AND
      ``telegram_chat_id`` set AND ``notifications_enabled``.
    * WhatsApp — ``tier`` in ('pro','elite'), ``whatsapp_verified`` AND
      ``whatsapp_digest_enabled``.

A user with both channels enabled gets the same body on both — we don't
drop one when the other succeeds, so reach goes up with redundancy.

Market-side data (regime / nifty / today's signals) is fetched once per
run and cached across the user loop — the per-user paths only hit
``positions`` / ``trades`` which are actually user-scoped.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .generator import (
    build_morning_brief,
    build_evening_summary,
    DigestPayload,
    _fetch_current_regime,
    _fetch_today_signals,
    _fetch_nifty_close,
)

logger = logging.getLogger(__name__)


_CONCURRENCY = 4  # LLM rate-limit friendly.


# ============================================================================
# Channel senders — each returns True on accepted delivery.
# ============================================================================


async def _send_telegram(chat_id: str, body: str) -> bool:
    try:
        from ...core.config import settings
        token = settings.TELEGRAM_BOT_TOKEN
        if not token:
            return False
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": body},
            )
        return 200 <= resp.status_code < 300
    except Exception as exc:
        logger.debug("telegram digest send failed chat=%s: %s", chat_id, exc)
        return False


async def _send_whatsapp(phone: str, body: str) -> bool:
    try:
        from ...platform import whatsapp as whatsapp_service
        # Use the approved digest template — BSPs require it for
        # business-initiated messages outside the 24h window.
        from ...core.config import settings
        return await whatsapp_service.send_text(
            phone, body, template=settings.WHATSAPP_DIGEST_TEMPLATE,
        )
    except Exception as exc:
        logger.debug("whatsapp digest send failed phone=%s: %s", phone, exc)
        return False


# ============================================================================
# Cohort query
# ============================================================================


def _eligible_users(sb) -> List[Dict[str, Any]]:
    """One query, select the minimum set of flags. We filter channels
    per-user during the loop so we don't duplicate rows for users with
    both Telegram + WhatsApp enabled."""
    try:
        rows = (
            sb.table("user_profiles")
            .select(
                "id, tier, notifications_enabled, "
                "telegram_connected, telegram_chat_id, "
                "whatsapp_verified, whatsapp_phone, whatsapp_digest_enabled"
            )
            .eq("onboarding_completed", True)
            .limit(10000)
            .execute()
        )
        return rows.data or []
    except Exception as exc:
        logger.error("digest cohort query failed: %s", exc)
        return []


def _channels_for(row: Dict[str, Any]) -> List[tuple]:
    """Return a list of (channel_name, send_fn_args) pairs for one user."""
    out: List[tuple] = []
    if (
        row.get("notifications_enabled")
        and row.get("telegram_connected")
        and row.get("telegram_chat_id")
    ):
        out.append(("telegram", str(row["telegram_chat_id"])))
    if (
        row.get("tier") in ("pro", "elite")
        and row.get("whatsapp_verified")
        and row.get("whatsapp_phone")
        and row.get("whatsapp_digest_enabled")
    ):
        out.append(("whatsapp", str(row["whatsapp_phone"])))
    return out


# ============================================================================
# Fan-out
# ============================================================================


def _audit_delivery(
    sb,
    *,
    user_id: str,
    kind: str,
    channel: str,
    status: str,
    body_preview: Optional[str] = None,
    error_detail: Optional[str] = None,
) -> None:
    """Best-effort write to ``user_digest_deliveries``. Never raises —
    audit failure must not cascade into a delivery failure."""
    try:
        sb.table("user_digest_deliveries").insert({
            "user_id": user_id,
            "kind": kind,
            "channel": channel,
            "status": status,
            "body_preview": (body_preview or "")[:200] or None,
            "error_detail": error_detail,
        }).execute()
    except Exception as exc:
        logger.debug(
            "digest audit insert failed user=%s channel=%s: %s",
            user_id, channel, exc,
        )


async def _deliver_one(
    sem: asyncio.Semaphore,
    user_row: Dict[str, Any],
    kind: str,
    market_cache: Dict[str, Any],
    sb,
) -> Dict[str, int]:
    """Build + send for one user. Returns per-channel send counts."""
    counts = {"telegram_sent": 0, "whatsapp_sent": 0, "failed": 0}
    channels = _channels_for(user_row)
    if not channels:
        return counts

    async with sem:
        try:
            if kind == "morning":
                payload: DigestPayload = await build_morning_brief(
                    user_id=user_row["id"],
                    supabase_client=sb,
                    market_cache=market_cache,
                )
            else:
                payload = await build_evening_summary(
                    user_id=user_row["id"],
                    supabase_client=sb,
                    market_cache=market_cache,
                )
        except Exception as exc:
            logger.warning("digest build failed user=%s: %s", user_row["id"], exc)
            counts["failed"] += 1
            for ch_name, _ in channels:
                _audit_delivery(
                    sb, user_id=user_row["id"], kind=kind, channel=ch_name,
                    status="failed", error_detail=f"build_failed: {exc}",
                )
            return counts

        # Fire channel sends in parallel for this user.
        tasks = []
        for ch, target in channels:
            if ch == "telegram":
                tasks.append(("telegram", _send_telegram(target, payload.body)))
            elif ch == "whatsapp":
                tasks.append(("whatsapp", _send_whatsapp(target, payload.body)))
        for name, coro in tasks:
            try:
                ok = await coro
                if ok:
                    counts[f"{name}_sent"] = counts.get(f"{name}_sent", 0) + 1
                    _audit_delivery(
                        sb, user_id=user_row["id"], kind=kind, channel=name,
                        status="sent", body_preview=payload.body,
                    )
                else:
                    counts["failed"] += 1
                    _audit_delivery(
                        sb, user_id=user_row["id"], kind=kind, channel=name,
                        status="failed", error_detail="provider_returned_false",
                    )
            except Exception as exc:
                counts["failed"] += 1
                _audit_delivery(
                    sb, user_id=user_row["id"], kind=kind, channel=name,
                    status="failed", error_detail=f"exception: {exc}",
                )

    return counts


async def _deliver_all(kind: str, supabase_client=None) -> Dict[str, int]:
    if supabase_client is None:
        from ...core.database import get_supabase_admin
        supabase_client = get_supabase_admin()

    # Shared market data — one query set per run.
    market_cache: Dict[str, Any] = {
        "regime": _fetch_current_regime(supabase_client),
        "nifty": _fetch_nifty_close(supabase_client),
    }
    if kind == "morning":
        # Signals only matter for the morning brief. Evening summary uses
        # per-user closed trades instead.
        from .generator import IST
        from datetime import datetime, timedelta, timezone
        since_utc = (datetime.now(IST) - timedelta(hours=16)).astimezone(timezone.utc).isoformat()
        market_cache["signals"] = _fetch_today_signals(supabase_client, since_utc)

    users = _eligible_users(supabase_client)
    sem = asyncio.Semaphore(_CONCURRENCY)
    results = await asyncio.gather(
        *[_deliver_one(sem, u, kind, market_cache, supabase_client) for u in users],
        return_exceptions=False,
    )

    totals = {"telegram_sent": 0, "whatsapp_sent": 0, "failed": 0, "n_users": len(users)}
    for r in results:
        totals["telegram_sent"] += r.get("telegram_sent", 0)
        totals["whatsapp_sent"] += r.get("whatsapp_sent", 0)
        totals["failed"] += r.get("failed", 0)

    return totals


async def deliver_morning_all(supabase_client=None) -> Dict[str, int]:
    return await _deliver_all("morning", supabase_client)


async def deliver_evening_all(supabase_client=None) -> Dict[str, int]:
    return await _deliver_all("evening", supabase_client)


__all__ = ["deliver_morning_all", "deliver_evening_all"]
