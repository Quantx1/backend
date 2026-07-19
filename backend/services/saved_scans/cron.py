"""Cron sweep — picks all enabled saved scans due for a re-run, executes
each, persists alerts on new hits, and dispatches notifications.

Schedule resolution:
  * `hourly`      — re-run if last_run_at is null OR > 55 min ago
  * `open_close`  — re-run at 09:30 (±5 min) and 15:25 (±5 min) only
  * `every_15min` — re-run if last_run_at is null OR > 13 min ago
  * `manual`      — skipped (user triggers from UI)

Market-hours gate: cron is a no-op outside 09:15-15:30 IST Mon-Fri.
Backend scheduler.py runs this every 5 min between 9:15 and 15:30 IST.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from .runner import run_saved_scan

logger = logging.getLogger(__name__)


_IST = ZoneInfo("Asia/Kolkata")


def _is_market_hours_now() -> bool:
    """True between 09:15 and 15:30 IST on a weekday."""
    now = datetime.now(_IST)
    if now.weekday() >= 5:    # Sat/Sun
        return False
    t = now.time()
    return (t.hour, t.minute) >= (9, 15) and (t.hour, t.minute) <= (15, 30)


def _is_due(scan: Dict[str, Any]) -> bool:
    """Decide whether this saved scan should re-run right now."""
    sched = scan.get("schedule", "hourly")
    if sched == "manual":
        return False

    last = scan.get("last_run_at")
    if last is None:
        return True
    if isinstance(last, str):
        last = datetime.fromisoformat(last.replace("Z", "+00:00"))

    now = datetime.now(timezone.utc)
    minutes_since = (now - last).total_seconds() / 60.0

    if sched == "every_15min":
        return minutes_since >= 13
    if sched == "hourly":
        return minutes_since >= 55
    if sched == "open_close":
        # Fire only if we're near 09:30 or 15:25 IST AND haven't run today
        ist = datetime.now(_IST)
        near_open = (ist.hour, ist.minute) >= (9, 28) and (ist.hour, ist.minute) <= (9, 35)
        near_close = (ist.hour, ist.minute) >= (15, 22) and (ist.hour, ist.minute) <= (15, 30)
        if not (near_open or near_close):
            return False
        # Already ran in this window?
        return minutes_since >= 60
    return False


async def sweep_due_scans() -> Dict[str, Any]:
    """Find + run every saved scan that's due. Returns a summary."""
    if not _is_market_hours_now():
        logger.debug("sweep_due_scans: outside market hours, skipping")
        return {"skipped": True, "reason": "outside market hours"}

    from backend.core.database import get_supabase_admin
    sb = get_supabase_admin()

    res = sb.table("saved_scans").select(
        "id,user_id,name,scanner_ids,universe,sectors,min_hits,schedule,"
        "notify_channels,last_run_at,last_hit_symbols"
    ).eq("enabled", True).neq("schedule", "manual").execute()
    candidates = res.data or []

    due = [s for s in candidates if _is_due(s)]
    if not due:
        return {"checked": len(candidates), "due": 0, "fired": 0}

    fired = 0
    alerts_inserted = 0

    for scan in due:
        result = await run_saved_scan(scan)
        if result.error:
            logger.warning("saved scan %s failed: %s", scan["id"], result.error)
            continue

        now_iso = datetime.now(timezone.utc).isoformat()

        # Update the scan row with new last-run state
        sb.table("saved_scans").update({
            "last_run_at": now_iso,
            "last_hit_symbols": result.matched_symbols,
            "last_hit_count": result.total_count,
            "updated_at": now_iso,
        }).eq("id", scan["id"]).execute()
        fired += 1

        # Insert an alert ONLY if there are new symbols
        if result.new_symbols:
            ins = sb.table("saved_scan_alerts").insert({
                "scan_id": scan["id"],
                "user_id": scan["user_id"],
                "new_symbols": result.new_symbols,
                "total_match_count": result.total_count,
            }).execute()
            if ins.data:
                alerts_inserted += 1
                # Notify in best-effort fashion (don't block the sweep)
                try:
                    await _dispatch_alert(
                        scan, result.new_symbols, ins.data[0]["id"],
                    )
                    sb.table("saved_scan_alerts").update({
                        "notified": True,
                    }).eq("id", ins.data[0]["id"]).execute()
                except Exception as e:
                    logger.warning("alert notify failed for %s: %s", scan["id"], e)
                    sb.table("saved_scan_alerts").update({
                        "notify_error": str(e)[:300],
                    }).eq("id", ins.data[0]["id"]).execute()

    return {
        "checked": len(candidates),
        "due": len(due),
        "fired": fired,
        "alerts_inserted": alerts_inserted,
    }


async def _dispatch_alert(
    scan: Dict[str, Any], new_symbols: List[str], alert_id: str,
) -> None:
    """Send the alert through the user's configured channels.

    Falls back gracefully — push always tries, email/whatsapp only if
    the user has those connected.
    """
    channels = list(scan.get("notify_channels") or ["push"])
    title = f"📊 {scan['name']}"
    body_lines = [
        f"{len(new_symbols)} new match{'es' if len(new_symbols) != 1 else ''}",
        ", ".join(new_symbols[:5]) + (f" +{len(new_symbols) - 5} more" if len(new_symbols) > 5 else ""),
    ]
    body = "\n".join(body_lines)

    # Push (web push) — best-effort
    if "push" in channels:
        try:
            from backend.platform.notifications import dispatch_push
            await dispatch_push(
                user_id=scan["user_id"],
                title=title, body=body,
                url=f"/scanner?saved={scan['id']}",
            )
        except Exception as e:
            logger.debug("push dispatch failed: %s", e)

    # Email — best-effort
    if "email" in channels:
        try:
            from backend.core.database import get_supabase_admin
            sb = get_supabase_admin()
            u = sb.table("user_profiles").select("email").eq(
                "user_id", scan["user_id"]
            ).limit(1).execute()
            email = (u.data or [{}])[0].get("email")
            if email:
                import os
                from backend.platform.push import EmailService
                api_key = os.environ.get("RESEND_API_KEY", "")
                from_email = os.environ.get("EMAIL_FROM", "Quant X <no-reply@quantx.app>")
                emailer = EmailService(api_key=api_key, from_email=from_email)
                if emailer.is_available:
                    html = (
                        f"<h3>{title}</h3>"
                        f"<p>{body.replace(chr(10), '<br>')}</p>"
                        f"<p><a href='https://quantx.app/scanner?saved={scan['id']}'>Open scan</a></p>"
                    )
                    await emailer.send(to=email, subject=title, html=html)
        except Exception as e:
            logger.debug("email dispatch failed: %s", e)
