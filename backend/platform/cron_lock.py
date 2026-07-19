"""System-level cron idempotency.

T1.2 (2026-05-31) — closes the catastrophic gap where a cron firing
twice (Railway restart / scheduler bug / pod auto-scale) would re-run
the AutoPilot daily rebalance and place every Elite user's portfolio
TWICE at the broker.

Pattern:
    async with cron_lock(supabase, 'autopilot_rebalance') as locked:
        if not locked:
            return  # another instance is already running today's job
        do_the_thing()

Locking is atomic at the DB layer via UNIQUE(job_id, run_date). The
second cron firing fails the INSERT and `cron_lock` yields False — no
distributed-lock library needed, no race window.

Errors during the work block are recorded on the lock row so the
admin dashboard can show "yesterday's autopilot_rebalance failed at
this step" instead of silently moving on.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import time
from datetime import datetime, date
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


def _pid_tag() -> str:
    """Hostname:pid string for the lock-holder identity column."""
    try:
        return f"{socket.gethostname()}:{os.getpid()}"
    except Exception:
        return f"unknown:{os.getpid()}"


def _ist_today() -> str:
    """India trading day in YYYY-MM-DD — matches the DB DEFAULT."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
    except Exception:
        return date.today().isoformat()


@contextlib.asynccontextmanager
async def cron_lock(
    supabase: Any,
    job_id: str,
    *,
    run_date: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> AsyncIterator[bool]:
    """Acquire a per-(job_id, run_date) lock; yield True on success.

    On second invocation of the same (job_id, run_date), the UNIQUE
    constraint causes the INSERT to fail and we yield False. The caller
    must check the yielded value and short-circuit:

        async with cron_lock(sb, 'autopilot_rebalance') as ok:
            if not ok:
                logger.info("already ran today, skipping")
                return
            ...

    On exception inside the block, the lock row is marked 'failed'
    with the error text. This means the SAME DAY's job can NOT be
    retried automatically — that's intentional. For trades-side work,
    "auto-retry on failure" is more dangerous than "alert ops + leave
    alone". Ops can manually clear the row to permit retry.
    """
    rd = run_date or _ist_today()
    pid = _pid_tag()
    start = time.monotonic()

    # Step 1: atomic INSERT — succeeds for first caller, fails for second.
    try:
        supabase.table("system_cron_runs").insert({
            "job_id": job_id,
            "run_date": rd,
            "status": "running",
            "pid": pid,
            "metadata": metadata or {},
        }).execute()
    except Exception as e:
        msg = str(e).lower()
        # Duplicate key violation -> another instance owns today's job
        if "duplicate" in msg or "unique" in msg or "23505" in msg:
            logger.warning(
                "cron_lock(%s, %s): already running/completed by another process",
                job_id, rd,
            )
            yield False
            return
        # Unknown error — log + skip safely (no destructive op outside the block)
        logger.error("cron_lock(%s, %s) acquire failed: %s", job_id, rd, e)
        yield False
        return

    # Step 2: caller does work
    err_text: Optional[str] = None
    try:
        yield True
    except Exception as e:
        err_text = f"{type(e).__name__}: {str(e)[:500]}"
        logger.exception("cron_lock(%s, %s) work raised: %s", job_id, rd, e)
        raise
    finally:
        # Step 3: settle the lock row to 'completed' or 'failed'.
        duration_ms = int((time.monotonic() - start) * 1000)
        try:
            supabase.table("system_cron_runs").update({
                "completed_at": datetime.utcnow().isoformat(),
                "status": "failed" if err_text else "completed",
                "duration_ms": duration_ms,
                "error": err_text,
            }).eq("job_id", job_id).eq("run_date", rd).execute()
        except Exception as e:
            logger.warning(
                "cron_lock(%s, %s) settle failed: %s — row may stay in 'running'",
                job_id, rd, e,
            )
