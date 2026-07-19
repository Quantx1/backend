"""
Admin system endpoints — health, command-center, kill switch, manual actions.

  GET  /admin/system/health                health probe + db/redis latency + metrics
  GET  /admin/scheduler/jobs               last N scheduler_job_runs rows
  GET  /admin/system/global-kill-switch    read kill-switch state
  POST /admin/system/global-kill-switch    flip kill switch (super_admin only)
  POST /admin/scan/trigger                 manual signal-gen run (super_admin only)
  POST /admin/scan/seed-demo               insert demo signals (super_admin, dev tool)
  POST /admin/kite/refresh-token           refresh Kite admin access_token
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from ...core.config import settings
from ._deps import (
    AdminRole,
    AdminUser,
    get_admin_user,
    get_supabase_admin,
    require_role,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================


class SystemHealthResponse(BaseModel):
    status: str
    timestamp: str
    database: str
    # PR 104 — DB round-trip latency on the health-check query. Supabase
    # speaks via PgBouncer over HTTPS, so we can't expose pg_stat_activity
    # pool counters directly; round-trip ms is the proxy ops cares about
    # ("technically up but slow during traffic spike").
    db_latency_ms: Optional[int] = None
    redis: str
    redis_latency_ms: Optional[int] = None
    scheduler_status: str
    last_signal_run: Optional[str]
    active_websocket_connections: int
    metrics: Dict[str, Any]


class GlobalKillSwitchPayload(BaseModel):
    active: bool
    reason: Optional[str] = None


# ============================================================================
# HEALTH
# ============================================================================


@router.get("/system/health", response_model=SystemHealthResponse)
async def get_system_health(admin: AdminUser = Depends(get_admin_user)):
    """Get system health status including scheduler and connections."""
    from ..app import scheduler_service, manager

    supabase = get_supabase_admin()

    # Database probe — PR 104 captures round-trip latency. The health-check
    # query itself is the probe; we time the full SDK round-trip.
    # >500ms downgrades status from "connected" to "slow".
    import time as _time
    db_status = "connected"
    db_latency_ms: Optional[int] = None
    _t0 = _time.perf_counter()
    try:
        supabase.table("subscription_plans").select("id").limit(1).execute()
        db_latency_ms = int((_time.perf_counter() - _t0) * 1000)
        if db_latency_ms > 500:
            db_status = "slow"
    except Exception:
        db_latency_ms = int((_time.perf_counter() - _t0) * 1000)
        db_status = "error"

    # Redis probe — same latency-capture pattern.
    redis_status = "disabled"
    redis_latency_ms: Optional[int] = None
    if settings.ENABLE_REDIS:
        _t0 = _time.perf_counter()
        try:
            if manager and manager.redis:
                await manager.redis.ping()
                redis_latency_ms = int((_time.perf_counter() - _t0) * 1000)
                redis_status = "slow" if redis_latency_ms > 200 else "connected"
            else:
                redis_status = "not_initialized"
        except Exception:
            redis_latency_ms = int((_time.perf_counter() - _t0) * 1000)
            redis_status = "error"

    # Scheduler status
    scheduler_status = "disabled"
    last_signal_run = None
    if settings.ENABLE_SCHEDULER and scheduler_service:
        scheduler_status = "running" if scheduler_service.scheduler.running else "stopped"
        last_signal = (
            supabase.table("signals")
            .select("generated_at")
            .order("generated_at", desc=True)
            .limit(1)
            .execute()
        )
        if last_signal.data:
            last_signal_run = last_signal.data[0].get("generated_at")

    # WebSocket connections
    ws_connections = manager.get_connection_count() if manager else 0

    # Metrics rollup
    metrics: Dict[str, Any] = {}
    metrics["total_users"] = (
        supabase.table("user_profiles").select("id", count="exact").execute().count or 0
    )
    metrics["active_subscribers"] = (
        supabase.table("user_profiles")
        .select("id", count="exact")
        .eq("subscription_status", "active")
        .execute()
        .count
        or 0
    )
    today = date.today().isoformat()
    metrics["today_signals"] = (
        supabase.table("signals")
        .select("id", count="exact")
        .eq("date", today)
        .execute()
        .count
        or 0
    )
    metrics["today_trades"] = (
        supabase.table("trades")
        .select("id", count="exact")
        .gte("created_at", today)
        .execute()
        .count
        or 0
    )
    metrics["active_positions"] = (
        supabase.table("positions")
        .select("id", count="exact")
        .eq("is_active", True)
        .execute()
        .count
        or 0
    )

    return SystemHealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        database=db_status,
        db_latency_ms=db_latency_ms,
        redis=redis_status,
        redis_latency_ms=redis_latency_ms,
        scheduler_status=scheduler_status,
        last_signal_run=last_signal_run,
        active_websocket_connections=ws_connections,
        metrics=metrics,
    )


# ============================================================================
# SCHEDULER HISTORY (N9 command center)
# ============================================================================


@router.get("/scheduler/jobs")
async def list_scheduler_jobs(
    job_id: Optional[str] = Query(None, description="Filter by job id (e.g. 'weekly_review_generate')"),
    status: Optional[str] = Query(None, description="Filter by status (ok | failed | skipped)"),
    limit: int = Query(50, ge=1, le=200),
    admin: AdminUser = Depends(get_admin_user),
):
    """Browse the last N scheduler_job_runs rows. Optional filters by
    job_id and status. Rows include items_processed / err_msg / metadata
    so ops can diagnose without SSH."""
    client = get_supabase_admin()
    try:
        q = (
            client.table("scheduler_job_runs")
            .select(
                "id, job_id, started_at, finished_at, status, "
                "err_msg, items_processed, metadata"
            )
            .order("started_at", desc=True)
            .limit(limit)
        )
        if job_id:
            q = q.eq("job_id", job_id)
        if status:
            q = q.eq("status", status)
        resp = q.execute()
        rows = resp.data or []
    except Exception as exc:
        logger.warning("scheduler jobs query failed: %s", exc)
        rows = []

    # Latest-per-job rollup so the UI can render a summary strip.
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        jid = row.get("job_id")
        if jid and jid not in latest:
            latest[jid] = row

    return {
        "rows": rows,
        "latest_by_job": list(latest.values()),
        "count": len(rows),
        "computed_at": datetime.utcnow().isoformat(),
    }


# ============================================================================
# GLOBAL KILL SWITCH
# ============================================================================


@router.get("/system/global-kill-switch")
async def get_global_kill_switch(admin: AdminUser = Depends(get_admin_user)):
    """Read current state of the platform-wide kill switch."""
    client = get_supabase_admin()
    try:
        rows = (
            client.table("system_flags")
            .select("key, value, description, updated_by, updated_at")
            .eq("key", "global_kill_switch")
            .limit(1)
            .execute()
        )
        row = (rows.data or [None])[0]
    except Exception as exc:
        logger.error("global kill switch read failed: %s", exc)
        row = None
    if not row:
        return {"active": False, "reason": None, "updated_by": None, "updated_at": None}

    value = row.get("value") or {}
    return {
        "active": bool(value.get("active", False)),
        "reason": value.get("reason"),
        "updated_by": row.get("updated_by"),
        "updated_at": row.get("updated_at"),
        "description": row.get("description"),
    }


@router.post("/system/global-kill-switch")
async def set_global_kill_switch(
    body: GlobalKillSwitchPayload,
    http_request: Request = None,
    admin: AdminUser = Depends(require_role(AdminRole.SUPER_ADMIN)),
):
    """Flip the global kill switch. Super-admin only — once active,
    every order-placing path stops until the flag is cleared."""
    client = get_supabase_admin()
    value = {"active": bool(body.active), "reason": body.reason}
    try:
        client.table("system_flags").upsert({
            "key": "global_kill_switch",
            "value": value,
            "updated_by": admin.id,
            "updated_at": datetime.utcnow().isoformat(),
        }, on_conflict="key").execute()
    except Exception as exc:
        logger.error("global kill switch write failed: %s", exc)
        raise HTTPException(status_code=500, detail="persist_failed")

    logger.warning(
        "GLOBAL_KILL_SWITCH flipped by admin=%s to active=%s reason=%s",
        admin.id, body.active, body.reason,
    )

    # PR 48 — invalidate the cached flag so every worker picks up the
    # new state on the next order attempt (TTL would otherwise lag 15s).
    try:
        from ...platform.system_flags import invalidate_cache
        invalidate_cache("global_kill_switch")
    except Exception:
        pass

    # Analytics — separate event from the per-user kill switch fired in app.py
    try:
        from ...observability import EventName, track
        track(EventName.KILL_SWITCH_FIRED, admin.id, {
            "scope": "global",
            "active": bool(body.active),
            "reason": body.reason or "",
        })
    except Exception:
        pass

    from ...platform.admin_audit import log_admin_action
    log_admin_action(
        actor_id=admin.id, actor_email=admin.email,
        action="global_kill_switch_flip",
        target_type="system_flag", target_id="global_kill_switch",
        payload={"active": body.active, "reason": body.reason},
        request=http_request, supabase_client=client,
    )

    return {
        "active": body.active,
        "reason": body.reason,
        "updated_by": admin.id,
        "updated_at": datetime.utcnow().isoformat(),
    }


# ============================================================================
# MANUAL SIGNAL SCAN (TESTING / ON-DEMAND)
# ============================================================================


@router.post("/scan/trigger")
async def trigger_manual_scan(
        symbols: Optional[str] = Query(
            None,
            description="Comma-separated symbols (e.g., RELIANCE,INFY). Leave empty for full universe."),
        max_stocks: int = Query(
            20,
            ge=1,
            le=300,
            description="Max stocks to scan"),
        http_request: Request = None,
        admin: AdminUser = Depends(
            require_role(
                AdminRole.SUPER_ADMIN)),
):
    """
    Manually trigger signal generation on-demand.
    Useful for testing outside market hours.
    Uses historical data — does NOT require live market.
    """
    from ..app import signal_generator

    if not signal_generator:
        raise HTTPException(status_code=503, detail="Signal generator not initialized. Start backend first.")

    candidates = None
    if symbols:
        candidates = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    # PR 95 — audit logging now wraps both success and failure paths.
    # The audit log is the source of truth for *attempted* admin actions,
    # not just successful ones. Previously a failed scan vanished from
    # the log entirely, so an operator reviewing later couldn't tell
    # who tried what or why it broke.
    from ...platform.admin_audit import log_admin_action
    target_id = (symbols or "full_universe")[:120]
    base_payload = {"symbols": symbols, "max_stocks": max_stocks}

    try:
        signals = await signal_generator.generate_intraday_signals(
            save=True,
            candidates=candidates[:max_stocks] if candidates else None,
        )

        log_admin_action(
            actor_id=admin.id, actor_email=admin.email,
            action="manual_scan_trigger", target_type="system",
            target_id=target_id,
            payload={
                **base_payload,
                "success": True,
                "signals_generated": len(signals),
            },
            request=http_request,
        )

        return {
            "status": "ok",
            "signals_generated": len(signals),
            "symbols_scanned": len(candidates) if candidates else "full universe",
            "signals": [
                {
                    "symbol": s.symbol,
                    "direction": s.direction,
                    "entry": s.entry_price,
                    "target": s.target_1,
                    "sl": s.stop_loss,
                    "confidence": s.confidence,
                    "strategies": s.reasons[:3] if s.reasons else [],
                }
                for s in signals
            ],
        }
    except Exception as e:
        logger.error(f"Manual scan failed: {e}", exc_info=True)
        try:
            log_admin_action(
                actor_id=admin.id, actor_email=admin.email,
                action="manual_scan_trigger", target_type="system",
                target_id=target_id,
                payload={
                    **base_payload,
                    "success": False,
                    "error": str(e)[:500],
                },
                request=http_request,
            )
        except Exception as audit_exc:
            logger.warning("manual_scan_trigger audit-log write failed: %s", audit_exc)
        raise HTTPException(status_code=500, detail=f"Scan failed: {e}")


@router.post("/scan/seed-demo")
async def seed_demo_signals(
    count: int = Query(10, ge=1, le=50, description="Number of demo signals to insert"),
    http_request: Request = None,
    admin: AdminUser = Depends(require_role(AdminRole.SUPER_ADMIN)),
):
    """Insert realistic demo signals into the database for testing
    frontend display without needing live data or Kite token.

    HARD-BLOCKED in production: these are randomly-fabricated signals (fake
    model scores / confidence / expected returns). Writing them into the live
    signals table would pollute the real, honest track record — a no-fabrication
    and no-misleading-performance violation. Non-prod environments only.
    """
    import random

    from ...core.config import settings
    if str(getattr(settings, "APP_ENV", "")).lower() in ("production", "prod"):
        raise HTTPException(
            status_code=403,
            detail="seed-demo is disabled in production — it fabricates signals "
                   "and would pollute the real track record.",
        )

    supabase = get_supabase_admin()
    today = date.today().isoformat()

    # Realistic NSE stocks
    stocks = [
        ("RELIANCE", 2890.50), ("INFY", 1845.30), ("TCS", 4120.75),
        ("HDFCBANK", 1672.40), ("ICICIBANK", 1245.80), ("KOTAKBANK", 1890.20),
        ("BHARTIARTL", 1580.90), ("ITC", 465.30), ("HINDUNILVR", 2340.60),
        ("BAJFINANCE", 7120.50), ("SBIN", 780.40), ("MARUTI", 12450.00),
        ("SUNPHARMA", 1780.20), ("TATASTEEL", 152.80), ("WIPRO", 485.60),
        ("LT", 3520.40), ("ADANIENT", 2890.70), ("TITAN", 3450.80),
        ("NESTLEIND", 2180.50), ("DRREDDY", 6780.30), ("ULTRACEMCO", 11200.00),
        ("HCLTECH", 1720.90), ("COALINDIA", 385.40), ("JSWSTEEL", 920.60),
        ("GRASIM", 2680.30), ("CIPLA", 1520.70), ("DIVISLAB", 5890.20),
        ("EICHERMOT", 4950.80), ("TATAPOWER", 425.30), ("DLF", 870.50),
    ]

    strategies = [
        "Consolidation_Breakout", "Trend_Pullback", "Reversal_Patterns",
        "Candle_Reversal", "BOS_Structure", "Volume_Reversal",
    ]

    inserted = []
    sample = random.sample(stocks, min(count, len(stocks)))

    for symbol, base_price in sample:
        direction = random.choice(["LONG", "SHORT"])
        confidence = round(random.uniform(65, 92), 1)
        strategy = random.choice(strategies)

        if direction == "LONG":
            entry = round(base_price * random.uniform(0.98, 1.02), 2)
            sl = round(entry * random.uniform(0.95, 0.97), 2)
            t1 = round(entry * random.uniform(1.03, 1.06), 2)
            t2 = round(entry * random.uniform(1.06, 1.10), 2)
            t3 = round(entry * random.uniform(1.10, 1.15), 2)
        else:
            entry = round(base_price * random.uniform(0.98, 1.02), 2)
            sl = round(entry * random.uniform(1.03, 1.05), 2)
            t1 = round(entry * random.uniform(0.94, 0.97), 2)
            t2 = round(entry * random.uniform(0.90, 0.94), 2)
            t3 = round(entry * random.uniform(0.85, 0.90), 2)

        rr = round(abs(t1 - entry) / abs(entry - sl), 2) if abs(entry - sl) > 0 else 2.0

        signal_data = {
            "symbol": symbol,
            "exchange": "NSE",
            "segment": "EQUITY",
            "direction": direction,
            "signal_type": "swing",
            "confidence": confidence,
            "catboost_score": round(random.uniform(0.5, 0.9), 2),
            "tft_score": round(random.uniform(0.4, 0.85), 2),
            "stockformer_score": round(random.uniform(55, 90), 1),
            "entry_price": entry,
            "stop_loss": sl,
            "target_1": t1,
            "target_2": t2,
            "target_3": t3,
            "risk_reward": rr,
            "expected_return": round(abs(t1 - entry) / entry * 100, 2),
            "max_loss_percent": round(abs(entry - sl) / entry * 100, 2),
            "reasons": [strategy, f"Confidence {confidence}%", f"R:R {rr}"],
            "strategy_names": [strategy],
            "status": "active",
            "date": today,
            "is_premium": random.choice([True, False]),
        }

        try:
            result = supabase.table("signals").insert(signal_data).execute()
            if result.data:
                inserted.append({"symbol": symbol, "direction": direction, "confidence": confidence})
        except Exception as e:
            logger.warning(f"Failed to insert demo signal for {symbol}: {e}")

    from ...platform.admin_audit import log_admin_action
    log_admin_action(
        actor_id=admin.id, actor_email=admin.email,
        action="seed_demo_signals", target_type="signal",
        payload={"count": count, "inserted": len(inserted)},
        request=http_request, supabase_client=supabase,
    )

    return {
        "status": "ok",
        "inserted": len(inserted),
        "date": today,
        "signals": inserted,
    }


# ============================================================================
# KITE ADMIN TOKEN REFRESH
# ============================================================================


@router.post("/kite/refresh-token", response_model=dict)
async def refresh_kite_admin_token(
    request_token: str = Query(..., description="Kite request_token from login callback"),
    http_request: Request = None,
    admin: AdminUser = Depends(get_admin_user),
):
    """Exchange a Kite request_token for a new access_token.

    Admin logs into Kite Connect, gets redirected with request_token,
    then calls this endpoint to refresh the app-wide access token.
    Token expires at 6 AM IST daily.
    """
    if settings.DATA_PROVIDER != "kite":
        raise HTTPException(status_code=400, detail="Kite token refresh not needed in free data mode")

    from ...data.providers.kite import get_kite_admin_client
    client = get_kite_admin_client()
    if not client.kite:
        raise HTTPException(status_code=500, detail="Kite admin client not initialized")

    try:
        session = client.kite.generate_session(request_token, settings.KITE_ADMIN_API_SECRET)
        new_token = session["access_token"]
        client.set_access_token(new_token)

        from ...platform.admin_audit import log_admin_action
        log_admin_action(
            actor_id=admin.id, actor_email=admin.email,
            action="kite_token_refresh", target_type="other", target_id="kite_admin",
            request=http_request,
        )

        return {
            "status": "ok",
            "message": "Kite access token refreshed successfully",
            "valid_until": "06:00 AM IST tomorrow",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Token refresh failed: {e}")
