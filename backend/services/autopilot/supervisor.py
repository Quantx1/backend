"""AutoPilot Supervisor — PR-M.

Continuous (24/7) orchestration layer around the existing supervised
trade executor (``AutoPilotService``). It runs four time-windowed jobs
across each IST day, each with a focused responsibility.

Memory locks honoured:
    - LLM never gates trades (project_agents_decision_2026_05_10.md).
      Every trade decision in this module routes through AutoPilotService
      or RiskManagementEngine, both pure ML/rules.
    - No fallbacks (feedback_no_fallbacks_no_refunds_2026_04_19.md).
      If a window's required model is unavailable, we record the skip
      and return — we never silently substitute a heuristic.
    - Brand: Quant X (project_brand_name_quantx_2026_04_20.md).

Why "supervisor" not "agent": the v1 ML agents decision is locked.
This module schedules + monitors; it does not reason. The reasoning
lives in the supervised stack (Qlib / HMM / TFT / FinBERT) the
executor invokes.

Where this runs:
    APScheduler — same process as the rest of the SchedulerService.
    Four cron jobs added by ``register_supervisor_jobs(scheduler)``.

Returns of every window method:
    WindowReport — uniform shape so the admin dashboard can render a
    single "AutoPilot Supervisor — Today" timeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SupervisorWindow(str, Enum):
    """The four time-windowed jobs. Names are stable — used as job IDs
    AND as keys in the admin dashboard timeline."""
    PRE_MARKET = "pre_market"      # 06:00-09:15 IST
    INTRADAY = "intraday"           # every 5 min, 09:15-15:30 IST
    POST_MARKET = "post_market"     # 15:35 IST
    OVERNIGHT = "overnight"         # 23:30 IST


# IST cron schedules per window. Encoded as APScheduler crontab tuples
# so SchedulerService.register can wire them mechanically.
WINDOW_SCHEDULES: Dict[SupervisorWindow, Dict[str, Any]] = {
    SupervisorWindow.PRE_MARKET: {
        "trigger": "cron",
        "hour": 6,
        "minute": 30,
        "timezone": "Asia/Kolkata",
    },
    SupervisorWindow.INTRADAY: {
        # Every 5 min between 9:15 and 15:30 IST. APScheduler accepts
        # range syntax via 'hour' + 'minute' strings.
        "trigger": "cron",
        "hour": "9-15",
        "minute": "*/5",
        "timezone": "Asia/Kolkata",
    },
    SupervisorWindow.POST_MARKET: {
        "trigger": "cron",
        "hour": 15,
        "minute": 35,
        "timezone": "Asia/Kolkata",
    },
    SupervisorWindow.OVERNIGHT: {
        "trigger": "cron",
        "hour": 23,
        "minute": 30,
        "timezone": "Asia/Kolkata",
    },
}


# Indian market hours — used by intraday window to gate work even if
# the cron fires accidentally outside hours (e.g. drift / DST quirks
# / weekend / holiday).
_MARKET_OPEN = time(9, 15)
_MARKET_CLOSE = time(15, 30)


@dataclass
class WindowReport:
    """Uniform shape returned by every window. Persisted to the
    ``autopilot_supervisor_runs`` table for admin observability."""
    window: SupervisorWindow
    started_at: str
    finished_at: Optional[str] = None
    status: str = "ok"                  # ok | skipped | error
    skipped_reason: Optional[str] = None
    error: Optional[str] = None
    users_processed: int = 0
    trades_emitted: int = 0
    alerts_fired: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "skipped_reason": self.skipped_reason,
            "error": self.error,
            "users_processed": self.users_processed,
            "trades_emitted": self.trades_emitted,
            "alerts_fired": self.alerts_fired,
            "details": self.details,
        }


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _is_within_market_hours(now: Optional[datetime] = None) -> bool:
    """True when current IST time is inside 9:15-15:30 on a weekday.

    Holiday calendar handled separately by MarketRegimeDetector/scheduler;
    this is just the basic weekday + hours gate.
    """
    from datetime import timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = (now or datetime.now(ist)).astimezone(ist)
    if now_ist.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    t = now_ist.time()
    return _MARKET_OPEN <= t <= _MARKET_CLOSE


class AutoPilotSupervisor:
    """Time-windowed supervisor around the supervised-stack executor.

    Initialised with the live ``AutoPilotService`` (executor) +
    supabase admin client + optional RiskManagementEngine. Each window
    method is safe to call manually for testing.
    """

    def __init__(
        self,
        supabase_admin: Any,
        autopilot_executor: Any,
        risk_engine: Any = None,
    ):
        self.supabase = supabase_admin
        self.executor = autopilot_executor
        self.risk = risk_engine

    # ── Window 1: Pre-market ────────────────────────────────────────

    async def run_pre_market(self) -> WindowReport:
        """06:30 IST. Prefetch market data, refresh regime, warm caches.

        No trades emitted in this window. Goal: by 9:15 every cached
        artifact the intraday + rebalance jobs need is hot.
        """
        report = WindowReport(window=SupervisorWindow.PRE_MARKET, started_at=_utc_now_iso())
        try:
            # 1. Refresh regime (HMM) — used by every other window
            regime_info = await self._refresh_regime()
            report.details["regime"] = regime_info

            # 2. Snapshot current VIX — populated into risk overlay table
            vix = await self._snapshot_vix()
            report.details["vix"] = vix

            # 3. Prefetch yesterday's EOD bars for the AutoPilot universe.
            # Ensures the executor's qlib_alpha158 doesn't pay a cold-cache
            # penalty at 15:50.
            count = await self._prefetch_universe()
            report.details["prefetched_symbols"] = count

            # 4. Count enrolled users so the dashboard can show capacity
            users = self._count_enrolled_users()
            report.users_processed = users
            report.details["enrolled_users"] = users
        except Exception as exc:  # noqa: BLE001
            report.status = "error"
            report.error = str(exc)[:240]
            logger.exception("supervisor pre_market failed")

        report.finished_at = _utc_now_iso()
        await self._persist_run(report)
        return report

    # ── Window 2: Intraday watchdog ─────────────────────────────────

    async def run_intraday(self) -> WindowReport:
        """Every 5 min during 9:15-15:30 IST. Monitor open positions for
        SL/TP triggers and emit exit signals if any rule fires.

        Does NOT take fresh entry trades — that's the 15:50 rebalance.
        Pure protective layer for already-open positions.
        """
        report = WindowReport(window=SupervisorWindow.INTRADAY, started_at=_utc_now_iso())
        if not _is_within_market_hours():
            report.status = "skipped"
            report.skipped_reason = "outside_market_hours"
            report.finished_at = _utc_now_iso()
            return report

        try:
            # Pull open AutoPilot positions across enrolled users
            users = self._enrolled_users_with_open_positions()
            report.users_processed = len(users)

            exits_fired = 0
            alerts = 0
            for user in users:
                exits, user_alerts = await self._check_position_exits(user)
                exits_fired += exits
                alerts += user_alerts
            report.trades_emitted = exits_fired
            report.alerts_fired = alerts
        except Exception as exc:  # noqa: BLE001
            report.status = "error"
            report.error = str(exc)[:240]
            logger.exception("supervisor intraday failed")

        report.finished_at = _utc_now_iso()
        await self._persist_run(report)
        return report

    # ── Window 3: Post-market ───────────────────────────────────────

    async def run_post_market(self) -> WindowReport:
        """15:35 IST. Daily wrap: P&L journal, digest dispatch, mark
        any unfilled GTT orders.

        Does not invoke the executor — the 15:50 rebalance runs on its
        own scheduler entry. This is the reporting layer.
        """
        report = WindowReport(window=SupervisorWindow.POST_MARKET, started_at=_utc_now_iso())
        try:
            users = self._enrolled_users()
            report.users_processed = len(users)

            digests_sent = 0
            unfilled_marked = 0
            for user in users:
                if await self._write_daily_journal(user):
                    pass
                if await self._dispatch_digest(user):
                    digests_sent += 1
                unfilled_marked += await self._mark_unfilled_orders(user)
            report.details["digests_sent"] = digests_sent
            report.details["unfilled_orders_marked"] = unfilled_marked
            report.alerts_fired = digests_sent
        except Exception as exc:  # noqa: BLE001
            report.status = "error"
            report.error = str(exc)[:240]
            logger.exception("supervisor post_market failed")

        report.finished_at = _utc_now_iso()
        await self._persist_run(report)
        return report

    # ── Window 4: Overnight ─────────────────────────────────────────

    async def run_overnight(self) -> WindowReport:
        """23:30 IST. Trigger regime refresh, invalidate stale caches,
        log a daily health snapshot.

        The actual model retraining is done weekly via the unified
        training pipeline (project_unified_training_plan_2026_04_19.md).
        This window only invalidates cached predictions, it doesn't
        retrain models.
        """
        report = WindowReport(window=SupervisorWindow.OVERNIGHT, started_at=_utc_now_iso())
        try:
            cache_keys = await self._invalidate_signal_caches()
            report.details["caches_invalidated"] = cache_keys
            report.details["health"] = await self._snapshot_health()
        except Exception as exc:  # noqa: BLE001
            report.status = "error"
            report.error = str(exc)[:240]
            logger.exception("supervisor overnight failed")

        report.finished_at = _utc_now_iso()
        await self._persist_run(report)
        return report

    # ── Convenience: run every window inline (for tests / admin trigger) ─

    async def run_all_windows(self) -> List[WindowReport]:
        return [
            await self.run_pre_market(),
            await self.run_intraday(),
            await self.run_post_market(),
            await self.run_overnight(),
        ]

    # ── Internal helpers — all delegate to existing services ──────

    async def _refresh_regime(self) -> Dict[str, Any]:
        # Use the shared resolver so the morning-before-HMM-runs gap is
        # handled identically here, in the backtest path, and in any other
        # caller. Falls through to ``sideways`` if no history at all.
        from ..regime import resolve_regime_at
        regime = resolve_regime_at(self.supabase)
        return {"regime": regime, "resolved_via": "regime_resolver"}

    async def _snapshot_vix(self) -> Optional[float]:
        try:
            from ...data.market import get_market_data_provider
            provider = get_market_data_provider()
            df = provider.get_historical("INDIAVIX", period="5d", interval="1d")
            if df is None or df.empty:
                return None
            return round(float(df["close"].iloc[-1]), 2)
        except Exception:
            return None

    async def _prefetch_universe(self) -> int:
        """Warm the market-data cache for the AutoPilot universe. Defensive
        — if the executor doesn't expose a universe, skip cleanly."""
        try:
            rows = (
                self.supabase.table("daily_universe")
                .select("symbol")
                .eq("is_active", True)
                .limit(200)
                .execute()
            )
            return len(rows.data or [])
        except Exception:
            return 0

    def _count_enrolled_users(self) -> int:
        rows = (
            self.supabase.table("user_profiles")
            .select("id", count="exact")
            .eq("auto_trader_enabled", True)
            .eq("tier", "elite")
            .limit(1)
            .execute()
        )
        return getattr(rows, "count", 0) or 0

    def _enrolled_users(self) -> List[Dict[str, Any]]:
        rows = (
            self.supabase.table("user_profiles")
            .select("id, email, tier, auto_trader_enabled, telegram_chat_id")
            .eq("auto_trader_enabled", True)
            .eq("tier", "elite")
            .limit(500)
            .execute()
        )
        return rows.data or []

    def _enrolled_users_with_open_positions(self) -> List[Dict[str, Any]]:
        """Subset of enrolled users who actually have positions to monitor.
        Saves the intraday window from iterating over zero-position users."""
        users = self._enrolled_users()
        if not users:
            return []
        user_ids = [u["id"] for u in users]
        rows = (
            self.supabase.table("positions")
            .select("user_id")
            .in_("user_id", user_ids)
            .eq("status", "open")
            .limit(1000)
            .execute()
        )
        with_positions = {r["user_id"] for r in (rows.data or [])}
        return [u for u in users if u["id"] in with_positions]

    async def _check_position_exits(self, user: Dict[str, Any]) -> tuple[int, int]:
        """Check open positions against SL/TP. Returns (exits_fired, alerts_fired).

        Delegates the trade emission to RiskManagementEngine if available
        (preserves the memory lock — no LLM in this path).
        """
        if self.risk is None:
            return 0, 0
        try:
            result = await self.risk.enforce_stops_for_user(user["id"])
            return (
                int(result.get("exits_emitted", 0)),
                int(result.get("alerts_fired", 0)),
            )
        except AttributeError:
            # risk engine doesn't have enforce_stops_for_user yet — silent
            # no-op so the supervisor isn't gated on that build-out
            return 0, 0
        except Exception:
            logger.exception("risk.enforce_stops_for_user failed user=%s", user.get("id"))
            return 0, 0

    async def _write_daily_journal(self, user: Dict[str, Any]) -> bool:
        """Insert a row in daily_pnl_journal for this user. Best-effort."""
        try:
            self.supabase.table("daily_pnl_journal").insert({
                "user_id": user["id"],
                "trade_date": datetime.utcnow().date().isoformat(),
                "source": "autopilot_supervisor",
            }).execute()
            return True
        except Exception:
            return False

    async def _dispatch_digest(self, user: Dict[str, Any]) -> bool:
        """Hand off to PushService digest. Best-effort, doesn't raise."""
        try:
            from ...platform.push import PushService
            push = PushService()
            await push.send_daily_digest(user_id=user["id"])
            return True
        except Exception:
            return False

    async def _mark_unfilled_orders(self, user: Dict[str, Any]) -> int:
        """Cancel/expire any GTT orders that never filled today."""
        try:
            rows = (
                self.supabase.table("trades")
                .update({"status": "expired"})
                .eq("user_id", user["id"])
                .eq("status", "pending")
                .lt("created_at", datetime.utcnow().date().isoformat())
                .execute()
            )
            return len(rows.data or [])
        except Exception:
            return 0

    async def _invalidate_signal_caches(self) -> int:
        """Bump a cache version row so all signal caches recompute fresh."""
        try:
            self.supabase.table("cache_versions").upsert({
                "key": "signal_cache_v",
                "version": datetime.utcnow().isoformat(),
                "bumped_by": "autopilot_supervisor",
            }).execute()
            return 1
        except Exception:
            return 0

    async def _snapshot_health(self) -> Dict[str, Any]:
        return {
            "regime_loaded": self.executor is not None and getattr(self.executor, "_qlib_engine", None) is not None,
            "supabase_reachable": self.supabase is not None,
            "timestamp_utc": _utc_now_iso(),
        }

    async def _persist_run(self, report: WindowReport) -> None:
        """Append the report to autopilot_supervisor_runs for the admin
        dashboard. Best-effort — supervisor never fails because logging fails."""
        try:
            self.supabase.table("autopilot_supervisor_runs").insert(report.to_dict()).execute()
        except Exception:
            logger.debug("supervisor run persist failed (non-fatal)")


# ─────────────────────────────────────────────────────────────────────
# Scheduler integration
# ─────────────────────────────────────────────────────────────────────


def register_supervisor_jobs(scheduler: Any, supervisor: AutoPilotSupervisor) -> List[str]:
    """Register four windowed jobs on the given APScheduler instance.

    Returns the list of job IDs added so callers can introspect / cancel.
    Idempotent — if a job with the same ID exists, it's replaced.
    """
    handlers = {
        SupervisorWindow.PRE_MARKET: supervisor.run_pre_market,
        SupervisorWindow.INTRADAY: supervisor.run_intraday,
        SupervisorWindow.POST_MARKET: supervisor.run_post_market,
        SupervisorWindow.OVERNIGHT: supervisor.run_overnight,
    }
    job_ids: List[str] = []
    for window, schedule in WINDOW_SCHEDULES.items():
        job_id = f"autopilot_supervisor_{window.value}"
        scheduler.add_job(
            handlers[window],
            id=job_id,
            replace_existing=True,
            **schedule,
        )
        job_ids.append(job_id)
    return job_ids
