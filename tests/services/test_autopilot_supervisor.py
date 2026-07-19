"""AutoPilot Supervisor tests — PR-M.

Pure unit tests with a mocked Supabase client + mocked executor.
No real scheduling — we invoke each window method directly.
"""

from __future__ import annotations

from datetime import datetime, time, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.services.autopilot.supervisor import (
    AutoPilotSupervisor,
    SupervisorWindow,
    WINDOW_SCHEDULES,
    WindowReport,
    _is_within_market_hours,
    register_supervisor_jobs,
)


@pytest.fixture
def supabase():
    """A mock supabase client where every chain returns empty data."""
    sb = MagicMock()
    chain = MagicMock()
    chain.execute.return_value = MagicMock(data=[], count=0)
    for attr in ("select", "eq", "neq", "in_", "lt", "order", "limit", "update", "insert", "upsert"):
        getattr(chain, attr).return_value = chain
    sb.table.return_value = chain
    return sb


@pytest.fixture
def executor():
    m = MagicMock()
    m._qlib_engine = MagicMock()  # so the health snapshot reports loaded
    return m


@pytest.fixture
def supervisor(supabase, executor):
    return AutoPilotSupervisor(
        supabase_admin=supabase,
        autopilot_executor=executor,
        risk_engine=None,
    )


# ─────────────────────────────────────────────────────────────────────
# Window schedules + enum
# ─────────────────────────────────────────────────────────────────────


class TestSchedules:

    def test_all_four_windows_have_schedules(self):
        for w in SupervisorWindow:
            assert w in WINDOW_SCHEDULES
            assert WINDOW_SCHEDULES[w]["trigger"] == "cron"
            assert WINDOW_SCHEDULES[w]["timezone"] == "Asia/Kolkata"

    def test_intraday_uses_5min_interval(self):
        sched = WINDOW_SCHEDULES[SupervisorWindow.INTRADAY]
        assert sched["minute"] == "*/5"
        assert sched["hour"] == "9-15"

    def test_pre_market_before_open(self):
        sched = WINDOW_SCHEDULES[SupervisorWindow.PRE_MARKET]
        assert sched["hour"] == 6
        assert 0 <= sched["minute"] <= 59
        # Must be before 9:15 open
        assert (sched["hour"], sched["minute"]) < (9, 15)


# ─────────────────────────────────────────────────────────────────────
# Market-hours gate
# ─────────────────────────────────────────────────────────────────────


class TestMarketHoursGate:

    def _ist(self, weekday: int, hh: int, mm: int) -> datetime:
        """Build a datetime that is `weekday` (0=Mon) at hh:mm IST."""
        ist = timezone(timedelta(hours=5, minutes=30))
        base = datetime(2026, 5, 25, hh, mm, tzinfo=ist)  # 2026-05-25 is Monday
        return base + timedelta(days=weekday - 0)

    def test_inside_market_hours_monday(self):
        # Monday 11:00 IST → inside
        assert _is_within_market_hours(self._ist(0, 11, 0)) is True

    def test_before_open(self):
        assert _is_within_market_hours(self._ist(0, 9, 0)) is False

    def test_after_close(self):
        assert _is_within_market_hours(self._ist(0, 15, 31)) is False

    def test_weekend_saturday(self):
        # Saturday inside-hours time → still False
        assert _is_within_market_hours(self._ist(5, 11, 0)) is False


# ─────────────────────────────────────────────────────────────────────
# Window: pre-market
# ─────────────────────────────────────────────────────────────────────


class TestPreMarket:

    @pytest.mark.asyncio
    async def test_pre_market_returns_ok_report(self, supervisor):
        report = await supervisor.run_pre_market()
        assert isinstance(report, WindowReport)
        assert report.window == SupervisorWindow.PRE_MARKET
        assert report.status == "ok"
        assert report.trades_emitted == 0
        assert "regime" in report.details

    @pytest.mark.asyncio
    async def test_pre_market_records_to_persist(self, supervisor, supabase):
        await supervisor.run_pre_market()
        # autopilot_supervisor_runs.insert must have been called
        supabase.table.assert_any_call("autopilot_supervisor_runs")


# ─────────────────────────────────────────────────────────────────────
# Window: intraday
# ─────────────────────────────────────────────────────────────────────


class TestIntraday:

    @pytest.mark.asyncio
    async def test_intraday_skips_outside_market_hours(self, supervisor, monkeypatch):
        # Force the gate to return False
        monkeypatch.setattr(
            "backend.services.autopilot.supervisor._is_within_market_hours",
            lambda *_: False,
        )
        report = await supervisor.run_intraday()
        assert report.status == "skipped"
        assert report.skipped_reason == "outside_market_hours"
        assert report.trades_emitted == 0

    @pytest.mark.asyncio
    async def test_intraday_inside_hours_processes(self, supervisor, monkeypatch):
        monkeypatch.setattr(
            "backend.services.autopilot.supervisor._is_within_market_hours",
            lambda *_: True,
        )
        report = await supervisor.run_intraday()
        # No users → 0 processed but status ok
        assert report.status == "ok"
        assert report.users_processed == 0


# ─────────────────────────────────────────────────────────────────────
# Window: post-market
# ─────────────────────────────────────────────────────────────────────


class TestPostMarket:

    @pytest.mark.asyncio
    async def test_post_market_ok_with_no_users(self, supervisor):
        report = await supervisor.run_post_market()
        assert report.status == "ok"
        assert report.users_processed == 0


# ─────────────────────────────────────────────────────────────────────
# Window: overnight
# ─────────────────────────────────────────────────────────────────────


class TestOvernight:

    @pytest.mark.asyncio
    async def test_overnight_logs_health(self, supervisor):
        report = await supervisor.run_overnight()
        assert report.status == "ok"
        assert "health" in report.details
        assert report.details["health"]["regime_loaded"] is True


# ─────────────────────────────────────────────────────────────────────
# Run all + scheduler registration
# ─────────────────────────────────────────────────────────────────────


class TestRunAllWindows:

    @pytest.mark.asyncio
    async def test_run_all_returns_four_reports(self, supervisor, monkeypatch):
        monkeypatch.setattr(
            "backend.services.autopilot.supervisor._is_within_market_hours",
            lambda *_: False,  # intraday will skip, but still returns a report
        )
        reports = await supervisor.run_all_windows()
        assert len(reports) == 4
        assert {r.window for r in reports} == set(SupervisorWindow)


class TestSchedulerRegistration:

    def test_register_adds_four_jobs(self, supervisor):
        scheduler = MagicMock()
        scheduler.add_job = MagicMock()
        ids = register_supervisor_jobs(scheduler, supervisor)
        assert len(ids) == 4
        assert scheduler.add_job.call_count == 4
        # Every job ID is prefixed predictably
        for jid in ids:
            assert jid.startswith("autopilot_supervisor_")

    def test_register_uses_replace_existing(self, supervisor):
        scheduler = MagicMock()
        scheduler.add_job = MagicMock()
        register_supervisor_jobs(scheduler, supervisor)
        for call in scheduler.add_job.call_args_list:
            assert call.kwargs.get("replace_existing") is True


# ─────────────────────────────────────────────────────────────────────
# Memory-lock guard
# ─────────────────────────────────────────────────────────────────────


class TestMemoryLockGuard:
    """Regression guard for project_agents_decision_2026_05_10.md:
    the supervisor must NOT import or invoke any LLM module."""

    def test_supervisor_does_not_import_llm(self):
        import backend.services.autopilot.supervisor as sup
        src = open(sup.__file__).read()
        for forbidden in ("from .copilot", "AssistantLLM", "from .assistant",
                           "import openai", "import anthropic", "langchain"):
            assert forbidden not in src, (
                f"Supervisor source contains '{forbidden}' — this would "
                f"breach the LLM-never-gates-trades lock."
            )
