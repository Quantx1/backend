"""PR-T — nightly model refresh tests.

Validates the new SchedulerService methods that wire the daily 22:00 IST
training cron to the unified runner. Three behaviours under test:

  1. Idempotency — _recent_successful_run_exists() reads training_runs.
  2. Pipeline call — _run_unified_pipeline() invokes ml.training.runner.run
     with the right args + persists the run row.
  3. Nightly job — nightly_model_refresh() skips when a recent run exists,
     otherwise delegates to _run_unified_pipeline.

Heavy stuff (Supabase, the actual trainer modules) is mocked at the
boundary so these run on CPU in <1s.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class _FakeReport:
    name: str
    status: str = "ok"


def _make_scheduler(supabase_rows=None, sb_raises: Exception | None = None):
    """Build a minimal SchedulerService with a stub Supabase client."""
    from backend.platform.scheduler import SchedulerService

    inst = SchedulerService.__new__(SchedulerService)  # bypass __init__
    inst.notification_service = None

    fake_sb = MagicMock()
    if sb_raises:
        fake_sb.table.side_effect = sb_raises
    else:
        chain = MagicMock()
        chain.select.return_value = chain
        chain.gte.return_value = chain
        chain.eq.return_value = chain
        chain.limit.return_value = chain
        chain.execute.return_value = SimpleNamespace(data=supabase_rows or [])
        chain.upsert.return_value = SimpleNamespace(
            execute=lambda: SimpleNamespace(data=None)
        )
        fake_sb.table.return_value = chain
    inst.supabase = fake_sb
    return inst, fake_sb


# ─────────────────────────────────────────────────────────────────────────
# _recent_successful_run_exists
# ─────────────────────────────────────────────────────────────────────────

def test_recent_run_lookback_returns_true_when_row_present():
    sched, _ = _make_scheduler(supabase_rows=[{"id": "abc"}])
    out = asyncio.run(sched._recent_successful_run_exists(within_hours=18))
    assert out is True


def test_recent_run_lookback_returns_false_when_empty():
    sched, _ = _make_scheduler(supabase_rows=[])
    out = asyncio.run(sched._recent_successful_run_exists(within_hours=18))
    assert out is False


def test_recent_run_lookback_returns_false_on_supabase_error():
    """Best-effort: a DB failure must NOT block the cron from running."""
    sched, _ = _make_scheduler(sb_raises=RuntimeError("supabase down"))
    out = asyncio.run(sched._recent_successful_run_exists(within_hours=18))
    assert out is False


# ─────────────────────────────────────────────────────────────────────────
# _run_unified_pipeline
# ─────────────────────────────────────────────────────────────────────────

def test_run_unified_pipeline_invokes_runner_with_skip_gpu_and_promote():
    sched, fake_sb = _make_scheduler()
    fake_reports = [_FakeReport(name="regime_hmm"), _FakeReport(name="qlib_alpha158")]

    with patch(
        "ml.training.runner.run", return_value=fake_reports
    ) as runner:
        asyncio.run(
            sched._run_unified_pipeline(
                only=["regime_hmm", "qlib_alpha158"],
                promote=True,
                triggered_by="test",
            )
        )

    runner.assert_called_once_with(
        only=["regime_hmm", "qlib_alpha158"],
        skip_gpu=True,
        promote=True,
        dry_run=False,
    )

    # Two upserts: initial "running" row + final "ok" row.
    upsert_calls = [
        c
        for c in fake_sb.table.return_value.upsert.call_args_list
    ]
    assert len(upsert_calls) >= 2
    final_payload = upsert_calls[-1][0][0]
    assert final_payload["status"] == "ok"
    assert final_payload["triggered_by"] == "test"
    assert final_payload["params"] == {
        "only": ["regime_hmm", "qlib_alpha158"],
        "promote": True,
        "skip_gpu": True,
    }


def test_run_unified_pipeline_records_partial_when_any_trainer_failed():
    sched, fake_sb = _make_scheduler()
    fake_reports = [
        _FakeReport(name="regime_hmm", status="ok"),
        _FakeReport(name="qlib_alpha158", status="failed"),
    ]
    with patch("ml.training.runner.run", return_value=fake_reports):
        asyncio.run(
            sched._run_unified_pipeline(only=["regime_hmm", "qlib_alpha158"])
        )
    final_payload = fake_sb.table.return_value.upsert.call_args_list[-1][0][0]
    assert final_payload["status"] == "partial"


def test_run_unified_pipeline_records_failed_when_runner_raises():
    sched, fake_sb = _make_scheduler()

    with patch("ml.training.runner.run", side_effect=RuntimeError("boom")):
        asyncio.run(sched._run_unified_pipeline(only=["regime_hmm"]))

    final_payload = fake_sb.table.return_value.upsert.call_args_list[-1][0][0]
    assert final_payload["status"] == "failed"
    assert "boom" in (final_payload["error"] or "")


# ─────────────────────────────────────────────────────────────────────────
# nightly_model_refresh — idempotency + delegation
# ─────────────────────────────────────────────────────────────────────────

def test_nightly_refresh_skips_when_recent_run_exists():
    sched, _ = _make_scheduler(supabase_rows=[{"id": "recent"}])
    sched._run_unified_pipeline = AsyncMock()
    asyncio.run(sched.nightly_model_refresh())
    sched._run_unified_pipeline.assert_not_called()


def test_nightly_refresh_runs_when_no_recent_run():
    sched, _ = _make_scheduler(supabase_rows=[])
    sched._run_unified_pipeline = AsyncMock()
    asyncio.run(sched.nightly_model_refresh())
    sched._run_unified_pipeline.assert_awaited_once()
    call = sched._run_unified_pipeline.await_args
    assert call.kwargs["only"] == ["regime_hmm", "qlib_alpha158"]
    assert call.kwargs["promote"] is True
    assert call.kwargs["triggered_by"] == "scheduler:nightly_model_refresh"
