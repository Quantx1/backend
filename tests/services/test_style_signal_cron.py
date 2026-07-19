"""Phase 2 — generate_style_signals cron unit tests.

Light, synthetic: engines are mocked (no LightGBM, no data plane, no
APScheduler run). Validates that the job exists, snapshots per engine, and
isolates a single engine's failure (never crashes the scheduler loop).
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from backend.platform.scheduler import STYLE_ENGINES, SchedulerService


def _make_scheduler():
    """Minimal SchedulerService with a stub Supabase client (telemetry sink)."""
    inst = SchedulerService.__new__(SchedulerService)  # bypass __init__
    inst.notification_service = None
    inst.supabase = MagicMock()
    return inst


class _FakeSignal:
    def __init__(self, symbol: str, rank: int):
        self.symbol = symbol
        self.rank = rank

    def to_dict(self):
        return {"symbol": self.symbol, "rank": self.rank}


class _FakeEngine:
    status = "ok"
    forecast_degraded = False

    def run(self, top_n=20):
        return [_FakeSignal("AAA", 1), _FakeSignal("BBB", 2)]


class _BoomEngine:
    status = "ok"

    def run(self, top_n=20):
        raise RuntimeError("boom")


def test_style_engines_roster():
    assert STYLE_ENGINES == ["momentum", "swing"]


def test_job_exists_and_writes_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_SNAPSHOT_DIR", str(tmp_path))
    sched = _make_scheduler()
    assert callable(getattr(sched, "generate_style_signals", None))

    with patch("backend.platform.scheduler._make_style_engine",
               side_effect=lambda name: _FakeEngine()), \
         patch("backend.platform.scheduler.is_trading_day",
               new=AsyncMock(return_value=True)):
        asyncio.run(sched.generate_style_signals())

    for name in STYLE_ENGINES:
        files = list(tmp_path.glob(f"{name}_*.json"))
        assert len(files) == 1, f"missing snapshot for {name}"
        snap = json.loads(files[0].read_text())
        assert snap["engine"] == name
        assert snap["count"] == 2
        assert snap["signals"][0] == {"symbol": "AAA", "rank": 1}
        assert snap["forecast_degraded"] is False


def test_one_engine_raising_never_crashes_the_job(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_SNAPSHOT_DIR", str(tmp_path))
    sched = _make_scheduler()

    def fake_make(name):
        return _BoomEngine() if name == "swing" else _FakeEngine()

    with patch("backend.platform.scheduler._make_style_engine",
               side_effect=fake_make), \
         patch("backend.platform.scheduler.is_trading_day",
               new=AsyncMock(return_value=True)):
        asyncio.run(sched.generate_style_signals())  # must not raise

    # the healthy engine still snapshotted; the broken one honest-empty
    assert len(list(tmp_path.glob("momentum_*.json"))) == 1
    assert list(tmp_path.glob("swing_*.json")) == []


def test_skips_on_non_trading_day(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALS_SNAPSHOT_DIR", str(tmp_path))
    sched = _make_scheduler()
    with patch("backend.platform.scheduler._make_style_engine") as mk, \
         patch("backend.platform.scheduler.is_trading_day",
               new=AsyncMock(return_value=False)):
        asyncio.run(sched.generate_style_signals())
    mk.assert_not_called()
    assert list(tmp_path.glob("*.json")) == []
