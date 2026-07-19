"""Paper window — outcome maturity math + evaluate_style_paper_window wiring.

Synthetic panel, mocked persistence + cron plumbing (no LightGBM, no data
plane, no APScheduler run). Mirrors tests/services/test_style_signal_cron.py
conventions.
"""
from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from backend.platform.scheduler import (
    STYLE_ENGINES,
    STYLE_HORIZONS,
    SchedulerService,
    _style_outcome_rows,
)

H = 3  # small synthetic horizon


def _panel(n_days: int) -> pd.DataFrame:
    """A/B/C universe over n_days business days. A and C rise, B falls."""
    dates = pd.bdate_range("2026-06-01", periods=n_days)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "symbol": "AAA", "close": 100.0 + 10.0 * i})
        rows.append({"date": d, "symbol": "BBB", "close": 100.0 - 5.0 * i})
        rows.append({"date": d, "symbol": "CCC", "close": 200.0 + 20.0 * i})
    return pd.DataFrame(rows)


BOOK = [{"symbol": "AAA", "rank": 1}, {"symbol": "BBB", "rank": 2}]


def test_horizon_map_single_source_of_truth():
    assert STYLE_HORIZONS == {"momentum": 20, "swing": 10}
    assert set(STYLE_HORIZONS) == set(STYLE_ENGINES)


def test_outcome_rows_mature_with_exactly_h_bars_after():
    panel = _panel(H + 1)  # t at index 0, exactly H bars after
    t0 = panel["date"].min().date()
    rows = _style_outcome_rows(panel, "momentum", H, t0, BOOK)
    assert rows is not None and len(rows) == 2
    # closes: AAA 100->130 (+0.30), BBB 100->85 (-0.15), CCC 200->260 (+0.30)
    fwd_a, fwd_b, fwd_c = 0.30, -0.15, 0.30
    bench = (fwd_a + fwd_b + fwd_c) / 3.0  # equal-weight over the FULL universe
    by_sym = {r["symbol"]: r for r in rows}
    assert by_sym["AAA"]["fwd_return_h"] == pytest.approx(fwd_a)
    assert by_sym["BBB"]["fwd_return_h"] == pytest.approx(fwd_b)
    for r in rows:
        assert r["bench_fwd_return_h"] == pytest.approx(bench)
        assert r["excess_h"] == pytest.approx(r["fwd_return_h"] - bench)
        assert r["horizon_days"] == H
        assert r["engine"] == "momentum"
        assert r["trade_date"] == t0.isoformat()
    # date-mean of per-row excess == book_mean_gross - bench
    book_gross = (fwd_a + fwd_b) / 2.0
    mean_excess = sum(r["excess_h"] for r in rows) / len(rows)
    assert mean_excess == pytest.approx(book_gross - bench)


def test_outcome_rows_do_not_mature_with_h_minus_1_bars():
    panel = _panel(H)  # only H-1 bars after the first date
    t0 = panel["date"].min().date()
    assert _style_outcome_rows(panel, "momentum", H, t0, BOOK) is None


def test_outcome_rows_none_when_trade_date_absent_from_panel():
    panel = _panel(H + 1)
    assert _style_outcome_rows(panel, "momentum", H, date(2020, 1, 1), BOOK) is None


def test_outcome_rows_skip_symbols_missing_closes():
    panel = _panel(H + 1)
    t0 = panel["date"].min().date()
    book = BOOK + [{"symbol": "ZZZ", "rank": 3}]  # not in the universe panel
    rows = _style_outcome_rows(panel, "momentum", H, t0, book)
    assert {r["symbol"] for r in rows} == {"AAA", "BBB"}


def _make_scheduler():
    inst = SchedulerService.__new__(SchedulerService)  # bypass __init__
    inst.notification_service = None
    inst.supabase = MagicMock()  # cron_lock insert + telemetry sink
    return inst


def test_evaluate_style_paper_window_end_to_end():
    """Full job: one candidate date matures for momentum; swing has none."""
    sched = _make_scheduler()
    panel = _panel(25)
    t0 = panel["date"].min().date()
    saved = []

    def fake_unmatured(engine, horizon, supabase=None):
        return [t0] if engine == "momentum" else []

    def fake_save(rows, supabase=None):
        saved.extend(rows)
        return len(rows)

    with patch("backend.platform.scheduler._load_style_panel", return_value=panel), \
         patch("backend.ai.signals.style_persistence.fetch_unmatured_dates",
               side_effect=fake_unmatured), \
         patch("backend.ai.signals.style_persistence.fetch_signal_rows",
               return_value=BOOK), \
         patch("backend.ai.signals.style_persistence.save_style_outcomes",
               side_effect=fake_save), \
         patch("backend.platform.scheduler.is_trading_day",
               new=AsyncMock(return_value=True)):
        asyncio.run(sched.evaluate_style_paper_window())

    # momentum H=20: panel has 25 bdays, t0 has 24 bars after -> matured
    assert {r["symbol"] for r in saved} == {"AAA", "BBB"}
    assert all(r["engine"] == "momentum" for r in saved)
    assert all(r["horizon_days"] == 20 for r in saved)
    # bench uses the FULL 3-symbol universe at t and t+20
    fwd = {"AAA": (100 + 10 * 20) / 100 - 1,
           "BBB": (100 - 5 * 20) / 100 - 1,
           "CCC": (200 + 20 * 20) / 200 - 1}
    bench = sum(fwd.values()) / 3.0
    for r in saved:
        assert r["fwd_return_h"] == pytest.approx(fwd[r["symbol"]])
        assert r["bench_fwd_return_h"] == pytest.approx(bench)


def test_evaluate_skips_when_lock_not_acquired():
    sched = _make_scheduler()

    # cron_lock: duplicate-key insert => yields False => job must bail out
    sched.supabase.table.return_value.insert.return_value.execute.side_effect = \
        Exception("duplicate key value violates unique constraint")

    with patch("backend.platform.scheduler._load_style_panel") as loader, \
         patch("backend.platform.scheduler.is_trading_day",
               new=AsyncMock(return_value=True)):
        asyncio.run(sched.evaluate_style_paper_window())
    loader.assert_not_called()


def test_evaluate_skips_on_non_trading_day():
    sched = _make_scheduler()
    with patch("backend.platform.scheduler._load_style_panel") as loader, \
         patch("backend.platform.scheduler.is_trading_day",
               new=AsyncMock(return_value=False)):
        asyncio.run(sched.evaluate_style_paper_window())
    loader.assert_not_called()
    sched.supabase.table.assert_not_called()  # no lock row either
