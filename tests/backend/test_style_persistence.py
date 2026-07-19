"""Paper window — style_persistence unit tests (mock supabase capture).

Covers: (a) save_style_signals upsert payload shape + idempotency key,
save_style_outcomes payload, and fetch_unmatured_dates set-difference +
calendar pre-filter. No network, no real client.
"""
from __future__ import annotations

from datetime import date, timedelta

from backend.ai.signals.style_persistence import (
    fetch_unmatured_dates,
    save_style_outcomes,
    save_style_signals,
)
from backend.ai.signals.style_types import MomentumSignal, Style


class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    """Chainable stub for select/eq/order/range/execute."""

    def __init__(self, rows):
        self._rows = rows
        self._slice = (0, len(rows) - 1)

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def range(self, start, end):
        self._slice = (start, end)
        return self

    def execute(self):
        s, e = self._slice
        return _Result(self._rows[s:e + 1])


class _FakeSupabase:
    """Captures upserts; serves canned rows per table for reads."""

    def __init__(self, rows_by_table=None):
        self.rows_by_table = rows_by_table or {}
        self.upserts = []  # (table, rows, on_conflict)
        self._table = None

    def table(self, name):
        self._table = name
        return self

    def upsert(self, rows, on_conflict=None):
        self.upserts.append((self._table, rows, on_conflict))
        return self

    # read chain
    def select(self, *a, **k):
        return _Query(self.rows_by_table.get(self._table, []))

    def execute(self):  # for the upsert chain
        return _Result([])


def _signal(symbol="AAA", rank=1):
    return MomentumSignal(
        symbol=symbol, style=Style.MOMENTUM, rank=rank, percentile=0.98,
        confidence=98.0, direction="BUY", entry_price=100.0, stop_loss=90.0,
        target=120.0, risk_reward=2.0, reasons=["r"],
        expected_return=0.031, top_decile_prob=1.0,
    )


def test_save_style_signals_payload_shape():
    sb = _FakeSupabase()
    n = save_style_signals(
        "momentum", date(2026, 7, 7), [_signal("AAA", 1), _signal("BBB", 2)],
        status="ok", forecast_degraded=True, supabase=sb,
    )
    assert n == 2
    assert len(sb.upserts) == 1
    table, rows, on_conflict = sb.upserts[0]
    assert table == "style_signals"
    assert on_conflict == "engine,trade_date,symbol"  # PK — same-day rerun overwrites
    row = rows[0]
    assert row["engine"] == "momentum"
    assert row["trade_date"] == "2026-07-07"
    assert row["symbol"] == "AAA"
    assert row["rank"] == 1
    assert row["percentile"] == 0.98
    assert row["confidence"] == 98.0
    assert row["direction"] == "BUY"
    assert row["entry_price"] == 100.0
    assert row["stop_loss"] == 90.0
    assert row["target"] == 120.0
    assert row["risk_reward"] == 2.0
    assert row["expected_return"] == 0.031
    assert row["top_decile_prob"] == 1.0
    assert row["status"] == "ok"
    assert row["forecast_degraded"] is True
    assert "generated_at" in row
    # reasons is NOT a table column — must not leak into the payload
    assert "reasons" not in row and "style" not in row
    assert rows[1]["symbol"] == "BBB" and rows[1]["rank"] == 2


def test_save_style_signals_never_raises_and_returns_zero():
    class _Boom:
        def table(self, name):
            raise RuntimeError("db down")

    n = save_style_signals(
        "momentum", date(2026, 7, 7), [_signal()],
        status="ok", forecast_degraded=False, supabase=_Boom(),
    )
    assert n == 0  # best-effort by contract


def test_save_style_signals_empty_book_writes_nothing():
    sb = _FakeSupabase()
    assert save_style_signals(
        "swing", date(2026, 7, 7), [], status="no_data",
        forecast_degraded=False, supabase=sb) == 0
    assert sb.upserts == []


def test_save_style_outcomes_payload():
    sb = _FakeSupabase()
    rows = [{
        "engine": "swing", "trade_date": date(2026, 6, 1), "symbol": "AAA",
        "rank": 1, "fwd_return_h": 0.02, "bench_fwd_return_h": 0.01,
        "excess_h": 0.01, "horizon_days": 10,
    }]
    assert save_style_outcomes(rows, supabase=sb) == 1
    table, out_rows, on_conflict = sb.upserts[0]
    assert table == "style_signal_outcomes"
    assert on_conflict == "engine,trade_date,symbol"
    assert out_rows[0]["trade_date"] == "2026-06-01"  # date -> ISO string
    assert out_rows[0]["horizon_days"] == 10


def test_fetch_unmatured_dates_set_difference_and_calendar_prefilter():
    old = (date.today() - timedelta(days=40)).isoformat()
    done = (date.today() - timedelta(days=35)).isoformat()
    recent = (date.today() - timedelta(days=2)).isoformat()  # < horizon calendar days
    sb = _FakeSupabase(rows_by_table={
        "style_signals": [
            {"trade_date": old}, {"trade_date": old},  # dupes collapse
            {"trade_date": done}, {"trade_date": recent},
        ],
        "style_signal_outcomes": [{"trade_date": done}],
    })
    out = fetch_unmatured_dates("momentum", 20, supabase=sb)
    assert out == [date.fromisoformat(old)]


def test_fetch_unmatured_dates_honest_empty_when_tables_missing():
    class _Boom:
        def table(self, name):
            raise RuntimeError("relation does not exist")

    assert fetch_unmatured_dates("momentum", 20, supabase=_Boom()) == []
