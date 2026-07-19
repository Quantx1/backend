"""Event-risk blackout gate — pure tests (fake Supabase, no network).

The gate is ENTRY-ONLY and must FAIL OPEN (a data outage never blocks all
trading — hard SL/target rails apply downstream).
"""
from datetime import date

from backend.services.scanners.event_risk import (
    filter_entry_weights,
    is_expiry_day,
    symbols_in_event_window,
)


class _FakeResp:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, rows, sink=None):
        self._rows = rows
        self._sink = sink

    def select(self, *a, **k):
        return self

    def in_(self, col, vals):
        if self._sink is not None:
            self._sink["in"] = (col, list(vals))
        return self

    def gte(self, *a, **k):
        return self

    def lte(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        return _FakeResp(self._rows)


class _FakeSB:
    def __init__(self, rows, sink=None):
        self._rows = rows
        self._sink = sink

    def table(self, name):
        return _FakeQuery(self._rows, self._sink)


def test_in_window_returns_matching_symbols():
    sb = _FakeSB([{"symbol": "TCS", "announce_date": "2026-06-14"}])
    got = symbols_in_event_window(["TCS", "RELIANCE"], days=3, supabase=sb)
    assert got == {"TCS"}


def test_uppercases_and_dedups_query():
    sink = {}
    sb = _FakeSB([], sink=sink)
    symbols_in_event_window(["tcs", "TCS", " reliance "], days=2, supabase=sb)
    # query was issued with normalized, de-duplicated symbols
    assert set(sink["in"][1]) == {"TCS", "RELIANCE"}


def test_empty_symbols_short_circuits_no_query():
    # passing no symbols must not even build a query → returns empty
    assert symbols_in_event_window([], supabase=_FakeSB([{"symbol": "X"}])) == set()


def test_zero_days_disables_gate():
    sb = _FakeSB([{"symbol": "TCS", "announce_date": "2026-06-14"}])
    assert symbols_in_event_window(["TCS"], days=0, supabase=sb) == set()


def test_fails_open_on_db_error():
    class _Boom:
        def table(self, *a, **k):
            raise RuntimeError("db down")

    assert symbols_in_event_window(["TCS"], supabase=_Boom()) == set()


def test_filter_entry_weights_drops_blackout_only():
    sb = _FakeSB([{"symbol": "TCS", "announce_date": "2026-06-14"}])
    kept, blocked = filter_entry_weights({"TCS": 0.05, "INFY": 0.04}, days=3, supabase=sb)
    assert kept == {"INFY": 0.04}
    assert blocked == {"TCS"}


def test_filter_entry_weights_empty_is_noop():
    kept, blocked = filter_entry_weights({}, supabase=_FakeSB([]))
    assert kept == {} and blocked == set()


def test_is_expiry_day_thursday():
    # 2026-06-18 is a Thursday
    assert is_expiry_day(date(2026, 6, 18)) is True


def test_is_expiry_day_monday_false():
    # 2026-06-15 is a Monday
    assert is_expiry_day(date(2026, 6, 15)) is False
