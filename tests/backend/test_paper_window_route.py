"""GET /api/signals/style/paper-window — aggregation + status-rule tests.

Mirrors tests/api/test_momentum_signals_route.py: exercises the compute
function directly with the persistence layer monkeypatched (no TestClient,
no network). Baseline expectations come from the real frozen file
data/paper/baseline_expectations.json (momentum p=0.5913, swing p=0.5291).
"""
from __future__ import annotations

from datetime import date, timedelta
from math import sqrt

import pytest

import backend.ai.signals.style_persistence as sp
from backend.api import signals_routes as sr

P_MOMENTUM = 0.5913  # frozen hit_rate_vs_universe in baseline_expectations.json


def _outcome_rows(engine: str, results: list) -> list:
    """One (date_gross, date_bench) pair per matured date -> outcome rows.
    Two symbols per date whose mean is date_gross, bench duplicated per row."""
    rows = []
    d0 = date(2026, 5, 1)
    for i, (gross, bench) in enumerate(results):
        d = (d0 + timedelta(days=i)).isoformat()
        for sym, fwd in (("AAA", gross + 0.01), ("BBB", gross - 0.01)):
            rows.append({
                "trade_date": d, "symbol": sym, "rank": 1,
                "fwd_return_h": fwd, "bench_fwd_return_h": bench,
                "excess_h": fwd - bench, "horizon_days": 20,
            })
    return rows


def _patch(monkeypatch, outcomes_by_engine, dates_by_engine):
    monkeypatch.setattr(
        sp, "fetch_outcomes",
        lambda engine, supabase=None: outcomes_by_engine.get(engine, []))
    monkeypatch.setattr(
        sp, "fetch_signal_dates",
        lambda engine, supabase=None: dates_by_engine.get(engine, []))


def test_honest_empty_shape_with_no_data(monkeypatch):
    _patch(monkeypatch, {}, {})
    payload = sr._compute_paper_window()
    assert payload["window_start"] is None
    assert payload["as_of"] == date.today().isoformat()
    assert set(payload["engines"]) == {"momentum", "swing"}
    mom, swi = payload["engines"]["momentum"], payload["engines"]["swing"]
    assert mom["horizon"] == 20 and swi["horizon"] == 10
    for eng in (mom, swi):
        assert eng["days_signaled"] == 0
        assert eng["days_matured"] == 0
        assert eng["live"] == {"hit_rate": None, "mean_excess_h": None,
                               "mean_gross_h": None, "n_dates": 0}
        assert eng["status"] == "collecting"
        assert eng["expected"]["source"] == "backtest 2023-07..2026-06"
    # frozen expectations are surfaced even before any live data exists
    assert mom["expected"]["hit_rate"] == pytest.approx(0.5913)
    assert mom["expected"]["mean_excess_h"] == pytest.approx(0.00716)
    assert swi["expected"]["hit_rate"] == pytest.approx(0.5291)
    assert swi["expected"]["mean_excess_h"] == pytest.approx(0.00365)


def test_collecting_below_ten_matured_dates(monkeypatch):
    results = [(0.02, 0.01)] * 5  # 5 matured dates, all hits
    sig_dates = [date(2026, 5, 1) + timedelta(days=i) for i in range(8)]
    _patch(monkeypatch, {"momentum": _outcome_rows("momentum", results)},
           {"momentum": sig_dates})
    mom = sr._compute_paper_window()["engines"]["momentum"]
    assert mom["days_signaled"] == 8
    assert mom["days_matured"] == 5
    assert mom["live"]["hit_rate"] == 1.0
    assert mom["live"]["mean_excess_h"] == pytest.approx(0.01)
    assert mom["live"]["mean_gross_h"] == pytest.approx(0.02)
    assert mom["status"] == "collecting"  # M < 10 regardless of performance


def test_on_track_and_off_track_around_binomial_bound(monkeypatch):
    m = 16
    bound = P_MOMENTUM - 2.0 * sqrt(P_MOMENTUM * (1 - P_MOMENTUM) / m)
    # 6/16 = 0.375 >= bound (~0.3455) -> on_track; 5/16 = 0.3125 -> off_track
    assert 5 / m < bound < 6 / m

    def run(hits: int) -> dict:
        results = [(0.02, 0.01)] * hits + [(0.00, 0.01)] * (m - hits)
        _patch(monkeypatch, {"momentum": _outcome_rows("momentum", results)},
               {"momentum": [date(2026, 5, 1)]})
        return sr._compute_paper_window()["engines"]["momentum"]

    on = run(6)
    assert on["days_matured"] == m
    assert on["live"]["hit_rate"] == pytest.approx(6 / m)
    assert on["status"] == "on_track"

    off = run(5)
    assert off["live"]["hit_rate"] == pytest.approx(5 / m)
    assert off["status"] == "off_track"


def test_aggregation_is_per_date_not_per_row(monkeypatch):
    # 2 rows per date must collapse to ONE (gross, bench) pair per date:
    # hit_rate counts dates, not symbols.
    results = [(0.03, 0.01), (0.00, 0.02)]  # one hit, one miss
    _patch(monkeypatch, {"momentum": _outcome_rows("momentum", results)},
           {"momentum": [date(2026, 5, 1), date(2026, 5, 2)]})
    mom = sr._compute_paper_window()["engines"]["momentum"]
    assert mom["days_matured"] == 2
    assert mom["live"]["hit_rate"] == 0.5
    assert mom["live"]["mean_excess_h"] == pytest.approx(((0.03 - 0.01) + (0.00 - 0.02)) / 2)
    assert mom["live"]["mean_gross_h"] == pytest.approx(0.015)


def test_window_start_is_min_trade_date_across_engines(monkeypatch):
    _patch(monkeypatch, {}, {
        "momentum": [date(2026, 7, 2), date(2026, 7, 3)],
        "swing": [date(2026, 7, 1)],
    })
    payload = sr._compute_paper_window()
    assert payload["window_start"] == "2026-07-01"
    assert payload["engines"]["momentum"]["days_signaled"] == 2
    assert payload["engines"]["swing"]["days_signaled"] == 1


def test_route_ttl_cache_and_registration_order():
    """The static /style/paper-window path must be registered before the
    dynamic /{signal_id} route, and the payload is cached for 60s."""
    paths = [r.path for r in sr.router.routes]
    assert "/api/signals/style/paper-window" in paths
    assert paths.index("/api/signals/style/paper-window") \
        < paths.index("/api/signals/{signal_id}")
    assert sr._PAPER_WINDOW_TTL_S == 60
