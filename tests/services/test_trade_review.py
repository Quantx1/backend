"""Per-trade AI Trade Review — pure facts + bullets (no DB, no network)."""
from backend.services.explain.trade_review import assemble_trade_facts, _review_points


def _target_hit_long():
    """LONG that hit its target. Entry 100 / stop 96 / exit 104.2."""
    return {
        "symbol": "tcs",
        "direction": "LONG",
        "entry_price": 100.0,
        "exit_price": 104.2,
        "stop_loss": 96.0,
        "net_pnl": 420.0,
        "gross_pnl": 440.0,
        "pnl_percent": 4.2,
        "exit_reason": "target_hit",
        "created_at": "2026-06-02T09:30:00+00:00",
        "closed_at": "2026-06-05T09:30:00+00:00",
        "signals": {"entry_price": 99.4, "stop_loss": 96.0, "target_1": 104.0},
    }


def _stop_hit_long():
    """LONG stopped out. Entry 200 / stop 196 / exit 196 = -1R."""
    return {
        "symbol": "INFY",
        "direction": "LONG",
        "entry_price": 200.0,
        "exit_price": 196.0,
        "stop_loss": 196.0,
        "net_pnl": -410.0,
        "pnl_percent": -2.0,
        "exit_reason": "stop_loss",
        "created_at": "2026-06-02T10:00:00+00:00",
        "closed_at": "2026-06-02T13:00:00+00:00",
    }


def _breakeven_no_stop():
    """Manual exit, flat, no stop on the row → no risk metrics."""
    return {
        "symbol": "SBIN",
        "direction": "LONG",
        "entry_price": 500.0,
        "exit_price": 500.0,
        "net_pnl": 0.0,
        "pnl_percent": 0.0,
        "exit_reason": "manual",
    }


def test_target_hit_long_facts_and_points():
    f = assemble_trade_facts(_target_hit_long())
    assert f["symbol"] == "TCS"
    assert f["side"] == "LONG"
    assert f["entry_price"] == 100.0 and f["exit_price"] == 104.2
    assert f["pnl_pct"] == 4.2
    # risk = 4 (100->96), move = +4.2 -> ~1.05R
    assert f["stop_distance_pct"] == 4.0
    assert f["r_multiple"] == 1.05
    assert f["hold_duration"] == "3 days"
    # entry quality vs signal entry 99.4
    assert f["entry_quality"]["signal_entry"] == 99.4
    assert f["entry_quality"]["slippage_pct"] == 0.6

    pts = _review_points(f)
    assert any("Hit target" in p and "+4.2%" in p for p in pts)
    assert any("Realized 1.05R" in p for p in pts)
    assert any("Held 3 days" in p for p in pts)
    assert any("within 0.6%" in p for p in pts)


def test_stop_hit_long_is_negative_r():
    f = assemble_trade_facts(_stop_hit_long())
    # exit == stop -> exactly -1R
    assert f["r_multiple"] == -1.0
    assert f["stop_distance_pct"] == 2.0
    assert f["hold_duration"] == "3 hr"
    # no originating signal on this row -> no entry_quality fabricated
    assert "entry_quality" not in f

    pts = _review_points(f)
    assert any("Stopped out" in p and "-2.0%" in p for p in pts)
    assert any("Lost 1.0R" in p for p in pts)


def test_breakeven_no_stop_has_no_risk_metrics():
    f = assemble_trade_facts(_breakeven_no_stop())
    assert f["pnl_pct"] == 0.0
    # no stop on the row -> never invent R-multiple / stop distance
    assert "r_multiple" not in f
    assert "stop_distance_pct" not in f
    assert "hold_duration" not in f  # no timestamps -> honest-empty

    pts = _review_points(f)
    assert any("Exited on manual" in p for p in pts)
    assert any("Net P&L +0.0" in p for p in pts)
    # bullets are always returned, never empty for a real trade
    assert len(pts) >= 1


def test_short_pnl_pct_derived_when_column_missing():
    # SHORT, no stored pnl_percent -> derived from entry/exit with side flip.
    trade = {
        "symbol": "RELIANCE",
        "direction": "SHORT",
        "entry_price": 100.0,
        "exit_price": 95.0,  # price fell -> short is +5%
        "net_pnl": 500.0,
    }
    f = assemble_trade_facts(trade)
    assert f["pnl_pct"] == 5.0


def test_empty_trade_yields_empty_facts_and_points():
    assert assemble_trade_facts({}) == {}
    assert _review_points({}) == []
