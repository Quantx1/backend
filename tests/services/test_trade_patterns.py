"""Trade Journal pattern mining — pure (#23)."""
from backend.services.explain.trade_patterns import mine_patterns


def _rec(symbol, pnl, minute, weekday, hold):
    return {"symbol": symbol, "pnl": pnl, "minute_of_day": minute, "weekday": weekday, "hold_min": hold}


def test_mine_patterns_sessions_and_symbols():
    records = [
        _rec("TCS", 100, 600, 0, 120),    # Open session, Mon, intraday, win
        _rec("TCS", 80, 580, 1, 90),      # Open session, Tue, win
        _rec("INFY", -50, 800, 2, 1200),  # Close session, Wed, swing, loss
        _rec("INFY", -30, 820, 3, 1500),  # Close session, loss
    ]
    p = mine_patterns(records)
    assert p["n"] == 4
    assert p["win_rate"] == 50
    # Open session should top the win-rate ranking (2/2 = 100%)
    assert p["by_session"][0]["label"].startswith("Open") and p["by_session"][0]["win_rate"] == 100
    # TCS is the best symbol, INFY the worst
    assert p["best_symbols"][0]["symbol"] == "TCS"
    assert any(s["symbol"] == "INFY" for s in p["worst_symbols"])
    holds = {b["label"] for b in p["by_hold"]}
    assert "Intraday" in holds and "Swing (>1 day)" in holds


def test_mine_patterns_empty():
    assert mine_patterns([]) == {"n": 0}
