from backend.services.news.earnings_preview import _drivers, _expected_move_pct, _run_up_pct


def test_expected_move_pct():
    assert _expected_move_pct(0.20, 30) == 5.73
    assert _expected_move_pct(None, 30) is None
    assert _expected_move_pct(0.20, 0) is None


def test_run_up_pct():
    closes = [100.0] * 10 + [110.0] * 15
    assert _run_up_pct(closes, window=20) is not None
    assert _run_up_pct([100.0, 101.0], window=20) is None


def test_drivers_full_facts():
    facts = {
        "symbol": "TCS",
        "earnings": {"announce_date": "2026-06-20", "days_to_earnings": 9,
                     "source": "yfinance_calendar"},
        "volatility": {"atm_iv_pct": 22.0, "iv_rank": 75.0, "iv_percentile": 80.0,
                       "expected_move_pct_to_expiry": 4.2, "expiry": "2026-06-26",
                       "days_to_expiry": 15},
        "run_up": {"pct_1m": 6.5},
        "relative_strength": {"rs_20d": 3.1, "rs_50d": 5.0, "outperforming": True},
    }
    txt = " ".join(_drivers(facts))
    assert "Earnings on 2026-06-20" in txt
    assert "±4.2%" in txt
    assert "IV Rank 75" in txt
    assert "run up 6.5%" in txt
    assert "Outperforming" in txt


def test_drivers_honest_empty_without_date():
    assert _drivers({}) == []
    assert _drivers({"symbol": "X", "run_up": {"pct_1m": 9.0}}) == []
