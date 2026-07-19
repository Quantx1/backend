"""Watchlist Daily Digest — deterministic bullets + summary (the grounded
narrative is integration-only). Pure tests with fabricated facts."""
from backend.services.news.watchlist_digest import build_summary, build_symbol_bullets


def test_symbol_bullets_full_facts():
    f = {"price": {"change_pct": 3.2}, "volume": {"x_avg": 2.1},
         "signal": {"direction": "LONG", "confidence": 0.74},
         "alerts": [{"type": "breakout", "message": "Price crossed its 20-day high."}]}
    txt = " ".join(build_symbol_bullets("TCS", f))
    assert "+3.2%" in txt
    assert "2.1×" in txt
    assert "Active LONG signal (74% conf)" in txt
    assert "Price crossed its 20-day high" in txt


def test_symbol_bullets_quiet_symbol_and_empty():
    assert build_symbol_bullets("X", {"volume": {"x_avg": 1.1}}) == []
    assert build_symbol_bullets("X", {}) == []


def test_summary_what_changed_today():
    facts = {"symbols": ["TCS", "INFY", "SBIN"],
             "per_symbol": {
                 "TCS": {"price": {"change_pct": 3.2}, "volume": {"x_avg": 2.1}},
                 "INFY": {"price": {"change_pct": -0.4}},
                 "SBIN": {"signal": {"direction": "LONG"}},
             },
             "regime": {"market": "bull"}}
    s = build_summary(facts)
    assert "1 of 3 moved" in s and "TCS" in s
    assert "unusual volume in TCS" in s
    assert "active signals on SBIN" in s
    assert "regime bull" in s


def test_summary_quiet_day_and_honest_empty():
    facts = {"per_symbol": {"TCS": {"price": {"change_pct": 0.1}}}}
    assert "No significant changes" in build_summary(facts)
    assert build_summary({}) is None
