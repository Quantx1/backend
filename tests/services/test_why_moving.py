"""'Why is X moving' — deterministic drivers (the grounded narrative is integration-only)."""
from backend.services.explain.why_moving import _drivers


def test_drivers_from_facts():
    facts = {
        "symbol": "RELIANCE",
        "price": {"change_pct": 2.4},
        "volume": {"x_avg": 3.1},
        "relative_strength": {"vs_nifty_pct": 1.8, "outperforming": True},
        "futures_oi": {"buildup": "long_buildup", "oi_change_pct": 12.0},
        "regime": {"market": "bull", "vix": 13.2},
    }
    txt = " ".join(_drivers(facts))
    assert "+2.4%" in txt
    assert "3.1×" in txt
    assert "Outperforming NIFTY by 1.8%" in txt
    assert "long buildup" in txt
    assert "regime: bull" in txt.lower()


def test_drivers_lagging_and_empty():
    txt = " ".join(_drivers({"relative_strength": {"vs_nifty_pct": -1.2, "outperforming": False}}))
    assert "Lagging NIFTY by 1.2%" in txt
    assert _drivers({"symbol": "X"}) == []
