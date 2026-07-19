"""AI Market Explainer — deterministic drivers (the grounded narrative is
integration-only). Pure test over `build_drivers` with fabricated facts."""
from backend.services.explain.market_explainer import build_drivers


def test_drivers_full_facts():
    facts = {
        "nifty": {"ltp": 24800.0, "change_pct": 0.82},
        "breadth": {"adv": 312, "dec": 145, "ratio": 2.15, "adv_pct": 78},
        "sectors": {"leading": ["Banking", "IT"], "lagging": ["Metal"]},
        "regime": {"market": "bull", "vix": 12.4},
    }
    txt = " ".join(build_drivers(facts))
    assert "NIFTY +0.82%" in txt
    assert "Breadth positive: 312 adv / 145 dec" in txt
    assert "78% advancing" in txt
    assert "Leading: Banking, IT" in txt
    assert "Lagging: Metal" in txt
    assert "Regime: Bull (VIX 12.4)" in txt


def test_drivers_negative_breadth_and_down_nifty():
    facts = {
        "nifty": {"change_pct": -1.1},
        "breadth": {"adv": 120, "dec": 340, "adv_pct": 26},
        "regime": {"market": "bear"},
    }
    txt = " ".join(build_drivers(facts))
    assert "NIFTY -1.1%" in txt
    assert "Breadth negative: 120 adv / 340 dec" in txt
    assert "Regime: Bear" in txt
    # No VIX -> no parenthetical.
    assert "VIX" not in txt


def test_drivers_partial_and_empty():
    # Only sectors present.
    txt = " ".join(build_drivers({"sectors": {"leading": ["Pharma"], "lagging": []}}))
    assert txt == "Leading: Pharma."
    # Nothing real -> honest-empty drivers list.
    assert build_drivers({}) == []
    assert build_drivers({"breadth": {}}) == []
