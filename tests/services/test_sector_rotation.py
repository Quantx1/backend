"""Sector rotation (RRG) — pure classify + aggregate (#8)."""
from backend.services.scanners.sector_rotation import classify_quadrant, aggregate


def test_classify_quadrant():
    assert classify_quadrant(2, 1) == "leading"
    assert classify_quadrant(2, -1) == "weakening"
    assert classify_quadrant(-2, -1) == "lagging"
    assert classify_quadrant(-2, 1) == "improving"
    assert classify_quadrant(None, 1) == "n/a"


def test_aggregate_rs_vs_market():
    rows = [
        {"symbol": "A", "ret_5d": 5, "ret_20d": 10},   # IT
        {"symbol": "B", "ret_5d": 1, "ret_20d": 2},    # IT
        {"symbol": "C", "ret_5d": -3, "ret_20d": -6},  # Auto
    ]
    secmap = {"A": "IT", "B": "IT", "C": "Auto"}
    out = aggregate(rows, secmap)
    it = next(r for r in out if r["sector"] == "IT")
    # mkt5 = (5+1-3)/3 = 1; IT s5 = (5+1)/2 = 3; rs_short = 2
    assert it["rs_short"] == 2.0
    assert it["rs_long"] == 4.0
    assert it["quadrant"] == "leading"
    auto = next(r for r in out if r["sector"] == "Auto")
    assert auto["quadrant"] == "lagging"
