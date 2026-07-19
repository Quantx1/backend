"""Portfolio analytics — correlation + rebalancing rules (#19)."""
from backend.services.portfolio.portfolio_analytics import compute_correlation, rebalancing_suggestions


def test_correlation_signs():
    a = [0.01, -0.02, 0.03, 0.0, 0.01] * 6   # 30 points
    rets = {"X": a, "Y": a, "Z": [-x for x in a]}
    c = compute_correlation(rets)
    assert c["avg_corr"] is not None
    xy = next(p for p in c["pairs"] if {p["a"], p["b"]} == {"X", "Y"})
    xz = next(p for p in c["pairs"] if {p["a"], p["b"]} == {"X", "Z"})
    assert xy["corr"] >= 0.99
    assert xz["corr"] <= -0.99


def test_correlation_too_few_holdings():
    assert compute_correlation({"X": [0.1] * 30})["avg_corr"] is None


def test_rebalancing_overweight_sector_and_corr():
    pos = [{"symbol": "A", "weight": 0.40}, {"symbol": "B", "weight": 0.35}, {"symbol": "C", "weight": 0.25}]
    sec = {"A": "IT", "B": "IT", "C": "Auto"}
    out = rebalancing_suggestions(pos, sector_by_symbol=sec, top_corr_pair={"a": "A", "b": "B", "corr": 0.9})
    actions = {x["action"] for x in out}
    assert "trim" in actions        # A is 40% overweight
    assert "diversify" in actions   # IT is 75% of the book
    assert "de-risk" in actions     # A & B correlated 0.9
