"""Deterministic L2 depth analysis (walls / imbalance / spread) — 0 LLM tokens."""
from backend.data.brokers.depth_models import MarketDepth, DepthLevel, analyze_depth


def test_analyze_depth_imbalance_walls_spread():
    d = MarketDepth(
        symbol="RELIANCE",
        bids=[DepthLevel(2500.0, 1000, 5), DepthLevel(2499.5, 300, 2)],
        asks=[DepthLevel(2500.5, 200, 3), DepthLevel(2501.0, 150, 1)],
    )
    a = analyze_depth(d)
    assert a["total_bid_qty"] == 1300 and a["total_ask_qty"] == 350
    assert a["imbalance"] == round((1300 - 350) / 1650, 4)
    assert a["pressure"] == "buy_pressure"
    assert a["best_bid"] == 2500.0 and a["best_ask"] == 2500.5
    assert a["spread"] == 0.5
    assert a["bid_wall"]["quantity"] == 1000
    assert a["ask_wall"]["quantity"] == 200


def test_analyze_depth_empty_is_safe():
    a = analyze_depth(MarketDepth(symbol="X"))
    assert a["imbalance"] == 0.0 and a["pressure"] == "balanced"
    assert a["bid_wall"] is None and a["best_bid"] is None
