# tests/data/brokers/test_depth_models.py
from backend.data.brokers.depth_models import DepthLevel, MarketDepth, from_kite_depth


def test_from_kite_depth_builds_bids_and_asks():
    raw = {
        "buy": [{"price": 100.0, "quantity": 500, "orders": 3},
                {"price": 99.95, "quantity": 1200, "orders": 5}],
        "sell": [{"price": 100.05, "quantity": 400, "orders": 2}],
    }
    md = from_kite_depth("RELIANCE", raw)
    assert md.symbol == "RELIANCE"
    assert md.levels() == 2
    assert md.bids[0] == DepthLevel(price=100.0, quantity=500, orders=3)
    assert md.asks[0].price == 100.05
    assert md.to_dict() == {
        "symbol": "RELIANCE", "levels": 2, "source": "broker",
        "bids": [{"price": 100.0, "quantity": 500, "orders": 3},
                 {"price": 99.95, "quantity": 1200, "orders": 5}],
        "asks": [{"price": 100.05, "quantity": 400, "orders": 2}],
    }


def test_from_kite_depth_honest_empty():
    assert from_kite_depth("X", {}).levels() == 0
    assert from_kite_depth("X", {"buy": [{"price": 0, "quantity": 0}], "sell": []}).levels() == 0
