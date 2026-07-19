"""Upstox V3 depth reuses the SAME canonical MarketDepth/to_dict shape as F6,
so it flows through the existing _on_tick -> DepthBus -> /ws path unchanged."""
from backend.data.brokers.depth_models import from_upstox_marketlevel, from_kite_depth


def test_upstox_and_kite_depth_share_to_dict_shape():
    up = from_upstox_marketlevel(
        "X", [{"bp": 1, "bq": 1, "bno": 1, "ap": 2, "aq": 1, "ano": 1}]
    ).to_dict()
    kite = from_kite_depth(
        "X",
        {"buy": [{"price": 1, "quantity": 1, "orders": 1}],
         "sell": [{"price": 2, "quantity": 1, "orders": 1}]},
    ).to_dict()
    assert set(up.keys()) == set(kite.keys())          # symbol/levels/source/bids/asks
    assert up["bids"][0].keys() == kite["bids"][0].keys()  # price/quantity/orders
