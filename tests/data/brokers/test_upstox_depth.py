from backend.data.brokers.depth_models import from_upstox_marketlevel, MarketDepth


def test_parses_upstox_marketlevel_to_depth():
    market_level = [
        {"bp": 100.0, "bq": 500, "bno": 3, "ap": 100.1, "aq": 400, "ano": 2},
        {"bp": 99.95, "bq": 1200, "bno": 5, "ap": 100.2, "aq": 900, "ano": 4},
        {"bp": 0, "bq": 0, "bno": 0, "ap": 0, "aq": 0, "ano": 0},  # padding -> dropped
    ]
    md = from_upstox_marketlevel("RELIANCE", market_level)
    assert isinstance(md, MarketDepth)
    assert md.symbol == "RELIANCE"
    assert md.levels() == 2
    assert md.bids[0].price == 100.0 and md.bids[0].quantity == 500 and md.bids[0].orders == 3
    assert md.asks[0].price == 100.1 and md.asks[0].quantity == 400 and md.asks[0].orders == 2
    assert md.bids[1].price == 99.95 and md.asks[1].price == 100.2


def test_upstox_depth_honest_empty():
    assert from_upstox_marketlevel("X", []).levels() == 0
    assert from_upstox_marketlevel("X", [{"bp": 0, "bq": 0, "ap": 0, "aq": 0}]).levels() == 0
