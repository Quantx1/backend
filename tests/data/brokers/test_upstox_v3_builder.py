from backend.data.brokers.ticker_mapping import build_upstox_v3_tick_data


def _feed():
    return {
        "ff": {"marketFF": {
            "ltpc": {"ltp": 100.0, "cp": 99.0},
            "marketOHLC": {"ohlc": [{"open": 98, "high": 101, "low": 97, "close": 99, "vol": 4200}]},
            "marketLevel": [
                {"bp": 99.9, "bq": 100, "bno": 1, "ap": 100.1, "aq": 80, "ano": 2},
                {"bp": 99.8, "bq": 200, "bno": 2, "ap": 100.2, "aq": 60, "ano": 1},
            ],
        }}
    }


def test_builder_extracts_price_and_30level_depth():
    td = build_upstox_v3_tick_data("RELIANCE", _feed())
    assert td["symbol"] == "RELIANCE"
    assert td["ltp"] == 100.0
    assert td["source"] == "broker"
    assert td["volume"] == 4200
    assert td["depth"]["levels"] == 2
    assert td["depth"]["bids"][0]["price"] == 99.9
    assert td["depth"]["asks"][1]["price"] == 100.2


def test_builder_omits_depth_when_no_levels():
    feed = {"ff": {"marketFF": {"ltpc": {"ltp": 5.0, "cp": 5.0}, "marketLevel": []}}}
    td = build_upstox_v3_tick_data("X", feed)
    assert "depth" not in td
    assert td["ltp"] == 5.0
