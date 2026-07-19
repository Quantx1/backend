from backend.data.brokers.ticker_mapping import build_zerodha_tick_data


def test_builder_includes_depth_when_present():
    tick = {
        "last_price": 100.0, "change": 1.0, "volume_traded": 12345,
        "ohlc": {"open": 99, "high": 101, "low": 98, "close": 99},
        "depth": {"buy": [{"price": 99.9, "quantity": 100, "orders": 1}],
                  "sell": [{"price": 100.1, "quantity": 80, "orders": 2}]},
    }
    td = build_zerodha_tick_data("RELIANCE", tick)
    assert td["symbol"] == "RELIANCE"
    assert td["ltp"] == 100.0
    assert td["source"] == "broker"
    assert td["depth"]["levels"] == 1
    assert td["depth"]["bids"][0]["price"] == 99.9


def test_builder_omits_depth_when_absent():
    td = build_zerodha_tick_data("X", {"last_price": 5.0, "ohlc": {"close": 5.0}})
    assert "depth" not in td  # honest-empty: no fabricated ladder
