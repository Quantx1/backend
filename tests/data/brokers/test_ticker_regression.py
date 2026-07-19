"""Depth capture must not change the existing LTP/price tick shape."""
from backend.data.brokers.ticker_mapping import build_zerodha_tick_data


def test_price_fields_unchanged():
    td = build_zerodha_tick_data("RELIANCE", {
        "last_price": 100.0, "change": 1.0, "volume_traded": 999,
        "ohlc": {"open": 99, "high": 101, "low": 98, "close": 99},
    })
    assert td == {
        "symbol": "RELIANCE", "ltp": 100.0, "open": 99, "high": 101, "low": 98,
        "change": 1.0, "change_percent": (1.0 / 99 * 100), "volume": 999, "source": "broker",
    }
