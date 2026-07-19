from datetime import datetime, timezone, timedelta
from backend.data.brokers.intraday_bars import IntradayBarAggregator

IST = timezone(timedelta(hours=5, minutes=30))


def _ts(h, m, s=0):
    return datetime(2026, 6, 8, h, m, s, tzinfo=IST)


def test_ticks_aggregate_into_5m_bar_and_close_on_rollover():
    agg = IntradayBarAggregator(interval_min=5, max_bars=10)
    assert agg.feed("X", 100.0, 100, _ts(10, 0, 10)) is None
    assert agg.feed("X", 102.0, 130, _ts(10, 2, 0)) is None
    assert agg.feed("X", 101.0, 150, _ts(10, 4, 59)) is None
    closed = agg.feed("X", 101.5, 160, _ts(10, 5, 1))
    assert closed is not None
    assert closed["open"] == 100.0 and closed["high"] == 102.0 and closed["low"] == 100.0 and closed["close"] == 101.0
    assert closed["volume"] == 50  # 150 - 100 cumulative delta within the bar


def test_frame_returns_completed_bars_with_ist_index():
    agg = IntradayBarAggregator(interval_min=5)
    agg.feed("X", 10.0, 10, _ts(9, 16))
    agg.feed("X", 11.0, 20, _ts(9, 21))   # closes 09:15 bar
    df = agg.frame("X")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 1
    assert df.iloc[0]["open"] == 10.0


def test_frame_none_when_no_completed_bars():
    agg = IntradayBarAggregator()
    agg.feed("X", 10.0, 10, _ts(9, 16))
    assert agg.frame("X") is None
