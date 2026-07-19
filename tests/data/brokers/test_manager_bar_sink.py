import pytest
from backend.data.brokers.ticker_mapping import BrokerTickerManager


@pytest.mark.asyncio
async def test_on_tick_feeds_bar_sink():
    fed = []
    def sink(symbol, price, volume):
        fed.append((symbol, price, volume))
    mgr = BrokerTickerManager(price_service=None, bar_sink=sink)
    await mgr._on_tick("u1", "RELIANCE", {"ltp": 100.5, "volume": 4200})
    assert fed == [("RELIANCE", 100.5, 4200)]


@pytest.mark.asyncio
async def test_on_tick_no_sink_is_noop():
    mgr = BrokerTickerManager(price_service=None)
    await mgr._on_tick("u1", "RELIANCE", {"ltp": 100.5, "volume": 4200})  # must not raise
