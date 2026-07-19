import pytest
from backend.data.brokers.ticker_mapping import BrokerTickerManager


class FakeBus:
    def __init__(self):
        self.published = []
    async def publish(self, symbol, depth):
        self.published.append((symbol, depth))


@pytest.mark.asyncio
async def test_on_tick_publishes_depth_when_present():
    bus = FakeBus()
    mgr = BrokerTickerManager(price_service=None, depth_bus=bus)
    await mgr._on_tick("u1", "RELIANCE", {"ltp": 100.0, "depth": {"levels": 1, "bids": [], "asks": []}})
    assert bus.published == [("RELIANCE", {"levels": 1, "bids": [], "asks": []})]


@pytest.mark.asyncio
async def test_on_tick_no_depth_no_publish():
    bus = FakeBus()
    mgr = BrokerTickerManager(price_service=None, depth_bus=bus)
    await mgr._on_tick("u1", "RELIANCE", {"ltp": 100.0})
    assert bus.published == []
