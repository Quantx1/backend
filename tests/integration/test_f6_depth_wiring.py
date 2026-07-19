"""End-to-end (in-process): a depth-bearing tick fans to a /ws symbol watcher."""
import pytest
from backend.data.brokers.ticker_mapping import BrokerTickerManager, build_zerodha_tick_data
from backend.platform.realtime import ConnectionManager, MessageType
from backend.platform.depth_bus import DepthBus, DEPTH_STREAM
from backend.platform.depth_to_ws import make_depth_handler


class _FakeRedis:
    def __init__(self):
        self.entries = []
        self._seq = 0
    async def xadd(self, stream, fields, maxlen=None, approximate=True):
        self._seq += 1
        self.entries.append((f"{self._seq}-0", dict(fields)))
        return f"{self._seq}-0"
    async def xread(self, streams, block=0, count=None):
        last = streams[DEPTH_STREAM]
        start = 0 if last == "0" else int(last.split("-")[0])
        new = [(eid, f) for (eid, f) in self.entries if int(eid.split("-")[0]) > start]
        return [(DEPTH_STREAM, new[:count] if count else new)] if new else []


class _FakeWS:
    def __init__(self): self.sent = []
    async def send_text(self, text): self.sent.append(text)


@pytest.mark.asyncio
async def test_tick_with_depth_reaches_ws_watcher():
    bus = DepthBus(_FakeRedis())
    mgr = ConnectionManager()
    ws = _FakeWS()
    mgr.active_connections["u1"] = ws
    mgr.subscribe_to_symbol("u1", "RELIANCE")

    ticker_mgr = BrokerTickerManager(price_service=None, depth_bus=bus)
    tick = {"last_price": 100.0, "ohlc": {"close": 99.0},
            "depth": {"buy": [{"price": 99.9, "quantity": 100, "orders": 1}],
                      "sell": [{"price": 100.1, "quantity": 80, "orders": 2}]}}
    td = build_zerodha_tick_data("RELIANCE", tick)
    await ticker_mgr._on_tick("u1", "RELIANCE", td)
    await bus.consume_once(make_depth_handler(mgr), last_id="0")

    assert len(ws.sent) == 1
    assert MessageType.DEPTH_UPDATE.value in ws.sent[0]
    assert "99.9" in ws.sent[0]
