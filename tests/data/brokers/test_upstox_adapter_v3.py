import asyncio
import pytest
from backend.data.brokers.ticker_mapping import UpstoxTickerAdapter


class FakeStreamerV3:
    def __init__(self):
        self.handlers = {}
        self.subscribed = []
        self.connected = False
    def on(self, event, cb):
        self.handlers[event] = cb
    def connect(self):
        self.connected = True
    def subscribe(self, keys, mode):
        self.subscribed.append((tuple(keys), mode))
    def unsubscribe(self, keys):
        pass
    def disconnect(self):
        self.connected = False
    def emit_message(self, msg):
        self.handlers["message"](msg)


@pytest.mark.asyncio
async def test_v3_message_routes_tick_with_depth_to_on_tick():
    got = []
    async def on_tick(symbol, tick_data):
        got.append((symbol, tick_data))

    fake = FakeStreamerV3()
    adapter = UpstoxTickerAdapter(
        access_token="t", on_tick=on_tick, loop=asyncio.get_event_loop(),
        streamer_factory=lambda token, keys, mode: fake,
    )
    await adapter.connect()
    await adapter.subscribe(["RELIANCE"])

    fake.emit_message({"feeds": {"NSE_EQ|RELIANCE": {"ff": {"marketFF": {
        "ltpc": {"ltp": 100.0, "cp": 99.0},
        "marketLevel": [{"bp": 99.9, "bq": 100, "bno": 1, "ap": 100.1, "aq": 80, "ano": 2}],
    }}}}})
    await asyncio.sleep(0.05)

    assert got, "on_tick was not called"
    sym, td = got[0]
    assert sym == "RELIANCE"
    assert td["ltp"] == 100.0
    assert td["depth"]["levels"] == 1
