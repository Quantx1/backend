import asyncio
import json
import pytest
from backend.platform.depth_bus import DepthBus, DEPTH_STREAM


class FakeRedis:
    """Minimal Redis-Streams stub (decode_responses=True semantics)."""
    def __init__(self):
        self.entries = []  # list of (id, fields)
        self._seq = 0

    async def xadd(self, stream, fields, maxlen=None, approximate=True):
        assert stream == DEPTH_STREAM
        self._seq += 1
        self.entries.append((f"{self._seq}-0", dict(fields)))
        return f"{self._seq}-0"

    async def xread(self, streams, block=0, count=None):
        # Yield to the event loop like a real blocking xread would (so a
        # consume_forever loop driving this fake doesn't starve other tasks).
        await asyncio.sleep(0)
        last = streams[DEPTH_STREAM]
        if last == "$":
            start = self._seq  # "$" = only entries added after this read
        else:
            start = 0 if last == "0" else int(last.split("-")[0])
        new = [(eid, f) for (eid, f) in self.entries if int(eid.split("-")[0]) > start]
        if not new:
            return []
        return [(DEPTH_STREAM, new[:count] if count else new)]


@pytest.mark.asyncio
async def test_publish_then_consume_roundtrips_depth():
    bus = DepthBus(FakeRedis())
    await bus.publish("RELIANCE", {"symbol": "RELIANCE", "levels": 1,
                                   "bids": [{"price": 99.9, "quantity": 100, "orders": 1}],
                                   "asks": [], "source": "broker"})
    got = {}
    async def handler(symbol, depth):
        got["symbol"] = symbol
        got["depth"] = depth
    await bus.consume_once(handler, last_id="0")
    assert got["symbol"] == "RELIANCE"
    assert got["depth"]["bids"][0]["price"] == 99.9


@pytest.mark.asyncio
async def test_consume_forever_dispatches_then_cancels():
    bus = DepthBus(FakeRedis())
    await bus.publish("X", {"symbol": "X", "levels": 1, "bids": [], "asks": []})
    got = []
    async def handler(symbol, depth):
        got.append((symbol, depth))
    task = asyncio.create_task(bus.consume_forever(handler, start_id="0"))
    for _ in range(50):
        await asyncio.sleep(0.005)
        if got:
            break
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert got and got[0][0] == "X"


@pytest.mark.asyncio
async def test_consume_forever_survives_xread_error():
    class BoomRedis(FakeRedis):
        async def xread(self, streams, block=0, count=None):
            raise RuntimeError("boom")
    async def handler(symbol, depth):
        pass
    bus = DepthBus(BoomRedis())
    task = asyncio.create_task(bus.consume_forever(handler, start_id="0"))
    await asyncio.sleep(0.05)  # loop catches the error, sleeps, stays alive
    assert not task.done()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
