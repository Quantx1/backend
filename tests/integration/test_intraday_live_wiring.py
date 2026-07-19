"""In-process: ticks across a 5-min boundary -> scanner stub -> /ws INTRADAY_SIGNAL."""
import pytest
from datetime import datetime, timezone, timedelta
from backend.data.brokers.intraday_bars import IntradayBarAggregator
from backend.services.intraday_scanner.live_consumer import IntradayLiveConsumer
from backend.services.intraday_scanner.scanner import IntradayMatch
from backend.platform.realtime import MessageType

IST = timezone(timedelta(hours=5, minutes=30))


class FakeManager:
    def __init__(self): self.sent = []
    async def broadcast_symbol_update(self, symbol, message): self.sent.append((symbol, message))


@pytest.mark.asyncio
async def test_bar_close_triggers_scan_and_emit():
    agg = IntradayBarAggregator(interval_min=5)
    mgr = FakeManager()
    match = IntradayMatch(symbol="X", setup_id="orb_long", direction="bullish",
                          detected_at="t", timeframe="5m", entry=1, stop=0.9, target=1.3,
                          risk_reward=3.0, last_price=1.0, volume_ratio=2.0,
                          confidence="high", reason="r")
    consumer = IntradayLiveConsumer(mgr, scan_fn=lambda sym, frame: [match], frame_fn=agg.frame)

    pending = []
    holder = {"ts": datetime(2026, 6, 8, 10, 0, 10, tzinfo=IST)}

    def bar_sink(symbol, price, volume):
        closed = agg.feed(symbol, price, volume, holder["ts"])
        if closed is not None:
            pending.append(symbol)

    bar_sink("X", 100.0, 100)
    holder["ts"] = datetime(2026, 6, 8, 10, 5, 1, tzinfo=IST)
    bar_sink("X", 101.0, 150)  # rolls -> closes 10:00 bar

    for sym in pending:
        await consumer.on_bar_close(sym)

    assert len(mgr.sent) == 1
    assert mgr.sent[0][1].type == MessageType.INTRADAY_SIGNAL
