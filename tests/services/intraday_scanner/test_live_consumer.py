import pytest
from backend.services.intraday_scanner.live_consumer import IntradayLiveConsumer
from backend.services.intraday_scanner.scanner import IntradayMatch
from backend.platform.realtime import MessageType


class FakeManager:
    def __init__(self):
        self.sent = []
    async def broadcast_symbol_update(self, symbol, message):
        self.sent.append((symbol, message))


def _match(setup_id="vwap_bounce"):
    return IntradayMatch(
        symbol="RELIANCE", setup_id=setup_id, direction="bullish",
        detected_at="2026-06-08T10:05:00+05:30", timeframe="5m",
        entry=100.0, stop=99.0, target=103.0, risk_reward=3.0,
        last_price=100.2, volume_ratio=1.8, confidence="high", reason="r",
    )


@pytest.mark.asyncio
async def test_emits_intraday_signal_on_match():
    mgr = FakeManager()
    consumer = IntradayLiveConsumer(mgr, scan_fn=lambda sym, frame: [_match()],
                                    frame_fn=lambda sym: object())
    await consumer.on_bar_close("RELIANCE")
    assert len(mgr.sent) == 1
    symbol, msg = mgr.sent[0]
    assert symbol == "RELIANCE"
    assert msg.type == MessageType.INTRADAY_SIGNAL
    assert msg.data["setup_id"] == "vwap_bounce"


@pytest.mark.asyncio
async def test_honest_empty_no_match_no_emit():
    mgr = FakeManager()
    consumer = IntradayLiveConsumer(mgr, scan_fn=lambda sym, frame: [],
                                    frame_fn=lambda sym: object())
    await consumer.on_bar_close("RELIANCE")
    assert mgr.sent == []


@pytest.mark.asyncio
async def test_dedups_same_setup_until_changed():
    mgr = FakeManager()
    consumer = IntradayLiveConsumer(mgr, scan_fn=lambda sym, frame: [_match()],
                                    frame_fn=lambda sym: object())
    await consumer.on_bar_close("RELIANCE")
    await consumer.on_bar_close("RELIANCE")
    assert len(mgr.sent) == 1


@pytest.mark.asyncio
async def test_no_frame_no_scan():
    mgr = FakeManager()
    called = {"n": 0}
    def scan(sym, frame):
        called["n"] += 1
        return []
    consumer = IntradayLiveConsumer(mgr, scan_fn=scan, frame_fn=lambda sym: None)
    await consumer.on_bar_close("RELIANCE")
    assert called["n"] == 0 and mgr.sent == []
