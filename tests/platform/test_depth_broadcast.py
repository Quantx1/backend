import pytest
from backend.platform.realtime import ConnectionManager, MessageType, WSMessage
from backend.platform.depth_to_ws import make_depth_handler


class FakeWS:
    def __init__(self):
        self.sent = []
    async def send_text(self, text):
        self.sent.append(text)


@pytest.mark.asyncio
async def test_depth_handler_broadcasts_to_symbol_watchers():
    mgr = ConnectionManager()
    ws = FakeWS()
    mgr.active_connections["u1"] = ws
    mgr.subscribe_to_symbol("u1", "RELIANCE")

    handler = make_depth_handler(mgr)
    await handler("RELIANCE", {"symbol": "RELIANCE", "levels": 1, "bids": [], "asks": []})

    assert len(ws.sent) == 1
    assert MessageType.DEPTH_UPDATE.value in ws.sent[0]
    assert "RELIANCE" in ws.sent[0]
