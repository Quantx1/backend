import pytest, asyncio
fakeredis = pytest.importorskip("fakeredis")
from backend.platform import oauth_state as os_mod

@pytest.fixture
def store(monkeypatch):
    # Each ``asyncio.run`` below spins up a fresh event loop. fakeredis' async
    # client binds its internal queue to the loop it's created in, so a single
    # shared client raises "bound to a different event loop" on the 2nd call.
    # Fix: hand out a fresh client per call, all backed by one in-memory
    # FakeServer so the data persists across loops (matches real Redis).
    from fakeredis import aioredis as fra, FakeServer
    server = FakeServer()
    monkeypatch.setattr(
        os_mod, "_get_redis", lambda: fra.FakeRedis(decode_responses=True, server=server)
    )
    return os_mod

def test_store_then_consume_roundtrips(store):
    state = asyncio.run(store.store_state("u1", "zerodha", return_to="onboarding"))
    data = asyncio.run(store.consume_state(state))
    assert data == {"user_id": "u1", "broker": "zerodha", "return_to": "onboarding"}

def test_consume_is_single_use(store):
    state = asyncio.run(store.store_state("u1", "upstox", return_to="settings"))
    assert asyncio.run(store.consume_state(state)) is not None
    assert asyncio.run(store.consume_state(state)) is None

def test_unknown_state_returns_none(store):
    assert asyncio.run(store.consume_state("nope")) is None
