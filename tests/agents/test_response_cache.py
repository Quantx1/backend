import pytest

import backend.ai.agents.response_cache as rc


@pytest.fixture(autouse=True)
def _no_l2(monkeypatch):
    # Pure unit test: disable the L2 Supabase path so cache_set/get never touch
    # the network. The dedicated L2 test re-enables it with a fake client.
    monkeypatch.setattr(rc, "_sb", lambda: None)
    rc._L1.clear()


def test_l1_set_then_get_returns_payload():
    rc.cache_set("k:2026-06-10", {"answer": "hi"}, ttl_seconds=3600, surface="t", model="m")
    assert rc.cache_get("k:2026-06-10") == {"answer": "hi"}


def test_l1_expired_entry_is_a_miss(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(rc.time, "monotonic", lambda: clock["t"])
    rc.cache_set("k", {"answer": "x"}, ttl_seconds=5, surface="t", model="m")
    clock["t"] = 1006.0   # advance past the 5s TTL
    assert rc.cache_get("k") is None


def test_missing_key_returns_none():
    assert rc.cache_get("does-not-exist") is None


def test_l2_hit_populates_l1(monkeypatch):
    class _SB:
        def table(self, *_):
            return self
        def select(self, *_):
            return self
        def eq(self, *_a, **_k):
            return self
        def gt(self, *_a, **_k):
            return self
        def limit(self, *_):
            return self
        def execute(self):
            class R:
                data = [{"payload": {"answer": "from-db"}}]
            return R()
    monkeypatch.setattr(rc, "_sb", lambda: _SB())
    assert rc.cache_get("cold:key") == {"answer": "from-db"}
    monkeypatch.setattr(rc, "_sb", lambda: None)
    assert rc.cache_get("cold:key") == {"answer": "from-db"}
