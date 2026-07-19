"""Pure tests for the AI Setup Finder aggregation/shaping.

No network, no DB, no live screener — we monkeypatch ``get_live_screener`` so
``run_scanner`` returns fabricated scanner payloads keyed by scanner id, then
assert the bucket counts / symbol lists / total / ok flag are shaped right.
"""
import pytest

import backend.ai.agents.response_cache as rc
from backend.services.scanners.setup_finder import (
    find_setups,
    _symbols_from_scan,
    SETUP_MAP,
)


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch):
    # find_setups now caches successful runs — keep tests pure + independent.
    monkeypatch.setattr(rc, "_sb", lambda: None)
    rc._L1.clear()

# scanner_id → fabricated run_scanner payload (mirrors format_for_frontend shape)
FAKE_RESULTS = {
    58: {"success": True, "results": [{"symbol": "RELIANCE"}, {"symbol": "TCS"}]},
    59: {"success": True, "results": [{"symbol": "INFY"}]},
    54: {"success": True, "results": [
        {"symbol": "HDFCBANK"}, {"symbol": "ICICIBANK"}, {"symbol": "SBIN"},
    ]},
    57: {"success": True, "results": []},  # honest-empty bucket
}


class _FakeScreener:
    def __init__(self, table):
        self._table = table
        self.calls = []

    async def run_scanner(self, scanner_id, exchange="N", index="12"):
        self.calls.append((scanner_id, exchange, index))
        return self._table.get(scanner_id, {"success": False, "results": []})


@pytest.fixture
def patch_screener(monkeypatch):
    def _install(table, cls=_FakeScreener):
        fake = cls(table) if cls is _FakeScreener else cls
        monkeypatch.setattr(
            "backend.data.screener.engine.get_live_screener",
            lambda: fake,
        )
        return fake
    return _install


# ── _symbols_from_scan (pure shaping) ────────────────────────────────


def test_symbols_extracts_in_order_and_dedupes():
    payload = {"results": [
        {"symbol": "AAA"}, {"symbol": "BBB"}, {"symbol": "AAA"}, {"symbol": ""},
        {"nope": 1}, "garbage", {"symbol": "  CCC  "},
    ]}
    assert _symbols_from_scan(payload) == ["AAA", "BBB", "CCC"]


def test_symbols_handles_missing_and_bad_shapes():
    assert _symbols_from_scan(None) == []
    assert _symbols_from_scan({}) == []
    assert _symbols_from_scan({"results": None}) == []
    assert _symbols_from_scan({"success": False, "results": []}) == []


# ── find_setups (aggregation) ────────────────────────────────────────


async def test_find_setups_shapes_four_buckets_in_order(patch_screener):
    fake = patch_screener(FAKE_RESULTS)
    out = await find_setups()

    # Always 4 buckets, in canonical order
    assert [s["key"] for s in out["setups"]] == [m[0] for m in SETUP_MAP]
    assert [s["label"] for s in out["setups"]] == [m[1] for m in SETUP_MAP]

    by_key = {s["key"]: s for s in out["setups"]}
    assert by_key["breakout"]["count"] == 2
    assert by_key["breakout"]["symbols"] == ["RELIANCE", "TCS"]
    assert by_key["pullback"]["count"] == 1
    assert by_key["trend"]["count"] == 3
    assert by_key["reversal"]["count"] == 0
    assert by_key["reversal"]["symbols"] == []

    # total == sum of real matches; ok True because scanners ran
    assert out["total"] == 6
    assert out["ok"] is True
    # Every canonical scanner id was dispatched exactly once
    assert sorted(c[0] for c in fake.calls) == sorted(m[2] for m in SETUP_MAP)


async def test_find_setups_universe_selects_full_index(patch_screener):
    fake = patch_screener(FAKE_RESULTS)
    await find_setups(universe="nse_all")
    # nse_all → full index breadth "0"; default → "12"
    assert all(c[2] == "0" for c in fake.calls)

    fake2 = patch_screener(FAKE_RESULTS)
    await find_setups()
    assert all(c[2] == "12" for c in fake2.calls)


async def test_find_setups_isolates_a_failing_scanner(patch_screener, monkeypatch):
    class _OneRaises(_FakeScreener):
        async def run_scanner(self, scanner_id, exchange="N", index="12"):
            self.calls.append((scanner_id, exchange, index))
            if scanner_id == 54:  # trend bucket blows up
                raise RuntimeError("boom")
            return self._table.get(scanner_id, {"results": []})

    fake = _OneRaises(FAKE_RESULTS)
    monkeypatch.setattr(
        "backend.data.screener.engine.get_live_screener", lambda: fake,
    )
    out = await find_setups()
    by_key = {s["key"]: s for s in out["setups"]}

    # Failing bucket isolated to 0; others intact
    assert by_key["trend"]["count"] == 0
    assert by_key["trend"]["symbols"] == []
    assert by_key["breakout"]["count"] == 2
    assert out["total"] == 3  # 2 + 1 + 0 + 0
    assert out["ok"] is True  # other scanners still ran


async def test_find_setups_all_fail_marks_not_ok(monkeypatch):
    class _AllRaise:
        async def run_scanner(self, *a, **k):
            raise RuntimeError("dead")

    monkeypatch.setattr(
        "backend.data.screener.engine.get_live_screener", lambda: _AllRaise(),
    )
    out = await find_setups()
    assert out["total"] == 0
    assert out["ok"] is False
    assert all(s["count"] == 0 and s["symbols"] == [] for s in out["setups"])
    assert len(out["setups"]) == 4
