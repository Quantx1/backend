"""AutoPilot per-stream toggle tests — PR-AS.

Pure-Python with mocked Supabase. Validates:
  - Cross-stream allocation sum ≤ 100% enforced at upsert time
  - is_stream_enabled defaults to False (opt-in, not opt-out)
  - list_streams_for_user always emits one row per built-in stream
  - Built-in stream name validation
  - no streams pending PROD (intraday LSTM stream removed with the model)
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from backend.services.autopilot.streams import (
    BUILTIN_STREAMS,
    NOT_YET_PROD_STREAMS,
    StreamState,
    is_builtin,
    is_prod_stream,
    is_stream_enabled,
    list_streams_for_user,
    total_allocated_pct,
    upsert_stream,
)


def _mock_sb(stored_rows: List[Dict[str, Any]] | None = None):
    """Mock supabase whose select returns ``stored_rows`` and whose
    upsert appends to it (so subsequent reads see the write)."""
    stored: List[Dict[str, Any]] = list(stored_rows or [])
    sb = MagicMock()

    class _Chain:
        def __init__(self, table_name: str):
            self.table_name = table_name
            self._filters: Dict[str, Any] = {}

        def select(self, *a, **kw): return self
        def eq(self, field, value): self._filters[field] = value; return self
        def lte(self, *a, **kw): return self
        def gte(self, *a, **kw): return self
        def order(self, *a, **kw): return self
        def limit(self, *a, **kw): return self

        def execute(self):
            rows = stored
            for f, v in self._filters.items():
                rows = [r for r in rows if r.get(f) == v]
            self._filters = {}
            return MagicMock(data=list(rows))

        def upsert(self, payload, on_conflict=None):
            self._pending_upsert = payload
            return self

    def _table(name: str):
        chain = _Chain(name)
        original_execute = chain.execute

        def _execute_with_upsert():
            if hasattr(chain, "_pending_upsert"):
                payload = chain._pending_upsert
                # Match on (user_id, stream, user_strategy_id)
                key = (payload["user_id"], payload["stream"], payload.get("user_strategy_id"))
                stored[:] = [
                    r for r in stored
                    if (r["user_id"], r["stream"], r.get("user_strategy_id")) != key
                ]
                stored.append(payload)
                del chain._pending_upsert
                return MagicMock(data=[payload])
            return original_execute()

        chain.execute = _execute_with_upsert
        return chain

    sb.table.side_effect = _table
    return sb


# ─────────────────────────────────────────────────────────────────────
# Constants + helpers
# ─────────────────────────────────────────────────────────────────────


class TestStreamConstants:

    def test_builtin_streams_complete(self):
        assert "swing" in BUILTIN_STREAMS
        assert "momentum" in BUILTIN_STREAMS
        assert "portfolio" in BUILTIN_STREAMS
        assert "options" in BUILTIN_STREAMS
        # intraday LSTM stream was removed when the model was dropped from v1
        assert "intraday" not in BUILTIN_STREAMS

    def test_no_streams_pending_prod(self):
        """The intraday LSTM stream (the only NOT_YET_PROD entry) was removed
        with the model — every built-in stream is now PROD-backed."""
        assert NOT_YET_PROD_STREAMS == ()
        assert is_prod_stream("swing") is True
        for stream in BUILTIN_STREAMS:
            assert is_prod_stream(stream) is True

    def test_is_builtin_rejects_unknown(self):
        assert is_builtin("user_strategy") is False  # special, not "built-in"
        assert is_builtin("crypto") is False
        assert is_builtin("swing") is True


# ─────────────────────────────────────────────────────────────────────
# is_stream_enabled (used by AutoPilotService gate)
# ─────────────────────────────────────────────────────────────────────


class TestIsStreamEnabled:

    def test_defaults_false_when_no_row(self):
        """Opt-in semantics: missing row → disabled."""
        sb = _mock_sb([])
        assert is_stream_enabled(sb, user_id="u1", stream="swing") is False

    def test_returns_true_when_enabled(self):
        sb = _mock_sb([
            {"user_id": "u1", "stream": "swing", "user_strategy_id": None, "enabled": True},
        ])
        assert is_stream_enabled(sb, user_id="u1", stream="swing") is True

    def test_returns_false_when_disabled(self):
        sb = _mock_sb([
            {"user_id": "u1", "stream": "swing", "user_strategy_id": None, "enabled": False},
        ])
        assert is_stream_enabled(sb, user_id="u1", stream="swing") is False

    def test_supabase_exception_returns_false(self):
        """Fail-safe: if the DB query crashes, return False (don't accidentally trade)."""
        sb = MagicMock()
        sb.table.side_effect = RuntimeError("simulated outage")
        assert is_stream_enabled(sb, user_id="u1", stream="swing") is False


# ─────────────────────────────────────────────────────────────────────
# list_streams_for_user
# ─────────────────────────────────────────────────────────────────────


class TestListStreams:

    def test_empty_user_gets_full_default_list(self):
        """Even a brand-new user should see one row per built-in stream."""
        sb = _mock_sb([])
        states = list_streams_for_user(sb, "u1")
        assert len(states) == len(BUILTIN_STREAMS)
        streams = {s.stream for s in states}
        assert streams == set(BUILTIN_STREAMS)
        for s in states:
            assert s.enabled is False
            assert s.allocated_capital_pct == 0

    def test_user_strategy_rows_included(self):
        """If the user has toggled a user_strategy stream, it's in the response too."""
        sb = _mock_sb([
            {"user_id": "u1", "stream": "user_strategy", "user_strategy_id": "abc",
             "enabled": True, "allocated_capital_pct": 20},
        ])
        states = list_streams_for_user(sb, "u1")
        us_rows = [s for s in states if s.stream == "user_strategy"]
        assert len(us_rows) == 1
        assert us_rows[0].user_strategy_id == "abc"


# ─────────────────────────────────────────────────────────────────────
# upsert_stream — validation rules
# ─────────────────────────────────────────────────────────────────────


class TestUpsertStream:

    def test_unknown_stream_rejected(self):
        sb = _mock_sb([])
        with pytest.raises(ValueError, match="unknown stream"):
            upsert_stream(sb, user_id="u1", stream="crypto",
                          user_strategy_id=None,
                          enabled=True, allocated_capital_pct=10)

    def test_user_strategy_requires_id(self):
        sb = _mock_sb([])
        with pytest.raises(ValueError, match="requires a user_strategy_id"):
            upsert_stream(sb, user_id="u1", stream="user_strategy",
                          user_strategy_id=None,
                          enabled=True, allocated_capital_pct=10)

    def test_allocation_bounds(self):
        sb = _mock_sb([])
        with pytest.raises(ValueError, match="allocated_capital_pct"):
            upsert_stream(sb, user_id="u1", stream="swing",
                          user_strategy_id=None,
                          enabled=True, allocated_capital_pct=120)
        with pytest.raises(ValueError, match="allocated_capital_pct"):
            upsert_stream(sb, user_id="u1", stream="swing",
                          user_strategy_id=None,
                          enabled=True, allocated_capital_pct=-5)

    def test_cross_stream_sum_enforced(self):
        """Sum of enabled allocations must not exceed 100%."""
        sb = _mock_sb([
            {"user_id": "u1", "stream": "swing", "user_strategy_id": None,
             "enabled": True, "allocated_capital_pct": 60},
            {"user_id": "u1", "stream": "momentum", "user_strategy_id": None,
             "enabled": True, "allocated_capital_pct": 30},
        ])
        # Total enabled = 60+30 = 90. Adding 15 on portfolio → 105% → reject.
        with pytest.raises(ValueError, match="100%"):
            upsert_stream(sb, user_id="u1", stream="portfolio",
                          user_strategy_id=None,
                          enabled=True, allocated_capital_pct=15)

    def test_disabling_doesnt_count_in_sum(self):
        """An enabled=False toggle never contributes to the 100% sum,
        regardless of allocated_capital_pct value."""
        sb = _mock_sb([
            {"user_id": "u1", "stream": "swing", "user_strategy_id": None,
             "enabled": True, "allocated_capital_pct": 90},
        ])
        # Setting momentum to disabled with 50% allocation should NOT trip the cap
        # (because disabled doesn't count toward the sum)
        state = upsert_stream(sb, user_id="u1", stream="momentum",
                              user_strategy_id=None,
                              enabled=False, allocated_capital_pct=50)
        assert state.enabled is False
        # Sum across enabled only = 90 (swing) — fine

    def test_updating_same_stream_uses_new_value_not_old(self):
        """When updating a stream we already have, the OLD allocation
        must be replaced — not added to the sum check."""
        sb = _mock_sb([
            {"user_id": "u1", "stream": "swing", "user_strategy_id": None,
             "enabled": True, "allocated_capital_pct": 90},
        ])
        # Bumping swing from 90 → 100 should succeed (we're replacing it,
        # not adding 100 to existing 90)
        state = upsert_stream(sb, user_id="u1", stream="swing",
                              user_strategy_id=None,
                              enabled=True, allocated_capital_pct=100)
        assert state.allocated_capital_pct == 100


# ─────────────────────────────────────────────────────────────────────
# total_allocated_pct
# ─────────────────────────────────────────────────────────────────────


class TestTotalAllocatedPct:

    def test_sums_enabled_only(self):
        states = [
            StreamState("swing", None, True, 60, True, None, None),
            StreamState("momentum", None, True, 25, True, None, None),
            StreamState("portfolio", None, False, 50, True, None, None),  # disabled, ignored
        ]
        assert total_allocated_pct(states) == 85
