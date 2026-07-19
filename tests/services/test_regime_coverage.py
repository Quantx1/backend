"""Regime-coverage tests — the accuracy guard.

A backtest that runs on carried-forward / defaulted regime (because
``regime_history`` doesn't reach back far enough) is NOT trustworthy for a
regime-gated strategy. ``resolve_regime_history_with_coverage`` reports what
fraction of the window mapped to a REAL detected regime (vs the pre-history
``sideways`` default), so the gate can fail-closed on fake regime.

Pure tests — a fake supabase stub returns canned rows, no network.
"""

from __future__ import annotations

from datetime import date

from backend.services.regime.resolver import (
    resolve_regime_history,
    resolve_regime_history_with_coverage,
)


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_a, **_k):
        return self

    def gte(self, *_a, **_k):
        return self

    def lte(self, *_a, **_k):
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        return type("R", (), {"data": self._rows})()


class _FakeSB:
    def __init__(self, rows):
        self._rows = rows

    def table(self, _name):
        return _FakeTable(self._rows)


class TestCoverage:
    def test_full_coverage_when_history_starts_before_window(self):
        rows = [{"regime": "bull", "detected_at": "2024-12-01"}]
        _map, cov = resolve_regime_history_with_coverage(
            _FakeSB(rows), start=date(2025, 1, 1), end=date(2025, 1, 31),
        )
        assert cov == 1.0  # a real regime exists ≤ every day in the window

    def test_zero_coverage_when_no_history_at_all(self):
        _map, cov = resolve_regime_history_with_coverage(
            _FakeSB([]), start=date(2025, 1, 1), end=date(2025, 1, 31),
        )
        assert cov == 0.0  # everything is the sideways default → fake

    def test_partial_coverage_when_history_starts_midway(self):
        # Real regime begins 2025-01-16 → first half of January is defaulted.
        rows = [{"regime": "bull", "detected_at": "2025-01-16"}]
        _map, cov = resolve_regime_history_with_coverage(
            _FakeSB(rows), start=date(2025, 1, 1), end=date(2025, 1, 31),
        )
        assert 0.4 < cov < 0.6  # ~16/31 days are real

    def test_defaulted_days_are_sideways(self):
        rows = [{"regime": "bull", "detected_at": "2025-01-16"}]
        m, _ = resolve_regime_history_with_coverage(
            _FakeSB(rows), start=date(2025, 1, 1), end=date(2025, 1, 31),
        )
        assert m[date(2025, 1, 1)] == "sideways"   # pre-history default
        assert m[date(2025, 1, 20)] == "bull"      # real, carried forward


class TestBackCompat:
    def test_plain_resolver_still_returns_map_only(self):
        rows = [{"regime": "bear", "detected_at": "2024-12-01"}]
        m = resolve_regime_history(_FakeSB(rows), start=date(2025, 1, 1), end=date(2025, 1, 5))
        assert isinstance(m, dict)
        assert m[date(2025, 1, 3)] == "bear"
