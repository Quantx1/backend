"""GET /api/strategies/{id}/gate — read-only gate verdict.

Calls the route function directly (like tests/backend/test_paper_window_route.py),
monkeypatching the registry fetch so there is no network/DB. Asserts the
verdict shape + the no-backtest fail-closed path.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import backend.api.strategies_routes as sr


def _call(strategy_id: str, user):
    return asyncio.run(sr.strategy_gate(strategy_id, user=user))


def test_no_backtest_fails_closed(monkeypatch):
    monkeypatch.setattr(sr, "get_supabase_admin", lambda: None)
    monkeypatch.setattr(
        sr.strat_registry, "get_strategy",
        lambda supabase, *, strategy_id, user_id: {"id": strategy_id, "last_backtest": None},
    )
    out = _call("s1", SimpleNamespace(id="u1"))
    assert out["has_backtest"] is False
    assert out["passed"] is False
    assert isinstance(out["failures"], list) and out["failures"]


def test_verdict_shape_with_backtest(monkeypatch):
    # A plausible OOS backtest blob — we assert the response SHAPE + that the
    # gate ran against it (has_backtest True), not a specific pass/fail (the
    # thresholds are exercised in the evaluation unit tests).
    monkeypatch.setattr(sr, "get_supabase_admin", lambda: None)
    monkeypatch.setattr(
        sr.strat_registry, "get_strategy",
        lambda supabase, *, strategy_id, user_id: {
            "id": strategy_id,
            "last_backtest": {
                "oos": {
                    "sharpe_ratio": 1.4, "total_trades": 40, "win_rate": 0.55,
                    "max_drawdown_pct": 8.0, "total_return_pct": 22.0,
                },
            },
        },
    )
    out = _call("s2", SimpleNamespace(id="u1"))
    assert out["has_backtest"] is True
    assert isinstance(out["passed"], bool)
    assert isinstance(out["failures"], list)
    assert isinstance(out["metrics"], dict)


def test_not_owner_404(monkeypatch):
    import pytest
    from fastapi import HTTPException
    monkeypatch.setattr(sr, "get_supabase_admin", lambda: None)
    monkeypatch.setattr(
        sr.strat_registry, "get_strategy",
        lambda supabase, *, strategy_id, user_id: None,
    )
    with pytest.raises(HTTPException) as ei:
        _call("nope", SimpleNamespace(id="u1"))
    assert ei.value.status_code == 404
