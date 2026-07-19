"""PR-U — tier cap enforcement tests.

Validates the new gates added to signals_routes + watchlist_routes:

  1. _apply_free_cap() truncates the signal list for Free non-admins,
     leaves Pro/Elite/admins alone.
  2. Watchlist POST raises 402 when Free user is at FREE_WATCHLIST_CAP,
     bypasses for Pro/Elite/admins.

These are unit-level — the route handlers are exercised directly with
mocked Supabase, no live FastAPI client needed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _user_tier(tier: str, is_admin: bool = False):
    from backend.core.tiers import Tier, UserTier
    return UserTier(user_id="u1", tier=Tier(tier), is_admin=is_admin)


def _fake_signal(symbol: str, direction: str = "LONG", segment: str = "EQUITY"):
    return {
        "symbol": symbol,
        "direction": direction,
        "segment": segment,
        "is_premium": False,
        "confidence": 0.7,
    }


# ─────────────────────────────────────────────────────────────────────────
# _apply_free_cap
# ─────────────────────────────────────────────────────────────────────────

def test_free_user_capped_to_one_signal():
    from backend.api.signals_routes import _apply_free_cap, FREE_DAILY_SIGNAL_CAP

    signals = [_fake_signal(s) for s in ("RELIANCE", "TCS", "INFY")]
    capped = _apply_free_cap(signals, _user_tier("free"))
    assert len(capped) == FREE_DAILY_SIGNAL_CAP == 1
    assert capped[0]["symbol"] == "RELIANCE"


def test_pro_user_uncapped():
    from backend.api.signals_routes import _apply_free_cap

    signals = [_fake_signal(s) for s in ("RELIANCE", "TCS", "INFY")]
    capped = _apply_free_cap(signals, _user_tier("pro"))
    assert len(capped) == 3
    assert capped == signals


def test_elite_user_uncapped():
    from backend.api.signals_routes import _apply_free_cap

    signals = [_fake_signal(s) for s in ("RELIANCE", "TCS", "INFY")]
    capped = _apply_free_cap(signals, _user_tier("elite"))
    assert len(capped) == 3


def test_free_admin_bypasses_cap():
    """Admin override: is_admin=True bypasses every tier gate."""
    from backend.api.signals_routes import _apply_free_cap

    signals = [_fake_signal(s) for s in ("RELIANCE", "TCS", "INFY")]
    capped = _apply_free_cap(signals, _user_tier("free", is_admin=True))
    assert len(capped) == 3


def test_empty_signal_list_returns_empty_regardless_of_tier():
    from backend.api.signals_routes import _apply_free_cap

    assert _apply_free_cap([], _user_tier("free")) == []
    assert _apply_free_cap([], _user_tier("pro")) == []


# ─────────────────────────────────────────────────────────────────────────
# Watchlist 402 cap
# ─────────────────────────────────────────────────────────────────────────

def _supabase_with_count(count: int):
    chain = MagicMock()
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.execute.return_value = SimpleNamespace(count=count, data=[])
    chain.insert.return_value = chain
    sb = MagicMock()
    sb.table.return_value = chain
    return sb, chain


def _watchlist_payload(symbol="RELIANCE"):
    """Minimal WatchlistAdd shape (the schema does .symbol.upper() etc)."""
    return SimpleNamespace(
        symbol=symbol,
        segment=SimpleNamespace(value="EQUITY"),
        alert_price_above=None,
        alert_price_below=None,
    )


def test_free_user_at_cap_raises_402():
    from backend.api.watchlist_routes import add_to_watchlist, FREE_WATCHLIST_CAP

    sb, _ = _supabase_with_count(FREE_WATCHLIST_CAP)
    with patch("backend.api.watchlist_routes._get_supabase_admin", return_value=sb):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(add_to_watchlist(
                _watchlist_payload(),
                user=SimpleNamespace(id="u1"),
                tier=_user_tier("free"),
            ))
    assert ei.value.status_code == 402
    detail = ei.value.detail
    assert detail["error"] == "watchlist_cap_reached"
    assert detail["cap"] == FREE_WATCHLIST_CAP
    assert detail["current_count"] == FREE_WATCHLIST_CAP


def test_free_user_below_cap_inserts():
    from backend.api.watchlist_routes import add_to_watchlist, FREE_WATCHLIST_CAP

    sb, _ = _supabase_with_count(FREE_WATCHLIST_CAP - 1)
    with patch("backend.api.watchlist_routes._get_supabase_admin", return_value=sb):
        out = asyncio.run(add_to_watchlist(
            _watchlist_payload(),
            user=SimpleNamespace(id="u1"),
            tier=_user_tier("free"),
        ))
    assert out == {"success": True}


def test_pro_user_at_5_symbols_does_not_raise():
    """Pro is uncapped → 100 existing symbols should still allow insert."""
    from backend.api.watchlist_routes import add_to_watchlist

    sb, _ = _supabase_with_count(100)
    with patch("backend.api.watchlist_routes._get_supabase_admin", return_value=sb):
        out = asyncio.run(add_to_watchlist(
            _watchlist_payload(),
            user=SimpleNamespace(id="u1"),
            tier=_user_tier("pro"),
        ))
    assert out == {"success": True}


def test_free_admin_at_cap_bypasses_check():
    from backend.api.watchlist_routes import add_to_watchlist, FREE_WATCHLIST_CAP

    sb, _ = _supabase_with_count(FREE_WATCHLIST_CAP + 10)
    with patch("backend.api.watchlist_routes._get_supabase_admin", return_value=sb):
        out = asyncio.run(add_to_watchlist(
            _watchlist_payload(),
            user=SimpleNamespace(id="u1"),
            tier=_user_tier("free", is_admin=True),
        ))
    assert out == {"success": True}
