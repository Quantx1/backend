"""Route-level integration tests (audit #6).

Every Section-4B bug shared one root cause: a wrong column / import / attribute
name behind a broad ``try/except``, so a headline route silently 500'd or
returned empty — and the unit tests never caught it because they bypassed the
route/schema seam. These tests drive the REAL FastAPI app through ``TestClient``:

1. **Boot** — importing the app loads every route module. A module-level import
   of a deleted symbol (e.g. the confluence ``PATTERN_SCANNERS`` import that
   hard-500'd the flagship screener) breaks this immediately.

2. **Registration inventory** — every critical user-action route is registered
   with the expected method + path. Catches an unregistered router or a path
   drift — the exact ambiguity that made the Bull/Bear debate look like dead
   code in the audit.

3. **Execution smoke** — the two GET routes that previously 500'd (F&O option
   chain on a ``regime_row`` NameError; confluence on the dead import) now
   execute end-to-end with the external boundary stubbed, returning a non-5xx
   status. This locks the actual fixes, not just the route's existence.

The harness overrides the auth leaf (``core.security.get_current_user``) and
stubs ``resolve_user_tier`` to an admin so tier gates pass, then stubs each
route's data boundary — exercising the real handler / schema in between.
"""
from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient


# ── Tier 1: boot + registration (no external calls, rock-solid) ─────────────

# (method, exact registered path) for each critical user action. Pinned from
# the live route table so a router-unregistration or path drift fails loudly.
CRITICAL_ROUTES = [
    ("POST", "/api/ai/copilot/chat"),                       # Main Chat (cap fix)
    ("POST", "/api/ai/debate/signal/{signal_id}"),          # Counterpoint debate
    ("GET", "/api/fo-strategies/chain/{symbol}"),           # chain tab (regime_row)
    ("POST", "/api/fo-strategies/paper/open"),              # F&O Deploy to paper
    ("POST", "/api/fo-strategies/backtest"),               # F&O backtest template
    ("POST", "/api/fo-strategies/ai-suggest"),             # Deploy AI suggestion
    ("GET", "/api/screener/v2/confluence"),                # Power Screener (import)
    ("POST", "/api/screener/pk/scan/batch"),               # batch scan (gating)
]


def _app():
    from backend.api.app import app
    return app


def test_app_boots_and_loads_every_route_module():
    """Importing the app must not raise — every route module loads. A
    module-level import of a deleted symbol would break this at load time."""
    from fastapi import FastAPI
    app = _app()
    assert isinstance(app, FastAPI)
    assert len(app.routes) > 50  # the app registers hundreds of routes


def _route_table():
    table = set()
    for r in _app().routes:
        for m in (getattr(r, "methods", None) or set()):
            table.add((m, getattr(r, "path", "")))
    return table


@pytest.mark.parametrize("method,path", CRITICAL_ROUTES)
def test_critical_route_registered(method, path):
    """Each critical user-action route must be registered with the right
    method+path — catches unregistered routers and path drift."""
    assert (method, path) in _route_table(), f"route not registered: {method} {path}"


# ── Tier 2: execution smoke (stubbed boundary, exercises real handler) ──────


@pytest.fixture
def authed_client(monkeypatch):
    """TestClient with auth overridden to an admin (tier gates pass).

    Overrides the auth leaf both route families depend on, and stubs
    resolve_user_tier so RequireFeature/RequireTier resolve to an admin.
    raise_server_exceptions=False so a handler 500 surfaces as a 500
    RESPONSE we can assert on, rather than re-raising into the test.
    """
    from backend.api.app import app
    from backend.core import security as core_security
    from backend.core import tiers as tiers_mod
    from backend.middleware import tier_gate

    fake_user = {"id": "itest-user", "email": "itest@quantx.app"}
    app.dependency_overrides[core_security.get_current_user] = lambda: fake_user
    # app.py defines its own get_current_user used by some routers — override it too.
    try:
        from backend.api.app import get_current_user as app_gcu
        app.dependency_overrides[app_gcu] = lambda: fake_user
    except Exception:
        pass

    admin = tiers_mod.UserTier(
        user_id="itest-user", tier=tiers_mod.Tier.ELITE, is_admin=True,
        email="itest@quantx.app",
    )
    monkeypatch.setattr(tier_gate, "resolve_user_tier", lambda *a, **k: admin)

    client = TestClient(app, raise_server_exceptions=False)
    yield client
    app.dependency_overrides.clear()


def test_fo_chain_unavailable_path_is_200(authed_client, monkeypatch):
    """No connected broker → get_option_chain returns None → the route must
    return a clean 200 'unavailable' payload (not a 500). Proves the auth gate,
    supabase fetch, and handler seam all execute."""
    import backend.api.fo_strategies_routes as fo

    monkeypatch.setattr(fo, "get_supabase_admin", lambda: object())
    monkeypatch.setattr(
        "backend.services.execution.option_chain.get_option_chain",
        lambda *a, **k: None,
    )
    res = authed_client.get("/api/fo-strategies/chain/NIFTY")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["source"] == "unavailable"
    assert body["rows"] == []


def test_fo_chain_with_rows_runs_greeks_path(authed_client, monkeypatch):
    """With real chain rows the handler runs the regime-spot-Greeks block —
    this is the path that 500'd on the undefined `regime_row`. Locks that fix."""
    import backend.api.fo_strategies_routes as fo

    row = types.SimpleNamespace(
        strike=22000.0, option_type="CE", expiry="2026-06-26", ltp=120.0,
        bid=119.0, ask=121.0, oi=10000, volume=5000, iv=0.0,
        tradingsymbol="NIFTY26JUN22000CE",
    )
    monkeypatch.setattr(fo, "get_supabase_admin", lambda: object())
    monkeypatch.setattr(
        "backend.services.execution.option_chain.get_option_chain",
        lambda *a, **k: [row],
    )
    # Stub the regime + spot helpers so we don't touch live data sources.
    monkeypatch.setattr(fo, "_load_latest_regime", lambda: {})
    monkeypatch.setattr(fo, "_spot_for", lambda sym, regime_row: 22000.0)

    res = authed_client.get("/api/fo-strategies/chain/NIFTY")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["source"] == "broker"
    assert len(body["rows"]) == 1
    assert body["rows"][0]["strike"] == 22000.0


def test_confluence_route_imports_and_executes(authed_client, monkeypatch):
    """The flagship Power Screener confluence route imports confluence_scan at
    call time (no try/except). The deleted PATTERN_SCANNERS import made this a
    hard 500. With the indicator cache stubbed empty it must reach its handled
    503 ('data not ready') — never a 500 from the import."""
    import backend.data.screener.engine as engine

    fake_screener = types.SimpleNamespace(_get_computed_data=lambda: (None, {}))
    monkeypatch.setattr(engine, "get_live_screener", lambda: fake_screener)

    res = authed_client.get("/api/screener/v2/confluence")
    assert res.status_code != 500, res.text
    assert res.status_code in (200, 503)


# ── Standalone on-demand sentiment ("Mood" for any stock) ───────────────────

def test_market_sentiment_honest_empty(monkeypatch):
    """GET /api/market/sentiment/{symbol} must return available=False (not a
    fabricated score) when the engine has no data for the symbol."""
    from fastapi.testclient import TestClient
    from backend.api.app import app
    import backend.ai.sentiment.engine as eng

    class _Stub:
        async def score_symbol(self, sym, **kw):
            return None
    monkeypatch.setattr(eng, "get_sentiment_engine", lambda: _Stub())

    res = TestClient(app, raise_server_exceptions=False).get("/api/market/sentiment/NOSUCH")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["available"] is False and body["mean_score"] is None


def test_market_sentiment_returns_label(monkeypatch):
    """With real sentiment, the endpoint maps mean_score → a bullish/bearish/
    neutral label for the standalone Mood card."""
    from fastapi.testclient import TestClient
    from backend.api.app import app
    import backend.ai.sentiment.engine as eng

    class _Stub:
        async def score_symbol(self, sym, **kw):
            return {"symbol": sym, "mean_score": 0.6, "headline_count": 4,
                    "positive_count": 3, "negative_count": 0, "neutral_count": 1,
                    "sample_headlines": [], "sources": ["ET"]}
    monkeypatch.setattr(eng, "get_sentiment_engine", lambda: _Stub())

    res = TestClient(app, raise_server_exceptions=False).get("/api/market/sentiment/RELTEST")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["available"] is True
    assert body["label"] == "bullish" and body["mean_score"] == 0.6
