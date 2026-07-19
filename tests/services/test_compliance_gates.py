"""Regression tests for the SEBI compliance gates.

Covers:
  * Path-A data-licensing entitlement (services/entitlement.py) — fail-closed in
    production, dev-open, licence flag, per-user broker grant, and the
    broker-vs-licence distinction for NSE-published aggregates.
  * Algo-order compliance gate (services/compliance_gate.py) — kill-switch,
    auto-trader pause, live-options block, and production empanelment requirement.

These enforce the deploy-blocking behavior; if any flips, a compliance control
has silently regressed.
"""

import pytest

from backend.services import entitlement as ent
from backend.services.entitlement import DataClass, check_entitlement
from backend.services import compliance_gate as cg
from backend.services.compliance_gate import check_algo_order


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    # Default all licence/empanelment flags OFF for every test.
    monkeypatch.setattr(ent.settings, "NSE_REALTIME_LICENSED", False, raising=False)
    monkeypatch.setattr(ent.settings, "NSE_EOD_LICENSED", False, raising=False)
    monkeypatch.setattr(cg.settings, "ALGO_TRADING_ENABLED", False, raising=False)
    monkeypatch.setattr(cg.settings, "ALGO_EMPANELMENT_ID", "", raising=False)
    monkeypatch.setattr(cg.settings, "ALGO_STATIC_IP", "", raising=False)
    monkeypatch.setattr(cg.settings, "ALLOW_LIVE_OPTIONS", False, raising=False)
    # Clear the per-user broker cache so tests don't leak into each other.
    ent._broker_cache.clear()


class _FakeSB:
    """Minimal supabase stub returning fixed user_profiles flags."""

    def __init__(self, **flags):
        self._flags = flags

    def table(self, *_a, **_k):
        return self

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def single(self):
        return self

    def execute(self):
        return type("R", (), {"data": dict(self._flags)})()


# ── Path-A entitlement ────────────────────────────────────────────────────

def test_entitlement_fails_closed_in_production(monkeypatch):
    monkeypatch.setattr(ent.settings, "APP_ENV", "production", raising=False)
    e = check_entitlement(DataClass.LIVE_QUOTE, None)
    assert e.allowed is False
    assert e.reason == "broker_required"


def test_entitlement_open_in_dev(monkeypatch):
    monkeypatch.setattr(ent.settings, "APP_ENV", "development", raising=False)
    e = check_entitlement(DataClass.LIVE_QUOTE, None)
    assert e.allowed is True
    assert e.source == "dev"


def test_entitlement_licence_flag_allows_in_production(monkeypatch):
    monkeypatch.setattr(ent.settings, "APP_ENV", "production", raising=False)
    monkeypatch.setattr(ent.settings, "NSE_REALTIME_LICENSED", True, raising=False)
    e = check_entitlement(DataClass.LIVE_QUOTE, None)
    assert e.allowed is True
    assert e.source == "licence"


def test_entitlement_broker_grant_allows_live_class(monkeypatch):
    monkeypatch.setattr(ent.settings, "APP_ENV", "production", raising=False)
    monkeypatch.setattr(ent, "_has_live_broker", lambda uid: True)
    e = check_entitlement(DataClass.LIVE_QUOTE, "user-1")
    assert e.allowed is True
    assert e.source == "broker"


def test_broker_does_not_grant_nse_aggregates(monkeypatch):
    # FII/DII is NSE-published, not in a broker feed → a connected broker must
    # NOT entitle it; only the EOD licence does.
    monkeypatch.setattr(ent.settings, "APP_ENV", "production", raising=False)
    monkeypatch.setattr(ent, "_has_live_broker", lambda uid: True)
    e = check_entitlement(DataClass.FII_DII, "user-1")
    assert e.allowed is False
    assert e.reason == "licence_required"


# ── Algo-order compliance gate ────────────────────────────────────────────

def test_kill_switch_blocks_order():
    sb = _FakeSB(kill_switch_active=True, auto_trader_enabled=True)
    d = check_algo_order(supabase=sb, user_id="u", segment="equity", automated=True, live=True)
    assert d.allowed is False
    assert d.reason == "kill_switch_active"


def test_auto_trader_disabled_blocks_automated_order():
    sb = _FakeSB(kill_switch_active=False, auto_trader_enabled=False)
    d = check_algo_order(supabase=sb, user_id="u", segment="equity", automated=True, live=True)
    assert d.allowed is False
    assert d.reason == "auto_trader_disabled"


def test_live_options_blocked_without_flag():
    sb = _FakeSB(kill_switch_active=False, auto_trader_enabled=True)
    d = check_algo_order(supabase=sb, user_id="u", segment="options", automated=True, live=True)
    assert d.allowed is False
    assert d.reason == "live_options_disabled_synthetic_backtest"


def test_production_live_requires_empanelment(monkeypatch):
    monkeypatch.setattr(cg.settings, "APP_ENV", "production", raising=False)
    sb = _FakeSB(kill_switch_active=False, auto_trader_enabled=True)
    d = check_algo_order(supabase=sb, user_id="u", segment="equity", automated=True, live=True)
    assert d.allowed is False
    assert d.reason == "algo_not_empanelled"


def test_production_live_still_blocked_without_static_ip(monkeypatch):
    # Empanelment ID + switch on, but no registered static IP → still refused.
    monkeypatch.setattr(cg.settings, "APP_ENV", "production", raising=False)
    monkeypatch.setattr(cg.settings, "ALGO_TRADING_ENABLED", True, raising=False)
    monkeypatch.setattr(cg.settings, "ALGO_EMPANELMENT_ID", "NSE-ALGO-1234", raising=False)
    sb = _FakeSB(kill_switch_active=False, auto_trader_enabled=True)
    d = check_algo_order(supabase=sb, user_id="u", segment="equity", automated=True, live=True)
    assert d.allowed is False
    assert d.reason == "algo_not_empanelled"


def test_production_live_allowed_when_fully_empanelled(monkeypatch):
    monkeypatch.setattr(cg.settings, "APP_ENV", "production", raising=False)
    monkeypatch.setattr(cg.settings, "ALGO_TRADING_ENABLED", True, raising=False)
    monkeypatch.setattr(cg.settings, "ALGO_EMPANELMENT_ID", "NSE-ALGO-1234", raising=False)
    monkeypatch.setattr(cg.settings, "ALGO_STATIC_IP", "203.0.113.7", raising=False)
    sb = _FakeSB(kill_switch_active=False, auto_trader_enabled=True)
    d = check_algo_order(supabase=sb, user_id="u", segment="equity", automated=True, live=True)
    assert d.allowed is True
    assert d.tags["algo_id"] == "NSE-ALGO-1234"
    assert d.tags["static_ip"] == "203.0.113.7"


def test_algo_readiness_report(monkeypatch):
    from backend.services.compliance_gate import algo_readiness

    r = algo_readiness()
    assert r["live_automated_ready"] is False
    assert r["checklist"]["exchange_empanelment_id"] is False
    monkeypatch.setattr(cg.settings, "ALGO_TRADING_ENABLED", True, raising=False)
    monkeypatch.setattr(cg.settings, "ALGO_EMPANELMENT_ID", "NSE-ALGO-1234", raising=False)
    monkeypatch.setattr(cg.settings, "ALGO_STATIC_IP", "203.0.113.7", raising=False)
    r2 = algo_readiness()
    assert r2["live_automated_ready"] is True
    assert all(r2["checklist"].values())


def test_paper_order_allowed_but_still_honours_kill_switch():
    # Paper is permitted even in production without empanelment...
    ok = _FakeSB(kill_switch_active=False, auto_trader_enabled=True)
    d = check_algo_order(supabase=ok, user_id="u", segment="equity", automated=True, live=False)
    assert d.allowed is True
    # ...but the durable kill-switch still blocks even paper.
    killed = _FakeSB(kill_switch_active=True, auto_trader_enabled=True)
    d2 = check_algo_order(supabase=killed, user_id="u", segment="equity", automated=True, live=False)
    assert d2.allowed is False
    assert d2.reason == "kill_switch_active"


def test_profile_read_failure_fails_closed(monkeypatch):
    monkeypatch.setattr(cg.settings, "APP_ENV", "production", raising=False)

    class _Boom:
        def table(self, *a, **k):
            return self

        def select(self, *a, **k):
            return self

        def eq(self, *a, **k):
            return self

        def single(self):
            return self

        def execute(self):
            raise RuntimeError("db down")

    d = check_algo_order(supabase=_Boom(), user_id="u", segment="equity", automated=True, live=True)
    assert d.allowed is False
    assert d.reason == "kill_switch_active"


# ── Bring-your-own-broker data (services/user_broker_data.py) ──────────────

def test_broker_quote_normalizes_kite_shape(monkeypatch):
    from backend.services import user_broker_data as ubd

    class _Kite:
        def get_quote(self, sym, exch):
            return {
                "last_price": 1250.5,
                "ohlc": {"open": 1240, "high": 1260, "low": 1235, "close": 1230},
                "volume": 543210,
                "net_change": 20.5,
            }

    monkeypatch.setattr(ubd, "_build_adapter", lambda uid: ("zerodha", _Kite()))
    q = ubd.quote("u1", "RELIANCE")
    assert q["ltp"] == 1250.5
    assert q["change"] == 20.5
    assert q["change_percent"] == round(20.5 / 1230 * 100, 2)
    assert q["close"] == 1230.0        # kite ohlc.close is PREV close
    assert q["source"] == "broker"


def test_broker_data_locks_when_no_broker(monkeypatch):
    from backend.services import user_broker_data as ubd

    monkeypatch.setattr(ubd, "_build_adapter", lambda uid: (None, None))
    assert ubd.quote("u1", "RELIANCE") is None
    assert ubd.indices("u1") is None
    assert ubd.option_chain("u1", "NIFTY") is None
    assert ubd.historical("u1", "RELIANCE") is None


def test_broker_historical_normalizes_kite(monkeypatch):
    from datetime import date
    from backend.services import user_broker_data as ubd

    class _Kite:
        def historical_data(self, token, frm, to, interval):
            return [
                {"date": date(2026, 7, 10), "open": 100, "high": 110, "low": 95, "close": 108, "volume": 12345},
                {"date": date(2026, 7, 11), "open": 108, "high": 112, "low": 104, "close": 106, "volume": 9876},
            ]

    class _Adapter:
        kite = _Kite()

    monkeypatch.setattr(ubd, "_build_adapter", lambda uid: ("zerodha", _Adapter()))
    monkeypatch.setattr(ubd, "_resolve_kite_token", lambda kite, sym: 738561)
    h = ubd.historical("u1", "RELIANCE", "1d", 30)
    assert h["source"] == "broker"
    assert len(h["data"]) == 2
    assert h["data"][0]["close"] == 108.0


def test_broker_historical_locks_for_non_kite(monkeypatch):
    from backend.services import user_broker_data as ubd

    class _NoKite:
        kite = None

    monkeypatch.setattr(ubd, "_build_adapter", lambda uid: ("upstox", _NoKite()))
    assert ubd.historical("u1", "RELIANCE") is None
