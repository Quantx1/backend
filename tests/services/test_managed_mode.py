"""Dual-mode (managed/pro) backend — ui_mode pref validation + managed overview.

`_health` / `_activity_lines` are exercised pure; `build_overview` with the
source readers monkeypatched (no DB, no network). The ui-preferences PUT
handler is called directly with a fake Supabase client.
"""
import backend.core.tiers as tiers
import backend.services.portfolio.managed_overview as mo
import backend.services.portfolio.user_risk as ur
from backend.api import user_routes
from backend.services.portfolio.managed_overview import _activity_lines, _health, build_overview


# ── health score ─────────────────────────────────────────────────────────


def test_health_clean_is_100():
    h = _health({"warnings": [], "ok": True})
    assert h["score"] == 100
    assert h["label"] == "Healthy"
    assert h["components"] == []


def test_health_deducts_by_severity():
    risk = {"warnings": [
        {"key": "day_loss", "severity": "high", "message": "x"},
        {"key": "single_name:TCS", "severity": "medium", "message": "y"},
    ]}
    h = _health(risk)
    assert h["score"] == 75  # 100 - 15 - 10
    assert {c["key"] for c in h["components"]} == {"day_loss", "single_name:TCS"}
    # every component carries its impact so the UI can explain the score
    assert all(c["impact"] < 0 for c in h["components"])


def test_health_warning_deductions_capped_at_45():
    warnings = [{"key": f"w{i}", "severity": "high", "message": "m"} for i in range(5)]
    h = _health({"warnings": warnings})  # -75 uncapped
    assert h["score"] == 55
    assert any(c["key"] == "warning_cap" for c in h["components"])


def test_health_day_loss_approach_deduction():
    # limit = 2% of 1,00,000 = 2,000; 80% = 1,600 → -1,700 deducts 10
    risk = {"warnings": [], "day_pnl": -1700.0,
            "daily_loss_limit_pct": 2.0, "capital": 100000.0}
    h = _health(risk)
    assert h["score"] == 90
    assert any(c["key"] == "day_loss_approach" for c in h["components"])


def test_health_no_double_count_when_breach_already_flagged():
    risk = {"warnings": [{"key": "day_loss", "severity": "high", "message": "m"}],
            "day_pnl": -2500.0, "daily_loss_limit_pct": 2.0, "capital": 100000.0}
    h = _health(risk)
    assert h["score"] == 85
    assert not any(c["key"] == "day_loss_approach" for c in h["components"])


def test_health_labels_by_band():
    assert _health({"warnings": []})["label"] == "Healthy"
    two_high = [{"key": "a", "severity": "high", "message": ""},
                {"key": "b", "severity": "high", "message": ""}]
    assert _health({"warnings": two_high})["label"] == "Watch"      # 70
    three_high = two_high + [{"key": "c", "severity": "high", "message": ""}]
    assert _health({"warnings": three_high})["label"] == "At risk"  # 55


# ── activity lines ───────────────────────────────────────────────────────


def test_activity_lines_closed_and_open_trades():
    trades = [
        {"symbol": "TCS", "status": "closed", "pnl_percent": 2.13,
         "net_pnl": 1234.5, "direction": "LONG", "quantity": 10},
        {"symbol": "RELIANCE", "status": "open", "direction": "LONG",
         "quantity": 12, "entry_price": 2456.0},
        {"symbol": "INFY", "status": "open", "direction": "SHORT",
         "quantity": 5, "entry_price": 1500.0},
    ]
    lines = _activity_lines(trades, {})
    assert lines[0].startswith("Closed TCS +2.1%")
    assert "₹1,234" in lines[0] or "₹1,235" in lines[0]
    assert lines[1] == "Bought 12 RELIANCE @ ₹2,456"
    assert lines[2].startswith("Sold short 5 INFY")


def test_activity_kill_switch_line_first():
    lines = _activity_lines([], {"kill_switch_active": True})
    assert len(lines) == 1
    assert "paused" in lines[0]


def test_activity_empty_is_empty():
    assert _activity_lines([], {}) == []


# ── build_overview composition ───────────────────────────────────────────


_RISK_OK = {
    "warnings": [], "ok": True, "day_pnl": -120.0, "capital": 50000.0,
    "positions_value": 10000.0, "positions_count": 2,
    "risk_profile": "moderate", "daily_loss_limit_pct": 3.0,
}


def _patch_sources(monkeypatch, *, tier=None):
    monkeypatch.setattr(ur, "risk_status", lambda uid: dict(_RISK_OK))
    monkeypatch.setattr(mo, "_sb", lambda: object())
    monkeypatch.setattr(mo, "_profile_row", lambda sb, uid: {
        "capital": 50000, "total_pnl": 1500, "total_trades": 10,
        "winning_trades": 6, "auto_trader_enabled": True,
        "kill_switch_active": False,
        "auto_trader_last_run_at": "2026-06-11T10:20:00Z",
    })
    monkeypatch.setattr(mo, "_unrealized_pnl", lambda sb, uid: 320.5)
    monkeypatch.setattr(mo, "_live_trades", lambda sb, uid, days=7, mode="live": [
        {"symbol": "INFY", "status": "closed", "pnl_percent": 1.2,
         "net_pnl": 300.0, "direction": "LONG", "quantity": 5},
    ])
    monkeypatch.setattr(mo, "_latest_regime", lambda sb: {
        "name": "sideways", "prob_bull": 0.2, "prob_sideways": 0.7,
        "prob_bear": 0.1, "as_of": "2026-06-11",
    })
    monkeypatch.setattr(mo, "_latest_drawdown", lambda sb, uid: {
        "current_pct": -3.2, "as_of": "2026-06-11",
    })
    if tier is not None:
        monkeypatch.setattr(
            tiers, "resolve_user_tier",
            lambda uid, **kw: tiers.UserTier(user_id=uid, tier=tier),
        )


def test_build_overview_elite_full_payload(monkeypatch):
    _patch_sources(monkeypatch, tier=tiers.Tier.ELITE)
    out = build_overview("u1")
    assert out["health"]["score"] == 100
    assert out["pnl"]["capital"] == 50000
    assert out["pnl"]["win_rate"] == 60.0
    assert out["pnl"]["unrealized_pnl"] == 320.5
    assert out["risk"]["level"] == "moderate"
    assert out["autopilot"]["available"] is True
    assert out["autopilot"]["mode"] == "live"
    assert out["autopilot"]["enabled"] is True
    assert out["autopilot"]["trades_7d"] == 1
    assert out["autopilot"]["activity"][0].startswith("Closed INFY")
    assert out["regime"]["name"] == "sideways"
    assert out["drawdown"]["current_pct"] == -3.2


def test_build_overview_free_tier_is_paper_and_unavailable(monkeypatch):
    # Pricing v2: auto_trader min tier is PRO; Free runs paper-only.
    _patch_sources(monkeypatch, tier=tiers.Tier.FREE)
    out = build_overview("u1")
    assert out["autopilot"]["available"] is False
    assert out["autopilot"]["mode"] == "paper"
    # config facts still reported honestly (tier gates the feature, not the data)
    assert out["autopilot"]["enabled"] is True


def test_build_overview_pro_tier_available_live(monkeypatch):
    # Pricing v2: Pro gets AutoPilot Lite (live).
    _patch_sources(monkeypatch, tier=tiers.Tier.PRO)
    out = build_overview("u1")
    assert out["autopilot"]["available"] is True
    assert out["autopilot"]["mode"] == "live"


def test_build_overview_pro_paper_opt_in_stays_paper(monkeypatch):
    _patch_sources(monkeypatch, tier=tiers.Tier.PRO)
    mo_profile = {
        "capital": 50000, "total_pnl": 1500, "total_trades": 10,
        "winning_trades": 6, "auto_trader_enabled": True,
        "kill_switch_active": False,
        "auto_trader_last_run_at": "2026-06-11T10:20:00Z",
        "auto_trader_config": {"mode": "paper"},
    }
    monkeypatch.setattr(mo, "_profile_row", lambda sb, uid: mo_profile)
    out = build_overview("u1")
    assert out["autopilot"]["available"] is True
    assert out["autopilot"]["mode"] == "paper"


def test_build_overview_db_down_is_honest_empty(monkeypatch):
    monkeypatch.setattr(ur, "risk_status", lambda uid: {"warnings": [], "ok": True})

    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(mo, "_sb", _boom)
    monkeypatch.setattr(
        tiers, "resolve_user_tier",
        lambda uid, **kw: (_ for _ in ()).throw(RuntimeError("down")),
    )
    out = build_overview("u1")
    assert out["health"]["score"] == 100  # no fabricated risk
    assert out["pnl"]["capital"] is None
    assert out["pnl"]["win_rate"] is None
    assert out["regime"] is None
    assert out["drawdown"] is None
    assert out["autopilot"]["available"] is False
    assert out["autopilot"]["activity"] == []


# ── ui_mode preference (PUT /api/user/ui-preferences) ────────────────────


class _FakeQ:
    def __init__(self, rec):
        self.rec = rec

    def update(self, payload):
        self.rec["update"] = payload
        return self

    def eq(self, col, val):
        return self

    def execute(self):
        return None


class _FakeSB:
    def __init__(self):
        self.rec = {}

    def table(self, name):
        return _FakeQ(self.rec)


class _User:
    id = "u1"


async def test_ui_mode_valid_value_persisted(monkeypatch):
    sb = _FakeSB()
    monkeypatch.setattr(user_routes, "_get_supabase_admin", lambda: sb)
    res = await user_routes.update_ui_preferences(
        {"ui_preferences": {"ui_mode": "managed"}}, user=_User())
    assert res["ui_preferences"]["ui_mode"] == "managed"
    assert sb.rec["update"]["ui_preferences"]["ui_mode"] == "managed"


async def test_ui_mode_invalid_value_dropped(monkeypatch):
    sb = _FakeSB()
    monkeypatch.setattr(user_routes, "_get_supabase_admin", lambda: sb)
    res = await user_routes.update_ui_preferences(
        {"ui_preferences": {"ui_mode": "yolo"}}, user=_User())
    assert "ui_mode" not in res["ui_preferences"]


async def test_ui_mode_wrong_type_dropped(monkeypatch):
    sb = _FakeSB()
    monkeypatch.setattr(user_routes, "_get_supabase_admin", lambda: sb)
    res = await user_routes.update_ui_preferences(
        {"ui_preferences": {"ui_mode": {"nested": True}}}, user=_User())
    assert "ui_mode" not in res["ui_preferences"]


async def test_ui_mode_coexists_with_watchlist_pins(monkeypatch):
    sb = _FakeSB()
    monkeypatch.setattr(user_routes, "_get_supabase_admin", lambda: sb)
    res = await user_routes.update_ui_preferences(
        {"ui_preferences": {
            "ui_mode": "pro",
            "watchlist_preset_pins": {"TCS": "pct5"},
            "hax": {"evil": 1},
        }},
        user=_User())
    assert res["ui_preferences"]["ui_mode"] == "pro"
    assert res["ui_preferences"]["watchlist_preset_pins"] == {"TCS": "pct5"}
    assert "hax" not in res["ui_preferences"]
