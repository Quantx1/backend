"""User-level Risk Manager — pure ``check_risk`` tests (no DB, no network).

Warn-only invariant: check_risk never raises and never returns anything a
caller could use to block — only {warnings, ok}.
"""
from backend.services.portfolio.user_risk import check_risk, effective_day_loss_limit_pct


def _keys(res):
    return [w["key"] for w in res["warnings"]]


# ── day-loss ──────────────────────────────────────────────────────────────


def test_day_loss_breach_by_profile_default_conservative():
    """Conservative default = 2% of capital. -2,000 on 1,00,000 == -2% → warn."""
    res = check_risk({"risk_profile": "conservative"}, -2000.0, 100000.0, [], None)
    assert "day_loss" in _keys(res)
    assert res["ok"] is False
    w = next(w for w in res["warnings"] if w["key"] == "day_loss")
    assert w["severity"] == "high"
    assert "2%" in w["message"]


def test_day_loss_not_breached_for_moderate_default():
    """Moderate default = 3% → -2,000 on 1,00,000 (-2%) is inside the limit."""
    res = check_risk({"risk_profile": "moderate"}, -2000.0, 100000.0, [], None)
    assert _keys(res) == []
    assert res["ok"] is True


def test_day_loss_explicit_override_beats_profile_default():
    """Explicit 1% override on an aggressive profile (default 5%) → -1,500 warns."""
    profile = {"risk_profile": "aggressive", "daily_loss_limit_pct": 1.0}
    assert effective_day_loss_limit_pct(profile) == 1.0
    res = check_risk(profile, -1500.0, 100000.0, [], None)
    assert "day_loss" in _keys(res)
    # without the override the same loss would be clean (5% = 5,000)
    res2 = check_risk({"risk_profile": "aggressive"}, -1500.0, 100000.0, [], None)
    assert _keys(res2) == []


# ── single-name concentration ─────────────────────────────────────────────


def test_single_name_25_pct_warns():
    positions = [{"symbol": "RELIANCE", "sector": "Energy", "value": 25000.0}]
    res = check_risk({"risk_profile": "moderate"}, 0.0, 100000.0, positions, None)
    assert _keys(res) == ["single_name:RELIANCE"]
    w = res["warnings"][0]
    assert w["severity"] == "medium"
    assert "25.0%" in w["message"]
    assert res["ok"] is False


def test_proposed_order_pushes_symbol_over_cap():
    """15% existing + 10% proposed in the same name → 25% warn, flagged
    as including the order."""
    positions = [{"symbol": "TCS", "value": 15000.0}]
    proposed = {"symbol": "TCS", "value": 10000.0}
    res = check_risk({"risk_profile": "moderate"}, 0.0, 100000.0, positions, proposed)
    assert "single_name:TCS" in _keys(res)
    w = next(w for w in res["warnings"] if w["key"] == "single_name:TCS")
    assert "including this order" in w["message"]


# ── sector concentration ──────────────────────────────────────────────────


def test_sector_45_pct_warns():
    """Three 15% positions in one sector → 45% sector warn, no single-name."""
    positions = [
        {"symbol": "HDFCBANK", "sector": "Financial Services", "value": 15000.0},
        {"symbol": "ICICIBANK", "sector": "Financial Services", "value": 15000.0},
        {"symbol": "SBIN", "sector": "Financial Services", "value": 15000.0},
    ]
    res = check_risk({"risk_profile": "moderate"}, 0.0, 100000.0, positions, None)
    assert _keys(res) == ["sector:Financial Services"]
    w = res["warnings"][0]
    assert w["severity"] == "medium"
    assert "45.0%" in w["message"]


# ── total exposure ────────────────────────────────────────────────────────


def test_total_exposure_over_100_pct_warns():
    """Six different names, different sectors, 18% each = 108% deployed."""
    positions = [
        {"symbol": f"SYM{i}", "sector": f"Sector {i}", "value": 18000.0}
        for i in range(6)
    ]
    res = check_risk({"risk_profile": "moderate"}, 0.0, 100000.0, positions, None)
    assert _keys(res) == ["total_exposure"]
    w = res["warnings"][0]
    assert w["severity"] == "high"
    assert "108.0%" in w["message"]


# ── clean + honest-empty ──────────────────────────────────────────────────


def test_clean_portfolio_is_ok():
    positions = [
        {"symbol": "INFY", "sector": "IT", "value": 10000.0},
        {"symbol": "TATASTEEL", "sector": "Metals", "value": 12000.0},
    ]
    res = check_risk({"risk_profile": "moderate"}, 500.0, 100000.0, positions, None)
    assert res == {"warnings": [], "ok": True}


def test_honest_empty_when_capital_missing():
    """No capital → no warnings at all, even with a big loss on the books."""
    assert check_risk({"risk_profile": "conservative"}, -99999.0, 0.0, [], None) == {
        "warnings": [], "ok": True,
    }
    assert check_risk({}, -99999.0, None, None, None) == {"warnings": [], "ok": True}


def test_honest_empty_when_profile_unknown_skips_day_loss_only():
    """No risk profile + no explicit limit → day-loss check is skipped, but
    exposure math still runs."""
    positions = [{"symbol": "RELIANCE", "value": 30000.0}]
    res = check_risk({}, -50000.0, 100000.0, positions, None)
    assert _keys(res) == ["single_name:RELIANCE"]


# ── loader honest-empty (DB unavailable) ──────────────────────────────────


def test_risk_status_honest_empty_when_db_unavailable(monkeypatch):
    """risk_status never raises: with Supabase down it returns the empty,
    ok=True shape with zeroed raw numbers."""
    import backend.services.portfolio.user_risk as ur

    def _boom():
        raise RuntimeError("no supabase in tests")

    monkeypatch.setattr(ur, "_sb", _boom)
    res = ur.risk_status("user-123")
    assert res["ok"] is True
    assert res["warnings"] == []
    assert res["day_pnl"] == 0.0
    assert res["capital"] == 0.0
    assert res["positions_count"] == 0
