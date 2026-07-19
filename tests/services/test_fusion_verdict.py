"""Fusion Verdict — pure fusion core + assembly (monkeypatched readers)."""
import backend.services.scanners.fusion_verdict as fv
from backend.services.scanners.fusion_verdict import build_verdict, fuse


def _f(key, score, weight=0.2):
    return {"key": key, "label": key, "weight": weight, "score": score, "lean": "x"}


# ── pure fuse() ────────────────────────────────────────────────────────────


def test_fuse_insufficient_data_under_two_factors():
    out = fuse([_f("a", 0.8)], gated=False)
    assert out["verdict"] == "Insufficient data"
    assert out["composite"] is None


def test_fuse_strong_setup_bullish():
    out = fuse([_f("a", 0.9), _f("b", 0.8), _f("c", 0.7)], gated=False)
    assert out["composite"] >= 72
    assert out["verdict"] == "Strong setup"
    assert out["direction"] == "bullish"


def test_fuse_avoid_bearish():
    out = fuse([_f("a", -0.9), _f("b", -0.8)], gated=False)
    assert out["composite"] < 30
    assert out["verdict"] == "Avoid"
    assert out["direction"] == "bearish"


def test_fuse_mixed_neutral_midband():
    out = fuse([_f("a", 0.05), _f("b", -0.05)], gated=False)
    assert 42 <= out["composite"] <= 58
    assert out["verdict"] == "Mixed"
    assert out["direction"] == "neutral"


def test_fuse_event_gate_overrides_bullish():
    out = fuse([_f("a", 0.9), _f("b", 0.8)], gated=True)
    assert out["verdict"] == "Hold off — event risk"
    assert out["gated"] is True
    # composite still computed (transparency) even though gated
    assert out["composite"] >= 72


def test_fuse_ignores_none_scored_factors_in_weighting():
    # a None-scored factor (e.g. event_risk marker) must not dilute the average
    out = fuse([_f("a", 0.9), _f("b", 0.9), _f("evt", None, weight=0.0)], gated=False)
    assert out["composite"] >= 90


# ── build_verdict() assembly ───────────────────────────────────────────────


def _patch_full(monkeypatch, *, gated=False):
    monkeypatch.setattr(fv, "_smart_money_score", lambda s: {"score": 0.6, "detail": "bullish OI", "lean": "bullish"})
    monkeypatch.setattr(fv, "_volume_score", lambda s: {"score": 0.5, "detail": "accumulation"})
    monkeypatch.setattr(fv, "_regime_score", lambda: {"name": "bull", "score": 0.7, "detail": "regime: bull"})

    def _scores(sym):
        return {"symbol": sym, "scores": [
            {"key": "alpha", "pct": 95.0, "note": "#5 of 100"},
            {"key": "momentum", "pct": 80.0},
            {"key": "trend", "pct": 70.0},
            {"key": "mood", "value": 0.4, "note": "12 headlines"},
        ], "composite": 80.0}
    monkeypatch.setattr(fv, "scores", _scores, raising=False)
    import backend.services.scanners.stock_scores as ss
    monkeypatch.setattr(ss, "scores", _scores)
    monkeypatch.setattr(
        "backend.services.scanners.event_risk.symbols_in_event_window",
        lambda syms, **k: ({"TCS"} if gated else set()),
    )


def test_build_verdict_full_bullish(monkeypatch):
    _patch_full(monkeypatch)
    out = build_verdict("TCS")
    assert out["symbol"] == "TCS"
    keys = {f["key"] for f in out["factors"]}
    assert {"alpha", "trend", "mood", "smart_money", "volume", "regime"} <= keys
    assert out["composite"] is not None and out["composite"] >= 58
    assert out["direction"] == "bullish"
    assert out["gated"] is False


def test_build_verdict_event_gated(monkeypatch):
    _patch_full(monkeypatch, gated=True)
    out = build_verdict("TCS")
    assert out["gated"] is True
    assert out["verdict"] == "Hold off — event risk"
    assert any(f["key"] == "event_risk" for f in out["factors"])


def test_build_verdict_empty_symbol():
    out = build_verdict("")
    assert out["verdict"] == "Insufficient data"
    assert out["factors"] == []


def test_build_verdict_honest_empty_when_no_sources(monkeypatch):
    monkeypatch.setattr(fv, "_smart_money_score", lambda s: None)
    monkeypatch.setattr(fv, "_volume_score", lambda s: None)
    monkeypatch.setattr(fv, "_regime_score", lambda: None)
    import backend.services.scanners.stock_scores as ss
    monkeypatch.setattr(ss, "scores", lambda sym: {"symbol": sym, "scores": [], "composite": None})
    monkeypatch.setattr(
        "backend.services.scanners.event_risk.symbols_in_event_window", lambda syms, **k: set()
    )
    out = build_verdict("ZZZ")
    assert out["verdict"] == "Insufficient data"
    assert out["composite"] is None
