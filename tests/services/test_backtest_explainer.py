"""AI Backtesting Assistant — deterministic drivers + suggestions over a
stored backtest summary. Pure: thresholds pinned to defaults, the grounded
narrative monkeypatched (no network / LLM / DB)."""
import backend.services.explain.backtest_explainer as bx
from backend.ai.strategy.evaluation import GateThresholds


STRONG = {
    "symbol": "RELIANCE",
    "total_trades": 84,
    "win_rate": 0.56,
    "total_return_pct": 21.4,
    "max_drawdown_pct": 12.0,
    "sharpe_ratio": 1.1,
    "out_of_sample": {
        "oos_trades": 34,
        "oos_mean_sharpe": 0.82,
        "oos_worst_drawdown_pct": 14.2,
        "oos_consistency": 0.75,
        "holdout_return_pct": 3.2,
        "oos_folds_profitable": 3,
        "n_folds": 4,
    },
}


def _pin_defaults(monkeypatch):
    """Pin the gate bars to the code defaults so env config can't skew tests."""
    monkeypatch.setattr(bx, "_thresholds", lambda: GateThresholds())


def test_strong_metrics_pass_gate_with_clear_drivers(monkeypatch):
    _pin_defaults(monkeypatch)
    out = bx.explain_backtest(STRONG, None)
    txt = " ".join(out["drivers"])
    assert "PASSES the out-of-sample quality gate" in txt
    assert "Out-of-sample Sharpe 0.82 — clears the 0.5 gate." in txt
    assert "Max drawdown 14.2% — within the 35% ceiling." in txt
    assert "34 out-of-sample trades — clears the 20-trade minimum." in txt
    assert "Profitable in 75% of walk-forward windows (3/4)" in txt
    assert "Holdout window +3.2%" in txt
    assert "In-sample: 56% win rate over 84 trades, +21.4% total return." in txt
    # Nothing failed -> nothing to fix, and no LLM was asked for.
    assert out["suggestions"] == []
    assert out["narrative"] is None


def test_failing_drawdown_yields_fail_driver_and_matching_suggestion(monkeypatch):
    _pin_defaults(monkeypatch)
    m = {**STRONG, "out_of_sample": {**STRONG["out_of_sample"],
                                     "oos_worst_drawdown_pct": 41.0}}
    out = bx.explain_backtest(m, {"universe": "single", "stop_loss_pct": 5.0})
    txt = " ".join(out["drivers"])
    assert "FAILS the quality gate (1 issue)" in txt
    assert "Max drawdown 41.0% — FAILS the 35% gate." in txt
    assert any("Tighten stops" in s for s in out["suggestions"])
    # Stop already in the DSL -> we don't tell them to add one.
    assert not any("Add a hard stop loss" in s for s in out["suggestions"])


def test_too_few_trades_driver_and_universe_suggestion(monkeypatch):
    _pin_defaults(monkeypatch)
    m = {**STRONG, "out_of_sample": {**STRONG["out_of_sample"], "oos_trades": 12}}
    out = bx.explain_backtest(m, {"universe": "single"})
    txt = " ".join(out["drivers"])
    assert "Only 12 out-of-sample trades — below the 20-trade minimum (low confidence)." in txt
    assert any("Widen the universe" in s for s in out["suggestions"])


def test_no_oos_block_is_called_out(monkeypatch):
    _pin_defaults(monkeypatch)
    out = bx.explain_backtest({"win_rate": 0.6, "total_trades": 12, "sharpe_ratio": 1.4})
    txt = " ".join(out["drivers"])
    assert "FAILS the quality gate" in txt
    assert "No walk-forward out-of-sample block" in txt
    assert any("walk-forward" in s for s in out["suggestions"])


def test_empty_metrics_honest_empty():
    out = bx.explain_backtest({}, None, use_llm=True)
    assert out == {"drivers": [], "suggestions": [], "narrative": None}


def test_narrative_grounded_with_hashed_daily_cache_key(monkeypatch):
    _pin_defaults(monkeypatch)
    seen = {}

    def _fake_grounded(facts, question, *, cache_key=None, **_kw):
        seen["cache_key"] = cache_key
        seen["facts"] = facts
        return "grounded read"

    monkeypatch.setattr("backend.ai.agents.grounded.grounded_reason", _fake_grounded)
    out = bx.explain_backtest(STRONG, {"universe": "nifty50"}, use_llm=True, user_id="u1")
    assert out["narrative"] == "grounded read"
    # btexplain:<16-char sha256 prefix>:<iso date>
    prefix, h, day = seen["cache_key"].split(":")
    assert prefix == "btexplain"
    assert len(h) == 16
    assert len(day) == 10
    # Narrative is grounded in the same deterministic facts the gate scored.
    assert seen["facts"]["gate"]["passed"] is True
    assert seen["facts"]["backtest"]["out_of_sample"]["oos_mean_sharpe"] == 0.82
