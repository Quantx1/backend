"""AI Options Copilot — fact-assembly + strategy-merge shaping.

Pure: a fake snapshot + a monkeypatched suggester. No network, no DB, no LLM.
Verifies (a) facts mirror the real snapshot numbers, (b) the descriptive
VIX-regime note is dropped from the trade list, (c) top-N capping, (d) the
deterministic strategies are ALWAYS returned, (e) honest-empty on no chain,
(f) the LLM narrative is gated on use_llm.
"""
from backend.services.fno_scanner import options_copilot as oc


class _FakeSnap:
    """Mimics the IndexSnapshot fields options_copilot reads."""
    symbol = "NIFTY"
    spot = 24010.0
    pcr_oi = 1.42
    pcr_tag = "extreme_bullish"
    max_pain = 24000.0
    max_pain_distance_pct = 0.04
    pull_to_max_pain_signal = False
    iv_atm = 0.1234
    days_to_expiry = 4


class _FakeStrat:
    def __init__(self, name, bias="neutral", confidence="high"):
        self.name = name
        self.bias = bias
        self.confidence = confidence

    def to_dict(self):
        return {"name": self.name, "bias": self.bias, "confidence": self.confidence}


def _patch(monkeypatch, suggestions, snap=_FakeSnap()):
    monkeypatch.setattr(oc, "fetch_index_snapshot", lambda s: snap)
    monkeypatch.setattr(oc, "_india_vix", lambda: 16.5)
    monkeypatch.setattr(oc, "_iv_rank", lambda s, iv: 62.0)
    monkeypatch.setattr(oc, "suggest_strategies", lambda snap, vix=None, iv_rank=None: suggestions)


def test_facts_mirror_snapshot(monkeypatch):
    _patch(monkeypatch, [_FakeStrat("Bull Call Spread", "bullish")])
    res = oc.best_trade("nifty", use_llm=False)
    f = res["facts"]
    assert res["symbol"] == "NIFTY"
    assert f["spot"] == 24010.0
    assert f["pcr_oi"] == 1.42
    assert f["pcr_tag"] == "extreme_bullish"
    assert f["bias"] == "bullish"          # mapped from pcr_tag
    assert f["max_pain"] == 24000.0
    assert f["atm_iv"] == 0.1234
    assert f["iv_rank"] == 62.0
    assert f["india_vix"] == 16.5
    assert f["vix_regime"] == "normal"     # 16.5 -> normal band
    assert f["days_to_expiry"] == 4


def test_regime_note_dropped_and_top_n(monkeypatch):
    suggestions = [
        _FakeStrat("VIX Regime: normal"),           # descriptive — must be dropped
        _FakeStrat("Bull Call Spread", "bullish"),
        _FakeStrat("Bull Put Spread", "bullish"),
        _FakeStrat("Cash-Secured Put", "bullish"),
        _FakeStrat("Synthetic Long", "bullish"),     # 4th trade — must be trimmed
    ]
    _patch(monkeypatch, suggestions)
    res = oc.best_trade("NIFTY", use_llm=False)
    names = [s["name"] for s in res["strategies"]]
    assert "VIX Regime: normal" not in names
    assert names == ["Bull Call Spread", "Bull Put Spread", "Cash-Secured Put"]
    # best trade is the head of the deterministic list
    assert res["strategies"][0]["name"] == "Bull Call Spread"


def test_strategies_always_returned_without_llm(monkeypatch):
    _patch(monkeypatch, [_FakeStrat("Iron Condor")])
    res = oc.best_trade("NIFTY", use_llm=False)
    assert res["strategies"]                # deterministic, 0 tokens
    assert res["narrative"] is None         # no LLM unless asked


def test_honest_empty_when_no_chain(monkeypatch):
    monkeypatch.setattr(oc, "fetch_index_snapshot", lambda s: None)
    res = oc.best_trade("NIFTY", use_llm=True)
    assert res == {"symbol": "NIFTY", "facts": None, "strategies": [], "narrative": None}


def test_narrative_uses_grounded_reason_only_when_use_llm(monkeypatch):
    _patch(monkeypatch, [_FakeStrat("Bull Call Spread", "bullish")])
    called = {}

    def _fake_grounded(facts, question, *, cache_key=None, role="responder", **kw):
        called["facts"] = facts
        called["cache_key"] = cache_key
        return "Bull Call Spread is the best trade: defined-risk bullish."

    # patch the symbol options_copilot imports lazily
    import backend.ai.agents.grounded as grounded_mod
    monkeypatch.setattr(grounded_mod, "grounded_reason", _fake_grounded)

    res = oc.best_trade("NIFTY", use_llm=True)
    assert res["narrative"].startswith("Bull Call Spread is the best trade")
    # the strategies list is forwarded into the grounded facts
    assert "strategies" in called["facts"]
    assert called["cache_key"].startswith("optcopilot:NIFTY:")
