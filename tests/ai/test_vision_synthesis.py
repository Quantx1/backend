from __future__ import annotations

from backend.ai.vision.analyzer import VisionAnalysis
from backend.ai.strategy.studio import synthesize_prompt_from_vision


def _mk(**kw):
    base = dict(symbol="RELIANCE", available=True)
    base.update(kw)
    return VisionAnalysis(**base)


def test_bullish_returns_long_prompt():
    p = synthesize_prompt_from_vision(_mk(setup="bullish continuation", trend="uptrend"),
                                      symbol="RELIANCE", timeframe="1d")
    assert p and "RELIANCE" in p and "long" in p.lower()


def test_bearish_returns_none():
    assert synthesize_prompt_from_vision(_mk(setup="bearish reversal", trend="downtrend"),
                                         symbol="RELIANCE") is None


def test_no_edge_returns_none():
    assert synthesize_prompt_from_vision(_mk(setup="no edge", trend="unclear"),
                                         symbol="RELIANCE") is None


def test_range_returns_mean_reversion():
    p = synthesize_prompt_from_vision(_mk(setup="range-bound", trend="range"),
                                      symbol="TCS")
    assert p and "TCS" in p and "mean-reversion" in p.lower()


def test_unavailable_returns_none():
    assert synthesize_prompt_from_vision(VisionAnalysis(symbol="X", available=False),
                                         symbol="X") is None


def test_missing_symbol_returns_none():
    assert synthesize_prompt_from_vision(_mk(setup="bullish continuation"),
                                         symbol="") is None


def test_synthesized_prompt_is_brand_safe():
    p = synthesize_prompt_from_vision(_mk(setup="bullish continuation", trend="uptrend"),
                                      symbol="INFY")
    banned = ["tradingview", "pine", "luxalgo", "tft", "qlib", "finbert", "hmm", "lightgbm"]
    assert p and not any(b in p.lower() for b in banned)
