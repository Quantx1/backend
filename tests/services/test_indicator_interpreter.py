"""Indicator Interpreter — pure rules + bias (#3)."""
from backend.services.explain.indicator_interpreter import interpret, bias


def test_interpret_overbought_uptrend():
    m = {"rsi_14": 75, "macd_hist": 0.5, "adx": 30, "di_plus": 25, "di_minus": 10,
         "ema_200": 100, "close": 120, "volume_ratio": 2.0}
    sigs = {n["indicator"]: n["signal"] for n in interpret(m)}
    assert sigs["RSI(14)"] == "overbought"
    assert sigs["MACD"] == "bullish"
    assert sigs["ADX"] == "bullish"
    assert sigs["Trend"] == "bullish"
    assert sigs["Volume"] == "high"


def test_interpret_oversold_downtrend_weak():
    m = {"rsi_14": 25, "macd_hist": -0.5, "adx": 15, "close": 80, "sma_200": 100}
    sigs = {n["indicator"]: n["signal"] for n in interpret(m)}
    assert sigs["RSI(14)"] == "oversold"
    assert sigs["MACD"] == "bearish"
    assert sigs["ADX"] == "neutral"
    assert sigs["Trend"] == "bearish"


def test_bias():
    assert bias([{"signal": "bullish"}, {"signal": "bullish"}, {"signal": "bearish"}]) == "bullish"
    assert bias([{"signal": "bearish"}, {"signal": "overbought"}]) == "bearish"
    assert bias([]) == "mixed"
