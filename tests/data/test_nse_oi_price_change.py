"""Regression: OI build-up scanners need a real price-direction leg.

The F&O bhavcopy carries each contract's OPEN/CLOSE, but get_participant_oi used
to hardcode change_pct=0.0 -> long/short build-up, unwinding & short-covering
buckets were permanently empty. _front_month_price_change supplies the real
front-month day move.
"""
import pandas as pd

from backend.data.screener.nse_data import _front_month_price_change, _coerce_float


def test_front_month_uses_prev_close_then_open_fallback():
    df = pd.DataFrame([
        # RELIANCE: two expiries — front month (earliest) is +5% via prev_close
        {"SYMBOL": "RELIANCE", "EXPIRY_DT": "26-Jun-2026", "OPEN": 100, "CLOSE": 105, "PREV_CLOSE": 100},
        {"SYMBOL": "RELIANCE", "EXPIRY_DT": "31-Jul-2026", "OPEN": 101, "CLOSE": 110, "PREV_CLOSE": 101},
        # TCS: no prev_close cell -> falls back to OPEN; -5% move
        {"SYMBOL": "TCS", "EXPIRY_DT": "26-Jun-2026", "OPEN": 200, "CLOSE": 190},
    ])
    out = _front_month_price_change(df)
    assert out["RELIANCE"] == 5.0     # front-month, via prev_close
    assert out["TCS"] == -5.0         # via open fallback (prev_close NaN)


def test_front_month_handles_empty_and_missing_columns():
    assert _front_month_price_change(pd.DataFrame()) == {}
    # missing OPEN/CLOSE -> symbol simply omitted, no crash
    df = pd.DataFrame([{"SYMBOL": "X", "EXPIRY_DT": "26-Jun-2026"}])
    assert _front_month_price_change(df) == {}


def test_coerce_float_rejects_nan():
    assert _coerce_float(float("nan")) is None
    assert _coerce_float("1,234.5") == 1234.5
    assert _coerce_float("") is None
