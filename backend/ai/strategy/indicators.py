"""
Closed-set technical indicator computation for the Strategy DSL.

Each name in ``INDICATOR_REGISTRY`` (see ``dsl.py``) maps to a pure
function that takes the recent price history and returns a single
float value AT THE CURRENT BAR. The interpreter calls these per-bar
to evaluate strategy Conditions.

Why a registry instead of dynamic ``getattr`` lookups:
  - The DSL never receives arbitrary strings — every indicator name
    is validated against the registry before reaching here.
  - We avoid the ``ta`` library's import-time side effects in hot paths.
  - Lets us hand-roll candle patterns + custom things (VWAP intraday
    reset) that ``ta`` doesn't have.

Inputs: a pandas DataFrame with required OHLCV columns:
  open · high · low · close · volume
all lowercase, datetime-indexed, oldest-to-newest. The DataFrame must
have at least ``MIN_LOOKBACK = 200`` rows for the long EMAs to settle.
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np
import pandas as pd


MIN_LOOKBACK = 200


# ─────────────────────────────────────────────────────────────────────
# Primitive helpers
# ─────────────────────────────────────────────────────────────────────


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window, min_periods=window).mean()


def _rsi(s: pd.Series, window: int) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return _true_range(df).ewm(alpha=1 / window, adjust=False).mean()


def _adx_components(df: pd.DataFrame, window: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX + DI+ + DI-. Caller picks which to return.

    Aaryansinha's repo exposes DI+ and DI- separately as ML features —
    important for directional strength + trend confirmation patterns.
    """
    h, low, _ = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _true_range(df).ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / tr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1 / window, adjust=False).mean()
    return adx, plus_di, minus_di


def _adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return _adx_components(df, window)[0]


def _di_plus(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return _adx_components(df, window)[1]


def _di_minus(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return _adx_components(df, window)[2]


def _macd(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Classic 12/26/9."""
    macd = _ema(s, 12) - _ema(s, 26)
    signal = _ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def _bbands(s: pd.Series, window: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = _sma(s, window)
    sd = s.rolling(window=window, min_periods=window).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def _stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> tuple[pd.Series, pd.Series]:
    h_max = df["high"].rolling(window=k_window, min_periods=k_window).max()
    l_min = df["low"].rolling(window=k_window, min_periods=k_window).min()
    k = 100 * (df["close"] - l_min) / (h_max - l_min).replace(0, np.nan)
    d = k.rolling(window=d_window, min_periods=d_window).mean()
    return k, d


def _williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h_max = df["high"].rolling(window=window, min_periods=window).max()
    l_min = df["low"].rolling(window=window, min_periods=window).min()
    return -100 * (h_max - df["close"]) / (h_max - l_min).replace(0, np.nan)


def _mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(window=window, min_periods=window).sum()
    neg_sum = neg.rolling(window=window, min_periods=window).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def _cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(window=window, min_periods=window).mean()
    md = (tp - sma).abs().rolling(window=window, min_periods=window).mean()
    return (tp - sma) / (0.015 * md.replace(0, np.nan))


def _vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP since first bar (use daily reset for intraday upstream)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_pv = (tp * df["volume"]).cumsum()
    cum_v = df["volume"].cumsum().replace(0, np.nan)
    return cum_pv / cum_v


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


# ─────────────────────────────────────────────────────────────────────
# PR-FEATURES additions (parity with aaryansinha16's macro feature list)
# ─────────────────────────────────────────────────────────────────────


def _roc(s: pd.Series, window: int) -> pd.Series:
    """Rate of Change — % change over ``window`` bars.
    ROC(10) = (close_now / close_10_ago - 1) * 100"""
    return (s / s.shift(window) - 1) * 100


def _stoch_rsi(s: pd.Series, rsi_window: int = 14, stoch_window: int = 14,
               d_window: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI — Stochastic applied to RSI values, not raw price.
    Returns (stoch_rsi_k, stoch_rsi_d). Range [0, 100] like Stoch.
    Use case: more sensitive than plain RSI for overbought/oversold."""
    rsi_series = _rsi(s, rsi_window)
    rsi_max = rsi_series.rolling(window=stoch_window, min_periods=stoch_window).max()
    rsi_min = rsi_series.rolling(window=stoch_window, min_periods=stoch_window).min()
    k = 100 * (rsi_series - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    d = k.rolling(window=d_window, min_periods=d_window).mean()
    return k, d


def _realized_volatility(s: pd.Series, window: int = 20) -> pd.Series:
    """Annualised realised volatility — std of log returns × sqrt(252).
    Aaryansinha uses both 20-bar and 60-bar versions as features."""
    log_ret = np.log(s / s.shift(1))
    return log_ret.rolling(window=window, min_periods=window).std() * np.sqrt(252) * 100


def _volatility_regime(s: pd.Series, window: int = 20, lookback: int = 252) -> pd.Series:
    """Classify volatility regime: 0=low, 1=normal, 2=high based on
    current vol vs trailing-year percentile. Returns int-valued series."""
    vol = _realized_volatility(s, window)
    p25 = vol.rolling(window=lookback, min_periods=lookback // 2).quantile(0.25)
    p75 = vol.rolling(window=lookback, min_periods=lookback // 2).quantile(0.75)
    regime = pd.Series(1, index=s.index)        # default normal
    regime = regime.where(vol >= p25, 0)         # below p25 → low
    regime = regime.where(vol <= p75, 2)         # above p75 → high
    return regime


def _obv_slope(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Slope of OBV over the last ``window`` bars. Positive = accumulation,
    negative = distribution. Normalised by current OBV magnitude."""
    obv = _obv(df)
    slope = (obv - obv.shift(window)) / window
    norm = obv.abs().replace(0, np.nan)
    return (slope / norm).fillna(0) * 100


def _volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Current bar's volume divided by trailing SMA20 of volume. 1.0 = avg,
    1.5+ = volume surge, < 0.5 = thin participation."""
    return df["volume"] / _sma(df["volume"], window).replace(0, np.nan)


def _volume_delta(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Signed volume — buy_vol - sell_vol over window. Buy = close > open."""
    sign = np.where(df["close"] > df["open"], 1, -1)
    signed = pd.Series(sign * df["volume"], index=df.index)
    return signed.rolling(window=window, min_periods=window).sum()


def _vwap_distance(df: pd.DataFrame) -> pd.Series:
    """% distance of close from VWAP. Positive = above (bullish), negative = below."""
    vwap = _vwap(df)
    return (df["close"] - vwap) / vwap.replace(0, np.nan) * 100


# ── Session features (timestamp-based) ─────────────────────────────────
# NSE market: 9:15 IST to 15:30 IST = 375 minutes


def _minutes_since_open(df: pd.DataFrame) -> pd.Series:
    """Minutes since 9:15 IST market open. Daily bars: returns 375 (full session)."""
    def _per_ts(ts):
        try:
            if hasattr(ts, "hour") and hasattr(ts, "minute"):
                if ts.hour == 0 and ts.minute == 0:
                    return 375  # daily bar — assume full session
                m = (ts.hour - 9) * 60 + (ts.minute - 15)
                return max(0, min(375, m))
        except Exception:
            pass
        return 375
    return pd.Series([_per_ts(t) for t in df.index], index=df.index, dtype=float)


def _session_progress(df: pd.DataFrame) -> pd.Series:
    """0.0-1.0 fraction through the trading day. Daily bars → 1.0."""
    return _minutes_since_open(df) / 375.0


def _is_first_hour(df: pd.DataFrame) -> pd.Series:
    """1.0 if current bar is in the first hour (9:15-10:15), else 0.0.
    Daily bars → 0.0 (we lose first-hour specificity on daily resolution)."""
    return (_minutes_since_open(df) < 60).astype(float)


def _is_last_hour(df: pd.DataFrame) -> pd.Series:
    """1.0 if current bar is in the last hour (14:30-15:30), else 0.0."""
    return (_minutes_since_open(df) >= 315).astype(float)


def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """SuperTrend line — returns the trend price. Caller compares close to this."""
    hl2 = (df["high"] + df["low"]) / 2
    atr = _atr(df, period)
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    st = pd.Series(index=df.index, dtype=float)
    direction = 1
    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = upper.iloc[i]
            continue
        prev = st.iloc[i - 1]
        c = df["close"].iloc[i]
        if direction == 1:
            st.iloc[i] = max(lower.iloc[i], prev)
            if c < st.iloc[i]:
                direction = -1
                st.iloc[i] = upper.iloc[i]
        else:
            st.iloc[i] = min(upper.iloc[i], prev)
            if c > st.iloc[i]:
                direction = 1
                st.iloc[i] = lower.iloc[i]
    return st


def _psar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """Parabolic SAR — Wilder 1978 implementation."""
    psar = df["close"].copy()
    bull = True
    af = step
    ep = df["high"].iloc[0]
    for i in range(2, len(df)):
        prev = psar.iloc[i - 1]
        if bull:
            psar.iloc[i] = prev + af * (ep - prev)
            if df["low"].iloc[i] < psar.iloc[i]:
                bull = False
                psar.iloc[i] = ep
                ep = df["low"].iloc[i]
                af = step
            else:
                if df["high"].iloc[i] > ep:
                    ep = df["high"].iloc[i]
                    af = min(af + step, max_step)
        else:
            psar.iloc[i] = prev + af * (ep - prev)
            if df["high"].iloc[i] > psar.iloc[i]:
                bull = True
                psar.iloc[i] = ep
                ep = df["high"].iloc[i]
                af = step
            else:
                if df["low"].iloc[i] < ep:
                    ep = df["low"].iloc[i]
                    af = min(af + step, max_step)
    return psar


# ─────────────────────────────────────────────────────────────────────
# Candle patterns — return 1.0 (present) or 0.0 (absent) on the LAST
# bar. Strategies compare with ``op: '==' , value: 1`` to fire.
# All ATR-relative so size sensitivity is consistent across stocks.
# ─────────────────────────────────────────────────────────────────────


def _candle_doji(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    rng = last["high"] - last["low"]
    if rng <= 0:
        return 0.0
    return 1.0 if body / rng < 0.10 else 0.0


def _candle_hammer(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    lower_wick = min(last["open"], last["close"]) - last["low"]
    upper_wick = last["high"] - max(last["open"], last["close"])
    if body <= 0:
        return 0.0
    return 1.0 if (lower_wick > 2 * body and upper_wick < body) else 0.0


def _candle_inverted_hammer(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    lower_wick = min(last["open"], last["close"]) - last["low"]
    upper_wick = last["high"] - max(last["open"], last["close"])
    if body <= 0:
        return 0.0
    return 1.0 if (upper_wick > 2 * body and lower_wick < body) else 0.0


def _candle_bullish_engulfing(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    p, c = df.iloc[-2], df.iloc[-1]
    return 1.0 if (
        p["close"] < p["open"] and
        c["close"] > c["open"] and
        c["open"] <= p["close"] and
        c["close"] >= p["open"]
    ) else 0.0


def _candle_bearish_engulfing(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    p, c = df.iloc[-2], df.iloc[-1]
    return 1.0 if (
        p["close"] > p["open"] and
        c["close"] < c["open"] and
        c["open"] >= p["close"] and
        c["close"] <= p["open"]
    ) else 0.0


def _candle_morning_star(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return 0.0
    d1, d2, d3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    return 1.0 if (
        d1["close"] < d1["open"] and
        abs(d2["close"] - d2["open"]) < abs(d1["close"] - d1["open"]) * 0.4 and
        d3["close"] > d3["open"] and
        d3["close"] > (d1["open"] + d1["close"]) / 2
    ) else 0.0


def _candle_evening_star(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return 0.0
    d1, d2, d3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    return 1.0 if (
        d1["close"] > d1["open"] and
        abs(d2["close"] - d2["open"]) < abs(d1["close"] - d1["open"]) * 0.4 and
        d3["close"] < d3["open"] and
        d3["close"] < (d1["open"] + d1["close"]) / 2
    ) else 0.0


def _candle_bullish_harami(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    p, c = df.iloc[-2], df.iloc[-1]
    return 1.0 if (
        p["close"] < p["open"] and
        c["close"] > c["open"] and
        c["open"] > p["close"] and c["close"] < p["open"]
    ) else 0.0


def _candle_bearish_harami(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    p, c = df.iloc[-2], df.iloc[-1]
    return 1.0 if (
        p["close"] > p["open"] and
        c["close"] < c["open"] and
        c["open"] < p["close"] and c["close"] > p["open"]
    ) else 0.0


def _candle_three_white_soldiers(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return 0.0
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    return 1.0 if (
        a["close"] > a["open"] and
        b["close"] > b["open"] and
        c["close"] > c["open"] and
        b["close"] > a["close"] and c["close"] > b["close"] and
        b["open"] > a["open"] and c["open"] > b["open"]
    ) else 0.0


def _candle_three_black_crows(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return 0.0
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    return 1.0 if (
        a["close"] < a["open"] and
        b["close"] < b["open"] and
        c["close"] < c["open"] and
        b["close"] < a["close"] and c["close"] < b["close"] and
        b["open"] < a["open"] and c["open"] < b["open"]
    ) else 0.0


# ─────────────────────────────────────────────────────────────────────
# Dispatch table — name → callable that returns the LAST-BAR value
# ─────────────────────────────────────────────────────────────────────


def _last(s: pd.Series) -> float:
    """Return the most recent non-NaN value, or NaN if all are NaN."""
    if s is None or len(s) == 0:
        return float("nan")
    v = s.iloc[-1]
    if isinstance(v, float) and math.isnan(v):
        # Walk backwards for the most recent non-NaN
        for x in reversed(s):
            if not (isinstance(x, float) and math.isnan(x)):
                return float(x)
        return float("nan")
    return float(v)


# ── Classic floor-trader pivots (from the PREVIOUS completed bar) ──────────
def _pivots(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) < 2:
        return {}
    hi = float(df["high"].iloc[-2])
    lo = float(df["low"].iloc[-2])
    c = float(df["close"].iloc[-2])
    p = (hi + lo + c) / 3.0
    rng = hi - lo
    return {
        "pivot_point": p,
        "pivot_r1": 2 * p - lo, "pivot_s1": 2 * p - hi,
        "pivot_r2": p + rng, "pivot_s2": p - rng,
        "pivot_r3": hi + 2 * (p - lo), "pivot_s3": lo - 2 * (hi - p),
    }


def _pivot(df: pd.DataFrame, key: str) -> float:
    return _pivots(df).get(key, float("nan"))


# ── Donchian channel (rolling extreme over the prior N completed bars) ─────
def _donchian(df: pd.DataFrame, window: int, side: str) -> float:
    if len(df) < window + 1:
        return float("nan")
    prior = df.iloc[-(window + 1):-1]   # exclude current bar so a breakout is real
    return float(prior["high"].max()) if side == "high" else float(prior["low"].min())


_INDICATOR_FNS: Dict[str, Callable[[pd.DataFrame], float]] = {
    # Momentum
    "rsi7": lambda df: _last(_rsi(df["close"], 7)),
    "rsi9": lambda df: _last(_rsi(df["close"], 9)),
    "rsi14": lambda df: _last(_rsi(df["close"], 14)),
    "stochastic_k": lambda df: _last(_stochastic(df)[0]),
    "stochastic_d": lambda df: _last(_stochastic(df)[1]),
    "williams_r": lambda df: _last(_williams_r(df)),
    "mfi": lambda df: _last(_mfi(df)),
    "cci": lambda df: _last(_cci(df)),
    # Trend / EMAs
    "ema5": lambda df: _last(_ema(df["close"], 5)),
    "ema8": lambda df: _last(_ema(df["close"], 8)),
    "ema13": lambda df: _last(_ema(df["close"], 13)),
    "ema21": lambda df: _last(_ema(df["close"], 21)),
    "ema50": lambda df: _last(_ema(df["close"], 50)),
    "ema100": lambda df: _last(_ema(df["close"], 100)),
    "ema200": lambda df: _last(_ema(df["close"], 200)),
    "sma10": lambda df: _last(_sma(df["close"], 10)),
    "sma20": lambda df: _last(_sma(df["close"], 20)),
    "sma50": lambda df: _last(_sma(df["close"], 50)),
    "sma100": lambda df: _last(_sma(df["close"], 100)),
    "sma200": lambda df: _last(_sma(df["close"], 200)),
    # MACD
    "macd": lambda df: _last(_macd(df["close"])[0]),
    "macd_signal": lambda df: _last(_macd(df["close"])[1]),
    "macd_hist": lambda df: _last(_macd(df["close"])[2]),
    # Other trend
    "adx": lambda df: _last(_adx(df)),
    "di_plus": lambda df: _last(_di_plus(df)),
    "di_minus": lambda df: _last(_di_minus(df)),
    "supertrend": lambda df: _last(_supertrend(df)),
    "psar": lambda df: _last(_psar(df)),
    # Momentum additions (PR-FEATURES)
    "roc_10": lambda df: _last(_roc(df["close"], 10)),
    "roc_20": lambda df: _last(_roc(df["close"], 20)),
    "stoch_rsi_k": lambda df: _last(_stoch_rsi(df["close"])[0]),
    "stoch_rsi_d": lambda df: _last(_stoch_rsi(df["close"])[1]),
    # Volatility
    "atr": lambda df: _last(_atr(df)),
    "bbands_upper": lambda df: _last(_bbands(df["close"])[0]),
    "bbands_middle": lambda df: _last(_bbands(df["close"])[1]),
    "bbands_lower": lambda df: _last(_bbands(df["close"])[2]),
    # Volatility additions (PR-FEATURES)
    "volatility_20": lambda df: _last(_realized_volatility(df["close"], 20)),
    "volatility_60": lambda df: _last(_realized_volatility(df["close"], 60)),
    "volatility_regime": lambda df: _last(_volatility_regime(df["close"])),
    # Volume / flow
    "vwap": lambda df: _last(_vwap(df)),
    "obv": lambda df: _last(_obv(df)),
    "volume_sma20": lambda df: _last(_sma(df["volume"], 20)),
    # Volume additions (PR-FEATURES)
    "obv_slope": lambda df: _last(_obv_slope(df)),
    "volume_ratio": lambda df: _last(_volume_ratio(df)),
    "volume_delta_20": lambda df: _last(_volume_delta(df, 20)),
    "vwap_distance_pct": lambda df: _last(_vwap_distance(df)),
    # Session features (PR-FEATURES)
    "minutes_since_open": lambda df: _last(_minutes_since_open(df)),
    "session_progress": lambda df: _last(_session_progress(df)),
    "is_first_hour": lambda df: _last(_is_first_hour(df)),
    "is_last_hour": lambda df: _last(_is_last_hour(df)),
    # Price refs
    "close": lambda df: float(df["close"].iloc[-1]),
    "open": lambda df: float(df["open"].iloc[-1]),
    "high": lambda df: float(df["high"].iloc[-1]),
    "low": lambda df: float(df["low"].iloc[-1]),
    "prev_close": lambda df: float(df["close"].iloc[-2]) if len(df) >= 2 else float("nan"),
    "prev_high": lambda df: float(df["high"].iloc[-2]) if len(df) >= 2 else float("nan"),
    "prev_low": lambda df: float(df["low"].iloc[-2]) if len(df) >= 2 else float("nan"),
    # Classic pivots (prior-bar derived)
    "pivot_point": lambda df: _pivot(df, "pivot_point"),
    "pivot_r1": lambda df: _pivot(df, "pivot_r1"),
    "pivot_s1": lambda df: _pivot(df, "pivot_s1"),
    "pivot_r2": lambda df: _pivot(df, "pivot_r2"),
    "pivot_s2": lambda df: _pivot(df, "pivot_s2"),
    "pivot_r3": lambda df: _pivot(df, "pivot_r3"),
    "pivot_s3": lambda df: _pivot(df, "pivot_s3"),
    # Donchian channel (prior-N-bar extremes)
    "donchian_high_20": lambda df: _donchian(df, 20, "high"),
    "donchian_low_20": lambda df: _donchian(df, 20, "low"),
    "donchian_high_55": lambda df: _donchian(df, 55, "high"),
    "donchian_low_55": lambda df: _donchian(df, 55, "low"),
    # Candle patterns
    "pattern_doji": _candle_doji,
    "pattern_hammer": _candle_hammer,
    "pattern_inverted_hammer": _candle_inverted_hammer,
    "pattern_bullish_engulfing": _candle_bullish_engulfing,
    "pattern_bearish_engulfing": _candle_bearish_engulfing,
    "pattern_morning_star": _candle_morning_star,
    "pattern_evening_star": _candle_evening_star,
    "pattern_bullish_harami": _candle_bullish_harami,
    "pattern_bearish_harami": _candle_bearish_harami,
    "pattern_three_white_soldiers": _candle_three_white_soldiers,
    "pattern_three_black_crows": _candle_three_black_crows,
}


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def list_indicators() -> tuple[str, ...]:
    return tuple(_INDICATOR_FNS.keys())


def compute_indicator(name: str, df: pd.DataFrame) -> float:
    """Compute the named indicator's CURRENT-BAR value.

    Returns NaN if there isn't enough data (caller treats NaN as
    "condition not met" — never True).
    """
    fn = _INDICATOR_FNS.get(name)
    if fn is None:
        raise ValueError(f"unknown indicator '{name}'")
    if len(df) < 2:
        return float("nan")
    return fn(df)


def compute_indicator_series(name: str, df: pd.DataFrame) -> pd.Series:
    """Used by crossover detection — we need the previous bar too."""
    series_fns: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "rsi7": lambda df: _rsi(df["close"], 7),
        "rsi9": lambda df: _rsi(df["close"], 9),
        "rsi14": lambda df: _rsi(df["close"], 14),
        "ema5": lambda df: _ema(df["close"], 5),
        "ema8": lambda df: _ema(df["close"], 8),
        "ema13": lambda df: _ema(df["close"], 13),
        "ema21": lambda df: _ema(df["close"], 21),
        "ema50": lambda df: _ema(df["close"], 50),
        "ema100": lambda df: _ema(df["close"], 100),
        "ema200": lambda df: _ema(df["close"], 200),
        "sma10": lambda df: _sma(df["close"], 10),
        "sma20": lambda df: _sma(df["close"], 20),
        "sma50": lambda df: _sma(df["close"], 50),
        "sma100": lambda df: _sma(df["close"], 100),
        "sma200": lambda df: _sma(df["close"], 200),
        "macd": lambda df: _macd(df["close"])[0],
        "macd_signal": lambda df: _macd(df["close"])[1],
        "macd_hist": lambda df: _macd(df["close"])[2],
        "vwap": _vwap,
        "supertrend": _supertrend,
        "psar": _psar,
        "close": lambda df: df["close"],
        "high": lambda df: df["high"],
        "low": lambda df: df["low"],
        "open": lambda df: df["open"],
    }
    fn = series_fns.get(name)
    if fn is None:
        # For non-series indicators (single scalar), return the scalar
        # broadcast to a 2-element series so prev/curr semantics still work.
        v = compute_indicator(name, df)
        return pd.Series([v, v], index=df.index[-2:] if len(df) >= 2 else df.index)
    return fn(df)


__all__ = [
    "MIN_LOOKBACK",
    "compute_indicator",
    "compute_indicator_series",
    "list_indicators",
]
