"""Intraday-specific indicators not in daily-bar TA libraries.

Required by the 12 setup scanners (ORB, VWAP family, IB, Anchored VWAP,
Power Hour, Mean Reversion, Squeeze, EOD Drift, etc.).

All functions are pure numpy/pandas — no broker calls, no I/O. The
caller fetches the intraday DataFrame (OHLCV with DatetimeIndex in IST)
and passes it in. This keeps the scanner testable without a market data
provider.

NSE session reference (IST):
    09:00-09:08  pre-open auction (don't trade)
    09:15-15:30  regular session
    12:30-13:30  lunch lull (avoid initiations — fake signals fire)
    14:30-15:30  power hour (fade HoD/LoD tests per Edgeful research)
    15:20-15:30  closing auction (flatten only)
"""

from __future__ import annotations

from datetime import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ── Session-aware helpers ─────────────────────────────────────────


def _ist_time(ts: pd.Timestamp) -> time:
    """Extract IST time-of-day from an index timestamp."""
    if ts.tz is None:
        return ts.time()
    try:
        return ts.tz_convert("Asia/Kolkata").time()
    except Exception:
        return ts.time()


def is_lunch_window(ts: pd.Timestamp) -> bool:
    """12:30-13:30 IST lunch lull — drop signals fired in this window."""
    t = _ist_time(ts)
    return time(12, 30) <= t < time(13, 30)


def is_power_hour(ts: pd.Timestamp) -> bool:
    """14:30-15:30 IST — Edgeful stats say new HoD/LoD prints only
    12-24% of sessions during power hour, so fade tests > chase breakouts.
    """
    t = _ist_time(ts)
    return time(14, 30) <= t < time(15, 30)


def is_closing_auction(ts: pd.Timestamp) -> bool:
    """15:20-15:30 IST — closing auction distorts price discovery;
    flatten by 15:25, never initiate."""
    t = _ist_time(ts)
    return time(15, 20) <= t < time(15, 30)


# ── VWAP family ────────────────────────────────────────────────────


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP — Σ(typical × volume) / Σvolume, reset 09:15 IST.

    Input: DataFrame with columns ['open','high','low','close','volume']
    and DatetimeIndex.

    Returns: same-length Series of VWAP values per bar.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    cum_pv = pv.groupby(df.index.date).cumsum()
    cum_vol = df["volume"].groupby(df.index.date).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def vwap_bands(df: pd.DataFrame, *, n_sigma: float = 2.0, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """VWAP ± n_sigma bands.

    sigma = rolling stdev of (close - VWAP) over `window` bars.
    Returns (vwap, upper, lower).
    """
    vwap = session_vwap(df)
    if vwap.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty
    residual = df["close"] - vwap
    sigma = residual.rolling(window, min_periods=max(5, window // 2)).std()
    upper = vwap + n_sigma * sigma
    lower = vwap - n_sigma * sigma
    return vwap, upper, lower


def anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> pd.Series:
    """Anchored VWAP (Brian Shannon) — VWAP computed from a chosen
    anchor bar to the end. Common anchors: earnings bar, swing high/low,
    IPO open, gap bar, FOMC/RBI policy bar.

    Returns a Series aligned to df.index, with NaN before anchor_idx.
    """
    if df is None or df.empty or anchor_idx >= len(df):
        return pd.Series(dtype=float, index=df.index if df is not None else None)
    seg = df.iloc[anchor_idx:].copy()
    typical = (seg["high"] + seg["low"] + seg["close"]) / 3.0
    pv = typical * seg["volume"]
    cum_pv = pv.cumsum()
    cum_vol = seg["volume"].cumsum().replace(0, np.nan)
    avwap = cum_pv / cum_vol
    out = pd.Series(np.nan, index=df.index)
    out.iloc[anchor_idx:] = avwap.values
    return out


# ── Opening Range / Initial Balance ───────────────────────────────


def opening_range(df: pd.DataFrame, *, minutes: int = 15) -> Optional[dict]:
    """First N minutes of the session (NSE: 09:15 IST onward).

    Returns dict {'high', 'low', 'range', 'bars_count'} for the FIRST
    `minutes` minutes; None when the slice is empty.

    Default 15-min ORB is the canonical NSE-tested window (TraderLion
    + QuantifiedStrategies converge on 15-min as best risk/noise).
    """
    if df is None or df.empty:
        return None
    open_bars = []
    for ts, row in df.iterrows():
        t = _ist_time(ts)
        if t < time(9, 15):
            continue
        # Minutes since session open (only counts bars on the same date)
        delta_min = (t.hour - 9) * 60 + (t.minute - 15)
        if delta_min < 0 or delta_min >= minutes:
            if delta_min >= minutes:
                break
            continue
        open_bars.append((ts, row))
    if not open_bars:
        return None
    highs = [r["high"] for _, r in open_bars]
    lows = [r["low"] for _, r in open_bars]
    return {
        "high": float(max(highs)),
        "low": float(min(lows)),
        "range": float(max(highs) - min(lows)),
        "bars_count": len(open_bars),
        "first_ts": open_bars[0][0],
        "last_ts": open_bars[-1][0],
    }


def initial_balance(df: pd.DataFrame) -> Optional[dict]:
    """First hour of trade — 09:15-10:15 IST (Dalton Market Profile).

    Trend Day = narrow IB (≤30-40% of typical 20-day range) + continuous
    range extension. Open Drive = price moves immediately in one
    direction in period A (first 30 min) and never returns into open.
    """
    return opening_range(df, minutes=60)


# ── Misc helpers ────────────────────────────────────────────────


def cumulative_delta_tickrule(df: pd.DataFrame) -> pd.Series:
    """Cumulative Volume Delta via tick-rule approximation.

    CRITICAL CAVEAT (per memory + research): standard NSE feeds do NOT
    expose true bid/ask aggression. The tick-rule (uptick = buy,
    downtick = sell) is a WEAK proxy. Restrict CVD-divergence scans to
    most-liquid F&O names (>10k trades/day) and gate behind a Pro+ tier
    where the data-quality caveat can be shown.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    sign = np.where(df["close"] > df["close"].shift(1), 1.0,
                    np.where(df["close"] < df["close"].shift(1), -1.0, 0.0))
    delta = sign * df["volume"].astype(float)
    return pd.Series(delta, index=df.index).cumsum()


def bb_squeeze_inside_kc(
    df: pd.DataFrame,
    *,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
) -> pd.Series:
    """TTM Squeeze detection (John Carter) — BB INSIDE KC.

    Squeeze ON when BB(20,2) is fully inside KC(20, 1.5×ATR).
    Returns boolean Series — True for bars where squeeze is firing.
    """
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    mid = df["close"].rolling(bb_period, min_periods=bb_period // 2).mean()
    std = df["close"].rolling(bb_period, min_periods=bb_period // 2).std()
    bb_up = mid + bb_std * std
    bb_lo = mid - bb_std * std
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(kc_period, min_periods=kc_period // 2).mean()
    kc_up = mid + kc_mult * atr
    kc_lo = mid - kc_mult * atr
    return (bb_lo > kc_lo) & (bb_up < kc_up)
