"""Microstructure feature computation — PR-DEPTH.

Pure-Python + numpy/pandas. No DB access here — caller passes a tick
DataFrame, this returns features. Lets the same code run in:
  - StrategyRunner gate (live tick feed from tick_data)
  - Backtest per-tick exit engine
  - Notebook research / admin tooling

Memory locks honoured: pure rules + math, no LLM, no broker calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Tick-rule classification thresholds
_BUY_PRESSURE_BPS = 1.0       # buy if price >= ask, fallback bps tolerance


@dataclass
class MicroFeatures:
    """One snapshot of micro features for one symbol at one timestamp."""
    timestamp: pd.Timestamp
    symbol: str
    last_price: float

    bid_ask_spread: float          # absolute, ₹
    bid_ask_spread_bps: float      # basis points relative to mid
    order_imbalance: float         # (bid_qty - ask_qty) / (bid_qty + ask_qty), [-1, 1]
    trade_size_spike: float        # vol_current / vol_rolling_mean
    volume_burst: float            # vol_short_window / vol_long_window
    tick_momentum: float           # Σ signed flow normalised by total vol, [-1, 1]
    buy_pressure_ratio: float      # buy_vol / total_vol, [0, 1]
    n_ticks_in_window: int

    @property
    def is_buying_pressure(self) -> bool:
        return self.tick_momentum > 0.1

    @property
    def is_selling_pressure(self) -> bool:
        return self.tick_momentum < -0.1

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else str(self.timestamp),
            "symbol": self.symbol,
            "last_price": round(self.last_price, 4),
            "bid_ask_spread": round(self.bid_ask_spread, 4),
            "bid_ask_spread_bps": round(self.bid_ask_spread_bps, 2),
            "order_imbalance": round(self.order_imbalance, 4),
            "trade_size_spike": round(self.trade_size_spike, 3),
            "volume_burst": round(self.volume_burst, 3),
            "tick_momentum": round(self.tick_momentum, 4),
            "buy_pressure_ratio": round(self.buy_pressure_ratio, 4),
            "n_ticks_in_window": self.n_ticks_in_window,
        }


def _classify_buy_sell(price: float, bid: float, ask: float) -> int:
    """Lee-Ready tick rule. Returns +1 for buyer-initiated, -1 for
    seller-initiated, 0 for ambiguous (price at midpoint)."""
    if ask > 0 and price >= ask:
        return 1
    if bid > 0 and price <= bid:
        return -1
    # Use midpoint comparison as tiebreaker
    if ask > 0 and bid > 0:
        mid = (ask + bid) / 2.0
        if price > mid:
            return 1
        if price < mid:
            return -1
    return 0


def compute_micro_features(
    tick_df: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    window_seconds: int = 30,
    short_window_seconds: int = 10,
) -> Optional[MicroFeatures]:
    """Compute one current-snapshot of micro features from the trailing
    ``window_seconds`` of ticks.

    Args:
        tick_df: DataFrame with columns timestamp, price, volume, and
            optionally bid_price, ask_price, bid_qty, ask_qty.
            Sorted ascending by timestamp.
        symbol: defaults to tick_df.symbol[-1] if column exists
        window_seconds: rolling window for the "current" features
        short_window_seconds: shorter window for volume_burst comparison

    Returns ``None`` if not enough data (< 3 ticks).
    """
    if tick_df is None or tick_df.empty or len(tick_df) < 3:
        return None

    df = tick_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    last = df.iloc[-1]
    last_ts = last["timestamp"]
    last_price = float(last["price"])

    if symbol is None:
        symbol = str(last.get("symbol", "UNKNOWN"))

    # Window the trailing N seconds
    win_start = last_ts - pd.Timedelta(seconds=window_seconds)
    short_start = last_ts - pd.Timedelta(seconds=short_window_seconds)
    win = df[df["timestamp"] >= win_start]
    short_win = df[df["timestamp"] >= short_start]
    if len(win) < 2:
        return None

    # ── Bid-Ask Spread ─────────────────────────────────────────
    have_book = ("bid_price" in df.columns and "ask_price" in df.columns)
    if have_book:
        bid = float(last.get("bid_price") or 0)
        ask = float(last.get("ask_price") or 0)
        if bid > 0 and ask > 0:
            spread = max(0.0, ask - bid)
            mid = (ask + bid) / 2.0
            spread_bps = (spread / mid) * 10_000 if mid > 0 else 0.0
        else:
            # Fall back to ±half-tick estimate
            spread = max(0.05, last_price * 0.001)
            spread_bps = (spread / last_price) * 10_000
    else:
        spread = max(0.05, last_price * 0.001)
        spread_bps = (spread / last_price) * 10_000

    # ── Order Imbalance ────────────────────────────────────────
    if have_book and "bid_qty" in df.columns and "ask_qty" in df.columns:
        bq = float(last.get("bid_qty") or 0)
        aq = float(last.get("ask_qty") or 0)
        denom = bq + aq
        imbalance = (bq - aq) / denom if denom > 0 else 0.0
    else:
        imbalance = 0.0

    # ── Trade Size Spike ───────────────────────────────────────
    # Current bar's volume vs rolling mean of last window's volume
    current_vol = float(last["volume"])
    rolling_mean_vol = float(win["volume"].mean()) if len(win) > 0 else 1.0
    trade_size_spike = current_vol / rolling_mean_vol if rolling_mean_vol > 0 else 1.0

    # ── Volume Burst ───────────────────────────────────────────
    short_vol_sum = float(short_win["volume"].sum())
    long_vol_sum = float(win["volume"].sum())
    # Normalise by window-length ratio so we're comparing rates not sums
    short_rate = short_vol_sum / max(short_window_seconds, 1)
    long_rate = long_vol_sum / max(window_seconds, 1)
    volume_burst = short_rate / long_rate if long_rate > 0 else 1.0

    # ── Tick Momentum + Buy-Pressure Ratio ─────────────────────
    if have_book:
        # Classify each tick in window via tick rule
        bids = win["bid_price"].fillna(0).astype(float).values
        asks = win["ask_price"].fillna(0).astype(float).values
        prices = win["price"].astype(float).values
        vols = win["volume"].astype(float).values

        signed_flow = 0.0
        buy_vol = 0.0
        sell_vol = 0.0
        for p, b, a, v in zip(prices, bids, asks, vols):
            sign = _classify_buy_sell(p, b, a)
            signed_flow += sign * v
            if sign > 0:
                buy_vol += v
            elif sign < 0:
                sell_vol += v

        total_signed_vol = buy_vol + sell_vol
        tick_momentum = signed_flow / total_signed_vol if total_signed_vol > 0 else 0.0
        buy_pressure_ratio = buy_vol / total_signed_vol if total_signed_vol > 0 else 0.5
    else:
        # No book data → use price-slope as a momentum proxy
        first_price = float(win["price"].iloc[0])
        slope = (last_price - first_price) / first_price if first_price > 0 else 0.0
        tick_momentum = float(np.clip(slope * 50, -1.0, 1.0))  # scale slope to [-1, 1]
        buy_pressure_ratio = 0.5 + tick_momentum / 2.0
        buy_pressure_ratio = float(np.clip(buy_pressure_ratio, 0.0, 1.0))

    return MicroFeatures(
        timestamp=last_ts,
        symbol=symbol,
        last_price=last_price,
        bid_ask_spread=spread,
        bid_ask_spread_bps=spread_bps,
        order_imbalance=imbalance,
        trade_size_spike=trade_size_spike,
        volume_burst=volume_burst,
        tick_momentum=tick_momentum,
        buy_pressure_ratio=buy_pressure_ratio,
        n_ticks_in_window=len(win),
    )


def compute_premium_slope(
    tick_df: pd.DataFrame,
    *,
    window_seconds: int = 30,
) -> Optional[float]:
    """Compute the % slope of the option premium over the trailing window.

    Used by the premium-confirmation gate ("falling knife" rejection):
    if a long-call entry signals but the option's premium has been
    falling >0.8% in the last 30 seconds, the gate rejects the trade.

    Returns the percentage slope (e.g. -0.012 = -1.2%) or ``None`` if
    insufficient data (< 5 ticks in window).
    """
    if tick_df is None or tick_df.empty or len(tick_df) < 5:
        return None

    df = tick_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    last_ts = df["timestamp"].iloc[-1]
    win = df[df["timestamp"] >= last_ts - pd.Timedelta(seconds=window_seconds)]
    if len(win) < 5:
        return None

    first_price = float(win["price"].iloc[0])
    last_price = float(win["price"].iloc[-1])
    if first_price <= 0:
        return None
    return (last_price - first_price) / first_price
