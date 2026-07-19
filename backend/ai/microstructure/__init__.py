"""Microstructure features — PR-DEPTH.

Tick-level features that signal short-horizon (next 2 minutes) pressure.
Used by:
  - StrategyRunner premium-confirmation gate (pre-entry sanity check)
  - Per-tick exit engine in options_backtest (mode='tick')
  - Future intraday signal generators when LSTM intraday reaches PROD

Pattern adapted from aaryansinha16/AI-trader's features/micro_features.py,
heavily refactored for Supabase Postgres + our async stack.

The 5 micro features (matching aaryansinha's contract):
  bid_ask_spread     — current ask - bid (₹)
  order_imbalance    — (bid_qty - ask_qty) / (bid_qty + ask_qty), ∈ [-1, 1]
  trade_size_spike   — current volume / rolling mean volume (>1 = spike)
  volume_burst       — short-window vol / long-window vol (>1 = burst)
  tick_momentum      — Σ(buy_vol - sell_vol) over window, normalised
"""

from .features import (
    MicroFeatures,
    compute_micro_features,
    compute_premium_slope,
)

__all__ = [
    "MicroFeatures",
    "compute_micro_features",
    "compute_premium_slope",
]
