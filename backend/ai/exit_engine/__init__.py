"""Per-tick walking exit engine — PR-DEPTH.

Pattern adapted from aaryansinha16/AI-trader's tick_replay_backtest.py
``_check_exit_tick`` method.

Why this exists:
  Bar-level exit checking (high <= sl, low >= tgt) approximates the
  intra-bar sequence. If both SL and target are touched within the same
  minute, the bar-level loop can't tell which one came first — so it
  has to make a worst-case assumption (SL wins). That biases backtests
  pessimistically.

  Tick-level checking walks every tick chronologically:
    for tick in window:
        check SL at current self.sl
        check target
        ratchet trailing using this tick

  First trigger wins. SL set during one tick protects the next.
  Honest about the one limitation: if a price spike activates trailing
  AND reverts within the same tick, we still miss it — but that's a
  sub-millisecond error window not a sub-minute one.

When this matters:
  - Daily-timeframe DSL strategies: not at all (one bar = one day)
  - Intraday LSTM strategies (when LSTM hits PROD): material
  - Options strategies: material (premiums re-price every tick)

Requires tick history — runs against ``tick_data`` Supabase table.
Returns empty handed if no tick data is available, falling back to
bar-level (caller's responsibility).
"""

from .tick_exit import (
    TickExitConfig,
    TickExitDecision,
    TickExitEngine,
    walk_ticks_for_exit,
)
from .stagnation_trailing import (
    StagnationTrailingState,
    update_stagnation_trailing,
)

__all__ = [
    "TickExitConfig",
    "TickExitDecision",
    "TickExitEngine",
    "walk_ticks_for_exit",
    "StagnationTrailingState",
    "update_stagnation_trailing",
]
