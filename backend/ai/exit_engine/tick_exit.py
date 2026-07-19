"""Per-tick walking exit logic — PR-DEPTH.

Adapted from aaryansinha16/AI-trader's tick_replay_backtest._check_exit_tick.

Contract:
  Caller maintains the position state (entry price, SL, target, trailing
  state). Each minute (or other tick window), pass us:
    - the window of ticks for this period
    - current position state
  We walk every tick in chronological order:
    1. Check SL at current self.sl (set by prior tick's trailing ratchet)
    2. Check target
    3. If neither: ratchet trailing using THIS tick as new peak candidate
  First trigger wins. Return TickExitDecision.

  Trailing logic mirrors aaryansinha's tiered-retention scheme:
    gain ≥ 50%  → retention 0.80
    gain ≥ 35%  → retention 0.70
    gain ≥ 25%  → retention 0.60
    gain ≥ trigger × 2.5 → retention 0.55
    gain ≥ trigger × 1.5 → retention 0.45
    else (above trigger) → retention 0.35
  Plus stagnation boost (separate module: stagnation_trailing.py).

Pure functions where possible. State object is the position itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TickExitConfig:
    """Per-strategy exit configuration. Pass into the engine."""
    stop_loss_pct: float                    # % of entry, e.g. 0.40 for 40%
    take_profit_pct: float                  # % of entry, e.g. 0.80 for 80%
    trailing_trigger_pct: float = 0.12      # activate trailing after +12% peak
    trailing_lock_pct: float = 0.08         # initial lock at +8% above entry
    enable_stagnation_boost: bool = True
    half_spread_pct: float = 0.003          # bid-side slippage on exits


@dataclass
class TickExitDecision:
    """Result of walking the ticks. exit_price=None means no exit fired."""
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None       # 'stop_loss' | 'target' | 'trailing_sl'
    new_sl: float = 0.0                     # updated SL after trailing ratchet
    new_peak: float = 0.0                   # updated peak premium
    trailing_active: bool = False
    ticks_walked: int = 0


@dataclass
class _PositionState:
    """Mutable state during the walk. Caller copies values back."""
    entry_price: float
    sl: float
    target: float
    peak: float
    trailing_active: bool
    config: TickExitConfig
    peak_bar_idx: int = 0
    current_bar_idx: int = 0


def _retention_tier(gain_pct: float, trigger_pct: float) -> float:
    """Tiered base retention as a fraction of peak gain to lock in.
    Mirrors aaryansinha's intra-bar exit sequence (commit 2026-04-16)."""
    if gain_pct >= 0.50:
        return 0.80
    if gain_pct >= 0.35:
        return 0.70
    if gain_pct >= 0.25:
        return 0.60
    if gain_pct >= trigger_pct * 2.5:
        return 0.55
    if gain_pct >= trigger_pct * 1.5:
        return 0.45
    return 0.35


def _ratchet_trailing(state: _PositionState, candidate_peak: float) -> None:
    """Update peak + ratchet SL. Pure mutation on state."""
    if candidate_peak > state.peak:
        state.peak = candidate_peak
        state.peak_bar_idx = state.current_bar_idx

    cfg = state.config
    gain_pct = (state.peak - state.entry_price) / state.entry_price if state.entry_price > 0 else 0

    if not state.trailing_active:
        if gain_pct >= cfg.trailing_trigger_pct:
            state.trailing_active = True
            lock_price = state.entry_price * (1 + cfg.trailing_lock_pct)
            state.sl = max(state.sl, lock_price)
        return

    # Trailing already active — tier-retain a fraction of peak gain
    gain_from_entry = state.peak - state.entry_price
    retention = _retention_tier(gain_pct, cfg.trailing_trigger_pct)

    # Optional stagnation boost: handled by caller via stagnation_trailing.py
    trail_sl = state.entry_price + retention * gain_from_entry
    state.sl = max(state.sl, trail_sl)


def walk_ticks_for_exit(
    tick_window: pd.DataFrame,
    *,
    entry_price: float,
    current_sl: float,
    current_target: float,
    current_peak: float,
    trailing_active: bool,
    config: TickExitConfig,
    current_bar_idx: int = 0,
    peak_bar_idx: int = 0,
) -> TickExitDecision:
    """Walk every tick in ``tick_window`` chronologically. First SL/TP hit
    wins. Trailing ratchets each tick.

    Args:
        tick_window: DataFrame with at least ['timestamp', 'price'] columns.
            If 'bid' present, use bid for exit slippage; else apply
            half_spread_pct.
        entry_price: original entry premium
        current_sl: SL going into this window
        current_target: target going into this window
        current_peak: peak premium seen so far
        trailing_active: True if trailing already activated
        config: TickExitConfig

    Returns: TickExitDecision. If exit_price is None, no exit fired and
    new_sl / new_peak / trailing_active reflect post-walk state.
    """
    if tick_window is None or tick_window.empty:
        return TickExitDecision(
            new_sl=current_sl, new_peak=current_peak,
            trailing_active=trailing_active, ticks_walked=0,
        )

    state = _PositionState(
        entry_price=entry_price,
        sl=current_sl,
        target=current_target,
        peak=current_peak,
        trailing_active=trailing_active,
        config=config,
        peak_bar_idx=peak_bar_idx,
        current_bar_idx=current_bar_idx,
    )

    df = tick_window.sort_values("timestamp") if "timestamp" in tick_window.columns else tick_window
    walked = 0

    for _, tick in df.iterrows():
        walked += 1
        price = float(tick["price"])

        # Exit-side slippage: use real bid if available, else half-spread fallback
        if "bid" in tick and tick["bid"] is not None and not pd.isna(tick["bid"]) and float(tick["bid"]) > 0:
            bid = float(tick["bid"]) * (1 - config.half_spread_pct)
        else:
            bid = price * (1 - config.half_spread_pct)

        # 1. SL — checked against the SL set by prior tick's trailing
        if bid <= state.sl:
            return TickExitDecision(
                exit_price=state.sl,
                exit_reason="trailing_sl" if state.trailing_active else "stop_loss",
                new_sl=state.sl, new_peak=state.peak,
                trailing_active=state.trailing_active, ticks_walked=walked,
            )

        # 2. Target — exit at limit (bid >= target)
        if bid >= state.target:
            return TickExitDecision(
                exit_price=state.target, exit_reason="target",
                new_sl=state.sl, new_peak=state.peak,
                trailing_active=state.trailing_active, ticks_walked=walked,
            )

        # 3. Ratchet trailing using THIS tick's price as the new peak candidate
        _ratchet_trailing(state, price)

    # No exit fired this window — return updated state for caller to persist
    return TickExitDecision(
        new_sl=state.sl, new_peak=state.peak,
        trailing_active=state.trailing_active, ticks_walked=walked,
    )


# ─────────────────────────────────────────────────────────────────────
# Engine class — keeps state across multiple windows for a single position
# ─────────────────────────────────────────────────────────────────────


class TickExitEngine:
    """Stateful per-position tick exit engine. Caller creates one per
    open position, calls ``step(tick_window)`` per minute, and reads
    ``last_decision`` to know what to do."""

    def __init__(
        self,
        *,
        entry_price: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        config: Optional[TickExitConfig] = None,
    ):
        self.entry_price = entry_price
        self.config = config or TickExitConfig(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        self.sl = entry_price * (1 - stop_loss_pct)
        self.target = entry_price * (1 + take_profit_pct)
        self.peak = entry_price
        self.trailing_active = False
        self.last_decision: Optional[TickExitDecision] = None
        self.total_ticks_walked = 0
        self.bar_idx = 0
        self.peak_bar_idx = 0
        self.is_closed = False
        self.close_price: Optional[float] = None
        self.close_reason: Optional[str] = None

    def step(self, tick_window: pd.DataFrame) -> TickExitDecision:
        """Walk one minute (or other window) of ticks. Returns the decision."""
        if self.is_closed:
            return TickExitDecision(
                exit_price=self.close_price, exit_reason=self.close_reason,
                new_sl=self.sl, new_peak=self.peak,
                trailing_active=self.trailing_active, ticks_walked=0,
            )

        self.bar_idx += 1
        decision = walk_ticks_for_exit(
            tick_window,
            entry_price=self.entry_price,
            current_sl=self.sl,
            current_target=self.target,
            current_peak=self.peak,
            trailing_active=self.trailing_active,
            config=self.config,
            current_bar_idx=self.bar_idx,
            peak_bar_idx=self.peak_bar_idx,
        )
        self.sl = decision.new_sl
        self.peak = decision.new_peak
        if decision.new_peak > self.peak:
            self.peak_bar_idx = self.bar_idx
        self.trailing_active = decision.trailing_active
        self.total_ticks_walked += decision.ticks_walked
        self.last_decision = decision

        if decision.exit_price is not None:
            self.is_closed = True
            self.close_price = decision.exit_price
            self.close_reason = decision.exit_reason

        return decision
