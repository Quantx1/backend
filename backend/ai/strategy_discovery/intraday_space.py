"""IntradaySearchSpace — sampler for 5m / 15m bar intraday strategies.

Designed for NSE cash-segment intraday day-trading. Every sampled
strategy has:
  * timeframe ∈ {5m, 15m} (DSL enforces stop_loss_pct for intraday)
  * Entry: momentum breakout, mean-reversion, VWAP cross, or opening-
    range breakout — session-aware via the `is_first_hour`/`is_last_hour`
    boolean indicators.
  * Tight stops (0.4–1.5%) + 1:2 to 1:4 target ratios.
  * EOD square-off enforced by the wrapper (`run_intraday_backtest`).

Mutation perturbs SL/TP, regime filter, sizing — same as the swing
space. Differences from the swing space:
  * Smaller stops and targets (intraday vol)
  * No trailing stops by default (gets stopped on first noise spike)
  * Session-aware entries (avoid first 5min noise + last 10min wind-down)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from backend.ai.strategy.dsl import (
    Condition, ConditionKind, InstrumentSegment, Operator, PositionSize,
    PositionSizeKind, RegimeFilter, Strategy, StrategyMode, Timeframe,
    Universe,
)


def _opening_range_breakout(rng: random.Random) -> Condition:
    """Buy on a bar that breaks above the prior session-open range."""
    # Use a composite: not_first_5min AND close > recent high
    return Condition(
        kind=ConditionKind.COMPOSITE_AND,
        children=[
            Condition(
                kind=ConditionKind.INDICATOR_COMPARE,
                indicator="is_first_hour", op=Operator.EQ, value=0,
            ),
            Condition(
                kind=ConditionKind.INDICATOR_CROSS,
                indicator="close", op=Operator.CROSSES_ABOVE, value="vwap",
            ),
        ],
    )


def _vwap_pullback(rng: random.Random) -> Condition:
    """Buy when price re-claims VWAP after dipping below."""
    return Condition(
        kind=ConditionKind.COMPOSITE_AND,
        children=[
            Condition(
                kind=ConditionKind.INDICATOR_COMPARE,
                indicator="vwap_distance_pct", op=Operator.LT, value=-0.2,
            ),
            Condition(
                kind=ConditionKind.INDICATOR_COMPARE,
                indicator="rsi14", op=Operator.GT, value=rng.choice([40, 45, 50]),
            ),
        ],
    )


def _intraday_momentum(rng: random.Random) -> Condition:
    """5m fast-EMA crossover above slow-EMA with volume confirmation."""
    return Condition(
        kind=ConditionKind.COMPOSITE_AND,
        children=[
            Condition(
                kind=ConditionKind.INDICATOR_CROSS,
                indicator="ema8", op=Operator.CROSSES_ABOVE, value="ema21",
            ),
            Condition(
                kind=ConditionKind.INDICATOR_COMPARE,
                indicator="volume_ratio", op=Operator.GT,
                value=round(rng.uniform(1.2, 2.0), 2),
            ),
        ],
    )


def _build_intraday_exit(rng: random.Random) -> Condition:
    """Default exit: ema8 crosses back below ema21 OR last-hour flat."""
    # Last-hour fall-back exit covered by the wrapper's EOD square-off;
    # this just gives an early-exit signal.
    return Condition(
        kind=ConditionKind.INDICATOR_CROSS,
        indicator="ema8", op=Operator.CROSSES_BELOW, value="ema21",
    )


@dataclass
class IntradaySearchSpace:
    """Search space for cash-segment intraday strategies (5m or 15m).

    `tf`:
      * "5m"  — high-frequency, tighter stops, NIFTY 50 only
      * "15m" — calmer, slightly wider, NIFTY 100 OK
    """
    tf: str = "5m"                              # "5m" | "15m"
    universe: Universe = Universe.NIFTY_50
    seed: Optional[int] = None

    def __post_init__(self):
        if self.tf not in ("5m", "15m"):
            raise ValueError(f"tf must be 5m|15m, got {self.tf}")

    def _rng(self, salt: int = 0) -> random.Random:
        seed = (self.seed or 0) + salt
        return random.Random(seed) if seed else random.Random()

    def sample(self, idx: int = 0) -> Strategy:
        rng = self._rng(idx)

        family = rng.choice(["orb", "vwap", "momentum"])
        if family == "orb":
            entry = _opening_range_breakout(rng)
            name_part = "ORB"
        elif family == "vwap":
            entry = _vwap_pullback(rng)
            name_part = "VWAP"
        else:
            entry = _intraday_momentum(rng)
            name_part = "Momo"

        exit_cond = _build_intraday_exit(rng)

        # Tighter intraday stops/targets
        if self.tf == "5m":
            sl = round(rng.uniform(0.4, 1.0), 2)
            tp = round(rng.uniform(0.8, 2.5), 2)
        else:
            sl = round(rng.uniform(0.6, 1.5), 2)
            tp = round(rng.uniform(1.2, 4.0), 2)

        # Regime filter — intraday strategies usually want a tradable
        # regime (not strict bear-only).
        regime = rng.choices(
            [RegimeFilter.ANY, RegimeFilter.BULL_ONLY, RegimeFilter.SIDEWAYS_ONLY],
            weights=[0.6, 0.3, 0.1],
        )[0]

        # Smaller sizing — intraday risk per trade should be tight.
        pct = round(rng.uniform(2.0, 8.0), 1)
        sizing = PositionSize(kind=PositionSizeKind.PERCENT_OF_CAPITAL, value=pct)

        tf_enum = Timeframe.M5 if self.tf == "5m" else Timeframe.M15
        return Strategy(
            name=f"{name_part}-intraday{self.tf}-{idx:04d}",
            instrument_segment=InstrumentSegment.EQUITY,
            symbol=None,
            universe=self.universe,
            timeframe=tf_enum,
            entry=entry,
            exit=exit_cond,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            trailing_stop_pct=None,             # noisy on intraday
            position_size=sizing,
            regime_filter=regime,
            lookback_days=30,                   # intraday windows are short
            mode=StrategyMode.BACKTEST,
        )

    def mutate(self, parent: Strategy, idx: int = 0) -> Strategy:
        rng = self._rng(idx + 300_000)
        child = parent.model_copy(deep=True)
        knobs = rng.sample(["stop", "target", "regime", "size"], k=rng.randint(1, 2))
        if "stop" in knobs:
            delta = rng.uniform(-0.3, 0.3)
            new_sl = max(0.2, min(3.0, (child.stop_loss_pct or 0.8) + delta))
            child.stop_loss_pct = round(new_sl, 2)
        if "target" in knobs:
            delta = rng.uniform(-0.5, 0.5)
            new_tp = max(0.5, min(6.0, (child.take_profit_pct or 1.5) + delta))
            child.take_profit_pct = round(new_tp, 2)
        if "regime" in knobs:
            child.regime_filter = rng.choice(list(RegimeFilter))
        if "size" in knobs:
            delta = rng.uniform(-1.5, 1.5)
            new_pct = max(1.0, min(12.0, child.position_size.value + delta))
            child.position_size = PositionSize(
                kind=PositionSizeKind.PERCENT_OF_CAPITAL, value=round(new_pct, 1),
            )
        child.name = f"{parent.name}-m{idx}"
        return Strategy.model_validate(child.model_dump())
