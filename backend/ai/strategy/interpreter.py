"""
Strategy interpreter — evaluates a Condition node against a price bar +
engine signal context, returns True/False.

This is the single hot-path called per bar during backtests AND live
trading. Stay branch-light; no logging in tight loops.

Threading model: interpreter is pure (no globals, no I/O). Caller
prepares ``InterpreterContext`` once per bar and passes it in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from .dsl import Condition, ConditionKind, EngineName, Operator
from .indicators import compute_indicator, compute_indicator_series


@dataclass(frozen=True)
class EngineSignals:
    """Latest engine outputs at the bar being evaluated.

    None values mean "engine unavailable for this symbol/regime/etc" —
    any engine_signal condition referring to a None engine evaluates
    to False (defensive default: never fire a trade on missing data).
    """
    alpha: Optional[float] = None    # cross-sectional rank (lower = stronger)
    mood: Optional[float] = None     # sentiment_5d_mean, [-1, 1]
    regime: Optional[str] = None     # 'bull' | 'sideways' | 'bear'
    # Trim history (all 2026-05-25):
    #   - `horizon` removed in PR-M cut (TimesFM dropped)
    #   - `vision`, `verdict`, `pulse` removed — no PROD model behind them
    # Re-add fields when their backing models reach PROD post-v1.


@dataclass
class InterpreterContext:
    """Everything the interpreter needs to evaluate a Condition.

    ``bars`` must be oldest-to-newest, with OHLCV columns lowercase. At
    minimum 2 rows for cross-detection; 200+ recommended for long EMAs.
    """
    bars: pd.DataFrame
    engines: EngineSignals = field(default_factory=EngineSignals)
    # Memoize indicator scalars within a single evaluate() call so a
    # composite_and with rsi14 in two children doesn't double-compute.
    _cache: Dict[str, Any] = field(default_factory=dict)

    def indicator(self, name: str) -> float:
        if name not in self._cache:
            self._cache[name] = compute_indicator(name, self.bars)
        return self._cache[name]

    def indicator_series(self, name: str) -> pd.Series:
        key = f"series:{name}"
        if key not in self._cache:
            self._cache[key] = compute_indicator_series(name, self.bars)
        return self._cache[key]


# ─────────────────────────────────────────────────────────────────────
# Operator implementations
# ─────────────────────────────────────────────────────────────────────


def _scalar_compare(left: float, op: Operator, right: Any) -> bool:
    """Generic scalar comparison. Both sides must be float-coercible."""
    if isinstance(left, float) and math.isnan(left):
        return False
    if op == Operator.BETWEEN:
        lo, hi = right
        return lo <= left <= hi
    if op == Operator.OUTSIDE:
        lo, hi = right
        return left < lo or left > hi
    if not isinstance(right, (int, float)):
        # caller error — already enforced at validate time, but be defensive
        return False
    if op == Operator.LT:
        return left < right
    if op == Operator.GT:
        return left > right
    if op == Operator.LTE:
        return left <= right
    if op == Operator.GTE:
        return left >= right
    if op == Operator.EQ:
        return left == right
    if op == Operator.NE:
        return left != right
    return False


def _string_compare(left: Optional[str], op: Operator, right: Any) -> bool:
    """For engine signals that return categorical strings."""
    if left is None:
        return False
    if op == Operator.EQ:
        return left == right
    if op == Operator.NE:
        return left != right
    return False


def _cross_detect(series: pd.Series, ref: pd.Series, direction: Operator) -> bool:
    """True iff the LAST bar crosses ``series`` above/below ``ref``."""
    if len(series) < 2 or len(ref) < 2:
        return False
    s_now, s_prev = series.iloc[-1], series.iloc[-2]
    r_now, r_prev = ref.iloc[-1], ref.iloc[-2]
    if any(isinstance(x, float) and math.isnan(x) for x in (s_now, s_prev, r_now, r_prev)):
        return False
    # Coerce numpy comparison results → Python bool so callers don't get
    # numpy.bool_ leaking through the interpreter API.
    if direction == Operator.CROSSES_ABOVE:
        return bool(s_prev <= r_prev and s_now > r_now)
    if direction == Operator.CROSSES_BELOW:
        return bool(s_prev >= r_prev and s_now < r_now)
    return False


# ─────────────────────────────────────────────────────────────────────
# Engine signal evaluator
# ─────────────────────────────────────────────────────────────────────


def _engine_value(engines: EngineSignals, engine: EngineName) -> Any:
    """Pull the raw value off the EngineSignals struct."""
    mapping = {
        EngineName.ALPHA: engines.alpha,
        EngineName.MOOD: engines.mood,
        EngineName.REGIME: engines.regime,
    }
    return mapping.get(engine)


# ─────────────────────────────────────────────────────────────────────
# Main entry point — recursive Condition evaluator
# ─────────────────────────────────────────────────────────────────────


def evaluate_condition(cond: Condition, ctx: InterpreterContext) -> bool:
    """Evaluate a Condition recursively. Returns True/False."""
    kind = cond.kind

    if kind == ConditionKind.COMPOSITE_AND:
        return all(evaluate_condition(c, ctx) for c in (cond.children or []))

    if kind == ConditionKind.COMPOSITE_OR:
        return any(evaluate_condition(c, ctx) for c in (cond.children or []))

    if kind == ConditionKind.INDICATOR_COMPARE:
        left = ctx.indicator(cond.indicator)  # type: ignore[arg-type]
        # value may be a literal number OR an indicator-name reference
        right = cond.value
        if isinstance(right, str):
            # Indicator-to-indicator comparison (e.g. close > vwap)
            right = ctx.indicator(right)
            if isinstance(right, float) and math.isnan(right):
                return False
        return _scalar_compare(left, cond.op, right)  # type: ignore[arg-type]

    if kind == ConditionKind.INDICATOR_CROSS:
        left_series = ctx.indicator_series(cond.indicator)  # type: ignore[arg-type]
        right_series = ctx.indicator_series(cond.value)  # type: ignore[arg-type]
        return _cross_detect(left_series, right_series, cond.op)  # type: ignore[arg-type]

    if kind == ConditionKind.ENGINE_SIGNAL:
        engine_val = _engine_value(ctx.engines, cond.engine)  # type: ignore[arg-type]
        if engine_val is None:
            return False
        if isinstance(engine_val, str):
            return _string_compare(engine_val, cond.op, cond.value)  # type: ignore[arg-type]
        return _scalar_compare(float(engine_val), cond.op, cond.value)  # type: ignore[arg-type]

    return False


def evaluate_entry(strategy, ctx: InterpreterContext) -> bool:
    """Returns True if entry condition fires AND regime filter passes."""
    # Regime gate first — cheaper than running indicators.
    if strategy.regime_filter.value != "any":
        rg = ctx.engines.regime
        wanted = strategy.regime_filter.value.replace("_only", "")
        if rg != wanted:
            return False
    return evaluate_condition(strategy.entry, ctx)


def evaluate_exit(strategy, ctx: InterpreterContext) -> bool:
    """Returns True if exit condition fires. (Regime filter does NOT
    block exits — you always want to be able to leave a position.)"""
    return evaluate_condition(strategy.exit, ctx)


__all__ = [
    "EngineSignals",
    "InterpreterContext",
    "evaluate_condition",
    "evaluate_entry",
    "evaluate_exit",
]
