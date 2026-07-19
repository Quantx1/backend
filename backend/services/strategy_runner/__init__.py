"""Strategy Fan-Out Engine — PR-FAN.

Iterates every (user, live strategy, symbol-in-universe) tuple per tick
and emits signals when the strategy's DSL evaluates true. Applies a
user-level AI overlay (regime gate + VIX overlay + alpha rank filter)
before allowing any new entry.

This is what makes the "deploy 50 strategies in parallel" UX real.

Memory locks honoured:
  - LLM never gates trades. The DSL interpreter + AI overlay are pure
    math/rules — no LLM calls anywhere in this code path.
  - No fallbacks. If regime is unknown we DO NOT trade
    (overlay treats unknown as "do not enter").
  - Brand: Quant X (no codename leakage).

Scaling note:
  v1 fan-out is naive — O(users × strategies × symbols × bars). At 100
  users × 10 strategies × 100 symbols = 100K evaluations per tick.
  Sustainable on a small Railway instance. At 1000+ users we'll need
  precomputed indicators per (symbol, bar) and event-driven fan-out.
  Left as a v2 perf PR; documented in
  ``docs/superpowers/specs/2026-05-25-pr-fan-strategy-runner.md``.
"""

from .runner import (
    StrategyRunner,
    StrategyRunnerReport,
    StrategySignalEvent,
)
from .ai_overlay import (
    AIOverlayDecision,
    AIOverlaySettings,
    apply_ai_overlay,
    load_overlay_settings,
)
from .universe_expander import expand_universe

__all__ = [
    "StrategyRunner",
    "StrategyRunnerReport",
    "StrategySignalEvent",
    "AIOverlayDecision",
    "AIOverlaySettings",
    "apply_ai_overlay",
    "load_overlay_settings",
    "expand_universe",
]
