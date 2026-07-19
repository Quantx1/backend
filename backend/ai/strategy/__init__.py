"""
Strategy Layer — PR-D/E/F per v2 design spec § 7.

The bridge between Quant X Engines (ML models that produce signals) and
user-deployable trading rules. Three components:

  - dsl.py        — Pydantic models for the constrained JSON DSL
  - indicators.py — closed-set technical indicator registry (no eval)
  - interpreter.py — evaluates a Condition node against a price bar +
                     engine context, returns True/False for entry/exit
  - registry.py    — strategy lifecycle (draft/backtest/paper/live/paused)
                     persistence + state machine

Locked design constraints (do not relax):
  - Indicators are a CLOSED SET — no arbitrary code execution
  - Engine references whitelist 7 names — Vision · Alpha · Verdict · Mood ·
    Regime · Pulse · Horizon (AutoPilot is the executor, not a signal source)
  - DSL must round-trip JSON ↔ Pydantic ↔ JSON cleanly so it stores in
    Supabase JSONB without serialization gymnastics
"""

from .dsl import (
    Strategy,
    Condition,
    PositionSize,
    StrategyMode,
    Timeframe,
    Universe,
    RegimeFilter,
    ConditionKind,
    Operator,
    EngineName,
    INDICATOR_REGISTRY,
)
from .indicators import compute_indicator, list_indicators
from .interpreter import (
    InterpreterContext,
    EngineSignals,
    evaluate_condition,
    evaluate_entry,
    evaluate_exit,
)

__all__ = [
    "Strategy",
    "Condition",
    "PositionSize",
    "StrategyMode",
    "Timeframe",
    "Universe",
    "RegimeFilter",
    "ConditionKind",
    "Operator",
    "EngineName",
    "INDICATOR_REGISTRY",
    "compute_indicator",
    "list_indicators",
    "InterpreterContext",
    "EngineSignals",
    "evaluate_condition",
    "evaluate_entry",
    "evaluate_exit",
]
