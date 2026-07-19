"""Outcome-trained per-strategy models — PR-DEPTH.

Pattern adapted from aaryansinha16/AI-trader's scripts/train_outcome_models.py.

The Big Idea:
  Don't train ML models on "will NIFTY go up 15 mins later?" — that's a
  synthetic label that ignores execution friction. Train models on
  "did THIS strategy actually win when it fired in this market state?"
  using REAL closed-trade outcomes from strategy_outcomes.

Per-template model: one LightGBM per template_slug (RSI Mean Reversion,
EMA Golden Cross, etc.). Trained when ≥30 closed trades exist for that
slug. Used at runtime as an additional filter in the AI overlay:

    if outcome_model[template_slug].predict_proba(features)[1] < 0.55:
        skip entry

This adds real per-strategy discrimination that pure rule-based systems
can't capture (path-dependent stuff like SL pickoff, regime drift).

Memory locks honoured: supervised ML, no LLM, no RL.
"""

from .trainer import (
    OutcomeModelConfig,
    OutcomeModelTrainer,
    OutcomeModelPredictor,
    train_all_strategy_outcome_models,
)
from .features import build_outcome_features

__all__ = [
    "OutcomeModelConfig",
    "OutcomeModelTrainer",
    "OutcomeModelPredictor",
    "train_all_strategy_outcome_models",
    "build_outcome_features",
]
