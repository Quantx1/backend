"""Regime resolver helpers — single source of truth for "what's the
regime at date X" across DSL backtest, signal generation, autopilot
supervisor, and the Copilot tool.

Why this exists: callers used to do their own ``supabase.table("regime_history")``
queries inline and each had a different fallback policy. Some failed open
(treating None as "any"), some failed closed (blocking trades silently),
and the morning-before-pre-market-runs gap behaved differently depending
on which surface was reading.

This module gives all callers the same chain:
    1. exact-date match in regime_history (if available)
    2. last known regime row ≤ requested date
    3. final fallback to ``sideways`` (the neutral default — won't satisfy
       bull_only / bear_only filters, won't get blocked by sideways_only)
"""

from .resolver import (
    DEFAULT_REGIME,
    resolve_regime_at,
    resolve_regime_history,
)

__all__ = ["DEFAULT_REGIME", "resolve_regime_at", "resolve_regime_history"]
