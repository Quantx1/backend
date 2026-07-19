"""Strategy promotion gate — the out-of-sample quality barrier that blocks
overfit strategies from reaching live money.

This is the enforcement point for the "only real, usable strategies execute
live" rule. A strategy can only be promoted to ``live`` (and, by config,
``paper``) once its stored ``last_backtest`` carries a walk-forward
``out_of_sample`` block that clears every threshold below.

Why out-of-sample, not in-sample:
    A rule strategy has no fitted parameters, so the overfit vector is
    *selection* — an LLM (or a user) generates many candidates and keeps the
    one whose full-history backtest looks best. The defence is to require the
    strategy to hold up across *multiple time windows* (consistency) and on
    the most-recent *holdout* window it was never selected against. In-sample
    Sharpe is exactly the number that's easiest to overfit, so the gate
    deliberately ignores it and scores the OOS block instead.

Pure functions only — no Supabase, no network. The API layer calls
``evaluate_gate(row["last_backtest"], thresholds)`` before allowing a
``→ live`` transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["GateThresholds", "GateResult", "evaluate_gate"]


@dataclass(frozen=True)
class GateThresholds:
    """Tunable bars a strategy must clear on its out-of-sample results.

    Defaults are deliberately conservative — a strategy that can't clear
    these should not be touching real money. Override per-deployment via
    config (env) so we can tune without a code change.
    """

    min_oos_sharpe: float = 0.5          # annualised, averaged across folds
    min_trades: int = 20                 # OOS trade count — stats need a sample
    max_drawdown_pct: float = 35.0       # worst per-fold drawdown ceiling
    min_consistency: float = 0.5         # fraction of folds that must be profitable
    require_holdout_positive: bool = True  # the most-recent window must not lose
    min_symbol_breadth: float = 0.5      # universe only: frac of symbols profitable
    min_regime_coverage: float = 0.8     # regime strategies only: frac of window on REAL regime


@dataclass
class GateResult:
    """Outcome of the gate. ``failures`` is a list of human-readable reasons
    suitable for returning to the UI (one per breached threshold)."""

    passed: bool
    failures: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def evaluate_gate(
    last_backtest: Optional[Dict[str, Any]],
    thresholds: Optional[GateThresholds] = None,
) -> GateResult:
    """Decide whether a strategy may be promoted to live, from its stored
    ``last_backtest`` summary.

    Args:
        last_backtest: the ``user_strategies.last_backtest`` JSON (or None if
            the strategy was never backtested). Must carry an
            ``out_of_sample`` block (from ``run_walk_forward``).
        thresholds: gate bars; defaults to :class:`GateThresholds`.

    Returns:
        GateResult(passed, failures, metrics).
    """
    th = thresholds or GateThresholds()

    if not last_backtest:
        return GateResult(
            passed=False,
            failures=["No backtest on record — run a backtest before deploying live."],
        )

    oos = last_backtest.get("out_of_sample")
    if not isinstance(oos, dict) or not oos:
        return GateResult(
            passed=False,
            failures=[
                "No out-of-sample (walk-forward) result — re-run the backtest. "
                "In-sample numbers cannot gate live money.",
            ],
        )

    failures: List[str] = []

    oos_trades = int(oos.get("oos_trades", 0) or 0)
    if oos_trades < th.min_trades:
        failures.append(
            f"Too few out-of-sample trades: {oos_trades} < {th.min_trades} "
            f"(not a statistically meaningful sample).",
        )

    oos_sharpe = float(oos.get("oos_mean_sharpe", 0.0) or 0.0)
    if oos_sharpe < th.min_oos_sharpe:
        failures.append(
            f"Out-of-sample Sharpe too low: {oos_sharpe:.2f} < {th.min_oos_sharpe:.2f}.",
        )

    worst_dd = float(oos.get("oos_worst_drawdown_pct", 0.0) or 0.0)
    if worst_dd > th.max_drawdown_pct:
        failures.append(
            f"Out-of-sample drawdown too deep: {worst_dd:.1f}% > {th.max_drawdown_pct:.1f}%.",
        )

    consistency = float(oos.get("oos_consistency", 0.0) or 0.0)
    if consistency < th.min_consistency:
        profitable = oos.get("oos_folds_profitable")
        n_folds = oos.get("n_folds")
        detail = (
            f" ({profitable}/{n_folds} windows profitable)"
            if profitable is not None and n_folds is not None
            else ""
        )
        failures.append(
            f"Inconsistent across time windows: {consistency:.0%} profitable "
            f"< {th.min_consistency:.0%} required{detail} — looks overfit to one regime.",
        )

    if th.require_holdout_positive:
        holdout = float(oos.get("holdout_return_pct", 0.0) or 0.0)
        if holdout < 0:
            failures.append(
                f"Most-recent (holdout) window lost money: {holdout:+.1f}% — "
                f"the strategy does not hold up on data it was never selected against.",
            )

    # Universe strategies: require breadth across symbols (only when >1 tested).
    # Blocks "great on one cherry-picked symbol, loses on the rest".
    symbols_tested = oos.get("symbols_tested")
    if isinstance(symbols_tested, int) and symbols_tested > 1:
        breadth = float(oos.get("breadth", 0.0) or 0.0)
        if breadth < th.min_symbol_breadth:
            profitable = oos.get("symbols_profitable")
            detail = f" ({profitable}/{symbols_tested} symbols profitable)" if profitable is not None else ""
            failures.append(
                f"Works on too few symbols: {breadth:.0%} < {th.min_symbol_breadth:.0%} "
                f"required{detail} — looks cherry-picked to one symbol, not a real edge.",
            )

    # Regime strategies: fail-closed when the backtest ran mostly on the
    # DEFAULT regime (because regime_history didn't cover the window). A
    # regime-gated strategy validated on fake regime is not trustworthy.
    regime = last_backtest.get("regime")
    if isinstance(regime, dict) and regime.get("used"):
        coverage = float(regime.get("coverage", 0.0) or 0.0)
        if coverage < th.min_regime_coverage:
            failures.append(
                f"Regime data covers only {coverage:.0%} of the backtest window "
                f"(< {th.min_regime_coverage:.0%}) — this regime-gated strategy ran mostly on the "
                f"default regime, so the result can't be trusted. Backfill regime_history for the "
                f"window or remove the regime filter.",
            )

    return GateResult(
        passed=not failures,
        failures=failures,
        metrics={
            "oos_trades": oos_trades,
            "oos_mean_sharpe": oos_sharpe,
            "oos_worst_drawdown_pct": worst_dd,
            "oos_consistency": consistency,
            "holdout_return_pct": float(oos.get("holdout_return_pct", 0.0) or 0.0),
            "symbols_tested": oos.get("symbols_tested"),
            "breadth": oos.get("breadth"),
        },
    )
