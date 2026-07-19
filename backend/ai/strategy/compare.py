"""
Strategy compare — head-to-head metrics across user strategies.

Closes the audit's one genuinely-missing capability (verified): there was no
way to diff/rank two-or-more user-authored strategies against each other.
Pure extraction over each strategy's stored ``last_backtest`` + the same
out-of-sample gate used for live promotion, so the comparison is apples-to-
apples with what actually decides go-live.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .evaluation import GateThresholds, evaluate_gate

# Metric → "higher is better?" — drives the per-metric winner highlight.
COMPARE_METRICS = {
    "oos_sharpe": True,
    "oos_consistency": True,
    "holdout_return_pct": True,
    "oos_worst_drawdown_pct": False,   # lower (shallower) is better
    "oos_trades": True,
}


def _extract(row: Dict[str, Any], thresholds: Optional[GateThresholds] = None) -> Dict[str, Any]:
    """One strategy's comparable card from its stored row."""
    lb = row.get("last_backtest") or {}
    oos = lb.get("out_of_sample") or {}
    gate = evaluate_gate(lb, thresholds)
    return {
        "id": row.get("id"),
        "name": row.get("name") or "Untitled",
        "status": row.get("status"),
        "has_backtest": bool(lb),
        "metrics": {
            "oos_sharpe": oos.get("oos_mean_sharpe"),
            "oos_consistency": oos.get("oos_consistency"),
            "holdout_return_pct": oos.get("holdout_return_pct"),
            "oos_worst_drawdown_pct": oos.get("oos_worst_drawdown_pct"),
            "oos_trades": oos.get("oos_trades"),
        },
        "gate_pass": gate.passed,
        "gate_failures": gate.failures,
    }


def compare_strategies(
    rows: List[Dict[str, Any]],
    thresholds: Optional[GateThresholds] = None,
) -> Dict[str, Any]:
    """Build the head-to-head table from raw strategy rows.

    Returns ``{strategies: [...], winners: {metric: id}, best_overall: id}``.
    ``winners`` flags the leader per metric (honest-empty for metrics no
    strategy has). ``best_overall`` = the gate-passing strategy with the
    highest OOS Sharpe, else the highest OOS Sharpe among all.
    """
    cards = [_extract(r, thresholds) for r in rows]

    winners: Dict[str, Optional[str]] = {}
    for metric, higher_better in COMPARE_METRICS.items():
        scored = [
            (c["id"], c["metrics"].get(metric))
            for c in cards
            if c["metrics"].get(metric) is not None
        ]
        if not scored:
            winners[metric] = None
            continue
        best = (max if higher_better else min)(scored, key=lambda kv: kv[1])
        winners[metric] = best[0]

    def _sharpe(c):
        return c["metrics"].get("oos_sharpe")

    passing = [c for c in cards if c["gate_pass"] and _sharpe(c) is not None]
    pool = passing or [c for c in cards if _sharpe(c) is not None]
    best_overall = max(pool, key=_sharpe)["id"] if pool else None

    return {"strategies": cards, "winners": winners, "best_overall": best_overall}


__all__ = ["compare_strategies", "COMPARE_METRICS"]
