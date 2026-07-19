"""AI Backtesting Assistant — explain a stored backtest + suggest improvements.

The backtest analogue of `why_moving` / `trade_review`: the strategy's persisted
``last_backtest`` summary (REAL numbers from the walk-forward runner) is scored
against the SAME deterministic gate thresholds that block live deployment
(`ai/strategy/evaluation.py`), turned into plain-English ``drivers`` bullets +
concrete ``suggestions`` (ALWAYS returned, 0 tokens), then OPTIONALLY narrated
over by the grounded agent (free-first model, cached per metrics-hash/day) only
when ``use_llm``.

The LLM never decides the verdict — `evaluate_gate` does. The narrative only
explains numbers that were computed deterministically. Honest-empty: no stored
backtest → empty drivers/suggestions, no narrative.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from typing import Any, Dict, List, Optional

from ...ai.strategy.evaluation import GateResult, GateThresholds, evaluate_gate

logger = logging.getLogger(__name__)


def _thresholds() -> GateThresholds:
    """The SAME env-tunable bars `transition_strategy` enforces — so the
    explanation always matches what the promotion gate will actually say.
    Falls back to the conservative defaults when settings can't load."""
    try:
        from ...core.config import settings
        return GateThresholds(
            min_oos_sharpe=settings.STRATEGY_GATE_MIN_OOS_SHARPE,
            min_trades=settings.STRATEGY_GATE_MIN_TRADES,
            max_drawdown_pct=settings.STRATEGY_GATE_MAX_DRAWDOWN_PCT,
            min_consistency=settings.STRATEGY_GATE_MIN_CONSISTENCY,
            require_holdout_positive=settings.STRATEGY_GATE_REQUIRE_HOLDOUT_POSITIVE,
            min_symbol_breadth=settings.STRATEGY_GATE_MIN_SYMBOL_BREADTH,
            min_regime_coverage=settings.STRATEGY_GATE_MIN_REGIME_COVERAGE,
        )
    except Exception:  # noqa: BLE001 — pure fallback for tests / partial envs
        return GateThresholds()


def _metrics_hash(metrics: Dict[str, Any]) -> str:
    """Stable 16-char content hash of the metrics dict — same backtest, same
    cache key; any re-run with different numbers gets a fresh narrative."""
    payload = json.dumps(metrics, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _drivers(
    metrics: Dict[str, Any],
    gate: GateResult,
    th: Optional[GateThresholds] = None,
) -> List[str]:
    """Deterministic plain bullet read of the backtest — always available,
    0 tokens. Headline verdict comes from the REAL promotion gate, each metric
    bullet states the number against the actual threshold it must clear."""
    th = th or GateThresholds()
    out: List[str] = []
    if not metrics:
        return out

    # Headline verdict — the gate decides, never the LLM.
    if gate.passed:
        out.append(
            "Verdict: PASSES the out-of-sample quality gate — eligible for deployment."
        )
    else:
        n = len(gate.failures)
        out.append(
            f"Verdict: FAILS the quality gate ({n} issue{'s' if n != 1 else ''}) "
            f"— not ready for live money."
        )

    oos = metrics.get("out_of_sample") or {}
    if not isinstance(oos, dict) or not oos:
        out.append(
            "No walk-forward out-of-sample block — in-sample numbers alone "
            "can't qualify a strategy for deployment."
        )
    else:
        sharpe = float(oos.get("oos_mean_sharpe", 0.0) or 0.0)
        out.append(
            f"Out-of-sample Sharpe {sharpe:.2f} — "
            + (f"clears the {th.min_oos_sharpe:g} gate."
               if sharpe >= th.min_oos_sharpe
               else f"FAILS the {th.min_oos_sharpe:g} gate."))

        dd = float(oos.get("oos_worst_drawdown_pct", 0.0) or 0.0)
        out.append(
            f"Max drawdown {dd:.1f}% — "
            + (f"within the {th.max_drawdown_pct:g}% ceiling."
               if dd <= th.max_drawdown_pct
               else f"FAILS the {th.max_drawdown_pct:g}% gate."))

        trades = int(oos.get("oos_trades", 0) or 0)
        if trades < th.min_trades:
            out.append(
                f"Only {trades} out-of-sample trades — below the "
                f"{th.min_trades}-trade minimum (low confidence)."
            )
        else:
            out.append(
                f"{trades} out-of-sample trades — clears the "
                f"{th.min_trades}-trade minimum."
            )

        cons = float(oos.get("oos_consistency", 0.0) or 0.0)
        profitable, n_folds = oos.get("oos_folds_profitable"), oos.get("n_folds")
        folds_tail = (
            f" ({profitable}/{n_folds})"
            if profitable is not None and n_folds is not None else ""
        )
        out.append(
            f"Profitable in {cons:.0%} of walk-forward windows{folds_tail} — "
            + ("clears" if cons >= th.min_consistency else "FAILS")
            + f" the {th.min_consistency:.0%} bar."
        )

        hr = oos.get("holdout_return_pct")
        if hr is not None:
            hr = float(hr or 0.0)
            out.append(
                f"Holdout window {hr:+.1f}% — "
                + ("holds up on data it was never selected against."
                   if hr >= 0
                   else "lost money on data it was never selected against."))

        symbols_tested = oos.get("symbols_tested")
        if isinstance(symbols_tested, int) and symbols_tested > 1:
            breadth = float(oos.get("breadth", 0.0) or 0.0)
            out.append(
                f"Profitable on {breadth:.0%} of the {symbols_tested} symbols tested — "
                + ("clears" if breadth >= th.min_symbol_breadth else "FAILS")
                + f" the {th.min_symbol_breadth:.0%} breadth bar."
            )

    regime = metrics.get("regime")
    if isinstance(regime, dict) and regime.get("used"):
        coverage = float(regime.get("coverage", 0.0) or 0.0)
        if coverage < th.min_regime_coverage:
            out.append(
                f"Regime data covers only {coverage:.0%} of the window "
                f"(< {th.min_regime_coverage:.0%}) — the regime-gated result "
                f"can't be trusted."
            )

    # In-sample context — informational, never the verdict.
    wr, n_trades = metrics.get("win_rate"), metrics.get("total_trades")
    if wr is not None and n_trades is not None:
        ret = metrics.get("total_return_pct")
        ret_tail = f", {float(ret):+.1f}% total return" if ret is not None else ""
        out.append(
            f"In-sample: {float(wr) * 100:.0f}% win rate over "
            f"{int(n_trades)} trades{ret_tail}."
        )

    return out


def _suggestions(
    metrics: Dict[str, Any],
    gate: GateResult,
    dsl: Optional[Dict[str, Any]] = None,
    th: Optional[GateThresholds] = None,
) -> List[str]:
    """Deterministic, concrete improvement suggestions mapped from exactly
    which gates failed (DSL-aware where it sharpens the advice). Empty when
    the gate passes — nothing to fix. 0 tokens."""
    th = th or GateThresholds()
    dsl = dsl or {}
    if not metrics:
        return []

    oos = metrics.get("out_of_sample") or {}
    if not isinstance(oos, dict) or not oos:
        return [
            "Re-run the backtest — only a walk-forward (out-of-sample) result "
            "can qualify the strategy for deployment."
        ]
    if gate.passed:
        return []

    out: List[str] = []
    regime_filter = str(dsl.get("regime_filter") or "any")

    trades = int(oos.get("oos_trades", 0) or 0)
    if trades < th.min_trades:
        if str(dsl.get("universe") or "single") == "single":
            out.append(
                "Widen the universe — backtest across NIFTY 50 instead of a "
                "single symbol to collect more out-of-sample trades."
            )
        else:
            out.append(
                "Extend the lookback window so the walk-forward collects more "
                "out-of-sample trades."
            )

    sharpe = float(oos.get("oos_mean_sharpe", 0.0) or 0.0)
    if sharpe < th.min_oos_sharpe:
        if regime_filter == "any":
            out.append(
                "Add a regime filter so the strategy only trades favourable "
                "markets — the edge per trade is too thin as-is."
            )
        else:
            out.append(
                "Tighten the entry conditions (add a confirming indicator) — "
                "the edge per trade is too thin even with the regime filter."
            )

    dd = float(oos.get("oos_worst_drawdown_pct", 0.0) or 0.0)
    if dd > th.max_drawdown_pct:
        if not dsl.get("stop_loss_pct"):
            out.append(
                f"Add a hard stop loss (stop_loss_pct) — the worst drawdown "
                f"breaches the {th.max_drawdown_pct:g}% ceiling."
            )
        else:
            out.append(
                f"Tighten stops or cut position size — the worst drawdown "
                f"breaches the {th.max_drawdown_pct:g}% ceiling."
            )

    cons = float(oos.get("oos_consistency", 0.0) or 0.0)
    if cons < th.min_consistency:
        if regime_filter == "any":
            out.append(
                "Add a regime filter — profits cluster in too few time "
                "windows, which suggests the edge only works in one market phase."
            )
        else:
            out.append(
                "Loosen the strategy's dependence on one market phase — it is "
                "profitable in too few walk-forward windows."
            )

    if th.require_holdout_positive and float(oos.get("holdout_return_pct", 0.0) or 0.0) < 0:
        out.append(
            "Re-validate on recent data — the most-recent holdout window lost "
            "money, so the edge may have decayed."
        )

    symbols_tested = oos.get("symbols_tested")
    if isinstance(symbols_tested, int) and symbols_tested > 1:
        if float(oos.get("breadth", 0.0) or 0.0) < th.min_symbol_breadth:
            out.append(
                "Drop symbol-specific tuning — the edge works on too few of "
                "the symbols tested, which looks cherry-picked."
            )

    regime = metrics.get("regime")
    if isinstance(regime, dict) and regime.get("used"):
        if float(regime.get("coverage", 0.0) or 0.0) < th.min_regime_coverage:
            out.append(
                "Backfill regime history for the backtest window or remove "
                "the regime filter — most of the window ran on default regime data."
            )

    # Two failures can map to the same fix — dedupe, keep order.
    return list(dict.fromkeys(out))


def explain_backtest(
    metrics: Dict[str, Any],
    dsl: Optional[Dict[str, Any]] = None,
    *,
    use_llm: bool = False,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """{drivers, suggestions, narrative} for a stored backtest summary.

    Drivers + suggestions are deterministic (the REAL promotion-gate math
    decides — the LLM never gates). Narrative is the grounded agent over the
    same facts, cached per (metrics hash, day), only when ``use_llm``.
    Honest-empty when there is no backtest to explain."""
    metrics = metrics or {}
    if not metrics:
        return {"drivers": [], "suggestions": [], "narrative": None}

    th = _thresholds()
    gate = evaluate_gate(metrics, th)
    drivers = _drivers(metrics, gate, th)
    suggestions = _suggestions(metrics, gate, dsl, th)

    narrative: Optional[str] = None
    if use_llm and drivers:
        # Ground the narrative in exactly the numbers the gate scored — strip
        # heavyweight keys defensively (full payloads carry trade lists).
        facts: Dict[str, Any] = {
            "backtest": {k: v for k, v in metrics.items()
                         if k not in ("trades", "equity_curve")},
            "gate": {"passed": gate.passed, "failures": gate.failures},
            "deterministic_suggestions": suggestions,
        }
        if dsl:
            strat = {k: dsl.get(k) for k in
                     ("name", "timeframe", "universe", "symbol", "regime_filter",
                      "stop_loss_pct", "take_profit_pct", "position_size")
                     if dsl.get(k) is not None}
            if strat:
                facts["strategy"] = strat
        from ...ai.agents.grounded import grounded_reason
        narrative = grounded_reason(
            facts,
            "Explain this strategy backtest to its creator: how good is it "
            "really, what the deployment-gate verdict means, and the single "
            "highest-leverage improvement to make next. Be specific to the "
            "numbers given.",
            cache_key=f"btexplain:{_metrics_hash(metrics)}:{date.today().isoformat()}",
            role="responder",
            user_id=user_id,
        )

    return {"drivers": drivers, "suggestions": suggestions, "narrative": narrative}
