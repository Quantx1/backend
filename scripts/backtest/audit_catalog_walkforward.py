#!/usr/bin/env python3
"""Catalog-wide walk-forward audit (2026-07-21) — every template, real numbers.

Why: the 49 stored backtest_* numbers in strategy_catalog were SEEDED
synthetic targets (scripts/backtest/backtest_options_strategies.py reads
them as calibration targets), and the other 51 templates had no stored
backtest at all. This script runs EVERY template through the SAME
walk-forward + gate pipeline the product uses for user strategies
(strategies_routes.py POST /{id}/backtest + evaluate_gate), then:

  --apply:
    * PASS  → write the REAL computed metrics into the backtest_* columns
    * FAIL  → is_active=false + NULL the fabricated backtest_* numbers
    * NO/INVALID DSL → is_active=false (unverifiable = out of the catalog)

Honesty notes recorded per result:
  * OPTIONS results use the synthetic Black-Scholes resolver (no real
    historical chains) — flagged `synthetic_options: true`.
  * Intraday timeframes run on yfinance-capped windows (5-30m ≈ 60d) —
    flagged `thin_intraday_window: true`.

Resume-safe: appends JSONL per template; reruns skip already-done slugs.

Usage (from the worktree root, PYTHONPATH=.):
    python3 scripts/backtest/audit_catalog_walkforward.py --results /tmp/catalog_audit.jsonl
    python3 scripts/backtest/audit_catalog_walkforward.py --results /tmp/catalog_audit.jsonl --apply
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from typing import Any, Dict, Optional

from pydantic import ValidationError


def _load_done(path: str) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    try:
        for line in open(path):
            try:
                r = json.loads(line)
                done[r["slug"]] = r
            except Exception:  # noqa: BLE001
                continue
    except FileNotFoundError:
        pass
    return done


def audit_one(sb, tpl: Dict[str, Any]) -> Dict[str, Any]:
    """Replicates strategies_routes.backtest_strategy dispatch for one template."""
    from backend.ai.strategy.backtest import (
        run_options_walk_forward,
        run_universe_walk_forward,
        run_walk_forward,
    )
    from backend.ai.strategy.dsl import Strategy as DSLStrategy
    from backend.ai.strategy.evaluation import GateThresholds, evaluate_gate
    from backend.ai.strategy.indicators import MIN_LOOKBACK
    from backend.ai.strategy.timeframes import annualization_periods, tf_config
    from backend.api.strategies_routes import (
        _date_of,
        _fetch_tf_ohlcv,
        _maybe_load_engine_signals,
        _regime_coverage_range,
        _strategy_uses_regime,
    )
    from backend.core.config import settings
    from backend.data.market import get_market_data_provider

    out: Dict[str, Any] = {
        "slug": tpl["slug"], "name": tpl["name"], "category": tpl.get("category"),
        "segment": tpl.get("segment"), "tier": tpl.get("tier_required"),
        "was_active": tpl.get("is_active"),
    }
    dsl = tpl.get("dsl")
    if not dsl:
        out["verdict"] = "NO_DSL"
        return out
    try:
        strategy = DSLStrategy.model_validate(dsl)
    except ValidationError as exc:
        out["verdict"] = "INVALID_DSL"
        out["error"] = str(exc)[:300]
        return out

    tf_cfg = tf_config(strategy.timeframe)
    ppy = annualization_periods(strategy.timeframe)
    lookback = strategy.lookback_days
    folds = settings.STRATEGY_GATE_FOLDS
    provider = get_market_data_provider()

    is_options = strategy.instrument_segment.value == "OPTIONS"
    is_universe = (not is_options) and strategy.universe.value != "single"
    uses_regime = _strategy_uses_regime(strategy)
    out["timeframe"] = strategy.timeframe.value
    out["universe"] = strategy.universe.value
    out["synthetic_options"] = bool(is_options)
    out["thin_intraday_window"] = strategy.timeframe.value in ("1m", "5m", "15m", "30m")

    regime_coverage: Optional[float] = None
    t0 = time.time()
    try:
        if is_universe:
            from backend.services.strategy_runner.universe_expander import expand_universe
            symbols = expand_universe(strategy.universe.value)[: settings.STRATEGY_GATE_UNIVERSE_MAX_SYMBOLS]
            ohlcv_by_symbol: Dict[str, Any] = {}
            for sym in symbols:
                if sym.upper().startswith("TATAMOTORS"):
                    continue  # delisted on Yahoo
                try:
                    df = _fetch_tf_ohlcv(provider, sym, tf_cfg=tf_cfg, lookback_days=lookback)
                except Exception:  # noqa: BLE001
                    continue
                if df is not None and len(df) >= MIN_LOOKBACK + 10:
                    ohlcv_by_symbol[sym] = df
            if not ohlcv_by_symbol:
                out["verdict"] = "NO_DATA"
                return out
            engine_signals_by_symbol = (
                {s: _maybe_load_engine_signals(sb, strategy, d) for s, d in ohlcv_by_symbol.items()}
                if uses_regime else None
            )
            if uses_regime:
                starts = [d.index[0] for d in ohlcv_by_symbol.values()]
                ends = [d.index[-1] for d in ohlcv_by_symbol.values()]
                regime_coverage = _regime_coverage_range(sb, _date_of(min(starts)), _date_of(max(ends)))
            result = run_universe_walk_forward(
                strategy, ohlcv_by_symbol,
                universe=strategy.universe.value, folds=folds,
                initial_capital=500_000.0, periods_per_year=ppy,
                engine_signals_by_symbol=engine_signals_by_symbol,
            )
            out["symbols_tested"] = len(ohlcv_by_symbol)
        else:
            sym = strategy.symbol or "NIFTY"
            ohlcv = _fetch_tf_ohlcv(provider, sym, tf_cfg=tf_cfg, lookback_days=lookback)
            if ohlcv is None or len(ohlcv) < MIN_LOOKBACK + 10:
                out["verdict"] = "NO_DATA"
                out["bars"] = 0 if ohlcv is None else len(ohlcv)
                return out
            engine_signals_by_date = _maybe_load_engine_signals(sb, strategy, ohlcv)
            if uses_regime:
                regime_coverage = _regime_coverage_range(sb, _date_of(ohlcv.index[0]), _date_of(ohlcv.index[-1]))
            wf = run_options_walk_forward if is_options else run_walk_forward
            result = wf(
                strategy, ohlcv, symbol=sym, folds=folds,
                initial_capital=500_000.0,
                engine_signals_by_date=engine_signals_by_date,
                periods_per_year=ppy,
            )
    except Exception as exc:  # noqa: BLE001
        out["verdict"] = "ERROR"
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return out

    summary = result.to_summary_dict()
    if regime_coverage is not None:
        summary["regime"] = {"used": True, "coverage": round(regime_coverage, 4)}
    gate = evaluate_gate(summary, GateThresholds())

    oos = summary.get("out_of_sample") or {}
    out.update({
        "verdict": "PASS" if gate.passed else "FAIL",
        "gate_failures": list(getattr(gate, "failures", []) or []),
        "elapsed_s": round(time.time() - t0, 1),
        "in_sample": {
            k: summary.get(k) for k in (
                "total_trades", "win_rate", "total_return_pct",
                "max_drawdown_pct", "sharpe_ratio", "profit_factor", "avg_hold_days",
            )
        },
        "oos": {
            k: oos.get(k) for k in (
                "oos_trades", "oos_consistency", "oos_mean_sharpe",
                "oos_worst_drawdown_pct", "holdout_return_pct", "holdout_sharpe",
            )
        },
    })
    return out


def apply_results(sb, results: Dict[str, Dict[str, Any]]) -> None:
    """Write the audit's verdicts back to strategy_catalog."""
    for slug, r in results.items():
        verdict = r.get("verdict")
        if verdict == "PASS":
            ins = r.get("in_sample") or {}
            # Real computed numbers replace the seeded ones. CAGR from total
            # return over the tested window is not directly available per
            # timeframe — store NULL rather than approximate dishonestly.
            sb.table("strategy_catalog").update({
                "is_active": True,
                "backtest_total_return": ins.get("total_return_pct"),
                "backtest_cagr": None,
                "backtest_win_rate": ins.get("win_rate"),
                "backtest_sharpe": ins.get("sharpe_ratio"),
                "backtest_max_drawdown": ins.get("max_drawdown_pct"),
                "backtest_total_trades": ins.get("total_trades"),
            }).eq("slug", slug).execute()
        elif verdict in ("FAIL", "NO_DSL", "INVALID_DSL", "NO_DATA", "ERROR"):
            sb.table("strategy_catalog").update({
                "is_active": False,
                "backtest_total_return": None,
                "backtest_cagr": None,
                "backtest_win_rate": None,
                "backtest_sharpe": None,
                "backtest_max_drawdown": None,
                "backtest_total_trades": None,
            }).eq("slug", slug).execute()
    print(f"applied {len(results)} verdicts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/tmp/catalog_audit.jsonl")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma-separated slugs")
    args = ap.parse_args()

    from backend.core.database import get_supabase_admin
    sb = get_supabase_admin()

    done = _load_done(args.results)
    if args.apply:
        apply_results(sb, done)
        return

    rows = (
        sb.table("strategy_catalog")
        .select("id,slug,name,category,segment,tier_required,risk_level,dsl,is_active")
        .order("slug").execute()
    ).data or []
    if args.only:
        want = set(args.only.split(","))
        rows = [r for r in rows if r["slug"] in want]
    if args.limit:
        rows = rows[: args.limit]

    todo = [r for r in rows if r["slug"] not in done]
    print(f"templates: {len(rows)} · already done: {len(rows) - len(todo)} · to run: {len(todo)}")
    with open(args.results, "a") as fh:
        for i, tpl in enumerate(todo, 1):
            try:
                r = audit_one(sb, tpl)
            except Exception as exc:  # noqa: BLE001
                r = {"slug": tpl["slug"], "verdict": "ERROR", "error": traceback.format_exc()[-300:]}
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] {tpl['slug']}: {r.get('verdict')} "
                  f"({r.get('elapsed_s', '-')}s) {';'.join(r.get('gate_failures', [])[:2])}")

    allr = _load_done(args.results)
    from collections import Counter
    print("\n== VERDICTS ==", Counter(v.get("verdict") for v in allr.values()))


if __name__ == "__main__":
    main()
