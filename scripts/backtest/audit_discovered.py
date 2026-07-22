#!/usr/bin/env python3
"""Re-validate top discovered_strategies through the product gate (2026-07-22).

The discovery pipeline (2026-05-30 run) left 1,172 candidates with its own
harness scores — none promoted. This script takes the credible tail
(trade_count >= 20, sharpe > 0.25), dedupes by canonical DSL, extends each
candidate's lookback to the product maximum (730d) for a full-length test,
and runs the SAME walk-forward + evaluate_gate pipeline the product applies
to user strategies. PASS candidates are eligible for catalog promotion via
--apply.

Honesty flags recorded per result:
  * selection_overlap: the discovery run selected these on data overlapping
    the re-test window — the paper-trading stage remains mandatory before
    any live deployment.

Usage (from the worktree root):
    PYTHONPATH=. python3 scripts/backtest/audit_discovered.py --results artifacts/backtests/discovered_audit.jsonl
    PYTHONPATH=. python3 scripts/backtest/audit_discovered.py --results ... --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from typing import Any, Dict

from pydantic import ValidationError


def _load_done(path: str) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    try:
        for line in open(path):
            try:
                r = json.loads(line)
                done[r["key"]] = r
            except Exception:  # noqa: BLE001
                continue
    except FileNotFoundError:
        pass
    return done


def _dsl_key(dsl: Dict[str, Any]) -> str:
    d = {k: v for k, v in dsl.items() if k != "name"}
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def fetch_candidates(sb) -> Dict[str, Dict[str, Any]]:
    rows = (
        sb.table("discovered_strategies")
        .select("id,label,kind,score,sharpe,trade_count,dsl")
        .eq("status", "candidate")
        .gte("trade_count", 20).gt("sharpe", 0.25)
        .order("score", desc=True).limit(200).execute()
    ).data or []
    uniq: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        dsl = r["dsl"]
        if not isinstance(dsl, dict):
            continue
        key = _dsl_key(dsl)
        if key not in uniq:
            r["key"] = key
            uniq[key] = r
    return uniq


def audit_one(sb, cand: Dict[str, Any]) -> Dict[str, Any]:
    from backend.ai.strategy.backtest import run_universe_walk_forward, run_walk_forward
    from backend.ai.strategy.dsl import Strategy as DSLStrategy
    from backend.ai.strategy.evaluation import GateThresholds, evaluate_gate
    from backend.ai.strategy.indicators import MIN_LOOKBACK
    from backend.ai.strategy.timeframes import annualization_periods, tf_config
    from backend.api.strategies_routes import _fetch_tf_ohlcv
    from backend.core.config import settings
    from backend.data.market import get_market_data_provider

    out: Dict[str, Any] = {
        "key": cand["key"], "discovered_id": cand["id"], "label": cand["label"],
        "kind": cand.get("kind"), "discovery_sharpe": cand.get("sharpe"),
        "discovery_trades": cand.get("trade_count"), "selection_overlap": True,
    }
    dsl = dict(cand["dsl"])
    dsl["lookback_days"] = 730  # product max — full-length honest window
    try:
        strategy = DSLStrategy.model_validate(dsl)
    except ValidationError as exc:
        out["verdict"] = "INVALID_DSL"
        out["error"] = str(exc)[:300]
        return out

    tf_cfg = tf_config(strategy.timeframe)
    ppy = annualization_periods(strategy.timeframe)
    folds = settings.STRATEGY_GATE_FOLDS
    provider = get_market_data_provider()
    out["timeframe"] = strategy.timeframe.value
    out["universe"] = strategy.universe.value

    t0 = time.time()
    try:
        if strategy.universe.value != "single":
            from backend.services.strategy_runner.universe_expander import expand_universe
            symbols = expand_universe(strategy.universe.value)[: settings.STRATEGY_GATE_UNIVERSE_MAX_SYMBOLS]
            ohlcv_by_symbol: Dict[str, Any] = {}
            for sym in symbols:
                if sym.upper().startswith("TATAMOTORS"):
                    continue  # delisted on Yahoo
                try:
                    df = _fetch_tf_ohlcv(provider, sym, tf_cfg=tf_cfg, lookback_days=730)
                except Exception:  # noqa: BLE001
                    continue
                if df is not None and len(df) >= MIN_LOOKBACK + 10:
                    ohlcv_by_symbol[sym] = df
            if not ohlcv_by_symbol:
                out["verdict"] = "NO_DATA"
                return out
            result = run_universe_walk_forward(
                strategy, ohlcv_by_symbol,
                universe=strategy.universe.value, folds=folds,
                initial_capital=500_000.0, periods_per_year=ppy,
            )
            out["symbols_tested"] = len(ohlcv_by_symbol)
        else:
            sym = strategy.symbol or "NIFTY"
            ohlcv = _fetch_tf_ohlcv(provider, sym, tf_cfg=tf_cfg, lookback_days=730)
            if ohlcv is None or len(ohlcv) < MIN_LOOKBACK + 10:
                out["verdict"] = "NO_DATA"
                return out
            result = run_walk_forward(
                strategy, ohlcv, symbol=sym, folds=folds,
                initial_capital=500_000.0, periods_per_year=ppy,
            )
    except Exception as exc:  # noqa: BLE001
        out["verdict"] = "ERROR"
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return out

    summary = result.to_summary_dict()
    gate = evaluate_gate(summary, GateThresholds())
    oos = summary.get("out_of_sample") or {}
    out.update({
        "verdict": "PASS" if gate.passed else "FAIL",
        "gate_failures": list(getattr(gate, "failures", []) or []),
        "elapsed_s": round(time.time() - t0, 1),
        "dsl": dsl,
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
    """Insert PASS candidates into strategy_catalog with their REAL numbers."""
    inserted = 0
    for key, r in results.items():
        if r.get("verdict") != "PASS":
            continue
        ins = r.get("in_sample") or {}
        slug = f"discovered-{r['label'].split(' ')[0].lower()}"[:60]
        existing = sb.table("strategy_catalog").select("id").eq("slug", slug).execute().data
        if existing:
            continue
        sb.table("strategy_catalog").insert({
            "slug": slug,
            "name": r["label"].split(" · ")[0],
            "category": "mean_reversion" if "MeanRev" in r["label"] else "momentum",
            "segment": "EQUITY",
            "tier_required": "pro",
            "risk_level": "medium",
            "dsl": r["dsl"],
            "is_active": True,
            "backtest_total_return": ins.get("total_return_pct"),
            "backtest_win_rate": ins.get("win_rate"),
            "backtest_sharpe": ins.get("sharpe_ratio"),
            "backtest_max_drawdown": ins.get("max_drawdown_pct"),
            "backtest_total_trades": ins.get("total_trades"),
        }).execute()
        sb.table("discovered_strategies").update({"status": "promoted"}).eq("id", r["discovered_id"]).execute()
        inserted += 1
    print(f"inserted {inserted} gate-passing discovered strategies")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/backtests/discovered_audit.jsonl")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from backend.core.database import get_supabase_admin
    sb = get_supabase_admin()

    done = _load_done(args.results)
    if args.apply:
        apply_results(sb, done)
        return

    uniq = fetch_candidates(sb)
    todo = [c for k, c in uniq.items() if k not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"unique candidates: {len(uniq)} · already done: {len(uniq) - len(todo)} · to run: {len(todo)}")
    with open(args.results, "a") as fh:
        for i, cand in enumerate(todo, 1):
            try:
                r = audit_one(sb, cand)
            except Exception:  # noqa: BLE001
                r = {"key": cand["key"], "verdict": "ERROR", "error": traceback.format_exc()[-300:]}
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] {cand['label'][:50]}: {r.get('verdict')} "
                  f"({r.get('elapsed_s', '-')}s) {';'.join(r.get('gate_failures', [])[:3])}")

    from collections import Counter
    allr = _load_done(args.results)
    print("\n== VERDICTS ==", Counter(v.get("verdict") for v in allr.values()))


if __name__ == "__main__":
    main()
