#!/usr/bin/env python3
"""Large discovery sweep (2026-07-22) — all asset classes, styles, timeframes.

Drives backend.ai.strategy_discovery.run_discovery across a matrix of
(kind, universe) with GA + walk-forward, persisting candidates into
discovered_strategies exactly like the nightly cron — but at a bigger
budget and with the enhanced panic-reversion-weighted search space.

Every candidate this writes is later re-validated through the STRICT
product gate by audit_discovered.py before any promotion. Nothing here
reaches the catalog on the discovery engine's internal score alone.

Honesty per asset class:
  * equity (swing/position) : daily bars, real prices — promotable.
  * intraday (5m/15m)       : thin yfinance window (≤60d), OOS sample is
                              small — flagged, rarely gate-eligible.
  * options (fo_weekly/monthly): SYNTHETIC Black-Scholes pricing, NOT
                              real historical chains — cannot prove real
                              money; run for coverage, never promoted as
                              verified.

Usage: PYTHONPATH=. python3 scripts/backtest/run_discovery_sweep.py --group equity
       PYTHONPATH=. python3 scripts/backtest/run_discovery_sweep.py --group intraday
       PYTHONPATH=. python3 scripts/backtest/run_discovery_sweep.py --group options
"""
from __future__ import annotations

import argparse
import time
import traceback

from backend.ai.strategy_discovery import DiscoveryConfig, run_discovery


def _ga(kind, universe, sym, hist, pop=20, gen=4):
    return DiscoveryConfig(
        kind=kind, mode="ga", universe=universe,
        symbols_per_candidate=sym, history_period=hist, seed=0,
        ga_pop_size=pop, ga_generations=gen, ga_elite=5, ga_children_per_elite=3,
        walk_forward_folds=3, initial_capital=500_000.0, max_workers=4,
    )


def _rand(kind, universe, sym, hist, n, wf=0):
    return DiscoveryConfig(
        kind=kind, mode="random", universe=universe,
        symbols_per_candidate=sym, sample_size=n, history_period=hist,
        seed=0, walk_forward_folds=wf, initial_capital=500_000.0, max_workers=4,
    )


GROUPS = {
    "equity": [
        _ga("equity_position", "nifty50", 8, "3y"),
        _ga("equity_position", "nifty100", 8, "3y"),
        _ga("equity_swing", "nifty50", 8, "3y"),
        _ga("equity_swing", "nifty100", 8, "3y"),
        _ga("equity_position", "nifty500", 10, "3y", pop=24, gen=4),
    ],
    "intraday": [
        _rand("intraday_15m", "nifty50", 5, "60d", 40),
        _rand("intraday_5m", "nifty50", 5, "30d", 40),
        _rand("intraday_15m", "nifty100", 5, "60d", 40),
    ],
    "options": [
        _rand("fo_weekly", "NIFTY", 1, "2y", 30, wf=3),
        _rand("fo_monthly", "NIFTY", 1, "2y", 30, wf=3),
        _rand("fo_weekly", "BANKNIFTY", 1, "2y", 30, wf=3),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS), required=True)
    args = ap.parse_args()

    configs = GROUPS[args.group]
    print(f"== discovery sweep: {args.group} ({len(configs)} batches) ==")
    for i, cfg in enumerate(configs, 1):
        t0 = time.time()
        tag = f"{cfg.kind}/{cfg.universe}"
        try:
            run_id = run_discovery(cfg)
            from backend.core.database import get_supabase_admin
            sb = get_supabase_admin()
            cnt = sb.table("discovered_strategies").select(
                "id", count="exact").eq("run_id", str(run_id)).execute().count
            print(f"[{i}/{len(configs)}] {tag}: run={run_id} persisted={cnt} ({time.time()-t0:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(configs)}] {tag}: ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
