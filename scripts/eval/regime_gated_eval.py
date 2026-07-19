#!/usr/bin/env python3
"""Regime-gated evaluation of a walk-forward backtest.

Joins the daily regime series (ml/regime ensemble, walk-forward, causal) onto a
backtest's dated per-period series and answers two questions:
  1. WHERE does the book bleed? (per-regime conditional stats)
  2. Does regime-GATING rescue it? (exposure scaled by state, iid evaluation)

Costs on the gated book: the per-period round-trip cost is charged only when
exposed (scaled by exposure); regime-switch liquidation costs are NOT modeled
(noted in output) — at ~2 switches/quarter this is second-order vs the 1/H
steady-state churn.

Usage:
  python3 scripts/eval/regime_gated_eval.py \
      --backtest artifacts/eval/momentum_e2_expanded.json \
      --regimes artifacts/regime/regime_series.parquet \
      [--exposure bull=1.0,sideways=0.5,bear=0.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.eval.walkforward_backtest import _iid_stats  # noqa: E402

DEFAULT_MAPS = {
    "gate_bear":      {"bull": 1.0, "sideways": 1.0, "bear": 0.0},
    "gate_bear_half": {"bull": 1.0, "sideways": 0.5, "bear": 0.0},
    "bull_only":      {"bull": 1.0, "sideways": 0.0, "bear": 0.0},
}


def load_joined(backtest_path: Path, regimes_path: Path) -> tuple[pd.DataFrame, dict]:
    bt = json.loads(backtest_path.read_text())
    if "per_date" not in bt or not bt["per_date"]:
        raise SystemExit(f"{backtest_path} has no per_date series — re-run the "
                         "backtest with the current harness")
    pdf = pd.DataFrame(bt["per_date"])
    pdf["date"] = pd.to_datetime(pdf["date"])
    reg = pd.read_parquet(regimes_path)[["date", "state_name"]]
    reg["date"] = pd.to_datetime(reg["date"]).astype("datetime64[ns]")
    joined = pdf.merge(reg, on="date", how="left")
    n_missing = int(joined["state_name"].isna().sum())
    joined["state_name"] = joined["state_name"].fillna("sideways")  # pre-series dates: neutral
    return joined, {"backtest": bt, "n_missing_regime": n_missing}


def conditional_report(joined: pd.DataFrame, horizon: int) -> dict:
    """Per-regime conditional stats of the UNGATED book (gross, per H-period)."""
    out = {}
    for state, g in joined.groupby("state_name"):
        excess = g["gross_h"] - g["bench_h"]
        out[state] = {
            "n_dates": int(len(g)),
            "mean_gross_h": round(float(g["gross_h"].mean()), 5),
            "mean_bench_h": round(float(g["bench_h"].mean()), 5),
            "mean_excess_h": round(float(excess.mean()), 5),
            "hit_rate_vs_bench": round(float((excess > 0).mean()), 3),
        }
    return out


def gated_iid(joined: pd.DataFrame, horizon: int, cost_bps_side: float,
              exposure: dict) -> dict:
    """iid (non-overlapping) stats of the gated book, per fold stride-H."""
    cost_rt = 2.0 * cost_bps_side / 10_000.0
    rows = []
    for _, fold_df in joined.groupby("fold"):
        fold_df = fold_df.sort_values("date")
        sub = fold_df.iloc[::horizon]
        for _, r in sub.iterrows():
            e = float(exposure.get(r["state_name"], 1.0))
            rows.append({
                "net": e * float(r["gross_h"]) - e * cost_rt,
                "excess": e * float(r["gross_h"]) - e * cost_rt - float(r["bench_h"]),
                "exposed": e,
            })
    net = _iid_stats((r["net"] for r in rows), horizon)
    exc = _iid_stats((r["excess"] for r in rows), horizon)
    return {"net_iid": net, "excess_iid": exc,
            "avg_exposure": round(float(np.mean([r["exposed"] for r in rows])), 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", required=True)
    ap.add_argument("--regimes", default="artifacts/regime/regime_series.parquet")
    ap.add_argument("--cost-bps", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    joined, meta = load_joined(Path(args.backtest), Path(args.regimes))
    bt = meta["backtest"]
    horizon = int(bt["horizon"])
    engine = bt["engine"]

    cond = conditional_report(joined, horizon)
    print(f"\n═══ {engine} (H={horizon}) — per-regime conditional (ungated, gross) ═══")
    for state in ("bull", "sideways", "bear"):
        c = cond.get(state)
        if c:
            print(f"  {state:9s} n={c['n_dates']:4d}  gross_H={c['mean_gross_h']:+.4f}  "
                  f"bench_H={c['mean_bench_h']:+.4f}  excess_H={c['mean_excess_h']:+.4f}  "
                  f"hit={c['hit_rate_vs_bench']:.0%}")

    results = {"engine": engine, "horizon": horizon, "conditional": cond,
               "n_missing_regime": meta["n_missing_regime"],
               "ungated": {"net_iid": bt.get("net_iid"), "excess_iid": bt.get("excess_iid")},
               "gated": {}}
    print(f"\n  {'variant':16s} {'netShp':>7} {'netDD':>7} {'excShp':>7} {'avgExp':>7}")
    u = bt.get("net_iid", {}); x = bt.get("excess_iid", {})
    print(f"  {'UNGATED':16s} {u.get('sharpe', 0):>7.2f} {u.get('max_drawdown', 0):>7.1%} "
          f"{x.get('sharpe', 0):>7.2f} {'1.000':>7}")
    for name, emap in DEFAULT_MAPS.items():
        g = gated_iid(joined, horizon, args.cost_bps, emap)
        results["gated"][name] = g
        print(f"  {name:16s} {g['net_iid']['sharpe']:>7.2f} "
              f"{g['net_iid']['max_drawdown']:>7.1%} "
              f"{g['excess_iid']['sharpe']:>7.2f} {g['avg_exposure']:>7.3f}")
    print("\n  note: switch-liquidation costs unmodeled (second-order at observed "
          "switch frequency); pre-regime-series dates treated as sideways "
          f"(n={meta['n_missing_regime']}).")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"  written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
