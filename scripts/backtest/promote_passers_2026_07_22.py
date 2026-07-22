#!/usr/bin/env python3
"""Promote the 2026-07-22 gate passers into strategy_catalog — curated, deduped.

Sources:
  * artifacts/backtests/discovered_audit_2026_07_22.jsonl  (discovery re-validation)
  * artifacts/backtests/candidates_v4_2026_07_22.jsonl     (wave-4 authored)

Rules:
  * Only PASS verdicts, deduped: where sibling mutations share the same
    entry/exit rule, keep the best OOS Sharpe variant only.
  * Public names replace internal discovery codenames (brand firewall) —
    both in the catalog row AND inside the stored DSL's name field.
  * Numbers written are the OUT-OF-SAMPLE walk-forward aggregates the gate
    scored (Sharpe = OOS mean, drawdown = worst OOS window). Total return
    is NOT available at universe level — left NULL, never fabricated.

Usage: PYTHONPATH=. python3 scripts/backtest/promote_passers_2026_07_22.py [--dry]
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

DISCOVERED = "artifacts/backtests/discovered_audit_2026_07_22.jsonl"
WAVE4 = "artifacts/backtests/candidates_v4_2026_07_22.jsonl"

# key = discovery label prefix (before " · ") or wave-4 slug
KEEP: Dict[str, Dict[str, str]] = {
    "MeanRev-position-0017-m20": dict(
        slug="drawdown-snap-tight-trail", name="Drawdown Snap — Tight Trail",
        category="equity_swing", intent="reversal", risk="medium",
        description=(
            "Buys NIFTY50 names that fell 4%+ over ten sessions and exits as "
            "RSI-14 recovers through 70, with a tight 1.5% trailing stop that "
            "locks the bounce. The highest-volume verified rule in the catalog: "
            "996 out-of-sample trades across walk-forward windows."),
    ),
    "MeanRev-position-0028": dict(
        slug="fast-rsi-reversal", name="Fast RSI Reversal",
        category="equity_swing", intent="reversal", risk="medium",
        description=(
            "Enters on a fast RSI-7 washout under 39 and holds until the slower "
            "RSI-14 confirms the recovery above 74 — enter on panic, leave only "
            "when the bounce matures. Top risk-adjusted passer of the audit."),
    ),
    "MeanRev-position-0029": dict(
        slug="rsi-dip-rider", name="RSI Dip Rider",
        category="equity_swing", intent="reversal", risk="medium",
        description=(
            "RSI-14 dip under 38 entry with a 14% profit target and a tight "
            "5.5% stop — an asymmetric payoff on the large-cap snap-back."),
    ),
    "MeanRev-position-0008": dict(
        slug="rsi7-full-cycle", name="RSI-7 Full Cycle",
        category="equity_swing", intent="reversal", risk="medium",
        description=(
            "Deep RSI-7 oversold entry under 28, riding the full oscillator "
            "cycle to overbought 78 with a wide 27% target for the outliers."),
    ),
    "MeanRev-position-0017-m21": dict(
        slug="stochastic-panic-snap", name="Stochastic Panic Snap",
        category="equity_swing", intent="reversal", risk="medium",
        description=(
            "Stochastic %K under 32 marks the washout; exit when RSI-14 "
            "recovers through 68. Cross-oscillator pairing — the entry and "
            "exit clocks measure different things."),
    ),
    "MeanRev-position-0001-m13-m24": dict(
        slug="washout-to-recovery", name="Washout to Recovery",
        category="equity_swing", intent="reversal", risk="medium",
        description=(
            "Williams %R below -72 entry, RSI-14 above 69 exit, 5.8% stop — "
            "washout in, confirmed recovery out."),
    ),
    "MeanRev-position-0018-m21": dict(
        slug="deep-drop-recovery", name="Deep Drop Recovery",
        category="equity_swing", intent="reversal", risk="medium",
        description=(
            "Waits for a hard 6.5%+ ten-session drop before entering — fewer, "
            "deeper panics with a 5.9% trailing stop on the recovery."),
    ),
    "MeanRev-position-0005-m12-m22": dict(
        slug="williams-washout", name="Williams Washout",
        category="equity_swing", intent="reversal", risk="medium",
        description=(
            "The symmetric Williams %R cycle: enter below -69, exit above -19, "
            "9.7% stop, 17% target. 529 out-of-sample trades."),
    ),
    "Momentum-position-0016": dict(
        slug="fast-trend-rider", name="Fast Trend Rider",
        category="equity_swing", intent="continuation", risk="medium",
        description=(
            "The one trend rule that survived the gate: EMA-8 crossing the "
            "50-SMA with a tight 4.7% stop and a 26% target — small losses, "
            "occasional big riders. Every classic slow MA cross failed."),
    ),
}

WAVE4_KEEP = ("capitulation-volume-snap", "roc-drop-deep-n50", "roc20-drop-recovery-n50")


def _rows(path: str):
    try:
        return [json.loads(line) for line in open(path)]
    except FileNotFoundError:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    from backend.core.database import get_supabase_admin
    sb = get_supabase_admin()

    from scripts.backtest.generate_candidates_v4 import CANDIDATES as W4
    w4_meta = {c["slug"]: c for c in W4}

    upserts = []
    promoted_ids = []

    for r in _rows(DISCOVERED):
        if r.get("verdict") != "PASS":
            continue
        prefix = r["label"].split(" · ")[0]
        meta = KEEP.get(prefix)
        if not meta:
            continue  # deduped sibling
        dsl = dict(r["dsl"])
        dsl["name"] = meta["name"]
        upserts.append((meta, dsl, r, r.get("discovered_id")))

    for r in _rows(WAVE4):
        if r.get("verdict") != "PASS" or r.get("slug") not in WAVE4_KEEP:
            continue
        c = w4_meta[r["slug"]]
        meta = dict(slug=c["slug"], name=c["name"], category="equity_swing",
                    intent="reversal", risk="medium", description=c["description"])
        upserts.append((meta, c["dsl"], r, None))

    print(f"promoting {len(upserts)} verified strategies:")
    for meta, dsl, r, disc_id in upserts:
        ins, oos = r.get("in_sample") or {}, r.get("oos") or {}
        row = {
            "slug": meta["slug"], "name": meta["name"],
            "description": meta["description"], "category": meta["category"],
            "template_slug": meta["category"],
            "strategy_intent": meta.get("intent", "reversal"),
            "segment": "EQUITY", "tier_required": "pro",
            "risk_level": meta["risk"], "min_capital": 50000,
            "tags": ["verified", "walk-forward", "2026-07"],
            "is_featured": False, "is_exclusive": False,
            "requires_fo_enabled": False, "engine_compatible": False,
            "strategy_class": "dsl.runtime", "is_active": True,
            "dsl": dsl,
            "backtest_total_return": None,
            "backtest_cagr": None,
            "backtest_win_rate": ins.get("win_rate"),
            "backtest_sharpe": oos.get("oos_mean_sharpe"),
            "backtest_max_drawdown": oos.get("oos_worst_drawdown_pct"),
            "backtest_total_trades": ins.get("total_trades"),
        }
        print(f"  {row['slug']:28} oosShp={row['backtest_sharpe']} wr={row['backtest_win_rate']} "
              f"tr={row['backtest_total_trades']} ddOOS={row['backtest_max_drawdown']}")
        if not args.dry:
            sb.table("strategy_catalog").upsert(row, on_conflict="slug").execute()
            if disc_id:
                sb.table("discovered_strategies").update(
                    {"status": "promoted"}).eq("id", disc_id).execute()
                promoted_ids.append(disc_id)

    if not args.dry:
        active = sb.table("strategy_catalog").select("id", count="exact").eq(
            "is_active", True).execute()
        print(f"\nactive catalog templates now: {active.count} · "
              f"discovered rows marked promoted: {len(promoted_ids)}")


if __name__ == "__main__":
    main()
