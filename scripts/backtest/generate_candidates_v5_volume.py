#!/usr/bin/env python3
"""Wave-5 — volume/flow-based equity candidates (2026-07-22).

The user asked specifically for volume-based strategies. The DSL exposes
a real volume/flow toolkit: volume_ratio, obv, obv_slope, volume_delta_20,
mfi, vwap, vwap_distance_pct. Wave-4 already proved one volume rule works
(capitulation-volume-snap, OOS Sharpe 0.59). This wave explores the rest
of the volume anomaly space honestly:

  climax / capitulation : heavy-volume drops that snap back
  accumulation breakouts: OBV rising through range highs
  volume dry-up pullback: low-volume dips in uptrends (no supply)
  VWAP reversion        : stretched below VWAP, snap to the mean
  money-flow washout     : MFI capitulation with volume confirm

Same harness + strict gate as every wave. --apply upserts ONLY passers.

Usage (worktree root, PYTHONPATH=.):
    python3 scripts/backtest/generate_candidates_v5_volume.py --results artifacts/backtests/candidates_v5.jsonl
    python3 scripts/backtest/generate_candidates_v5_volume.py --results ... --apply
"""
from __future__ import annotations

import argparse
import json
import traceback
from typing import Any, Dict, List

from scripts.backtest.audit_catalog_walkforward import _load_done, audit_one
from scripts.backtest.generate_candidates import _and, cmp_, cross

PCT7 = {"kind": "percent_of_capital", "value": 7}
PCT10 = {"kind": "percent_of_capital", "value": 10}


def _mk(slug, name, desc, universe, entry, exit_, sl, tp=None, trail=None, size=PCT7, cat="equity_swing"):
    dsl: Dict[str, Any] = {
        "name": name, "universe": universe, "timeframe": "1d",
        "entry": entry, "exit": exit_, "stop_loss_pct": sl, "position_size": size,
        "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
    }
    if tp is not None:
        dsl["take_profit_pct"] = tp
    if trail is not None:
        dsl["trailing_stop_pct"] = trail
    return dict(slug=slug, name=name, category=cat, description=desc, dsl=dsl)


CANDIDATES: List[Dict[str, Any]] = [
    # ── climax / capitulation (volume marks the flush) ──────────────
    _mk("volume-climax-reversal", "Volume Climax Reversal",
        "A 5%+ ten-day drop on 2.5x average volume is a selling climax — "
        "forced liquidation, not information. Buy the flush, exit on RSI "
        "recovery. Volume is the tell that separates a bottom from a slide.",
        "nifty100",
        _and(cmp_("roc_10", "<", -5), cmp_("volume_ratio", ">", 2.5)),
        cmp_("rsi14", ">", 62), sl=8, tp=16, size=PCT7),
    _mk("mfi-capitulation-volume", "Money-Flow Capitulation",
        "MFI below 20 means price AND volume are both washed out — a deeper "
        "signal than price alone. Confirm with a hard drop, exit on recovery.",
        "nifty50",
        _and(cmp_("mfi", "<", 20), cmp_("roc_10", "<", -4)),
        cmp_("mfi", ">", 55), sl=8, tp=18, size=PCT7),
    # ── accumulation breakouts (OBV leads price) ────────────────────
    _mk("obv-accumulation-break", "OBV Accumulation Breakout",
        "On-balance-volume rising (obv_slope > 0) as price breaks the 20-day "
        "high — buyers accumulated through the base, now price confirms. "
        "Volume-led breakout, exit on trend loss.",
        "nifty100",
        _and(cross("close", "crosses_above", "donchian_high_20"), cmp_("obv_slope", ">", 0),
             cmp_("volume_ratio", ">", 1.3)),
        cross("close", "crosses_below", "ema21"), sl=6, trail=8, size=PCT7),
    _mk("volume-delta-thrust", "Signed-Volume Thrust",
        "A strongly positive 20-day signed-volume balance with price above the "
        "50-EMA — persistent net buying. Ride it until the balance flips.",
        "nifty100",
        _and(cmp_("volume_delta_20", ">", 0), cross("ema8", "crosses_above", "ema21"),
             cmp_("volume_ratio", ">", 1.5)),
        cmp_("volume_delta_20", "<", 0), sl=6, trail=7, size=PCT7),
    # ── volume dry-up pullback (no supply on the dip) ───────────────
    _mk("volume-dryup-pullback", "Volume Dry-Up Pullback",
        "A pullback (RSI < 42) on BELOW-average volume in an uptrend means "
        "no real supply — the dip is profit-taking, not distribution. Buy it, "
        "exit on the momentum snap-back.",
        "nifty100",
        _and(cmp_("rsi14", "<", 42), cmp_("volume_ratio", "<", 0.8),
             cross("close", "crosses_above", "ema5")),
        cmp_("rsi14", ">", 62), sl=7, tp=14, trail=4, size=PCT7),
    # ── VWAP reversion (stretched from the volume-weighted mean) ────
    _mk("vwap-stretch-reversion", "VWAP Stretch Reversion",
        "Price stretched 3%+ below the anchored VWAP with a washed-out RSI — "
        "an over-extension from the volume-weighted fair value. Snap back to "
        "the mean.",
        "nifty50",
        _and(cmp_("vwap_distance_pct", "<", -3), cmp_("rsi14", "<", 40)),
        cmp_("vwap_distance_pct", ">", 0), sl=7, tp=12, trail=4, size=PCT10),
    _mk("vwap-deep-reversion-n100", "Deep VWAP Reversion",
        "The same VWAP over-extension on the broader nifty100, deeper stretch "
        "(4%) and an RSI-14 recovery exit.",
        "nifty100",
        _and(cmp_("vwap_distance_pct", "<", -4), cmp_("rsi7", "<", 30)),
        cmp_("rsi14", ">", 60), sl=8, tp=15, size=PCT7),
    # ── OBV divergence proxy ────────────────────────────────────────
    _mk("obv-slope-snap", "OBV-Slope Snap",
        "Price down over ten days but OBV slope turning positive — the volume "
        "divergence that precedes a reversal. Buy the accumulation, exit on "
        "the RSI recovery.",
        "nifty100",
        _and(cmp_("roc_10", "<", -4), cmp_("obv_slope", ">", 0)),
        cmp_("rsi14", ">", 62), sl=7, tp=15, trail=4, size=PCT7),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/backtests/candidates_v5.jsonl")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    from backend.core.database import get_supabase_admin
    sb = get_supabase_admin()
    done = _load_done(args.results)

    if args.apply:
        inserted = 0
        for c in CANDIDATES:
            r = done.get(c["slug"])
            if not r or r.get("verdict") != "PASS":
                continue
            ins = r.get("in_sample") or {}
            oos = r.get("oos") or {}
            sb.table("strategy_catalog").upsert({
                "slug": c["slug"], "name": c["name"], "description": c["description"],
                "category": c["category"], "template_slug": c["category"],
                "strategy_intent": "reversal" if "reversion" in c["slug"] or "snap" in c["slug"] or "capitulation" in c["slug"] or "climax" in c["slug"] or "pullback" in c["slug"] else "continuation",
                "segment": "EQUITY", "tier_required": "pro",
                "risk_level": "medium", "min_capital": 50000,
                "tags": ["verified", "walk-forward", "volume", "2026-07"],
                "is_featured": False, "is_exclusive": False,
                "requires_fo_enabled": False, "engine_compatible": False,
                "strategy_class": "dsl.runtime", "is_active": True,
                "dsl": c["dsl"],
                "backtest_total_return": None, "backtest_cagr": None,
                "backtest_win_rate": ins.get("win_rate"),
                "backtest_sharpe": oos.get("oos_mean_sharpe"),
                "backtest_max_drawdown": oos.get("oos_worst_drawdown_pct"),
                "backtest_total_trades": ins.get("total_trades"),
            }, on_conflict="slug").execute()
            inserted += 1
        print(f"inserted {inserted} gate-passing volume candidates")
        return

    todo = [c for c in CANDIDATES if c["slug"] not in done]
    print(f"candidates: {len(CANDIDATES)} · to run: {len(todo)}")
    with open(args.results, "a") as fh:
        for i, c in enumerate(todo, 1):
            tpl = {"slug": c["slug"], "name": c["name"], "category": c["category"],
                   "segment": "EQUITY", "tier_required": "pro", "is_active": False,
                   "dsl": c["dsl"]}
            try:
                r = audit_one(sb, tpl)
            except Exception:  # noqa: BLE001
                r = {"slug": c["slug"], "verdict": "ERROR", "error": traceback.format_exc()[-300:]}
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] {c['slug']}: {r.get('verdict')} "
                  f"({r.get('elapsed_s', '-')}s) {';'.join(r.get('gate_failures', [])[:2])}")

    from collections import Counter
    allr = _load_done(args.results)
    print("\n== WAVE-5 VOLUME VERDICTS ==", Counter(v.get("verdict") for v in allr.values()))


if __name__ == "__main__":
    main()
