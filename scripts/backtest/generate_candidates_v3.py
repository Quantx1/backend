#!/usr/bin/env python3
"""Wave-3 algo candidates (2026-07-22) — new indicator families + refinements.

Built after the 2026-07-21 purge (126/126 failed) taught us where the edge
is NOT (generic TA crosses on large caps), and the discovered-strategies
re-validation showed where it IS (tight-trailing RSI mean reversion on
liquid NSE names). Wave 3 covers:

  52-week-high momentum : George-Hwang anomaly — strong published evidence
                          in Indian equities. Needs the new
                          dist_52w_high_pct / donchian_*_252 indicators.
  medium-term momentum  : Jegadeesh-Titman 3-6 month horizon via roc_63 /
                          roc_126 — the academic momentum horizon, which
                          roc_10/roc_20 (wave 2, failed) never expressed.
  low-volatility        : low-vol anomaly documented on NSE top-100.
  overnight gaps        : gap-down overreaction reversion + gap-up
                          continuation via the new gap_pct.
  mean-rev refinements  : the PASSING discovery family (RSI snap, tight
                          trailing) re-authored on the broader nifty100
                          universe and with sibling oscillators.
  Indian-app classics   : fast EMA cross with ADX arm (Streak's most
                          deployed template family) — untested in waves 1-2.

Same harness as wave 1-2: every candidate runs the product walk-forward +
gate (audit_catalog_walkforward.audit_one); --apply upserts ONLY passers
into strategy_catalog with their REAL computed metrics.

Usage (worktree root, PYTHONPATH=.):
    python3 scripts/backtest/generate_candidates_v3.py --results artifacts/backtests/candidates_v3.jsonl
    python3 scripts/backtest/generate_candidates_v3.py --results ... --apply
"""
from __future__ import annotations

import argparse
import json
import traceback
from typing import Any, Dict, List

from scripts.backtest.audit_catalog_walkforward import _load_done, audit_one
from scripts.backtest.generate_candidates import _and, cmp_, cross


def _or(*children: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "composite_or", "children": list(children)}


PCT5 = {"kind": "percent_of_capital", "value": 5}
PCT10 = {"kind": "percent_of_capital", "value": 10}


CANDIDATES: List[Dict[str, Any]] = [
    # ── 52-week-high momentum (George-Hwang) ────────────────────────
    dict(
        slug="52w-high-momentum", name="52-Week-High Momentum",
        category="momentum", description=(
            "Hold names sitting within 5% of their 52-week high with strong "
            "6-month momentum and a live trend (ADX). The George-Hwang anomaly, "
            "repeatedly documented in Indian equities: proximity to the 52-week "
            "high predicts continuation because traders anchor on it."),
        dsl={
            "name": "52-Week-High Momentum", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("dist_52w_high_pct", ">", -5),
                cmp_("roc_126", ">", 15),
                cmp_("adx", ">", 18),
            ),
            "exit": cmp_("dist_52w_high_pct", "<", -15),
            "stop_loss_pct": 8, "trailing_stop_pct": 10, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="52w-high-breakout", name="52-Week-High Breakout",
        category="momentum", description=(
            "Enter on a close through the prior 52-week high on expanded volume "
            "— the highest-conviction breakout there is: every holder is in "
            "profit, no overhead supply. Exit on a close under the 55-bar low."),
        dsl={
            "name": "52-Week-High Breakout", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cross("close", "crosses_above", "donchian_high_252"),
                cmp_("volume_ratio", ">", 1.5),
            ),
            "exit": cross("close", "crosses_below", "donchian_low_55"),
            "stop_loss_pct": 7, "trailing_stop_pct": 12, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="52w-high-pullback", name="Pullback Near 52-Week High",
        category="momentum", description=(
            "Buy the dip in leaders: name within 12% of its 52-week high, "
            "long-term uptrend intact, short-term oversold (RSI < 45). "
            "Momentum entry at a mean-reversion price."),
        dsl={
            "name": "Pullback Near 52-Week High", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("dist_52w_high_pct", ">", -12),
                cmp_("dist_52w_high_pct", "<", -3),
                cmp_("rsi14", "<", 45),
                cmp_("roc_126", ">", 10),
            ),
            "exit": cmp_("rsi14", ">", 65),
            "stop_loss_pct": 7, "take_profit_pct": 15, "trailing_stop_pct": 6,
            "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="donchian55-near-52w-high", name="Range Break in Leader Territory",
        category="momentum", description=(
            "A 55-bar breakout only counts when it happens near the 52-week "
            "high — breakouts from the middle of a range fail; breakouts into "
            "open air continue. Structure filter on top of the turtle rule."),
        dsl={
            "name": "Range Break in Leader Territory", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cross("close", "crosses_above", "donchian_high_55"),
                cmp_("dist_52w_high_pct", ">", -8),
                cmp_("volume_ratio", ">", 1.2),
            ),
            "exit": cross("close", "crosses_below", "ema21"),
            "stop_loss_pct": 6, "trailing_stop_pct": 9, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    # ── medium-term momentum (Jegadeesh-Titman horizon) ─────────────
    dict(
        slug="dual-horizon-momentum", name="Dual-Horizon Momentum",
        category="momentum", description=(
            "6-month AND 3-month momentum both positive and strong, price above "
            "the 200-SMA. The academic momentum premium at its documented "
            "horizon — not the 10/20-day noise most retail bots trade."),
        dsl={
            "name": "Dual-Horizon Momentum", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("roc_126", ">", 20),
                cmp_("roc_63", ">", 8),
                cross("close", "crosses_above", "ema21"),
            ),
            "exit": cmp_("roc_63", "<", 0),
            "stop_loss_pct": 8, "trailing_stop_pct": 10, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="momentum-low-vol", name="Quiet Momentum",
        category="momentum", description=(
            "Strong 6-month momentum delivered calmly (low realized-vol "
            "regime). Low-vol momentum carries the momentum premium with a "
            "fraction of the crash risk — both anomalies documented on NSE."),
        dsl={
            "name": "Quiet Momentum", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("roc_126", ">", 15),
                cmp_("volatility_regime", "<", 1),
                cmp_("close", ">", 0),
            ),
            "exit": _or(
                cmp_("roc_126", "<", 0),
                cmp_("volatility_regime", ">", 1),
            ),
            "stop_loss_pct": 8, "trailing_stop_pct": 12, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    # ── low-volatility anomaly ──────────────────────────────────────
    dict(
        slug="low-vol-trend", name="Low-Volatility Trend",
        category="equity_positional", description=(
            "The NSE low-vol anomaly: quiet names in uptrends outperform "
            "risk-adjusted. Hold while the trend and the calm both persist."),
        dsl={
            "name": "Low-Volatility Trend", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("volatility_regime", "<", 1),
                cmp_("roc_63", ">", 5),
                cross("ema13", "crosses_above", "ema50"),
            ),
            "exit": cross("close", "crosses_below", "sma100"),
            "stop_loss_pct": 8, "trailing_stop_pct": 12, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    # ── overnight gaps ──────────────────────────────────────────────
    dict(
        slug="gap-down-reversion", name="Gap-Down Overreaction",
        category="mean_reversion", description=(
            "A >2% overnight gap down in a name still above its 200-SMA is "
            "usually overreaction — buy the fear, exit on the snap-back. "
            "Tight stop because a real breakdown keeps falling."),
        dsl={
            "name": "Gap-Down Overreaction", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("gap_pct", "<", -2),
                cmp_("rsi14", "<", 40),
                cross("close", "crosses_above", "prev_close"),
            ),
            "exit": cmp_("rsi14", ">", 55),
            "stop_loss_pct": 5, "take_profit_pct": 8, "trailing_stop_pct": 4,
            "position_size": PCT10,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="gap-up-continuation", name="Gap-Up Continuation",
        category="momentum", description=(
            "A >2% gap up on heavy volume near the 52-week high is "
            "institutional repricing, not noise — ride the continuation with "
            "a fast-EMA exit."),
        dsl={
            "name": "Gap-Up Continuation", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("gap_pct", ">", 2),
                cmp_("volume_ratio", ">", 2),
                cmp_("dist_52w_high_pct", ">", -10),
            ),
            "exit": cross("ema8", "crosses_below", "ema21"),
            "stop_loss_pct": 5, "trailing_stop_pct": 8, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    # ── mean-reversion refinements (the passing discovery family) ───
    dict(
        slug="rsi7-snap-nifty100", name="RSI-7 Snap (Broad)",
        category="mean_reversion", description=(
            "The gate-passing discovery scaffold — deep RSI-7 oversold entry, "
            "overbought exit, tight trailing stop — re-authored on the broader "
            "nifty100 universe for breadth."),
        dsl={
            "name": "RSI-7 Snap (Broad)", "universe": "nifty100", "timeframe": "1d",
            "entry": cmp_("rsi7", "<", 26),
            "exit": cmp_("rsi7", ">", 76),
            "stop_loss_pct": 8.5, "take_profit_pct": 25, "trailing_stop_pct": 2.5,
            "position_size": PCT10,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="rsi14-uptrend-reversion", name="RSI-14 Dip in Uptrend",
        category="mean_reversion", description=(
            "Slower oscillator, stricter trend filter: RSI-14 oversold while "
            "the name holds its 200-SMA. Fewer, higher-quality snaps."),
        dsl={
            "name": "RSI-14 Dip in Uptrend", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("rsi14", "<", 32),
                cross("close", "crosses_above", "ema5"),
            ),
            "exit": cmp_("rsi14", ">", 60),
            "stop_loss_pct": 8, "take_profit_pct": 20, "trailing_stop_pct": 3,
            "position_size": PCT10,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="williams-snap-tight-trail", name="Williams %R Snap",
        category="mean_reversion", description=(
            "Deep Williams %R washout with the discovery family's tight-trail "
            "exit scaffold — sibling oscillator, same anomaly."),
        dsl={
            "name": "Williams %R Snap", "universe": "nifty50", "timeframe": "1d",
            "entry": cmp_("williams_r", "<", -88),
            "exit": cmp_("williams_r", ">", -30),
            "stop_loss_pct": 8, "take_profit_pct": 22, "trailing_stop_pct": 2.5,
            "position_size": PCT10,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="cci-washout-reversion", name="CCI Washout",
        category="mean_reversion", description=(
            "CCI < -120 marks a statistical washout; exit when it recovers "
            "through +50. Tight trailing stop caps the losers."),
        dsl={
            "name": "CCI Washout", "universe": "nifty50", "timeframe": "1d",
            "entry": cmp_("cci", "<", -120),
            "exit": cmp_("cci", ">", 50),
            "stop_loss_pct": 8, "take_profit_pct": 22, "trailing_stop_pct": 2.5,
            "position_size": PCT10,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    # ── Indian-app classic, untested in waves 1-2 ───────────────────
    dict(
        slug="fast-ema-cross-adx", name="Fast EMA Cross (ADX-armed)",
        category="equity_swing", description=(
            "The most-deployed retail template in India: 8/21 EMA cross with "
            "an ADX arm and volume confirmation. Included so the catalog's "
            "verdict on it is measured, not assumed."),
        dsl={
            "name": "Fast EMA Cross (ADX-armed)", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cross("ema8", "crosses_above", "ema21"),
                cmp_("adx", ">", 22),
                cmp_("volume_ratio", ">", 1.2),
            ),
            "exit": cross("ema8", "crosses_below", "ema21"),
            "stop_loss_pct": 6, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/backtests/candidates_v3.jsonl")
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
            sb.table("strategy_catalog").upsert({
                "slug": c["slug"], "name": c["name"], "description": c["description"],
                "category": c["category"], "segment": "EQUITY",
                "tier_required": "pro",
                "risk_level": "medium", "min_capital": 50000,
                "tags": ["verified", "walk-forward", "2026-07"],
                "is_featured": False, "is_exclusive": False,
                "requires_fo_enabled": False, "engine_compatible": False,
                "strategy_class": "dsl.runtime", "is_active": True,
                "dsl": c["dsl"],
                "backtest_total_return": ins.get("total_return_pct"),
                "backtest_win_rate": ins.get("win_rate"),
                "backtest_sharpe": ins.get("sharpe_ratio"),
                "backtest_max_drawdown": ins.get("max_drawdown_pct"),
                "backtest_total_trades": ins.get("total_trades"),
            }, on_conflict="slug").execute()
            inserted += 1
        print(f"inserted {inserted} gate-passing wave-3 candidates into strategy_catalog")
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
    print("\n== WAVE-3 VERDICTS ==", Counter(v.get("verdict") for v in allr.values()))


if __name__ == "__main__":
    main()
