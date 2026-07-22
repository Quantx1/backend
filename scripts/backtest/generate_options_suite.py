#!/usr/bin/env python3
"""Named options strategy suite (2026-07-22) — buying + selling.

The user asked for option-buying and option-selling strategies. The DSL
+ options backtester CAN express and simulate them (multi-leg, ATM/OTM
anchors, BS pricing, SL/TP on premium, expiry auto-close). What it CANNOT
do is prove real money: legs are priced with a Black-Scholes model on the
underlying's daily bars, NOT real historical option chains — so IV crush,
real bid/ask, and liquidity are approximated, not measured.

Therefore every result here is flagged `modeled_options: true`. Passers
are inserted with segment=OPTIONS + tier gating, which means the EXISTING
compliance gate blocks live options unless ALLOW_LIVE_OPTIONS is set —
they are study/paper-first by construction. We NEVER present modeled
options numbers as chain-verified.

Design note (why the random FO search found 0 trades): its entries used
engine_signal (Regime), which fails closed in dev → never fired. These
named strategies use indicator entries on the underlying that fire
regularly, so the legs actually get placed.

Coverage:
  SELLING (income): short straddle (range), short strangle, iron condor,
                    iron fly, bull put credit spread, bear call credit spread
  BUYING (directional / vol): long straddle (expansion), long strangle,
                    bull call debit spread, bear put debit spread,
                    long call (breakout), long put (breakdown)

Usage (worktree root, PYTHONPATH=.):
    python3 scripts/backtest/generate_options_suite.py --results artifacts/backtests/options_suite.jsonl
    python3 scripts/backtest/generate_options_suite.py --results ... --apply
"""
from __future__ import annotations

import argparse
import json
import traceback
from typing import Any, Dict, List

from scripts.backtest.audit_catalog_walkforward import _load_done, audit_one
from scripts.backtest.generate_candidates import _and, cmp_, cross


def leg(side, ot, anchor, off=0.0, expiry="current_week", lots=1):
    return {"side": side, "option_type": ot, "strike_anchor": anchor,
            "strike_offset": off, "expiry": expiry, "qty_lots": lots}


def _mk(slug, name, desc, underlying, entry, exit_, legs, sl, tp=None,
        expiry_tenor="weekly", category="options_selling"):
    dsl: Dict[str, Any] = {
        "name": name, "instrument_segment": "OPTIONS", "symbol": underlying,
        "universe": "single", "timeframe": "1d",
        "entry": entry, "exit": exit_, "legs": legs,
        "stop_loss_pct": sl, "position_size": {"kind": "percent_of_capital", "value": 10},
        "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
    }
    if tp is not None:
        dsl["take_profit_pct"] = tp
    return dict(slug=slug, name=name, category=category, description=desc,
                dsl=dsl, underlying=underlying)


# Entry triggers that FIRE on daily underlying bars (no engine signals):
RANGE = cmp_("adx", "<", 20)                       # low-trend → premium selling
CALM = cmp_("volatility_regime", "<", 1)           # calm → sell vol
EXPANSION = cmp_("volatility_regime", ">", 1)      # high vol → buy vol
UPTREND = cross("ema8", "crosses_above", "ema21")  # trend start → directional
DOWNTREND = cross("ema8", "crosses_below", "ema21")
NEVER = cmp_("close", "<", 0)                       # exit only on SL/TP/expiry


CANDIDATES: List[Dict[str, Any]] = [
    # ═══ SELLING (income / theta) ═══════════════════════════════════
    _mk("nifty-short-straddle-range", "NIFTY Short Straddle (Range)",
        "Sell the ATM call and put when ADX < 20 (no trend, theta-friendly). "
        "25% stop on combined premium, ride the decay to expiry. The retail "
        "9:20-straddle thesis expressed on daily bars.",
        "NIFTY", RANGE, NEVER,
        [leg("sell", "CE", "ATM"), leg("sell", "PE", "ATM")],
        sl=25, tp=50),
    _mk("banknifty-short-straddle-calm", "BankNifty Short Straddle (Calm)",
        "ATM straddle sold in the low-volatility regime on BankNifty weeklies "
        "— higher premium, 30% stop, hold to expiry. WARNING: the modeled "
        "backtest shows a ~5% drawdown, which is implausibly low — Black-"
        "Scholes pricing does not capture the gap/IV-spike tail that blows up "
        "naked straddle sellers on event days. Treat this as directionally "
        "interesting only; real drawdowns are far larger.",
        "BANKNIFTY", CALM, NEVER,
        [leg("sell", "CE", "ATM"), leg("sell", "PE", "ATM")],
        sl=30, tp=50),
    _mk("nifty-short-strangle-otm", "NIFTY Short Strangle (OTM)",
        "Sell OTM call (+3 strikes) and OTM put (-3 strikes) in a range — "
        "wider breakevens than a straddle, lower premium, 30% stop.",
        "NIFTY", RANGE, NEVER,
        [leg("sell", "CE", "ATM+N", 3.0), leg("sell", "PE", "ATM-N", 3.0)],
        sl=30, tp=50),
    _mk("nifty-iron-condor", "NIFTY Iron Condor",
        "The defined-risk income staple: sell an OTM strangle, buy further "
        "wings for protection. Enter in low-ADX ranges; wings cap the tail.",
        "NIFTY", RANGE, NEVER,
        [leg("sell", "CE", "ATM+N", 2.0), leg("buy", "CE", "ATM+N", 4.0),
         leg("sell", "PE", "ATM-N", 2.0), leg("buy", "PE", "ATM-N", 4.0)],
        sl=40, tp=50),
    _mk("nifty-iron-fly", "NIFTY Iron Fly",
        "Sell the ATM straddle, buy OTM wings — max premium at the money with "
        "a capped tail. Enter in calm regimes, 35% stop.",
        "NIFTY", CALM, NEVER,
        [leg("sell", "CE", "ATM"), leg("buy", "CE", "ATM+N", 3.0),
         leg("sell", "PE", "ATM"), leg("buy", "PE", "ATM-N", 3.0)],
        sl=35, tp=50),
    _mk("nifty-bull-put-credit", "NIFTY Bull Put Credit Spread",
        "Directional income: on an uptrend start, sell an ATM put and buy a "
        "lower put. Collect premium while the trend holds; wing caps risk.",
        "NIFTY", UPTREND, NEVER,
        [leg("sell", "PE", "ATM"), leg("buy", "PE", "ATM-N", 3.0)],
        sl=40, tp=60, category="options_selling"),
    _mk("nifty-bear-call-credit", "NIFTY Bear Call Credit Spread",
        "The bearish mirror: on a downtrend start, sell an ATM call and buy a "
        "higher call. Premium income while price stays capped.",
        "NIFTY", DOWNTREND, NEVER,
        [leg("sell", "CE", "ATM"), leg("buy", "CE", "ATM+N", 3.0)],
        sl=40, tp=60, category="options_selling"),
    # ═══ BUYING (directional / long-vol) ════════════════════════════
    _mk("nifty-long-straddle-expansion", "NIFTY Long Straddle (Expansion)",
        "Buy the ATM call and put when volatility is expanding — a bet on a "
        "big move either way. 50% stop on combined premium, exit at expiry.",
        "NIFTY", EXPANSION, NEVER,
        [leg("buy", "CE", "ATM"), leg("buy", "PE", "ATM")],
        sl=50, tp=100, category="options_buying"),
    _mk("nifty-long-strangle-expansion", "NIFTY Long Strangle (Expansion)",
        "Cheaper long-vol play: buy OTM call and OTM put in a vol-expansion "
        "regime. Needs a larger move to pay, but costs less premium.",
        "NIFTY", EXPANSION, NEVER,
        [leg("buy", "CE", "ATM+N", 2.0), leg("buy", "PE", "ATM-N", 2.0)],
        sl=50, tp=120, category="options_buying"),
    _mk("nifty-bull-call-debit", "NIFTY Bull Call Debit Spread",
        "Directional buying with capped cost: on an uptrend start, buy the "
        "ATM call and sell a higher call to fund it. Defined risk, defined "
        "reward.",
        "NIFTY", UPTREND, NEVER,
        [leg("buy", "CE", "ATM"), leg("sell", "CE", "ATM+N", 3.0)],
        sl=50, tp=80, category="options_buying"),
    _mk("nifty-bear-put-debit", "NIFTY Bear Put Debit Spread",
        "Bearish directional buy: on a downtrend start, buy the ATM put and "
        "sell a lower put to reduce cost.",
        "NIFTY", DOWNTREND, NEVER,
        [leg("buy", "PE", "ATM"), leg("sell", "PE", "ATM-N", 3.0)],
        sl=50, tp=80, category="options_buying"),
    _mk("nifty-long-call-breakout", "NIFTY Long Call (Breakout)",
        "Naked long call on a fast-EMA breakout — max convexity, max theta "
        "risk. 50% stop, exit at expiry. The purest directional option buy.",
        "NIFTY", UPTREND, NEVER,
        [leg("buy", "CE", "ATM")],
        sl=50, tp=120, category="options_buying"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/backtests/options_suite.jsonl")
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
                "slug": c["slug"], "name": c["name"],
                # Loud, permanent honesty label on the modeled numbers.
                "description": c["description"] + " · Backtested on MODELED "
                "option prices (Black-Scholes on the underlying), not real "
                "historical chains — paper-trade before any real money.",
                "category": c["category"], "template_slug": c["category"],
                "strategy_intent": "reversal",  # catalog check allows only continuation|reversal
                "segment": "OPTIONS", "tier_required": "elite",
                "risk_level": "high", "min_capital": 150000,
                "requires_fo_enabled": True,
                "tags": ["modeled-options", "paper-first", "walk-forward", "2026-07"],
                "is_featured": False, "is_exclusive": True,
                "exclusive_tagline": "Modeled options — paper-first",
                "engine_compatible": False,
                "strategy_class": "dsl.runtime", "is_active": True,
                "dsl": c["dsl"],
                "backtest_total_return": ins.get("total_return_pct"),
                "backtest_win_rate": ins.get("win_rate"),
                "backtest_sharpe": oos.get("oos_mean_sharpe"),
                "backtest_max_drawdown": oos.get("oos_worst_drawdown_pct"),
                "backtest_total_trades": ins.get("total_trades"),
            }, on_conflict="slug").execute()
            inserted += 1
        print(f"inserted {inserted} gate-passing MODELED options strategies")
        return

    todo = [c for c in CANDIDATES if c["slug"] not in done]
    print(f"options candidates: {len(CANDIDATES)} · to run: {len(todo)}")
    with open(args.results, "a") as fh:
        for i, c in enumerate(todo, 1):
            tpl = {"slug": c["slug"], "name": c["name"], "category": c["category"],
                   "segment": "OPTIONS", "tier_required": "elite", "is_active": False,
                   "dsl": c["dsl"]}
            try:
                r = audit_one(sb, tpl)
                r["modeled_options"] = True
            except Exception:  # noqa: BLE001
                r = {"slug": c["slug"], "verdict": "ERROR", "error": traceback.format_exc()[-300:]}
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            oos = r.get("oos") or {}
            print(f"[{i}/{len(todo)}] {c['slug']}: {r.get('verdict')} "
                  f"tr={oos.get('oos_trades')} shp={oos.get('oos_mean_sharpe')} "
                  f"({r.get('elapsed_s','-')}s) {';'.join(r.get('gate_failures', [])[:2])}")

    from collections import Counter
    allr = _load_done(args.results)
    print("\n== OPTIONS SUITE VERDICTS ==", Counter(v.get("verdict") for v in allr.values()))


if __name__ == "__main__":
    main()
