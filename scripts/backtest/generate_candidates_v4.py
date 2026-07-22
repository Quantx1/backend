#!/usr/bin/env python3
"""Wave-4 algo candidates (2026-07-22) — mining the VALIDATED panic-reversion region.

The discovered-strategies re-validation produced 8 gate passers, all one
family: deep short-term weakness on liquid NSE names (roc_10 < -4..-6.5,
williams_r < -69, rsi7 < 39, stoch_k < 32), exit on oscillator recovery
(rsi14 > 68-74), SL 6-10%, TP 12-21%, nifty50. Wave 4 mines *around* that
validated region — NOT a blind grid:

  breadth       : the same passing rules moved to nifty100 (the key
                  generalization test)
  confluence    : double-oscillator panic, capitulation volume
  regime guards : skip the high-vol regime where knives keep falling
  siblings      : MFI / Bollinger-cross / gap variants of the same anomaly

Multiple-testing honesty: this wave exploits a family already validated
out-of-sample by 8 independent passers; each variant still runs the full
walk-forward + gate individually, and the product's mandatory paper stage
remains the final arbiter before live.

Usage (worktree root, PYTHONPATH=.):
    python3 scripts/backtest/generate_candidates_v4.py --results artifacts/backtests/candidates_v4.jsonl
    python3 scripts/backtest/generate_candidates_v4.py --results ... --apply
"""
from __future__ import annotations

import argparse
import json
import traceback
from typing import Any, Dict, List

from scripts.backtest.audit_catalog_walkforward import _load_done, audit_one
from scripts.backtest.generate_candidates import _and, cmp_, cross

PCT5 = {"kind": "percent_of_capital", "value": 5}
PCT7 = {"kind": "percent_of_capital", "value": 7}


def _mk(slug: str, name: str, desc: str, universe: str, entry, exit_,
        sl: float, tp: float | None = None, trail: float | None = None,
        size=PCT5) -> Dict[str, Any]:
    dsl: Dict[str, Any] = {
        "name": name, "universe": universe, "timeframe": "1d",
        "entry": entry, "exit": exit_,
        "stop_loss_pct": sl, "position_size": size,
        "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
    }
    if tp is not None:
        dsl["take_profit_pct"] = tp
    if trail is not None:
        dsl["trailing_stop_pct"] = trail
    return dict(slug=slug, name=name, category="mean_reversion",
                description=desc, dsl=dsl)


CANDIDATES: List[Dict[str, Any]] = [
    # ── breadth: passing rules on nifty100 ──────────────────────────
    _mk("roc-drop-recovery-n100", "10-Day Drop Recovery (Broad)",
        "The champion discovered rule — a 4%+ drop over 10 sessions, exit when "
        "RSI-14 recovers through 70 — tested on the broader nifty100 universe.",
        "nifty100",
        cmp_("roc_10", "<", -4), cmp_("rsi14", ">", 70),
        sl=6.2, tp=20),
    _mk("williams-recovery-n100", "Williams %R Washout (Broad)",
        "Williams %R below -70 marks a washout; hold until it recovers through "
        "-20. The second discovered passer family, on nifty100.",
        "nifty100",
        cmp_("williams_r", "<", -70), cmp_("williams_r", ">", -20),
        sl=9.5, tp=17, size=PCT7),
    _mk("stoch-panic-rsi-exit-n100", "Stochastic Panic / RSI Exit (Broad)",
        "Stochastic %K under 32 entry with an asymmetric RSI-14 > 68 exit — "
        "cross-oscillator pairing from the discovery passers, on nifty100.",
        "nifty100",
        cmp_("stochastic_k", "<", 32), cmp_("rsi14", ">", 68),
        sl=5.6, tp=19, size=PCT7),
    _mk("rsi7-asym-recovery-n100", "RSI-7 Panic / RSI-14 Exit (Broad)",
        "Fast oscillator entry (RSI-7 < 35), slow oscillator exit (RSI-14 > 72): "
        "enter on panic, leave only when the bounce matures.",
        "nifty100",
        cmp_("rsi7", "<", 35), cmp_("rsi14", ">", 72),
        sl=9.5, tp=21),
    # ── deeper / slower drop windows ────────────────────────────────
    _mk("roc-drop-deep-n50", "Deep 10-Day Capitulation",
        "An 8%+ drop over 10 sessions on a NIFTY50 name is capitulation "
        "territory — earlier exit (RSI-14 > 65) banks the sharper snap-back.",
        "nifty50",
        cmp_("roc_10", "<", -8), cmp_("rsi14", ">", 65),
        sl=10, tp=15, size=PCT7),
    _mk("roc20-drop-recovery-n50", "20-Day Slide Recovery",
        "The same anomaly on a slower clock: an 8%+ slide over 20 sessions, "
        "exit on RSI-14 recovery through 70.",
        "nifty50",
        cmp_("roc_20", "<", -8), cmp_("rsi14", ">", 70),
        sl=8, tp=18, size=PCT7),
    # ── confluence: higher-quality panic ────────────────────────────
    _mk("double-oscillator-panic", "Double-Oscillator Panic",
        "Both the 10-day ROC and Williams %R confirm the washout before "
        "entering — fewer trades, cleaner panic.",
        "nifty100",
        _and(cmp_("roc_10", "<", -5), cmp_("williams_r", "<", -75)),
        cmp_("rsi14", ">", 68),
        sl=8, tp=18, size=PCT7),
    _mk("capitulation-volume-snap", "Capitulation Volume Snap",
        "A hard 10-day drop on 1.5x volume — forced selling, not drift. "
        "Volume marks the flush; the bounce follows.",
        "nifty50",
        _and(cmp_("roc_10", "<", -5), cmp_("volume_ratio", ">", 1.5)),
        cmp_("rsi14", ">", 68),
        sl=8, tp=18, size=PCT7),
    _mk("gap-down-panic-combo", "Gap Into Panic",
        "An overnight gap down landing on an already-weak name (10-day ROC "
        "< -4) — the overreaction compounds; buy it, exit on RSI recovery.",
        "nifty100",
        _and(cmp_("gap_pct", "<", -1.5), cmp_("roc_10", "<", -4)),
        cmp_("rsi14", ">", 65),
        sl=7, tp=15, size=PCT7),
    # ── regime guards ───────────────────────────────────────────────
    _mk("panic-snap-calm-regime", "Panic Snap (Calm Regimes Only)",
        "The champion rule gated to low/normal volatility regimes — in "
        "crash regimes, knives keep falling; stand aside.",
        "nifty50",
        _and(cmp_("roc_10", "<", -4), cmp_("volatility_regime", "<", 2)),
        cmp_("rsi14", ">", 70),
        sl=6.2, tp=20, size=PCT7),
    _mk("panic-snap-uptrend-only", "Panic Snap (Uptrend Only)",
        "Only buy 10-day panics in names that were in 6-month uptrends before "
        "the drop — dips in leaders, not breakdowns in laggards.",
        "nifty100",
        _and(cmp_("roc_10", "<", -4.5), cmp_("roc_126", ">", 5)),
        cmp_("rsi14", ">", 70),
        sl=7, tp=18, size=PCT7),
    # ── sibling oscillators / structures ────────────────────────────
    _mk("mfi-washout-recovery", "Money-Flow Washout",
        "MFI under 25 = price AND volume both washed out. Exit on RSI-14 "
        "recovery — the flow version of the validated anomaly.",
        "nifty50",
        cmp_("mfi", "<", 25), cmp_("rsi14", ">", 68),
        sl=8, tp=18, size=PCT7),
    _mk("bband-break-panic-snap", "Band-Break Panic Snap",
        "Close breaking below the lower Bollinger band while RSI-7 is under "
        "35 — a statistical excursion plus a momentum washout.",
        "nifty100",
        _and(cross("close", "crosses_below", "bbands_lower"), cmp_("rsi7", "<", 35)),
        cmp_("rsi14", ">", 68),
        sl=7, tp=16, size=PCT7),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/backtests/candidates_v4.jsonl")
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
        print(f"inserted {inserted} gate-passing wave-4 candidates into strategy_catalog")
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
    print("\n== WAVE-4 VERDICTS ==", Counter(v.get("verdict") for v in allr.values()))


if __name__ == "__main__":
    main()
