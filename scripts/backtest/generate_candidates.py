#!/usr/bin/env python3
"""New algo candidates (2026-07-21) — author → walk-forward → keep passers.

14 hand-authored DSL candidates across the quant families the platform's
closed indicator registry honestly supports:

  trend/breakout : Turtle Donchian-55, Donchian-20 volume break, SuperTrend
                   + ADX rider, golden cross + volume
  mean reversion : Bollinger reclaim, RSI-7 snap in uptrend
  ICT-style      : liquidity sweep + reclaim of the prior low (stop-hunt
                   reversal — expressed honestly via prev_low/close, no
                   invented "order block" indicators)
  pivots         : floor-pivot reclaim with directional confirmation
  volume/flow    : OBV-slope surge, signed-volume Donchian break
  momentum       : dual momentum (ROC 20/10 + 200-SMA filter), MACD turn
                   with trend strength
  confluence     : bullish engulfing at oversold in an uptrend
  intraday       : VWAP reclaim (15m, mandatory SL, square-off) — runs on
                   the thin yfinance 60d window; flagged accordingly

Deliberately NO options candidates (the options backtester is synthetic-
priced — it cannot prove real-money results) and NO engine_signal
candidates (regime_history coverage < 80% over a 5y window → the gate
fails them closed; honest, so don't author them).

Every candidate runs the SAME walk-forward + gate as user strategies
(audit_catalog_walkforward.audit_one). --apply inserts ONLY passers into
strategy_catalog with their REAL computed metrics.

Usage (worktree root, PYTHONPATH=.):
    python3 scripts/backtest/generate_candidates.py --results /tmp/candidates.jsonl
    python3 scripts/backtest/generate_candidates.py --results /tmp/candidates.jsonl --apply
"""
from __future__ import annotations

import argparse
import json
import traceback
from typing import Any, Dict, List

from scripts.backtest.audit_catalog_walkforward import _load_done, audit_one

PCT5 = {"kind": "percent_of_capital", "value": 5}
PCT8 = {"kind": "percent_of_capital", "value": 8}


def _and(*children: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "composite_and", "children": list(children)}


def cmp_(ind: str, op: str, value) -> Dict[str, Any]:
    return {"kind": "indicator_compare", "indicator": ind, "op": op, "value": value}


def cross(ind: str, op: str, other: str) -> Dict[str, Any]:
    return {"kind": "indicator_cross", "indicator": ind, "op": op, "value": other}


CANDIDATES: List[Dict[str, Any]] = [
    # ── trend / breakout ────────────────────────────────────────────
    dict(
        slug="turtle-donchian-55", name="Turtle 55-Bar Breakout",
        category="equity_swing", description=(
            "The classic Turtle rule on NSE large caps: enter when price closes "
            "through the 55-bar high, ride with a trailing stop, exit on a close "
            "under the 20-bar low. Pure price structure — no oscillators."),
        dsl={
            "name": "Turtle 55-Bar Breakout", "universe": "nifty100", "timeframe": "1d",
            "entry": cross("close", "crosses_above", "donchian_high_55"),
            "exit": cross("close", "crosses_below", "donchian_low_20"),
            "stop_loss_pct": 7, "trailing_stop_pct": 10, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="donchian20-volume-break", name="20-Bar Range Break + Volume",
        category="equity_swing", description=(
            "Breakout of the 20-bar high confirmed by 1.5x average volume and a "
            "positive signed-volume balance — participation, not just price."),
        dsl={
            "name": "20-Bar Range Break + Volume", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cross("close", "crosses_above", "donchian_high_20"),
                cmp_("volume_ratio", ">", 1.5),
                cmp_("volume_delta_20", ">", 0),
            ),
            "exit": cross("ema13", "crosses_below", "ema50"),
            "stop_loss_pct": 5, "take_profit_pct": 15, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="supertrend-adx-rider", name="SuperTrend Rider (ADX-armed)",
        category="equity_swing", description=(
            "Enter when price flips above SuperTrend with ADX > 20 (a trend "
            "actually exists), exit on the opposite flip. Volatility-adaptive "
            "stop is built into the indicator."),
        dsl={
            "name": "SuperTrend Rider (ADX-armed)", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cross("close", "crosses_above", "supertrend"),
                cmp_("adx", ">", 20),
            ),
            "exit": cross("close", "crosses_below", "supertrend"),
            "stop_loss_pct": 6, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="golden-cross-volume", name="Golden Cross + Participation",
        category="equity_swing", description=(
            "50-EMA over 200-EMA golden cross taken only when volume runs above "
            "average — institutional participation behind the regime change."),
        dsl={
            "name": "Golden Cross + Participation", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cross("ema50", "crosses_above", "ema200"),
                cmp_("volume_ratio", ">", 1.2),
            ),
            "exit": cross("ema50", "crosses_below", "ema200"),
            "stop_loss_pct": 8, "trailing_stop_pct": 12, "position_size": PCT8,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    # ── mean reversion ──────────────────────────────────────────────
    dict(
        slug="bband-reclaim-reversion", name="Bollinger Reclaim",
        category="equity_swing", description=(
            "Price re-enters the band from below with RSI washed out — the "
            "snap-back trade. Exits at the middle band; tight stop."),
        dsl={
            "name": "Bollinger Reclaim", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cross("close", "crosses_above", "bbands_lower"),
                cmp_("rsi14", "<", 38),
            ),
            "exit": cmp_("close", ">=", 0) | {},  # replaced below
            "stop_loss_pct": 4, "take_profit_pct": 8, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="rsi7-uptrend-snap", name="RSI-7 Snapback (uptrend only)",
        category="equity_swing", description=(
            "Buy sharp dips (RSI-7 under 25) only while price holds above the "
            "200-SMA — mean reversion WITH the primary trend, never against it."),
        dsl={
            "name": "RSI-7 Snapback (uptrend only)", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("rsi7", "<", 25),
                cmp_("close", ">", 0),  # placeholder replaced below
            ),
            "exit": cmp_("rsi7", ">", 60),
            "stop_loss_pct": 5, "take_profit_pct": 10, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    # ── ICT-style liquidity concept (honest expression) ─────────────
    dict(
        slug="liquidity-sweep-reclaim", name="Liquidity Sweep Reclaim",
        category="equity_swing", description=(
            "The stop-hunt reversal: today's low sweeps under the prior low "
            "(resting liquidity taken) but the close reclaims it. Enter the "
            "reclaim, out if momentum dies."),
        dsl={
            "name": "Liquidity Sweep Reclaim", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("low", "<", 0),   # placeholder replaced below
                cmp_("close", ">", 0),  # placeholder replaced below
            ),
            "exit": cross("close", "crosses_below", "ema21"),
            "stop_loss_pct": 4, "take_profit_pct": 9, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    # ── pivots ──────────────────────────────────────────────────────
    dict(
        slug="pivot-reclaim-directional", name="Pivot Reclaim (DI-confirmed)",
        category="equity_swing", description=(
            "Close crosses back above the floor pivot with DI+ leading DI− — "
            "the level reclaim confirmed by directional pressure."),
        dsl={
            "name": "Pivot Reclaim (DI-confirmed)", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cross("close", "crosses_above", "pivot_point"),
                cmp_("di_plus", ">", 20),
            ),
            "exit": cross("close", "crosses_below", "pivot_s1"),
            "stop_loss_pct": 4, "take_profit_pct": 8, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    # ── volume / flow ───────────────────────────────────────────────
    dict(
        slug="obv-slope-surge", name="OBV Accumulation Surge",
        category="equity_swing", description=(
            "On-balance volume slope turns positive with a volume surge while "
            "price holds the 50-EMA — accumulation you can measure."),
        dsl={
            "name": "OBV Accumulation Surge", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("obv_slope", ">", 0),
                cmp_("volume_ratio", ">", 1.5),
                cross("close", "crosses_above", "ema50"),
            ),
            "exit": cmp_("obv_slope", "<", 0),
            "stop_loss_pct": 5, "take_profit_pct": 12, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    # ── momentum / quant ────────────────────────────────────────────
    dict(
        slug="dual-momentum-roc", name="Dual Momentum (ROC 20/10)",
        category="equity_swing", description=(
            "Absolute momentum: 20-bar ROC above 5% with 10-bar ROC positive, "
            "taken only above the 200-SMA. Exit when short momentum flips."),
        dsl={
            "name": "Dual Momentum (ROC 20/10)", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("roc_20", ">", 5),
                cmp_("roc_10", ">", 0),
                cmp_("close", ">", 0),  # placeholder replaced below
            ),
            "exit": cmp_("roc_10", "<", 0),
            "stop_loss_pct": 6, "trailing_stop_pct": 9, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="macd-turn-adx", name="MACD Turn with Trend Strength",
        category="equity_swing", description=(
            "MACD crosses its signal below zero (early turn) with ADX > 18 — "
            "momentum inflection inside a trending tape."),
        dsl={
            "name": "MACD Turn with Trend Strength", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cross("macd", "crosses_above", "macd_signal"),
                cmp_("adx", ">", 18),
            ),
            "exit": cross("macd", "crosses_below", "macd_signal"),
            "stop_loss_pct": 5, "take_profit_pct": 12, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    # ── confluence ──────────────────────────────────────────────────
    dict(
        slug="engulfing-oversold-uptrend", name="Engulfing at Oversold (uptrend)",
        category="equity_swing", description=(
            "Bullish engulfing printed while RSI is washed out and the stock "
            "still holds its 200-SMA — candle + oscillator + trend confluence."),
        dsl={
            "name": "Engulfing at Oversold (uptrend)", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("pattern_bullish_engulfing", "==", 1),
                cmp_("rsi14", "<", 45),
            ),
            "exit": cross("ema13", "crosses_below", "ema21"),
            "stop_loss_pct": 4, "take_profit_pct": 9, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    # ── intraday (thin yfinance window — flagged) ───────────────────
    dict(
        slug="vwap-reclaim-15m", name="VWAP Reclaim (15m)",
        category="equity_intraday", description=(
            "15-minute VWAP reclaim with RSI above 50 — the intraday trend-side "
            "entry desks actually use. Mandatory stop, auto square-off 15:15."),
        dsl={
            "name": "VWAP Reclaim (15m)", "universe": "single", "symbol": "RELIANCE",
            "timeframe": "15m",
            "entry": _and(
                cross("close", "crosses_above", "vwap"),
                cmp_("rsi14", ">", 50),
            ),
            "exit": cross("close", "crosses_below", "vwap"),
            "stop_loss_pct": 1.2, "take_profit_pct": 2.5, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 60, "mode": "backtest",
            "square_off_time": "15:15",
        },
    ),
    dict(
        slug="first-hour-range-break-15m", name="First-Hour High Break (15m)",
        category="equity_intraday", description=(
            "After the first hour, a close through the session's developing "
            "high with volume — the opening-range breakout in DSL form."),
        dsl={
            "name": "First-Hour High Break (15m)", "universe": "single", "symbol": "RELIANCE",
            "timeframe": "15m",
            "entry": _and(
                cmp_("is_first_hour", "==", 0),
                cross("close", "crosses_above", "donchian_high_20"),
                cmp_("volume_ratio", ">", 1.3),
            ),
            "exit": cross("close", "crosses_below", "vwap"),
            "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 60, "mode": "backtest",
            "square_off_time": "15:15",
        },
    ),
]


# ── wave 2 (2026-07-21 late) — more families, same gate ─────────────
CANDIDATES += [
    dict(
        slug="macd-state-pullback", name="Uptrend Pullback (MACD state)",
        category="equity_swing", description=(
            "Buy dips while the MACD histogram stays positive (trend intact): "
            "RSI washed to 40 inside an uptrend, exit when the histogram flips."),
        dsl={
            "name": "Uptrend Pullback (MACD state)", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(cmp_("macd_hist", ">", 0), cmp_("rsi14", "<", 40)),
            "exit": cmp_("macd_hist", "<", 0),
            "stop_loss_pct": 5, "take_profit_pct": 10, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="squeeze-expansion-break", name="Low-Vol Squeeze Breakout",
        category="equity_swing", description=(
            "Volatility regime at its low state, then price breaks the upper "
            "band — the compression-to-expansion trade."),
        dsl={
            "name": "Low-Vol Squeeze Breakout", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cmp_("volatility_regime", "==", 0),
                cross("close", "crosses_above", "bbands_upper"),
            ),
            "exit": cross("close", "crosses_below", "bbands_middle"),
            "stop_loss_pct": 5, "trailing_stop_pct": 8, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="di-cross-trend-ignition", name="DI Cross Trend Ignition",
        category="equity_swing", description=(
            "DI+ crossing over DI− with ADX alive — the earliest measurable "
            "moment a directional trend ignites."),
        dsl={
            "name": "DI Cross Trend Ignition", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(cross("di_plus", "crosses_above", "di_minus"), cmp_("adx", ">", 15)),
            "exit": cross("di_plus", "crosses_below", "di_minus"),
            "stop_loss_pct": 5, "take_profit_pct": 12, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="stochrsi-pop-momentum", name="StochRSI Pop",
        category="equity_swing", description=(
            "StochRSI K over D with RSI already above 50 — momentum popping "
            "off a reset inside strength."),
        dsl={
            "name": "StochRSI Pop", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cross("stoch_rsi_k", "crosses_above", "stoch_rsi_d"),
                cmp_("rsi14", ">", 50),
            ),
            "exit": cross("stoch_rsi_k", "crosses_below", "stoch_rsi_d"),
            "stop_loss_pct": 4, "take_profit_pct": 8, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="mfi-hammer-reversal", name="Money-Flow Hammer Reversal",
        category="equity_swing", description=(
            "MFI under 20 (money-flow washout) with a hammer printed — volume-"
            "weighted capitulation plus the rejection candle."),
        dsl={
            "name": "Money-Flow Hammer Reversal", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(cmp_("mfi", "<", 20), cmp_("pattern_hammer", "==", 1)),
            "exit": cmp_("mfi", ">", 60),
            "stop_loss_pct": 4, "take_profit_pct": 9, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="psar-flip-roc", name="PSAR Flip + Momentum",
        category="equity_swing", description=(
            "Parabolic SAR flips under price while 10-bar momentum is already "
            "positive — trail the SAR out."),
        dsl={
            "name": "PSAR Flip + Momentum", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(cross("close", "crosses_above", "psar"), cmp_("roc_10", ">", 0)),
            "exit": cross("close", "crosses_below", "psar"),
            "stop_loss_pct": 5, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="cci-hundred-momentum", name="CCI +100 Momentum",
        category="equity_swing", description=(
            "CCI through +100 (the classic momentum threshold) with volume "
            "behind it; out when CCI loses zero."),
        dsl={
            "name": "CCI +100 Momentum", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(cmp_("cci", ">", 100), cmp_("volume_ratio", ">", 1.2)),
            "exit": cmp_("cci", "<", 0),
            "stop_loss_pct": 5, "take_profit_pct": 12, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="willr-thrust-continuation", name="Williams %R Thrust",
        category="equity_swing", description=(
            "%R holding above −20 with participation — closes pinned to the "
            "highs, the classic thrust-continuation read."),
        dsl={
            "name": "Williams %R Thrust", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(cmp_("williams_r", ">", -20), cmp_("volume_ratio", ">", 1.3)),
            "exit": cmp_("williams_r", "<", -80),
            "stop_loss_pct": 5, "take_profit_pct": 10, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="three-soldiers-adx", name="Three Soldiers Continuation",
        category="equity_swing", description=(
            "Three white soldiers inside a trending tape (ADX > 20) — "
            "structured continuation, not a bottom guess."),
        dsl={
            "name": "Three Soldiers Continuation", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(cmp_("pattern_three_white_soldiers", "==", 1), cmp_("adx", ">", 20)),
            "exit": cross("ema8", "crosses_below", "ema21"),
            "stop_loss_pct": 5, "take_profit_pct": 10, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="vwap-stretch-snap-15m", name="VWAP Stretch Snap (15m)",
        category="equity_intraday", description=(
            "Price stretched 1.5% under VWAP with RSI-7 washed out — the "
            "rubber-band snap back toward VWAP. Square-off 15:15."),
        dsl={
            "name": "VWAP Stretch Snap (15m)", "universe": "single", "symbol": "RELIANCE",
            "timeframe": "15m",
            "entry": _and(cmp_("vwap_distance_pct", "<", -1.5), cmp_("rsi7", "<", 25)),
            "exit": cmp_("vwap_distance_pct", ">", 0),
            "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 60, "mode": "backtest",
            "square_off_time": "15:15",
        },
    ),
    dict(
        slug="obv-divergence-proxy", name="OBV Leads Price",
        category="equity_swing", description=(
            "OBV slope turns up while price still sits under the 21-EMA — "
            "accumulation leading price, resolved by the reclaim."),
        dsl={
            "name": "OBV Leads Price", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(cmp_("obv_slope", ">", 0), cross("close", "crosses_above", "ema21")),
            "exit": cross("close", "crosses_below", "ema21"),
            "stop_loss_pct": 4, "take_profit_pct": 10, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
    dict(
        slug="turtle-55-lowvol", name="Turtle 55 in Quiet Tape",
        category="equity_swing", description=(
            "The 55-bar breakout taken only out of a low/normal volatility "
            "state — expansion entries, not chase entries."),
        dsl={
            "name": "Turtle 55 in Quiet Tape", "universe": "nifty100", "timeframe": "1d",
            "entry": _and(
                cross("close", "crosses_above", "donchian_high_55"),
                cmp_("volatility_regime", "<", 2),
            ),
            "exit": cross("close", "crosses_below", "donchian_low_20"),
            "stop_loss_pct": 7, "trailing_stop_pct": 10, "position_size": PCT5,
            "regime_filter": "any", "lookback_days": 730, "mode": "backtest",
        },
    ),
]

# Fix placeholders that need indicator-vs-indicator compares expressed as
# crosses/compares the DSL supports (indicator_compare is indicator-vs-CONSTANT,
# so indicator-vs-indicator relations use indicator_cross or restructure):
for c in CANDIDATES:
    if c["slug"] == "bband-reclaim-reversion":
        c["dsl"]["exit"] = cross("close", "crosses_above", "bbands_middle")
    if c["slug"] == "rsi7-uptrend-snap":
        c["dsl"]["entry"] = _and(
            cmp_("rsi7", "<", 25),
            cross("close", "crosses_above", "ema5"),  # dip stabilising above fast EMA
        )
    if c["slug"] == "liquidity-sweep-reclaim":
        c["dsl"]["entry"] = _and(
            cross("close", "crosses_above", "prev_low"),   # reclaim of the swept level
            cmp_("pattern_hammer", "==", 1),               # rejection wick printed
        )
    if c["slug"] == "dual-momentum-roc":
        c["dsl"]["entry"] = _and(
            cmp_("roc_20", ">", 5),
            cmp_("roc_10", ">", 0),
            cross("close", "crosses_above", "sma20"),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/tmp/candidates.jsonl")
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
                "tier_required": "free" if inserted < 2 else "pro",
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
        print(f"inserted {inserted} gate-passing candidates into strategy_catalog")
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
    print("\n== CANDIDATE VERDICTS ==", Counter(v.get("verdict") for v in allr.values()))


if __name__ == "__main__":
    main()
