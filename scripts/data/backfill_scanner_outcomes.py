#!/usr/bin/env python
"""PR-S10 backfill — replay every scanner over the last N days, score each
hit's forward return, populate `scanner_outcomes` + `scanner_stats`.

For each scanner_id in SCANNER_FILTERS:
  1. For each trading day in [today-lookback, today-21]:
     - Recompute summary_df at that historical date
     - Run the scanner filter against it
     - For each match: record entry_price + forward returns
       (5d / 10d / 20d) + drawdown
  2. After all hits collected, refresh `scanner_stats` with aggregates.

Run modes:
  python scripts/data/backfill_scanner_outcomes.py --scanner 52 --lookback 90
  python scripts/data/backfill_scanner_outcomes.py --all --lookback 180

This is slow (per-day indicator recompute × 180 days × 52 scanners). The
backfill is meant to run nightly as part of the unified training pipeline,
not on user-request. Output is cached in scanner_stats forever (next-day
backfill only adds the new day's hits + recomputes aggregates).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
logger = logging.getLogger("backfill_scanner_outcomes")


# Won = forward return crossed +1.5% on close
WIN_THRESHOLD_PCT = 1.5


def _safe_pct(after: float, before: float) -> float:
    if before is None or before <= 0:
        return 0.0
    return (after / before - 1.0) * 100.0


def _per_symbol_summary_at_date(
    bars: pd.DataFrame, target_date: pd.Timestamp,
) -> Dict[str, Any] | None:
    """Build a summary row equivalent to what live LiveScreenerEngine
    produces, but computed AT target_date (no look-ahead)."""
    # Slice to bars up to and including target_date
    hist = bars.loc[bars.index <= target_date]
    if len(hist) < 50:
        return None
    last = hist.iloc[-1]
    close = float(last["close"])
    prev_close = float(hist["close"].iloc[-2]) if len(hist) >= 2 else close
    change_pct = (close / prev_close - 1) * 100 if prev_close > 0 else 0.0

    # Cheap indicators (RSI, MAs, ATR, volume_sma20)
    def _rsi(closes: pd.Series, n: int = 14) -> float:
        if len(closes) < n + 1: return 50.0
        delta = closes.diff().dropna().tail(n)
        gain = float(delta[delta > 0].sum() / n)
        loss = float(-delta[delta < 0].sum() / n)
        if loss == 0: return 100.0
        return 100.0 - 100.0 / (1 + gain / loss)

    def _ema(s: pd.Series, n: int) -> float:
        if len(s) < n: return float(s.iloc[-1])
        return float(s.ewm(span=n, adjust=False).mean().iloc[-1])

    def _sma(s: pd.Series, n: int) -> float:
        if len(s) < n: return float(s.iloc[-1])
        return float(s.tail(n).mean())

    closes = hist["close"]
    highs = hist["high"]
    lows = hist["low"]
    vols = hist["volume"]

    atr = float((highs.tail(14) - lows.tail(14)).mean()) if len(hist) >= 14 else float(closes.std())

    return {
        "symbol": "",  # filled by caller
        "close": close,
        "change_pct": round(change_pct, 2),
        "volume": float(last["volume"]),
        "volume_ratio": float(last["volume"]) / max(1.0, float(vols.tail(20).mean())),
        "volume_sma20": float(vols.tail(20).mean()),
        "rsi_14": _rsi(closes),
        "ema_9": _ema(closes, 9),
        "ema_21": _ema(closes, 21),
        "ema_200": _ema(closes, 200),
        "sma_20": _sma(closes, 20),
        "sma_50": _sma(closes, 50),
        "sma_200": _sma(closes, 200),
        "atr_14": atr,
        "macd": _ema(closes, 12) - _ema(closes, 26),
        "macd_signal": 0.0,            # simplified
        "macd_hist": 0.0,
        "adx": 25.0,                   # simplified
        "bb_upper": float(closes.tail(20).mean() + 2 * closes.tail(20).std()),
        "bb_lower": float(closes.tail(20).mean() - 2 * closes.tail(20).std()),
        "high_52w": float(highs.tail(252).max()),
        "low_52w": float(lows.tail(252).min()),
        "high_10d": float(highs.tail(10).max()),
        "low_10d": float(lows.tail(10).min()),
    }


def _forward_returns(bars: pd.DataFrame, entry_date: pd.Timestamp, entry_price: float):
    """Compute 5d / 10d / 20d forward returns + max drawdown."""
    future = bars.loc[bars.index > entry_date]
    if len(future) < 5:
        return None

    def _ret_at(n: int):
        if len(future) < n:
            return None
        return _safe_pct(float(future["close"].iloc[n - 1]), entry_price)

    # Max drawdown in next 20 bars
    window = future.head(20)
    if window.empty:
        return None
    rolling_min = float(window["low"].min())
    max_dd = _safe_pct(rolling_min, entry_price)

    return {
        "return_5d_pct": _ret_at(5),
        "return_10d_pct": _ret_at(10),
        "return_20d_pct": _ret_at(20),
        "max_drawdown_pct": round(max_dd, 4),
    }


def backfill_scanner(
    scanner_id: int, symbols: List[str], lookback_days: int = 180,
):
    """Replay one scanner across history. Returns a list of outcome dicts."""
    from backend.data.screener.filters import (
        SCANNER_FILTERS, PATTERN_SCANNERS, EXTERNAL_DATA_SCANNERS,
    )
    from backend.data.market import get_market_data_provider

    if scanner_id in PATTERN_SCANNERS or scanner_id in EXTERNAL_DATA_SCANNERS:
        logger.info("scanner %d: pattern/external, skipping", scanner_id)
        return []

    filter_fn = SCANNER_FILTERS.get(scanner_id)
    if filter_fn is None:
        return []

    mp = get_market_data_provider()

    # Pull bars for all symbols
    bars_by: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = mp.get_historical(sym, period="2y", interval="1d")
            if df is not None and len(df) >= 60:
                df = df.copy()
                df.columns = [c.lower() for c in df.columns]
                df.index = pd.to_datetime(df.index)
                bars_by[sym] = df
        except Exception:
            continue

    logger.info("scanner %d: loaded bars for %d/%d symbols",
                scanner_id, len(bars_by), len(symbols))

    if not bars_by:
        return []

    # Iterate days (we want entry_date + at least 20 bars of forward)
    today = pd.Timestamp.now().normalize()
    earliest = today - pd.Timedelta(days=lookback_days)
    latest_entry = today - pd.Timedelta(days=21)    # need 20 fwd bars

    # Build a common date index from one symbol
    sample = next(iter(bars_by.values()))
    test_dates = sample.loc[
        (sample.index >= earliest) & (sample.index <= latest_entry)
    ].index.tolist()

    outcomes: List[Dict[str, Any]] = []
    for entry_date in test_dates:
        # Build summary_df AT this date
        rows = []
        for sym, bars in bars_by.items():
            row = _per_symbol_summary_at_date(bars, entry_date)
            if row:
                row["symbol"] = sym
                rows.append(row)
        if not rows:
            continue
        summary_df = pd.DataFrame(rows)

        try:
            matched = filter_fn(summary_df.copy())
        except Exception:
            continue
        if matched is None or matched.empty:
            continue

        # For each hit, compute forward returns
        for _, m in matched.iterrows():
            sym = m["symbol"]
            entry_price = float(m["close"])
            fwd = _forward_returns(bars_by[sym], entry_date, entry_price)
            if fwd is None:
                continue
            r5 = fwd.get("return_5d_pct")
            r10 = fwd.get("return_10d_pct")
            outcomes.append({
                "scanner_id": scanner_id,
                "symbol": sym,
                "hit_date": entry_date.date().isoformat(),
                "entry_price": round(entry_price, 2),
                "return_5d_pct": round(r5, 4) if r5 is not None else None,
                "return_10d_pct": round(r10, 4) if r10 is not None else None,
                "return_20d_pct": round(fwd.get("return_20d_pct") or 0, 4),
                "max_drawdown_pct": round(fwd.get("max_drawdown_pct") or 0, 4),
                "won_5d": (r5 is not None and r5 >= WIN_THRESHOLD_PCT),
                "won_10d": (r10 is not None and r10 >= WIN_THRESHOLD_PCT),
            })

    return outcomes


def compute_stats(outcomes: List[Dict[str, Any]], scanner_id: int,
                  lookback_days: int) -> Dict[str, Any]:
    if not outcomes:
        return {
            "scanner_id": scanner_id, "total_hits": 0,
            "win_rate_5d": 0, "win_rate_10d": 0,
            "avg_return_5d_pct": 0, "avg_return_10d_pct": 0,
            "median_return_5d_pct": 0, "median_return_10d_pct": 0,
            "avg_drawdown_pct": 0, "lookback_days": lookback_days,
        }

    rets_5d = [o["return_5d_pct"] for o in outcomes if o.get("return_5d_pct") is not None]
    rets_10d = [o["return_10d_pct"] for o in outcomes if o.get("return_10d_pct") is not None]
    dds = [o["max_drawdown_pct"] for o in outcomes if o.get("max_drawdown_pct") is not None]
    wins_5d = sum(1 for o in outcomes if o.get("won_5d"))
    wins_10d = sum(1 for o in outcomes if o.get("won_10d"))

    return {
        "scanner_id": scanner_id,
        "total_hits": len(outcomes),
        "win_rate_5d": round(wins_5d / len(outcomes), 4),
        "win_rate_10d": round(wins_10d / len(outcomes), 4),
        "avg_return_5d_pct": round(float(np.mean(rets_5d)), 4) if rets_5d else 0,
        "avg_return_10d_pct": round(float(np.mean(rets_10d)), 4) if rets_10d else 0,
        "median_return_5d_pct": round(float(np.median(rets_5d)), 4) if rets_5d else 0,
        "median_return_10d_pct": round(float(np.median(rets_10d)), 4) if rets_10d else 0,
        "avg_drawdown_pct": round(float(np.mean(dds)), 4) if dds else 0,
        "lookback_days": lookback_days,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scanner", type=int, help="single scanner_id")
    parser.add_argument("--all", action="store_true", help="run all eligible scanners")
    parser.add_argument("--lookback", type=int, default=180)
    parser.add_argument("--universe", default="nifty100",
                        help="nifty50/100/500/nse_all")
    args = parser.parse_args()

    if not (args.scanner is not None or args.all):
        parser.error("either --scanner ID or --all required")

    from backend.data.screener.filters import (
        SCANNER_FILTERS, PATTERN_SCANNERS, EXTERNAL_DATA_SCANNERS,
    )
    from backend.ai.qlib.data_handler import load_universe
    from backend.core.database import get_supabase_admin

    symbols = load_universe(args.universe)[:200]    # cap for backfill speed
    logger.info("Backfill universe: %d symbols", len(symbols))

    targets = (
        [args.scanner] if args.scanner is not None
        else [s for s in SCANNER_FILTERS.keys()
              if s not in PATTERN_SCANNERS and s not in EXTERNAL_DATA_SCANNERS and s != 0]
    )

    sb = get_supabase_admin()
    for sid in targets:
        logger.info("=== Scanner %d ===", sid)
        outcomes = backfill_scanner(sid, symbols, lookback_days=args.lookback)
        if outcomes:
            # Upsert outcomes (chunked)
            for i in range(0, len(outcomes), 200):
                chunk = outcomes[i:i + 200]
                try:
                    sb.table("scanner_outcomes").upsert(
                        chunk, on_conflict="scanner_id,symbol,hit_date",
                    ).execute()
                except Exception as e:
                    logger.warning("upsert chunk failed: %s", e)
            logger.info("scanner %d: %d outcomes persisted", sid, len(outcomes))

        # Refresh stats
        stats = compute_stats(outcomes, sid, args.lookback)
        try:
            sb.table("scanner_stats").upsert(stats, on_conflict="scanner_id").execute()
            logger.info(
                "scanner %d stats: n=%d wr5d=%.0f%% avg5d=%.2f%%",
                sid, stats["total_hits"], stats["win_rate_5d"] * 100,
                stats["avg_return_5d_pct"],
            )
        except Exception as e:
            logger.warning("stats upsert failed: %s", e)

    logger.info("Backfill complete.")


if __name__ == "__main__":
    main()
