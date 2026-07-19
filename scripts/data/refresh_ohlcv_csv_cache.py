#!/usr/bin/env python3
"""Refresh the offline OHLCV CSV cache (data/cache/*.csv) through today.

Full-history REWRITE per symbol — never append. yfinance auto_adjust re-bases
the whole series at every fetch, so appending fresh adjusted rows onto an old
adjusted file corrupts prices across any split/bonus that happened in between.
Rewriting the file keeps the entire series consistently adjusted as of today.

Safety: a symbol's file is only replaced when the fresh fetch (a) is non-empty,
(b) has >= 90% of the old row count, and (c) ends on/after the old last date.
Failures keep the old file and are listed in the summary (stale > corrupt).

Usage:
  python3 scripts/data/refresh_ohlcv_csv_cache.py            # all cache files
  python3 scripts/data/refresh_ohlcv_csv_cache.py --only RELIANCE NSEI
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
COLS = ["open", "high", "low", "close", "volume"]


def _ticker_for(path: Path) -> str:
    name = path.name
    if name == "NSEI_10y.csv":
        return "^NSEI"
    return name[: -len("_NS_10y.csv")] + ".NS"


def _fetch(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf  # noqa: PLC0415

    raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    # flatten single-ticker MultiIndex columns; lowercase
    raw.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in raw.columns]
    if not set(COLS).issubset(raw.columns):
        return pd.DataFrame()
    df = raw[COLS].dropna(subset=["close"]).copy()
    # match the existing cache stamp format: tz-aware IST midnight
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("Asia/Kolkata")
    else:
        idx = idx.tz_convert("Asia/Kolkata")
    df.index = idx.normalize()
    df.index.name = "date"
    df["volume"] = df["volume"].fillna(0).astype("int64")
    return df


def refresh_one(path: Path) -> tuple[str, str]:
    """Returns (symbol, status) — status in {refreshed, failed:<why>}."""
    ticker = _ticker_for(path)
    old = pd.read_csv(path)
    old_n = len(old)
    old_first = pd.to_datetime(old["date"].iloc[0], utc=True)
    old_last = pd.to_datetime(old["date"].iloc[-1], utc=True)
    start = old_first.tz_convert("Asia/Kolkata").date().isoformat()

    try:
        fresh = _fetch(ticker, start)
    except Exception as exc:  # noqa: BLE001
        return ticker, f"failed:fetch:{type(exc).__name__}"
    if fresh.empty:
        return ticker, "failed:empty"
    if len(fresh) < int(old_n * 0.9):
        return ticker, f"failed:truncated({len(fresh)}<{old_n})"
    fresh_last = fresh.index[-1].tz_convert("UTC")
    if fresh_last < old_last:
        return ticker, "failed:not_fresher"

    tmp = path.with_suffix(".csv.tmp")
    fresh.reset_index().to_csv(tmp, index=False)
    tmp.replace(path)
    return ticker, "refreshed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="bare symbols (RELIANCE, NSEI, ...)")
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    files = sorted(CACHE.glob("*.csv"))
    if args.only:
        want = {s.upper() for s in args.only}
        files = [f for f in files
                 if (f.name == "NSEI_10y.csv" and "NSEI" in want)
                 or f.name.replace("_NS_10y.csv", "") in want]

    ok, failed = [], []
    for i, f in enumerate(files, 1):
        sym, status = refresh_one(f)
        (ok if status == "refreshed" else failed).append((sym, status))
        print(f"[{i}/{len(files)}] {sym}: {status}", flush=True)
        time.sleep(args.sleep)

    print(f"\nSUMMARY: refreshed={len(ok)} failed={len(failed)}")
    for sym, status in failed:
        print(f"  KEPT-OLD {sym}: {status}")
    return 0 if len(ok) > 0 and len(failed) < len(files) * 0.1 else 1


if __name__ == "__main__":
    sys.exit(main())
