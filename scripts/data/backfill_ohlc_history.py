#!/usr/bin/env python3
"""Backfill multi-year daily OHLC into the candles store, chunked by ~1 year.

nselib's price_volume_and_deliverable_position_data caps a single request at
~1 year, so we walk back year-by-year. Idempotent (upsert_candles ON CONFLICT),
honest-empty per chunk. Backtesting needs >=210 bars/symbol; the short Kite load
only had ~201, so this fills the history that makes backtests runnable.

Usage:
  .venv/bin/python scripts/data/backfill_ohlc_history.py --file /tmp/bt_symbols.txt --years 3
"""
import argparse
import datetime as dt
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
from backend.core.database import get_supabase_admin  # noqa: E402
from backend.data.providers.nselib_source import get_nselib_provider  # noqa: E402
from backend.data.ohlc_store import upsert_candles  # noqa: E402


def yearly_chunks(years: int):
    """[(from, to)] DD-MM-YYYY, newest first, each spanning ~1 year."""
    out, end = [], dt.date.today()
    for _ in range(years):
        start = end - dt.timedelta(days=365)
        out.append((start.strftime("%d-%m-%Y"), end.strftime("%d-%m-%Y")))
        end = start - dt.timedelta(days=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="file of comma/newline-separated symbols")
    ap.add_argument("--years", type=int, default=3)
    args = ap.parse_args()

    with open(args.file) as f:
        raw = f.read().replace("\n", ",")
    syms = [s.strip() for s in raw.split(",") if s.strip() and s.strip().lower() != "none"]

    sb = get_supabase_admin()
    prov = get_nselib_provider()
    chunks = yearly_chunks(args.years)
    print(f"Backfilling {len(syms)} symbols x {args.years}yr ({len(chunks)} chunks each)", flush=True)

    total, ok = 0, 0
    for i, sym in enumerate(syms, 1):
        n = 0
        for frm, to in chunks:
            try:
                rows = prov.get_daily_ohlc(sym, frm, to)
                if rows:
                    n += upsert_candles(sb, rows)
            except Exception as exc:
                print(f"  {sym} {frm}->{to}: ERR {type(exc).__name__}: {str(exc)[:60]}", flush=True)
        total += n
        if n > 0:
            ok += 1
        print(f"[{i}/{len(syms)}] {sym}: +{n} rows (cum {total}, {ok} ok)", flush=True)
        time.sleep(0.25)  # be gentle on NSE
    print(f"DONE: {total} rows across {ok}/{len(syms)} symbols", flush=True)


if __name__ == "__main__":
    main()
