#!/usr/bin/env python3
"""Reliable multi-year daily OHLC backfill via DIRECT Postgres (psycopg2).

Why this exists: the PostgREST/Supabase-pooler path silently drops bulk upserts
to the partitioned `candles` table for symbols that already have rows (the call
echoes res.data but nothing persists, and the optimistic count lies). This
loader holds ONE psycopg2 connection, upserts with execute_values + ON CONFLICT,
and prints the TRUE persisted bar count per symbol so coverage is honest.

nselib caps a single price_volume request at ~1 year, so we walk back
year-by-year. Empty fetches (NSE throttling) are retried with backoff.
Backtesting needs >=210 bars/symbol; the short Kite load only had ~201.

Requires DATABASE_URL in .env (Supabase Session pooler, port 5432).

Usage:
  .venv/bin/python scripts/data/backfill_ohlc_pg.py --file /tmp/bt_symbols.txt --years 3
  .venv/bin/python scripts/data/backfill_ohlc_pg.py --symbols ABB,RELIANCE --years 3
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

from backend.data.providers.nselib_source import get_nselib_provider  # noqa: E402
from backend.data.ohlc_store import pg_connect, pg_upsert_candles  # noqa: E402


def yearly_chunks(years):
    """[(from, to)] DD-MM-YYYY, newest first, each spanning ~1 year."""
    out, end = [], dt.date.today()
    for _ in range(years):
        start = end - dt.timedelta(days=365)
        out.append((start.strftime("%d-%m-%Y"), end.strftime("%d-%m-%Y")))
        end = start - dt.timedelta(days=1)
    return out


def fetch_with_retry(prov, sym, frm, to, tries=3):
    """nselib returns honest-empty on throttling; retry a few times with backoff."""
    for i in range(tries):
        rows = prov.get_daily_ohlc(sym, frm, to)
        if rows:
            return rows
        time.sleep(1.0 + i)
    return []


def load_symbols(args):
    syms = []
    if args.file:
        with open(args.file) as f:
            raw = f.read().replace("\n", ",")
        syms += [s.strip() for s in raw.split(",") if s.strip() and s.strip().lower() != "none"]
    if args.symbols:
        syms += [s.strip() for s in args.symbols.split(",") if s.strip()]
    seen = set()
    return [s for s in syms if not (s in seen or seen.add(s))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="file of comma/newline-separated symbols")
    ap.add_argument("--symbols", help="comma-separated symbols")
    ap.add_argument("--years", type=int, default=3)
    args = ap.parse_args()

    syms = load_symbols(args)
    if not syms:
        print("ERROR: no symbols (use --file or --symbols)", flush=True)
        sys.exit(2)

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set in .env", flush=True)
        sys.exit(2)

    prov = get_nselib_provider()
    chunks = yearly_chunks(args.years)
    conn = pg_connect(dsn)
    print(f"Backfill(PG) {len(syms)} symbols x {args.years}yr ({len(chunks)} chunks each)", flush=True)

    total, ok, empties = 0, 0, []
    for i, sym in enumerate(syms, 1):
        rows = []
        for frm, to in chunks:
            rows += fetch_with_retry(prov, sym, frm, to)
            time.sleep(0.4)  # be gentle on NSE between chunks
        n = 0
        if rows:
            try:
                n = pg_upsert_candles(rows, conn=conn)
                conn.commit()
            except Exception as exc:
                conn.rollback()
                print(f"[{i}/{len(syms)}] {sym}: DB ERR {type(exc).__name__}: {str(exc)[:80]}", flush=True)
                continue
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM candles WHERE stock_symbol=%s AND interval='1d'",
                (sym,))
            have = cur.fetchone()[0]
        total += n
        if have >= 210:
            ok += 1
        if not rows:
            empties.append(sym)
        print(f"[{i}/{len(syms)}] {sym}: upserted {n} (now {have} bars)", flush=True)

    conn.close()
    print(f"DONE: {total} rows upserted; {ok}/{len(syms)} symbols now have >=210 bars", flush=True)
    if empties:
        print(f"EMPTY (no nselib data, retried): {','.join(empties)}", flush=True)


if __name__ == "__main__":
    main()
