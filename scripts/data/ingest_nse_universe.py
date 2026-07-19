#!/usr/bin/env python3
"""Ingest the full NSE reference universe into Supabase.

Populates:
  - instruments        : all NSE main-board equities (EQUITY_L) + sector +
                         mcap_category derived from index membership.
  - index_constituents : broad-market + sectoral + F&O-eligible memberships
                         from NSE's public archive CSVs.
  - data/nse_all_symbols.json : refreshed full-universe symbol cache.

Reference data only (symbol lists + index membership + sector) — freely
publishable, distinct from the licensed live price plane. Uses the reliable
direct-Postgres upsert path (psycopg2). Honest-empty per index on failure.

Usage: .venv/bin/python scripts/data/ingest_nse_universe.py
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))

from psycopg2.extras import execute_values  # noqa: E402
from backend.data.ohlc_store import pg_connect  # noqa: E402
from backend.data.reference import nse_reference as nref  # noqa: E402


def _upsert(conn, table, rows, pk_cols, update_cols):
    """Reliable bulk upsert via execute_values; returns affected rowcount."""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    values = [[r.get(c) for c in cols] for r in rows]
    set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
    sql = (
        f"INSERT INTO {table} ({','.join(cols)}) VALUES %s "
        f"ON CONFLICT ({','.join(pk_cols)}) DO UPDATE SET {set_clause}"
    )
    with conn.cursor() as cur:
        execute_values(cur, sql, values, page_size=1000)
        n = cur.rowcount
    conn.commit()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fno", action="store_true")
    args = ap.parse_args()

    conn = pg_connect()
    print("connected to Postgres", flush=True)

    # ── 1) equity master -> instruments ──
    eq_rows = nref.map_equity_master_rows(nref.fetch_equity_master())
    for r in eq_rows:
        r["expiry"] = "1900-01-01"
        r["strike"] = 0
    n = _upsert(
        conn, "instruments", eq_rows,
        ["symbol", "exchange", "instrument_type", "expiry", "strike"],
        ["isin", "series", "name", "face_value", "listing_date", "status", "source"],
    )
    print(f"instruments: {len(eq_rows)} rows (affected {n})", flush=True)

    # ── 2) index constituents ──
    cbi = {}
    total_mem = 0
    for index_name, (fn, _cat) in nref.INDEX_CSV_MAP.items():
        rows = nref.map_index_constituent_rows(
            nref.fetch_index_constituents_csv(fn), index_name)
        cbi[index_name] = rows
        if rows:
            _upsert(conn, "index_constituents", rows,
                    ["index_name", "symbol"], ["weight", "industry", "source"])
            total_mem += len(rows)
        print(f"  {index_name:28s} {len(rows):4d}", flush=True)
        time.sleep(0.4)

    # F&O-eligible stocks as a synthetic "index"
    if not args.skip_fno:
        try:
            frows = [{"index_name": nref.FNO_INDEX_NAME, "symbol": s,
                      "weight": None, "industry": None, "source": "nselib"}
                     for s in nref.fetch_fno_stock_symbols()]
            cbi[nref.FNO_INDEX_NAME] = frows
            if frows:
                _upsert(conn, "index_constituents", frows,
                        ["index_name", "symbol"], ["weight", "industry", "source"])
                total_mem += len(frows)
            print(f"  {nref.FNO_INDEX_NAME:28s} {len(frows):4d}", flush=True)
        except Exception as e:
            print(f"  F&O fetch failed: {e}", flush=True)

    # ── 3) derive sector + mcap -> instruments ──
    sector = nref.build_sector_map(cbi)
    mcap = nref.build_mcap_map(cbi)
    syms = set(sector) | set(mcap)
    upd = [[s, sector.get(s), mcap.get(s)] for s in syms]
    if upd:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "UPDATE instruments AS i SET "
                "sector = COALESCE(v.sector::text, i.sector), "
                "mcap_category = COALESCE(v.mcap::text, i.mcap_category), "
                "updated_at = now() "
                "FROM (VALUES %s) AS v(symbol, sector, mcap) "
                "WHERE i.symbol = v.symbol::text AND i.instrument_type='EQ'",
                upd, template="(%s,%s,%s)", page_size=1000)
            conn.commit()
    print(f"sectors set: {len(sector)} | mcap set: {len(mcap)}", flush=True)

    # ── 4) regenerate nse_all_symbols.json ──
    all_syms = sorted({r["symbol"] for r in eq_rows})
    cache = {"updated_at": dt.date.today().isoformat(),
             "count": len(all_syms), "symbols": all_syms}
    with open(os.path.join(ROOT, "data", "nse_all_symbols.json"), "w") as f:
        json.dump(cache, f, indent=2)
    print(f"nse_all_symbols.json: {len(all_syms)} symbols", flush=True)

    conn.close()
    print(f"DONE: {len(eq_rows)} instruments, {total_mem} memberships "
          f"across {len(cbi)} indices", flush=True)


if __name__ == "__main__":
    main()
