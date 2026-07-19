#!/usr/bin/env python3
"""
End-to-end data-source test + backfill for the F0-F5 centralized plane.

Runs each free source through the REAL app code (fetch -> map -> store -> Supabase)
and reports rows fetched/written per source. Bhavcopy-based sources try the most
recent completed trading days (today's EOD isn't published until ~18:00 IST).

Usage: .venv/bin/python scripts/data/backfill_data.py [--full]
  (default uses a small symbol sample for per-symbol sources; --full uses nifty500)
"""
import sys, os, time, datetime as dt, argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from backend.core.database import get_supabase_admin
from backend.data.reference import nse_reference, nse_orderflow as of, nse_derivatives as nd
from backend.data.providers.nselib_source import get_nselib_provider
from backend.data.ohlc_store import upsert_candles
from backend.data.orderflow_store import upsert_rows
from backend.platform.scheduler import equity_master_to_rows, fundamentals_to_row
from backend.data.fundamentals.screener_in import get_fundamentals

ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true")
args = ap.parse_args()

sb = get_supabase_admin()
SAMPLE = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "LT"]
results = []


def recent_weekdays(n=6, skip_today=True):
    out, d = [], dt.date.today()
    if skip_today:
        d -= dt.timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out


def section(name):
    print(f"\n{'='*70}\n{name}\n{'='*70}")


def rec(name, fetched, written, note=""):
    results.append((name, fetched, written, note))
    print(f"  -> {name}: fetched={fetched} written={written} {note}")


# 1) REFERENCE — instrument master (nselib, not date-based) -------------------
section("1) nselib REFERENCE — instrument master")
try:
    t = time.time()
    df = nse_reference.fetch_equity_master()
    rows = equity_master_to_rows(df)
    w = 0
    for i in range(0, len(rows), 500):
        try:
            sb.table("instruments").upsert(
                rows[i:i+500], on_conflict="symbol,exchange,instrument_type,expiry,strike").execute()
            w += len(rows[i:i+500])
        except Exception as e:
            print(f"    upsert chunk failed: {str(e)[:80]}")
    rec("instruments", len(rows), w, f"({time.time()-t:.1f}s)")
except Exception as e:
    rec("instruments", 0, 0, f"ERR {type(e).__name__}: {str(e)[:80]}")

# 2) OHLC — nselib daily for a few symbols ------------------------------------
section("2) nselib OHLC — daily candles (sample)")
try:
    prov = get_nselib_provider()
    days = recent_weekdays(8, skip_today=True)
    frm, to = days[-1].strftime("%d-%m-%Y"), days[0].strftime("%d-%m-%Y")
    fetched = written = 0
    for sym in SAMPLE[:5]:
        try:
            r = prov.get_daily_ohlc(sym, frm, to)
            fetched += len(r)
            written += upsert_candles(sb, r)
        except Exception as e:
            print(f"    {sym}: {str(e)[:70]}")
    rec("candles", fetched, written, f"({frm}..{to}, 5 syms)")
except Exception as e:
    rec("candles", 0, 0, f"ERR {type(e).__name__}: {str(e)[:80]}")

# 3) ORDER FLOW — FII/DII + participant-OI + bulk/block + short + ban ----------
section("3) nselib ORDER FLOW (most recent published day)")
for label, fetch, mapper, table, conflict in [
    ("fii_dii_flow_eod", lambda d: of.fetch_fii_dii(), of.map_fii_dii_rows, "fii_dii_flow_eod", "date,segment"),
    ("participant_oi_eod", lambda d: of.fetch_participant_oi(d), of.map_participant_oi_rows, "participant_oi_eod", "date,participant"),
    ("short_selling", lambda d: of.fetch_short_selling(d, d), of.map_short_selling_rows, "short_selling", "date,symbol"),
    ("fno_ban", lambda d: of.fetch_fno_ban(d), of.map_fno_ban_symbols, "fno_ban", "date,symbol"),
]:
    try:
        got = 0
        for day in recent_weekdays(4, skip_today=False):
            d_nse, d_iso = day.strftime("%d-%m-%Y"), day.isoformat()
            try:
                raw = fetch(d_nse)
                rows = mapper(raw, d_iso) if table != "fno_ban" else mapper(raw, d_iso)
                if rows:
                    got = upsert_rows(sb, table, rows, conflict)
                    rec(table, len(rows), got, f"({d_iso})")
                    break
            except Exception as e:
                continue
        if got == 0:
            rec(table, 0, 0, "(no data on recent days)")
    except Exception as e:
        rec(table, 0, 0, f"ERR {type(e).__name__}: {str(e)[:70]}")

# bulk/block deals (date range)
try:
    got = 0
    for day in recent_weekdays(4, skip_today=False):
        d_nse = day.strftime("%d-%m-%Y")
        try:
            raw = of.fetch_bulk_deals(d_nse, d_nse)
            rows = of.map_bulk_block_rows(raw, "BULK")
            if rows:
                got = upsert_rows(sb, "bulk_block_deals", rows, "date,symbol,deal_type,client_name,buy_sell,qty")
                rec("bulk_block_deals", len(rows), got, f"({day.isoformat()})")
                break
        except Exception:
            continue
    if got == 0:
        rec("bulk_block_deals", 0, 0, "(no data on recent days)")
except Exception as e:
    rec("bulk_block_deals", 0, 0, f"ERR {str(e)[:70]}")

# 4) DERIVATIVES — F&O bhavcopy -> option chain + futures + metrics -----------
section("4) nselib DERIVATIVES — F&O bhavcopy (most recent published day)")
try:
    done = False
    for day in recent_weekdays(5, skip_today=False):
        d_nse, d_iso = day.strftime("%d-%m-%Y"), day.isoformat()
        try:
            df = nd.fetch_fno_bhavcopy(d_nse)
            opt = nd.map_fno_options_rows(df, d_iso)
            fut = nd.map_fno_futures_rows(df, d_iso)
            met = nd.build_derivatives_metrics(opt)
            if opt or fut:
                w = upsert_rows(sb, "options_chain_eod", opt, "date,symbol,expiry,strike,option_type")
                w += upsert_rows(sb, "futures_eod", fut, "date,symbol,expiry")
                w += upsert_rows(sb, "derivatives_metrics_eod", met, "date,symbol,expiry")
                rec("derivatives (opt+fut+met)", len(opt)+len(fut)+len(met), w, f"({d_iso}: opt={len(opt)} fut={len(fut)} met={len(met)})")
                done = True
                break
        except Exception as e:
            continue
    if not done:
        rec("derivatives", 0, 0, "(no F&O bhavcopy on recent days)")
except Exception as e:
    rec("derivatives", 0, 0, f"ERR {type(e).__name__}: {str(e)[:80]}")

# 5) FUNDAMENTALS — screener.in (sample) --------------------------------------
section("5) screener.in FUNDAMENTALS (sample)")
try:
    snap = dt.date.today().isoformat()
    syms = SAMPLE if not args.full else SAMPLE  # keep sample for the test
    rows = []
    for sym in syms:
        try:
            data = get_fundamentals(sym)
            row = fundamentals_to_row(sym, data, snap)
            if row:
                rows.append(row)
                print(f"    {sym}: pe={row.get('pe')} roe={row.get('roe')} mcap_cr={row.get('market_cap_cr')}")
        except Exception as e:
            print(f"    {sym}: {str(e)[:70]}")
    w = upsert_rows(sb, "fundamentals_history", rows, "snapshot_date,symbol")
    rec("fundamentals_history", len(rows), w, f"({snap})")
except Exception as e:
    rec("fundamentals_history", 0, 0, f"ERR {type(e).__name__}: {str(e)[:80]}")

# SUMMARY ---------------------------------------------------------------------
section("SUMMARY")
ok = sum(1 for _, f, w, _ in results if w > 0)
for name, f, w, note in results:
    flag = "OK " if w > 0 else ("-- " if "no data" in note else "ERR")
    print(f"  [{flag}] {name:30s} fetched={f:<7} written={w:<7} {note}")
print(f"\n{ok}/{len(results)} sources wrote data.")
