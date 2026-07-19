"""Read/upsert helpers for the durable `candles` daily OHLC store (Supabase).

Pure transforms (`df_to_candle_rows`, `rows_to_df`) are tested; the Supabase
read/upsert wrappers are thin and honest-empty on failure."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

_CANDLE_COLS = (
    "stock_symbol", "exchange", "interval", "timestamp", "open", "high",
    "low", "close", "volume", "delivery_qty", "delivery_pct", "source",
)
_TS_FORMATS = ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S")


def df_to_candle_rows(symbol: str, df: "pd.DataFrame", interval: str = "1d",
                      source: str = "nselib") -> List[Dict]:
    rows: List[Dict] = []
    if df is None or df.empty:
        return rows
    for idx, r in df.iterrows():
        ts = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        rows.append({
            "stock_symbol": symbol, "exchange": "NSE", "interval": interval,
            "timestamp": ts,
            "open": _f(r.get("open")), "high": _f(r.get("high")),
            "low": _f(r.get("low")), "close": _f(r.get("close")),
            "volume": _i(r.get("volume")), "source": source,
        })
    return rows


def rows_to_df(rows: List[Dict]) -> "pd.DataFrame":
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("ts").sort_index()
    df.index.name = None
    return df[["open", "high", "low", "close", "volume"]]


def read_candles(supabase, symbol: str, interval: str = "1d", limit: int = 500) -> List[Dict]:
    try:
        res = (supabase.table("candles").select("*")
               .eq("stock_symbol", symbol).eq("interval", interval)
               .order("timestamp", desc=True).limit(limit).execute())
        return list(reversed(res.data or []))
    except Exception as e:
        logger.debug("read_candles failed for %s: %s", symbol, e)
        return []


def _parse_ts(v) -> str:
    """nselib hands back 'DD-Mon-YYYY' (e.g. '30-Apr-2026') or ISO; Postgres wants
    an unambiguous date. Return an ISO date string (Postgres parses it directly)."""
    s = str(v).strip()
    head = s[:19] if "T" in s else s
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(head, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, dayfirst=True).date().isoformat()
    except Exception:
        return s  # last resort: let Postgres try to parse it


def _candle_tuple(r: Dict) -> tuple:
    return (
        r.get("stock_symbol"), r.get("exchange", "NSE"), r.get("interval", "1d"),
        _parse_ts(r.get("timestamp")),
        r.get("open"), r.get("high"), r.get("low"), r.get("close"),
        r.get("volume"), r.get("delivery_qty"), r.get("delivery_pct"),
        r.get("source", "nselib"),
    )


def pg_connect(dsn: str = None):
    """Open a psycopg2 connection from a DSN via PARSED kwargs.

    psycopg2 mangles non-ASCII bytes (e.g. a '£' in the password) when they are
    embedded in a DSN *string* passed to connect(), but handles them correctly
    as discrete keyword arguments. So we always parse the DSN and connect via
    **kwargs. Requires DATABASE_URL when no explicit dsn is given."""
    import psycopg2
    from psycopg2.extensions import parse_dsn
    dsn = dsn or os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(**parse_dsn(dsn))


def pg_upsert_candles(rows: List[Dict], conn=None) -> int:
    """Reliable bulk upsert via DIRECT Postgres (psycopg2 + execute_values).

    The PostgREST/Supabase-pooler path silently drops bulk upserts to the
    partitioned `candles` table for symbols that already have rows (the call
    echoes res.data but nothing persists). This path persists reliably and
    returns the true affected rowcount. Requires DATABASE_URL. Caller may pass
    an already-open `conn` (the caller then owns commit); otherwise we open,
    commit, and close our own connection."""
    if not rows:
        return 0
    from psycopg2.extras import execute_values

    # de-dupe within the batch on the PK (last write wins) — nselib yearly
    # chunks overlap a day at the boundaries.
    dedup: Dict[tuple, tuple] = {}
    for r in rows:
        t = _candle_tuple(r)
        dedup[(t[0], t[2], t[3])] = t
    values = list(dedup.values())

    owns = conn is None
    if owns:
        conn = pg_connect()
    try:
        sql = (
            "INSERT INTO candles (stock_symbol,exchange,interval,timestamp,"
            "open,high,low,close,volume,delivery_qty,delivery_pct,source) "
            "VALUES %s ON CONFLICT (stock_symbol,interval,timestamp) DO UPDATE SET "
            "open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, "
            "close=EXCLUDED.close, volume=EXCLUDED.volume, "
            "delivery_qty=EXCLUDED.delivery_qty, delivery_pct=EXCLUDED.delivery_pct, "
            "source=EXCLUDED.source"
        )
        with conn.cursor() as cur:
            execute_values(cur, sql, values, page_size=1000)
            n = cur.rowcount
        if owns:
            conn.commit()
        return n if (n is not None and n >= 0) else len(values)
    except Exception:
        if owns:
            conn.rollback()
        raise
    finally:
        if owns:
            conn.close()


def upsert_candles(supabase, rows: List[Dict]) -> int:
    """Upsert daily candles. Prefers the reliable direct-Postgres path when
    DATABASE_URL is configured (the only path that actually persists bulk
    upserts to the partitioned table); falls back to PostgREST otherwise."""
    if not rows:
        return 0
    if os.getenv("DATABASE_URL"):
        try:
            return pg_upsert_candles(rows)
        except Exception as e:
            logger.warning("pg_upsert_candles failed, falling back to PostgREST: %s", e)
    try:
        supabase.table("candles").upsert(rows, on_conflict="stock_symbol,interval,timestamp").execute()
        return len(rows)
    except Exception as e:
        logger.debug("upsert_candles failed (%d rows): %s", len(rows), e)
        return 0


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
