"""nselib-backed daily OHLC + delivery source (F1, centralized EOD plane).

`normalize_bhavcopy_rows` is pure (tested); `get_daily_ohlc` lazily fetches via
nselib and normalizes. Honest-empty on any failure (never fabricates bars)."""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def _num(v):
    try:
        f = float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    # float('nan') parses fine but breaks int() and pollutes the DB — reject it
    # (NSE leaves blank int fields like DeliverableQty on some days -> NaN).
    return None if f != f else f


def _intnum(v):
    n = _num(v)
    return int(n) if n is not None else None


# BOM + zero-width + non-breaking-space chars NSE prepends to the FIRST CSV
# column (so "Symbol" arrives as "﻿Symbol" and name-matching silently
# fails, leaving the symbol unread -> rows land under the string "None").
_ZW = "﻿​‌‍ "


def _norm_col(s) -> str:
    return str(s).strip().strip(_ZW).strip().lower().replace("%", "").replace(" ", "")


def normalize_bhavcopy_rows(df: "pd.DataFrame", symbol: str = None) -> List[Dict]:
    """Normalize an nselib bhavcopy/price-volume DataFrame to `candles` rows.
    Column-name-tolerant (nselib revisions vary; first column carries a BOM).
    When `symbol` is given it is stamped on every row (bulletproof: the caller
    requested one symbol), instead of trusting the BOM-prone Symbol column."""
    if df is None or len(df) == 0:
        return []

    def pick(row, *names):
        for n in names:
            target = _norm_col(n)
            for k in row.index:
                if _norm_col(k) == target:
                    return row[k]
        return None

    forced = (symbol or "").strip().upper() or None
    rows: List[Dict] = []
    for _, r in df.iterrows():
        if forced:
            sym = forced
        else:
            sym = (str(pick(r, "Symbol", "SYMBOL")) or "").strip()
            if sym.lower() in ("", "none", "nan"):
                sym = ""
        date = str(pick(r, "Date", "DATE1", "TIMESTAMP") or "").strip()
        close = _num(pick(r, "ClosePrice", "close", "CLOSE_PRICE"))
        if not sym or not date or close is None:
            continue
        rows.append({
            "stock_symbol": sym, "exchange": "NSE", "interval": "1d",
            "timestamp": date,
            "open": _num(pick(r, "OpenPrice", "open")),
            "high": _num(pick(r, "HighPrice", "high")),
            "low": _num(pick(r, "LowPrice", "low")),
            "close": close,
            "volume": _intnum(pick(r, "TotalTradedQuantity", "volume", "TTL_TRD_QNTY")),
            "delivery_qty": _intnum(pick(r, "DeliverableQty", "DELIV_QTY")),
            "delivery_pct": _num(pick(r, "DlyQttoTradedQty", "DELIV_PER")),
            "source": "nselib",
        })
    return rows


class NselibProvider:
    """Daily OHLC+delivery via nselib (lazy import). Honest-empty on failure."""

    def get_daily_ohlc(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        try:
            from nselib import capital_market
            df = capital_market.price_volume_and_deliverable_position_data(
                symbol=symbol, from_date=from_date, to_date=to_date)
            return normalize_bhavcopy_rows(df, symbol=symbol)
        except Exception as e:
            logger.debug("nselib get_daily_ohlc failed for %s: %s", symbol, e)
            return []  # honest-empty


_provider = None


def get_nselib_provider() -> "NselibProvider":
    global _provider
    if _provider is None:
        _provider = NselibProvider()
    return _provider
