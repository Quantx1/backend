"""
Qlib binary format writer — NSE variant.

Qlib stores each (symbol, field) time series as a single little-endian
float32 file::

    features/<SYMBOL>/<field>.day.bin
    bytes 0-3     : start_index (float32) — position in day.txt where
                    this symbol's data begins
    bytes 4-end   : float32 values, one per trading session

For the official reference, see
https://github.com/microsoft/qlib/blob/main/scripts/dump_bin.py

Our version stays tiny — we write from pandas DataFrames produced by the
NSE ingestion pipeline (see ``scripts/data/ingest_nse_to_qlib.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIELDS = ("open", "high", "low", "close", "volume", "factor", "vwap")


def write_instrument_bin(
    *,
    provider_uri: Path,
    symbol: str,
    ohlcv: pd.DataFrame,
    calendar: List[str],
) -> int:
    """Write one symbol's seven Qlib ``.day.bin`` files.

    Returns the number of trading-day bars written (= len(ohlcv) clipped
    to the calendar). Silently skips bars outside the calendar range.
    """
    features_dir = provider_uri / "features" / symbol.lower()
    features_dir.mkdir(parents=True, exist_ok=True)

    cal_map = {d: i for i, d in enumerate(calendar)}

    df = ohlcv.copy()
    df.index = pd.to_datetime(df.index).normalize()
    df["date_str"] = df.index.strftime("%Y-%m-%d")
    df = df[df["date_str"].isin(cal_map)]
    if df.empty:
        logger.warning("No calendar-aligned rows for %s", symbol)
        return 0

    df = df.sort_values("date_str")
    positions = df["date_str"].map(cal_map).values
    start_index = int(positions[0])
    # Qlib expects contiguous — validate.
    expected = np.arange(start_index, start_index + len(df))
    if not np.array_equal(positions, expected):
        logger.warning(
            "Non-contiguous calendar positions for %s — will gap-fill via ffill",
            symbol,
        )
        df = _gap_fill(df, calendar, start_index)
        if df.empty:
            return 0
        start_index = int(df["date_str"].map(cal_map).iloc[0])

    # VWAP proxy when raw vwap column missing.
    if "vwap" not in df.columns:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    # Factor column — multiplicative adjustment factor. If absent, 1.0.
    if "factor" not in df.columns:
        df["factor"] = 1.0

    for field in FIELDS:
        if field not in df.columns:
            continue
        values = df[field].astype("float32").values
        header = np.array([start_index], dtype="float32")
        out = np.concatenate([header, values]).astype("<f4")
        out.tofile(features_dir / f"{field}.day.bin")

    return int(len(df))


def _gap_fill(df: pd.DataFrame, calendar: List[str], start_index: int) -> pd.DataFrame:
    """Forward-fill missing trading days between first and last observed
    bar so Qlib's contiguous-position invariant holds."""
    last_date = df["date_str"].iloc[-1]
    try:
        end_index = calendar.index(last_date)
    except ValueError:
        return pd.DataFrame()
    full_dates = calendar[start_index: end_index + 1]
    full = pd.DataFrame({"date_str": full_dates})
    merged = full.merge(df, on="date_str", how="left")
    for col in ("open", "high", "low", "close", "volume", "factor", "vwap"):
        if col in merged.columns:
            merged[col] = merged[col].ffill()
    return merged.dropna(subset=["close"])


def write_instrument_file(
    provider_uri: Path,
    name: str,
    members: Dict[str, tuple],
) -> None:
    """Write ``<provider_uri>/instruments/<name>.txt``.

    ``members`` is ``{symbol: (start_date_str, end_date_str)}``. Qlib
    reads this tab-separated.
    """
    inst_dir = provider_uri / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{sym.lower()}\t{start}\t{end}"
        for sym, (start, end) in sorted(members.items())
    ]
    (inst_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")


def write_calendar(provider_uri: Path, calendar: List[str]) -> None:
    """Write ``<provider_uri>/calendars/day.txt``."""
    cal_dir = provider_uri / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "day.txt").write_text("\n".join(calendar) + "\n")
