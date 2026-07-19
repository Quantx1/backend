#!/usr/bin/env python
"""Quick data-layer diagnostic.

Tests in isolation:
  1. bhavcopy single-symbol fetch (jugaad-data via production_ohlcv)
  2. yfinance single-symbol fetch
  3. yfinance batch fetch (10 symbols × 5y)
  4. production_ohlcv full path (10 symbols × 5y)

Identifies which layer is rate-limiting the lgbm_signal_gate trainer.
Run inline (not nohup) so timing is visible.

Usage: python -u scripts/data/diag_data_layer.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def t1_bhavcopy() -> None:
    print("=== test 1: bhavcopy single symbol RELIANCE 1 month ===", flush=True)
    t0 = time.time()
    try:
        from ml.data.bhavcopy_source import bhavcopy_download_with_fallback
        df, source = bhavcopy_download_with_fallback(
            ["RELIANCE"], "2024-01-01", "2024-02-01",
        )
        print(f"  OK source={source} rows={len(df)} time={time.time()-t0:.1f}s",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {type(e).__name__}: {e}  time={time.time()-t0:.1f}s",
              flush=True)


def t2_yf_single() -> None:
    print("\n=== test 2: yfinance single symbol RELIANCE 1 month ===", flush=True)
    t0 = time.time()
    try:
        import yfinance as yf
        df = yf.download("RELIANCE.NS",
                         start="2024-01-01", end="2024-02-01",
                         progress=False, timeout=30)
        print(f"  OK rows={len(df)} time={time.time()-t0:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {type(e).__name__}: {e}  time={time.time()-t0:.1f}s",
              flush=True)


def t3_yf_batch() -> None:
    print("\n=== test 3: yfinance BATCH 10 symbols 5y ===", flush=True)
    t0 = time.time()
    try:
        import yfinance as yf
        syms = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "BHARTIARTL.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "ITC.NS",
        ]
        df = yf.download(syms,
                         start="2019-01-01", end="2024-01-01",
                         progress=False, timeout=60, group_by="ticker")
        print(f"  OK shape={df.shape} time={time.time()-t0:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {type(e).__name__}: {e}  time={time.time()-t0:.1f}s",
              flush=True)


def t4_production_ohlcv() -> None:
    print("\n=== test 4: production_ohlcv 10 symbols 5y ===", flush=True)
    t0 = time.time()
    try:
        from ml.data.production_ohlcv import production_ohlcv
        syms = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "BHARTIARTL", "SBIN", "AXISBANK", "KOTAKBANK", "ITC",
        ]
        df = production_ohlcv(
            syms, start="2019-01-01", end="2024-01-01",
            include_delisted=False,
        )
        print(f"  OK shape={df.shape} time={time.time()-t0:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {type(e).__name__}: {e}  time={time.time()-t0:.1f}s",
              flush=True)


def t5_qlib_provider() -> None:
    print("\n=== test 5: Qlib provider read 10 symbols ===", flush=True)
    t0 = time.time()
    try:
        from backend.ai.qlib import get_qlib_engine, load_history_many
        engine = get_qlib_engine()
        engine.init_qlib()
        prices = load_history_many(
            ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
             "BHARTIARTL", "SBIN", "AXISBANK", "KOTAKBANK", "ITC"],
            start="2019-01-01", end="2024-01-01",
        )
        nrows = sum(len(df) for df in prices.values())
        print(f"  OK n_symbols={len(prices)} total_rows={nrows} "
              f"time={time.time()-t0:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {type(e).__name__}: {e}  time={time.time()-t0:.1f}s",
              flush=True)


if __name__ == "__main__":
    t1_bhavcopy()
    t2_yf_single()
    t3_yf_batch()
    t4_production_ohlcv()
    t5_qlib_provider()
    print("\n=== diagnostic complete ===", flush=True)
