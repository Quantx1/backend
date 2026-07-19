"""
PR 185 — fundamentals backfill driver.

Pre-warms the PIT fundamentals parquet cache so trainers (lgbm_signal_gate
in particular) get real fundamentals features instead of zero-fills on
the first --all run.

Usage:
    python scripts/data/backfill_fundamentals.py
    python scripts/data/backfill_fundamentals.py --top-n 50    # quick test
    python scripts/data/backfill_fundamentals.py --symbols RELIANCE,TCS,HDFCBANK

Time budget: ~10s per symbol via yfinance (4-8 quarters fetched), so
200 symbols → ~30 minutes worst-case. Each run is upsert-safe; re-runs
overwrite stale entries from yfinance with the latest restated values.

After this completes successfully, ml/data/cache/fundamentals_pit.parquet
will exist and lgbm_signal_gate's 8 fundamentals features will populate
with real data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# PR 210 — auto-add repo root to sys.path so the script works without
# the user having to remember `export PYTHONPATH=...` after shell restart.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import time
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill_fundamentals")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backfill PIT fundamentals cache")
    p.add_argument("--top-n", type=int, default=500,
                   help="Top-N liquid NSE symbols to ingest (default 500)")
    p.add_argument("--symbols", type=str, default=None,
                   help="Comma-separated symbol list, overrides --top-n")
    p.add_argument("--lag-days", type=int, default=60,
                   help="Publication lag for PIT discipline (default 60)")
    args = p.parse_args(argv)

    from ml.data import LiquidUniverseConfig, liquid_universe
    from ml.data.fundamentals_pit import (
        FUNDAMENTALS_CACHE_FILE,
        ingest_yfinance_fundamentals,
    )

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        logger.info("explicit symbols: %d", len(symbols))
    else:
        symbols = liquid_universe(LiquidUniverseConfig(top_n=args.top_n))
        logger.info("liquid universe top-%d: %d symbols", args.top_n, len(symbols))

    if not symbols:
        logger.error("no symbols to ingest")
        return 1

    t0 = time.time()
    records = ingest_yfinance_fundamentals(
        symbols=symbols,
        publication_lag_days=args.lag_days,
        persist=True,
    )
    elapsed = time.time() - t0

    logger.info(
        "wrote %d records in %.1fs → %s",
        len(records), elapsed, FUNDAMENTALS_CACHE_FILE,
    )
    if not FUNDAMENTALS_CACHE_FILE.exists():
        logger.error("cache file not created — ingest may have failed silently")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
