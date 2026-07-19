"""
PR 185 — FII/DII historical backfill driver.

Pre-warms the FII/DII parquet cache with daily institutional flow
history. Trainers (lgbm_signal_gate) use these for the 4 flow features
(fii_5d_sum, dii_5d_sum, fii_5d_z, dii_5d_z) — without backfill they
zero-fill across the entire 8-year training window.

Usage:
    python scripts/data/backfill_fii_dii.py
    python scripts/data/backfill_fii_dii.py --start 2018-01-01 --end 2025-12-31

Time budget: ~0.5s per business day via jugaad-data archive scraper, so
8 years (~2000 business days) → ~20 minutes worst-case. Idempotent — re-runs
upsert into the existing parquet.

Failure modes:
    - jugaad-data not installed → script exits with code 2.
    - NSE archive 403/rate-limit → individual days skipped, partial cache
      still useful.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# PR 210 — auto-add repo root to sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill_fii_dii")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backfill FII/DII parquet cache")
    p.add_argument("--start", type=str, default=None,
                   help="ISO start date. Default: today - 8 years")
    p.add_argument("--end", type=str, default=None,
                   help="ISO end date. Default: today")
    args = p.parse_args(argv)

    from ml.data.fii_dii_history import (
        FII_DII_CACHE_FILE,
        backfill_from_jugaad,
    )

    end_d = date.fromisoformat(args.end) if args.end else date.today()
    start_d = (
        date.fromisoformat(args.start) if args.start
        else end_d - timedelta(days=365 * 8)
    )

    logger.info("backfilling FII/DII from %s to %s", start_d, end_d)
    df = backfill_from_jugaad(start_d, end_d, persist=True)
    logger.info("retrieved %d rows → %s", len(df), FII_DII_CACHE_FILE)

    if df.empty:
        logger.error(
            "no rows ingested — jugaad-data may be missing or NSE blocked archive access",
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
