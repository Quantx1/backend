"""
PR 194 — sentiment-cache backfill driver.

Walks the liquid universe via Google News RSS + FinBERT-India and
populates ml/data/cache/sentiment_history.parquet. Designed to run
nightly in production (cron) so the cache accumulates depth before
each retrain.

Usage:
    python scripts/data/backfill_sentiment.py
    python scripts/data/backfill_sentiment.py --top-n 50      # quick test
    python scripts/data/backfill_sentiment.py --symbols RELIANCE,TCS

Time budget: ~2s per symbol (1s news fetch + 30ms FinBERT × 100 headlines
+ 1s rate limit), so 200 symbols → ~7 minutes per run. Idempotent — re-runs
upsert into the cache.
"""

from __future__ import annotations

import argparse
import logging
import sys
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
logger = logging.getLogger("backfill_sentiment")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backfill sentiment parquet cache")
    p.add_argument("--top-n", type=int, default=500,
                   help="Top-N liquid NSE symbols to ingest (default 500)")
    p.add_argument("--symbols", type=str, default=None,
                   help="Comma-separated symbol list, overrides --top-n")
    p.add_argument("--rate-limit", type=float, default=1.5,
                   help="Sleep seconds between Google News calls (default 1.5)")
    args = p.parse_args(argv)

    from ml.data import LiquidUniverseConfig, liquid_universe
    from ml.data.news_ingester import backfill_sentiment_cache

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = liquid_universe(LiquidUniverseConfig(top_n=args.top_n))
    logger.info("ingesting sentiment for %d symbols", len(symbols))

    result = backfill_sentiment_cache(
        symbols=symbols, rate_limit_seconds=args.rate_limit,
    )
    logger.info("done: %s", result)
    if result["n_scored"] == 0:
        logger.error("no symbols produced scored sentiment — check FinBERT load + Google News access")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
