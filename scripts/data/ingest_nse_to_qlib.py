#!/usr/bin/env python
"""
Convert NSE daily OHLCV into Qlib's binary provider directory layout.

Produces::

    <provider_uri>/
        calendars/day.txt                 — NSE trading calendar
        instruments/nifty50.txt
        instruments/nifty100.txt
        instruments/nifty250.txt
        instruments/nifty500.txt
        instruments/nse_all.txt
        features/<symbol>/open.day.bin
        features/<symbol>/high.day.bin
        features/<symbol>/low.day.bin
        features/<symbol>/close.day.bin
        features/<symbol>/volume.day.bin
        features/<symbol>/factor.day.bin  — dividend/split adjustment
        features/<symbol>/vwap.day.bin    — (h+l+c)/3 proxy

Usage::

    # Full NSE All + 5 years
    python scripts/data/ingest_nse_to_qlib.py \\
        --tier nse_all \\
        --lookback-years 5 \\
        --provider-uri ~/.qlib/qlib_data/nse_data

    # Smaller (Nifty 500) for dev
    python scripts/data/ingest_nse_to_qlib.py --tier nifty500 --lookback-years 3

After ingestion, ``scripts/train/train_qlib_alpha158.py`` reads the resulting
directory via ``qlib.init(provider_uri=...)``.

Indian-market adaptations built in:
  * NSE session calendar (pandas_market_calendars['NSE'])
  * Circuit-breaker bar cleanup (|daily_return| > 19.5% → dropped)
  * Adjustment factor from yfinance adjclose vs raw close
  * Multiple tier instrument files (market-cap stratification)
  * Contiguous Qlib calendar positions (gap-fill on market halts)

yfinance ``.NS`` suffix appended automatically — seed files list base
symbols without suffix.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
logger = logging.getLogger("ingest_nse_to_qlib")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.ai.qlib.calendar import build_qlib_calendar  # noqa: E402
from backend.ai.qlib.data_handler import (  # noqa: E402
    NSE_TIER_FILES,
    load_history_many,
    load_universe,
)
from backend.ai.qlib.dump_bin import (  # noqa: E402
    write_calendar,
    write_instrument_bin,
    write_instrument_file,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="nse_all",
                        choices=sorted(NSE_TIER_FILES.keys()))
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=10,
        help=(
            "Years of history to ingest. Default bumped to 10y on "
            "2026-05-19 (pre-training audit) for richer regime coverage. "
            "Set to 5 for a faster smoke ingest."
        ),
    )
    parser.add_argument("--min-bars", type=int, default=200)
    parser.add_argument("--provider-uri",
                        default="~/.qlib/qlib_data/nse_data")
    parser.add_argument("--calendar-start", default="2015-01-01")
    args = parser.parse_args()

    provider_uri = Path(args.provider_uri).expanduser().resolve()
    provider_uri.mkdir(parents=True, exist_ok=True)

    # ── 1. Calendar ────────────────────────────────────────────────
    logger.info("Building NSE calendar from %s", args.calendar_start)
    calendar = build_qlib_calendar(start=args.calendar_start)
    write_calendar(provider_uri, calendar)
    logger.info("Wrote %d trading days to calendars/day.txt", len(calendar))

    # ── 2. Symbols to fetch (superset of requested tier) ───────────
    primary = load_universe(args.tier)
    logger.info("Primary tier %s: %d symbols", args.tier, len(primary))

    # ── 3. Historical OHLCV ────────────────────────────────────────
    lookback_days = args.lookback_years * 252
    logger.info("Fetching ~%d days / symbol", lookback_days)
    history = load_history_many(
        primary, lookback_days=lookback_days, min_rows=args.min_bars,
    )
    if not history:
        raise RuntimeError("No history loaded — check provider config.")

    # ── 4. Write binary features ───────────────────────────────────
    members: Dict[str, tuple] = {}
    bars_written = 0
    for symbol, df in history.items():
        try:
            n = write_instrument_bin(
                provider_uri=provider_uri,
                symbol=symbol,
                ohlcv=df,
                calendar=calendar,
            )
            if n > 0:
                start_date = df.index[0].strftime("%Y-%m-%d")
                end_date = df.index[-1].strftime("%Y-%m-%d")
                members[symbol] = (start_date, end_date)
                bars_written += n
        except Exception as e:
            logger.warning("Failed to write %s: %s", symbol, e)

    logger.info(
        "Wrote features for %d symbols / %d total bars",
        len(members), bars_written,
    )

    # ── 5. Instrument files (every tier, not just primary) ─────────
    for tier in NSE_TIER_FILES:
        tier_symbols = {s.upper() for s in load_universe(tier)}
        tier_members = {
            s: v for s, v in members.items()
            if s.upper() in tier_symbols
        }
        if tier_members:
            write_instrument_file(provider_uri, tier, tier_members)
            logger.info("Wrote instruments/%s.txt (%d symbols)", tier, len(tier_members))

    logger.info("Done. provider_uri = %s", provider_uri)
    logger.info(
        "Next: python scripts/train/train_qlib_alpha158.py "
        "--provider-uri %s --instruments %s",
        provider_uri, args.tier,
    )


if __name__ == "__main__":
    main()
