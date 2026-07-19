"""Run one saved scan and return the diff vs last run."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Outcome of executing one saved scan."""
    scan_id: str
    matched_symbols: List[str]      # all current matches
    new_symbols: List[str]          # symbols not in previous run (the alert trigger)
    total_count: int
    error: Optional[str] = None


async def run_saved_scan(scan_row: Dict[str, Any]) -> RunResult:
    """Execute a single saved scan via the v2 confluence engine.

    Returns the symbol diff vs `scan_row['last_hit_symbols']` so the
    caller can fire an alert only when something new shows up.
    """
    import asyncio
    from backend.services.screener_v2 import confluence_scan
    from backend.data.screener.engine import NSE_STOCK_INFO, get_live_screener

    scan_id = str(scan_row["id"])
    scanner_ids: List[int] = list(scan_row["scanner_ids"] or [])
    sectors: List[str] = list(scan_row.get("sectors") or [])
    min_hits: int = int(scan_row.get("min_hits") or 1)
    last_symbols: set[str] = set(scan_row.get("last_hit_symbols") or [])

    try:
        screener = get_live_screener()
        summary_df, _ = await asyncio.to_thread(screener._get_computed_data)
        if summary_df is None or summary_df.empty:
            return RunResult(
                scan_id=scan_id, matched_symbols=[], new_symbols=[],
                total_count=0, error="screener data not ready",
            )

        # Sector pre-filter on the dataframe
        if sectors:
            tagged = {s for s, info in NSE_STOCK_INFO.items()
                      if info.get("sector") in sectors}
            if "symbol" in summary_df.columns:
                summary_df = summary_df[summary_df["symbol"].isin(tagged)]
            if summary_df.empty:
                return RunResult(
                    scan_id=scan_id, matched_symbols=[], new_symbols=[],
                    total_count=0,
                )

        matches = await asyncio.to_thread(
            confluence_scan,
            summary_df,
            scanner_ids=scanner_ids,
            stock_info=NSE_STOCK_INFO,
            min_hits=min_hits,
            limit=50,
        )

        matched_syms = [m.symbol for m in matches]
        new_syms = [s for s in matched_syms if s not in last_symbols]

        return RunResult(
            scan_id=scan_id,
            matched_symbols=matched_syms,
            new_symbols=new_syms,
            total_count=len(matched_syms),
        )

    except Exception as e:
        logger.exception("run_saved_scan %s failed: %s", scan_id, e)
        return RunResult(
            scan_id=scan_id, matched_symbols=[], new_symbols=[],
            total_count=0, error=str(e)[:300],
        )
