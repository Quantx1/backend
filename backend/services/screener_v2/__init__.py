"""Technical Screener v2 — cutting-edge trader surface (PR-S7).

What's new vs the legacy 50+ scanners:
  * **Confluence scoring** — surface stocks matched by N scanners at once,
    not just one
  * **Deep-dive** per symbol — every indicator currently firing, ATR-
    derived entry/stop/target, regime + sector context, news + earnings
    nearness
  * **Streaming** — fan results out as scanners complete so the UI shows
    live progress instead of waiting for a 60s blocking call
  * **Full NSE universe via streaming** — bypasses the 500-cap of the
    legacy synchronous bulk pipeline

Module layout:
  confluence.py — multi-scanner aggregation + composite scoring
  enrich.py     — deep-dive enrichment (news, earnings, levels, AI thesis)
"""

from .confluence import (
    confluence_scan, ConfluenceMatch, SCANNER_CATEGORIES,
)
from .enrich import enrich_symbol, EnrichedMatch
from .multi_timeframe import scan_multi_timeframe, MTFMatch
from .sector_heatmap import build_sector_heatmap, SectorRow
from .comparable import comparable_setups, ComparableResult

__all__ = [
    "confluence_scan", "ConfluenceMatch", "SCANNER_CATEGORIES",
    "enrich_symbol", "EnrichedMatch",
    "scan_multi_timeframe", "MTFMatch",
    "build_sector_heatmap", "SectorRow",
    "comparable_setups", "ComparableResult",
]
