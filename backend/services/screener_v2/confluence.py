"""Confluence scoring — surface stocks matched by N scanners at once.

The legacy scanner returns "stocks that match ONE filter" — that's noisy.
Real traders care about *confluence*: when the same stock fires on 3+
independent setups (e.g. breakout + volume surge + RSI cross), the
probability of a real move is materially higher.

This module:
  1. Runs N scanner filters against the same cached summary_df
  2. Joins matches by symbol
  3. Scores each symbol by (a) how many scanners fired, (b) the category
     diversity of those scanners (breakout + momentum + volume = stronger
     than 3 redundant momentum scanners), (c) a per-scanner quality
     weight
  4. Returns ranked ConfluenceMatch records the UI can render as cards

The scoring formula favours diverse multi-category confluence over
high-count single-category matches.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# ── Scanner → category map ──────────────────────────────────────────
# Each scanner_id is tagged with its primary category. Confluence score
# weights category diversity, so a (breakout + volume + momentum) hit
# beats three momentum hits.

SCANNER_CATEGORIES: Dict[int, str] = {
    # Trend / breakout
    1: "breakout", 5: "breakout", 6: "breakout", 16: "breakout",
    33: "breakout", 58: "breakout", 66: "breakout",
    # Momentum
    2: "momentum", 17: "momentum", 26: "momentum", 30: "momentum",
    31: "momentum", 32: "momentum",
    52: "momentum", 56: "momentum", 70: "momentum", 71: "momentum",
    # Reversal / mean reversion
    3: "reversal", 9: "reversal", 19: "reversal",
    27: "reversal", 57: "reversal", 65: "reversal",
    # Volume / flow
    4: "volume", 8: "volume", 34: "volume", 35: "volume",
    # Pattern / volatility
    14: "pattern", 28: "pattern", 48: "pattern",
    # Candle pattern
    12: "candle", 13: "candle",
    # Moving averages / trend continuation
    11: "ma", 15: "ma",
    54: "ma", 59: "ma", 63: "ma", 67: "ma",
    # Volatility
    18: "volatility", 21: "volatility", 22: "volatility", 29: "volatility",
    53: "volatility", 55: "volatility", 60: "volatility", 68: "volatility",
    # Smart money + F&O
    36: "smart_money", 37: "smart_money", 38: "smart_money",
    39: "fo", 40: "fo", 41: "fo", 42: "fo", 87: "fo", 88: "fo",
    # Weakness (mirror of strength)
    7: "weakness", 10: "weakness", 64: "weakness",
    # Relative strength
    61: "rs_leader", 69: "rs_laggard",
    # PR-S18 — institutional setups
    72: "pullback", 73: "reversal", 74: "breakout", 75: "momentum",
    76: "volatility", 77: "momentum", 78: "momentum", 79: "reversal",
    80: "reversal", 81: "breakout",
    82: "breakout", 83: "breakout", 84: "breakout", 85: "event",
    86: "reversal",
    # PR-S20 — F&O stock scanners
    87: "fo", 88: "fo",
    # 0 = full-screen (skip in confluence)
}


# Per-scanner quality weight — bumps high-conviction scanners (VCP,
# Power Setup, FII buying) above noisy ones (top gainers, 52w highs).
SCANNER_WEIGHTS: Dict[int, float] = {
    # High-conviction setups
    14: 1.5,   # VCP
    52: 1.5,   # Power Setup (bullish 4-of-4)
    62: 1.5,   # Power Setup Short (bearish 4-of-4)
    # Smart money (when data is real)
    36: 1.6,   # FII Net Buyers
    37: 1.4,   # DII Net Buyers
    38: 1.5,   # FII+DII Positive
    # F&O confirmation
    40: 1.3,   # Long Buildup
    42: 1.2,   # Short Covering
    # Volume confirmation
    8: 1.2,   # Volume Surge
    4: 1.1,   # Volume Breakout
    # Momentum
    1: 1.2,   # Breakout (Consolidation)
    17: 1.1,   # Bull Momentum
    30: 1.1,   # Momentum Burst
    70: 1.1,   # Bear Momentum
    71: 1.1,   # Momentum Crash
    # Trend confirmation (MA stacks)
    54: 1.2,
    63: 1.2,
    # Fresh trend starts
    56: 1.3,
    64: 1.3,
    # Decisive breakouts/breakdowns with volume
    58: 1.3,
    66: 1.3,
    # MACD
    26: 1.1,
    # PR-S18 institutional setups (weights reflect literature confidence)
    72: 1.4,   # Pocket Pivot — published Kacher/Morales backtest data
    74: 1.5,   # Episodic Pivot — Qullamaggie 30%×10R skewed payoff
    75: 1.3,   # Holy Grail — Raschke published 70%+ in-sample
    76: 1.3,   # Coiled Spring — Crabel-era volatility expansion edge
    78: 1.2,   # Three White Soldiers — Bulkowski 82% reversal
    81: 1.2,   # Weekly Pivot Reclaim — Varsity narrow-CPR signal
    82: 1.5,   # Stage 2 Acceleration — Weinstein top-tier setup
    83: 1.4,   # CAN SLIM Base Breakout — O'Neil canonical
    84: 1.4,   # Cup-Handle Volume — O'Neil
    85: 1.3,   # PEAD — academic effect
    # Default 1.0 for everything not listed
}


# Bearish scanners — flag them so the UI can show
# "matched 3 bullish, 1 bearish — mixed signal".
BEARISH_SCANNERS = {
    3, 7, 10, 13, 27, 41,                  # legacy bearish-only
    62, 63, 64, 65, 66, 67, 68, 69, 70, 71,  # PR-S17 bearish parity
    80,                                     # PR-S18 Gap Fill Reversal (short bias)
}


# PR-S18 — horizon tag per scanner.
# Lets the UI filter "show me only swing setups", or AutoPilot route
# only positional signals (longer hold = lower trading frequency).
SCANNER_HORIZON: Dict[int, str] = {
    # Intraday-leaning (work on EOD bars but signal is day-tactical)
    2: "intraday", 3: "intraday", 20: "intraday",
    # Swing (3-15 day hold) — the bulk of the pool
    1: "swing", 4: "swing", 5: "swing", 6: "swing", 7: "swing", 8: "swing", 9: "swing",
    10: "swing", 11: "swing", 12: "swing", 13: "swing", 15: "swing",
    # PR-S20 — F&O stock scanners (swing-horizon by default)
    87: "swing", 88: "swing",
    17: "swing", 19: "swing", 21: "swing", 22: "swing", 26: "swing",
    27: "swing", 28: "swing", 29: "swing", 30: "swing", 32: "swing",
    33: "swing", 36: "swing", 37: "swing", 38: "swing", 39: "swing",
    40: "swing", 41: "swing", 42: "swing", 48: "swing",
    52: "swing", 53: "swing", 55: "swing", 56: "swing", 57: "swing",
    58: "swing", 59: "swing", 60: "swing", 61: "swing",
    62: "swing", 64: "swing", 65: "swing", 66: "swing", 67: "swing",
    68: "swing", 69: "swing", 70: "swing", 71: "swing",
    72: "swing", 73: "swing", 74: "swing", 75: "swing", 76: "swing",
    77: "swing", 78: "swing", 79: "swing", 80: "swing", 81: "swing",
    # Positional (3+ week hold, weekly/monthly thesis)
    14: "positional", 16: "positional", 18: "positional", 31: "positional",
    34: "positional", 35: "positional", 54: "positional", 63: "positional",
    82: "positional", 83: "positional", 84: "positional",
    85: "positional", 86: "positional",
    # Scanner 0 = full screening (skip; no horizon)
}


# PR-S18 — setup-type tag per scanner.
# Useful when the user wants "all breakout setups" or "all reversal setups"
# regardless of timeframe.
SCANNER_SETUP_TYPE: Dict[int, str] = {
    # Breakout / range expansion
    1: "breakout", 5: "breakout", 6: "breakout", 16: "breakout",
    33: "breakout", 58: "breakout", 66: "breakout",
    74: "breakout", 81: "breakout", 82: "breakout", 83: "breakout", 84: "breakout",
    # Momentum / trend continuation
    2: "momentum", 17: "momentum", 26: "momentum", 30: "momentum",
    31: "momentum", 32: "momentum", 52: "momentum", 56: "momentum",
    62: "momentum",
    70: "momentum", 71: "momentum", 75: "momentum", 77: "momentum", 78: "momentum",
    # Mean reversion / reversal
    3: "reversal", 9: "reversal", 10: "reversal", 19: "reversal",
    27: "reversal", 57: "reversal", 65: "reversal",
    73: "reversal", 79: "reversal", 80: "reversal", 86: "reversal",
    # Pullback (in-trend dip)
    59: "pullback", 67: "pullback", 72: "pullback",
    # Pattern (chart shape)
    14: "pattern", 28: "pattern", 48: "pattern",
    # Volatility (compression / squeeze)
    18: "volatility", 21: "volatility", 22: "volatility", 29: "volatility",
    53: "volatility", 55: "volatility", 60: "volatility", 68: "volatility",
    76: "volatility",
    # Candle / candlestick patterns
    12: "candle", 13: "candle",
    # Trend (stacked MA / structural)
    11: "trend", 15: "trend", 54: "trend", 63: "trend",
    # Volume / order flow
    4: "volume", 8: "volume", 34: "volume", 35: "volume",
    # Smart money / F&O positioning
    36: "smart_money", 37: "smart_money", 38: "smart_money",
    39: "fo", 40: "fo", 41: "fo", 42: "fo", 87: "fo", 88: "fo",
    # Weakness / failed-strength tag
    7: "weakness", 64: "weakness",
    # Event-driven
    85: "event",
    # Relative strength (cross-sectional)
    61: "relative_strength", 69: "relative_strength",
}


@dataclass
class ScannerHit:
    """One scanner-on-this-symbol match."""
    scanner_id: int
    scanner_name: str
    category: str
    weight: float
    bullish: bool


@dataclass
class ConfluenceMatch:
    """One stock with all the scanners it matched."""
    symbol: str
    name: str
    sector: Optional[str]
    last_price: float
    change_pct: float
    volume: int
    rsi: float
    # Confluence stats
    hit_count: int                          # how many scanners matched
    bull_count: int
    bear_count: int
    category_diversity: int                 # distinct bullish categories
    composite_score: float                  # ranking number
    hits: List[ScannerHit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hits"] = [asdict(h) for h in self.hits]
        return d


def _composite_score(
    hits: List[ScannerHit], change_pct: float, volume_ratio: float,
) -> float:
    """Blend hit count + category diversity + weight + price/vol confirm.

    Formula:
        weighted_hits = sum(weight * (1 if bullish else -0.6) for h in hits)
        diversity_bonus = 0.2 * distinct_bullish_categories
        confirm_bonus = clip(change_pct/10, -0.3, 0.3) + clip((vol_ratio-1)/4, 0, 0.3)
        score = weighted_hits + diversity_bonus + confirm_bonus

    Bearish scanners pull score down rather than just being ignored —
    a confluence of bullish + bearish hits is a mixed signal, not a buy.
    """
    weighted = sum(h.weight * (1.0 if h.bullish else -0.6) for h in hits)
    bull_cats = {h.category for h in hits if h.bullish}
    diversity_bonus = 0.2 * len(bull_cats)
    chg = max(-0.3, min(0.3, change_pct / 10.0))
    vol = max(0.0, min(0.3, (volume_ratio - 1.0) / 4.0))
    return weighted + diversity_bonus + chg + vol


def confluence_scan(
    summary_df: pd.DataFrame,
    *,
    scanner_ids: Sequence[int],
    stock_info: Dict[str, Dict[str, str]],
    min_hits: int = 2,
    limit: int = 50,
) -> List[ConfluenceMatch]:
    """Run N scanner filters in parallel against summary_df, aggregate
    hits per symbol, rank by composite score.

    `summary_df` — pre-computed indicator dataframe from LiveScreenerEngine
                   (one row per symbol with all indicators)
    `scanner_ids` — which scanners to include in the confluence
    `stock_info`  — symbol → {name, sector} lookup (e.g. NSE_STOCK_INFO)
    `min_hits`    — drop symbols matched by fewer than this many scanners
    """
    from backend.data.screener.filters import SCANNER_FILTERS
    from backend.data.screener.engine import SCANNER_MENU

    if summary_df is None or summary_df.empty:
        return []

    submenu = SCANNER_MENU["scan_types"]["X"]["submenu"]

    # Map: symbol → list of (scanner_id, ScannerHit)
    by_symbol: Dict[str, List[ScannerHit]] = defaultdict(list)

    for sid in scanner_ids:
        # Legacy chart-pattern scanner IDs (the old PATTERN_SCANNERS set) were
        # removed 2026-05-31; any scanner without a summary-df filter is skipped
        # by the filter_fn guard below (pattern v2 lives in services.chart_patterns).
        filter_fn = SCANNER_FILTERS.get(sid)
        if filter_fn is None or sid == 0:    # 0 = full-screen, skip
            continue
        try:
            matched = filter_fn(summary_df.copy())
        except Exception as e:
            logger.debug("confluence scanner %d filter failed: %s", sid, e)
            continue
        if matched is None or matched.empty:
            continue
        scanner_name = submenu.get(sid, {}).get("name", f"Scanner {sid}")
        category = SCANNER_CATEGORIES.get(sid, "other")
        weight = SCANNER_WEIGHTS.get(sid, 1.0)
        bullish = sid not in BEARISH_SCANNERS

        for sym in matched["symbol"].tolist():
            by_symbol[sym].append(ScannerHit(
                scanner_id=sid,
                scanner_name=scanner_name,
                category=category,
                weight=weight,
                bullish=bullish,
            ))

    if not by_symbol:
        return []

    # Build ConfluenceMatch per symbol
    matches: List[ConfluenceMatch] = []
    df_indexed = summary_df.set_index("symbol") if "symbol" in summary_df.columns else summary_df

    for sym, hits in by_symbol.items():
        if len(hits) < min_hits:
            continue
        try:
            row = df_indexed.loc[sym]
        except KeyError:
            continue

        info = stock_info.get(sym, {})
        bull_count = sum(1 for h in hits if h.bullish)
        bear_count = sum(1 for h in hits if not h.bullish)
        bull_cats = {h.category for h in hits if h.bullish}

        change_pct = float(row.get("change_pct", 0) or 0)
        volume = int(row.get("volume", 0) or 0)
        volume_sma = float(row.get("volume_sma20", 1) or 1)
        vol_ratio = volume / volume_sma if volume_sma > 0 else 1.0

        matches.append(ConfluenceMatch(
            symbol=sym,
            name=info.get("name", sym),
            sector=info.get("sector"),
            last_price=float(row.get("close", row.get("ltp", 0)) or 0),
            change_pct=round(change_pct, 2),
            volume=volume,
            rsi=round(float(row.get("rsi", 0) or 0), 1),
            hit_count=len(hits),
            bull_count=bull_count,
            bear_count=bear_count,
            category_diversity=len(bull_cats),
            composite_score=round(_composite_score(hits, change_pct, vol_ratio), 4),
            hits=sorted(hits, key=lambda h: -h.weight),
        ))

    matches.sort(key=lambda m: m.composite_score, reverse=True)
    return matches[:limit]
