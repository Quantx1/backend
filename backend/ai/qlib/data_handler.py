"""
NSE universe loader + market-cap tier resolver.

The Qlib Alpha158 handler ranks stocks cross-sectionally — pooling a
₹30 small-cap with HDFCBANK biases the model, so we maintain five
tiered instrument files:

    nifty50     — Nifty 50 (~50 largecaps)
    nifty100    — Nifty 100 (Nifty 50 + Nifty Next 50)
    nifty250    — Nifty 100 + Nifty Midcap 150
    nifty500    — top 500 by full market cap
    nse_all     — every listed NSE equity with enough liquidity

Seed lists ship in ``data/nse_tiers/``. A quarterly refresh job pulls
updated constituent CSVs from NSE's published index files; v1 ships a
static snapshot dated in the filenames.

Training + inference both read these lists via ``load_universe(tier)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_TIERS_DIR = _ROOT / "data" / "nse_tiers"


NSE_TIER_FILES: Dict[str, str] = {
    "nifty50": "nifty50.txt",
    "nifty100": "nifty100.txt",
    "nifty250": "nifty250.txt",
    "nifty500": "nifty500.txt",
    "nse_all": "nse_all.txt",
}


def load_universe(tier: str = "nse_all") -> List[str]:
    """Read a tier file (one symbol per line, `#` = comment).

    Resolution:
      1. For ``nse_all``: prefer ``data/nse_all_symbols.json`` (the live
         2,136-symbol cache regenerated nightly by ``UniverseScreener``)
         over the static ``nse_tiers/nse_all.txt`` seed (~551 syms).
         PR-S2.1 (2026-05-31): unlocks the full NSE universe for every
         downstream caller — discovery, news scanner, technical screeners,
         and pattern v2.
      2. ``data/nse_tiers/<tier>.txt``
      3. ``data/alpha_universe.txt`` (legacy fallback)
      4. Hardcoded Nifty 50 minimal list
    """
    # JSON-first for nse_all — the 2,136 cache supersedes the 551 seed file
    if tier == "nse_all":
        json_syms = _load_nse_all_json()
        if json_syms and len(json_syms) > 100:
            return json_syms

    fname = NSE_TIER_FILES.get(tier)
    if fname:
        path = _TIERS_DIR / fname
        if path.exists():
            return _parse_symbol_file(path)

    legacy = _ROOT / "data" / "alpha_universe.txt"
    if legacy.exists():
        logger.warning("Tier %s not found, falling back to alpha_universe.txt", tier)
        return _parse_symbol_file(legacy)

    return [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "KOTAKBANK", "LT", "SBIN",
        "BHARTIARTL", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
        "HCLTECH", "WIPRO", "SUNPHARMA", "ULTRACEMCO", "TITAN",
    ]


def _load_nse_all_json() -> List[str]:
    """Read the live nse_all_symbols.json cache. Returns [] on any failure
    so the caller falls back to the static tier file."""
    import json
    path = _ROOT / "data" / "nse_all_symbols.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        syms = data.get("symbols") or []
        return [s.strip().upper() for s in syms if s]
    except Exception as e:
        logger.warning("Failed to read nse_all_symbols.json: %s", e)
        return []


def _parse_symbol_file(path: Path) -> List[str]:
    symbols = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip().upper()
        if line:
            symbols.append(line)
    return symbols


def tier_membership(symbol: str) -> List[str]:
    """Return every tier this symbol belongs to (smallest tier first).
    Used as a categorical feature during training."""
    symbol = symbol.upper()
    return [tier for tier, fname in NSE_TIER_FILES.items()
            if symbol in set(load_universe(tier))]


def load_history(symbol: str, *, lookback_days: int = 500) -> Optional[Tuple]:
    """Load NSE OHLCV for one symbol via the existing market-data
    provider. Returns (DataFrame, dividends_df) tuple; ``dividends_df``
    may be ``None`` if the provider doesn't surface corporate actions.

    yfinance: ``.NS`` suffix resolves to NSE. We request un-adjusted
    prices so the Qlib ``factor`` column carries dividend/split
    adjustments explicitly.
    """

    try:
        from ...data.market import get_market_data_provider

        provider = get_market_data_provider()
        period = "2y" if lookback_days <= 500 else "5y" if lookback_days <= 1300 else "10y"
        df = provider.get_historical(symbol.upper(), period=period, interval="1d")
    except Exception as e:
        logger.debug("load_history(%s) failed: %s", symbol, e)
        return None

    if df is None or len(df) == 0:
        return None

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None

    # Compute the adjustment factor when the provider surfaces an
    # ``adj close`` column; otherwise factor = 1.0 (no adjustment).
    if "adj close" in df.columns:
        df["factor"] = df["adj close"] / df["close"]
    elif "adj_close" in df.columns:
        df["factor"] = df["adj_close"] / df["close"]
    else:
        df["factor"] = 1.0

    keep = ["open", "high", "low", "close", "volume", "factor"]
    df = df[keep].tail(lookback_days).copy()

    # Drop suspect bars that look like NSE circuit-breaker days —
    # |daily return| > 19.5% = likely circuit hit, price did not
    # discover fair value. Small loss of data, big gain in signal.
    df = _clip_circuit_breaker_days(df)

    return df, None


def _clip_circuit_breaker_days(df):
    """Remove rows where absolute daily return > 19.5%. Conservative
    threshold covers all NSE circuit tiers (2/5/10/20%)."""

    ret = df["close"].pct_change()
    mask = ret.abs() > 0.195
    if mask.any():
        logger.debug("Dropping %d circuit-breaker bars", int(mask.sum()))
        df = df.loc[~mask].copy()
    return df


def load_history_many(
    symbols: List[str],
    *,
    lookback_days: int = 500,
    min_rows: int = 100,
) -> Dict:
    """Bulk loader — ``{symbol: DataFrame}``. Skips failures and symbols
    with fewer than ``min_rows`` clean bars."""
    out: Dict = {}
    for sym in symbols:
        res = load_history(sym, lookback_days=lookback_days)
        if res is None:
            continue
        df, _ = res
        if df is not None and len(df) >= min_rows:
            out[sym] = df
    logger.info(
        "load_history_many: %d/%d symbols with >=%d bars",
        len(out), len(symbols), min_rows,
    )
    return out
