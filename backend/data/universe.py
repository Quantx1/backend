"""
================================================================================
SWING AI UNIVERSE SCREENER
================================================================================
Dynamically fetches ALL NSE-listed stocks, applies fast price/volume/momentum filters,
and returns the top 50-70 swing-tradeable candidates.

Pipeline:
  1. Get all NSE symbols (cached JSON → NSE CSV → expanded local fallback)
  2. Batch-download 3-month OHLCV via Kite Connect
  3. Filter: price range, avg volume, momentum, volatility cap
  4. Rank by composite score (liquidity + momentum + trend)
  5. Return top N candidates

Designed to complete in < 5 minutes for 2000+ symbols.
================================================================================
"""

import io
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# Repo root for data/ directory access
ROOT_DIR = Path(__file__).resolve().parents[2]

# Lazy-load settings to avoid circular imports
_settings = None


def _get_settings():
    global _settings
    if _settings is None:
        try:
            from ..core.config import settings
            _settings = settings
        except Exception:
            _settings = None
    return _settings


class UniverseScreener:
    """
    Lightweight dynamic stock screener for NSE.
    Dynamic stock universe for signal generation.
    Returns top 50-70 swing-tradeable candidates.
    """

    def __init__(self):
        self.data_dir = ROOT_DIR / "data"
        self.symbol_cache_file = self.data_dir / "nse_all_symbols.json"

        s = _get_settings()
        self.max_candidates = getattr(s, "SCREENER_MAX_CANDIDATES", 70) if s else 70
        self.batch_size = getattr(s, "SCREENER_BATCH_SIZE", 200) if s else 200
        self.data_period = getattr(s, "SCREENER_DATA_PERIOD", "3mo") if s else "3mo"
        self.min_trading_days = getattr(s, "SCREENER_MIN_TRADING_DAYS", 20) if s else 20
        self.max_volatility = getattr(s, "SCREENER_MAX_VOLATILITY", 0.08) if s else 0.08
        self.cache_max_age_days = getattr(s, "SCREENER_SYMBOL_CACHE_DAYS", 7) if s else 7

        self.min_price = getattr(s, "EOD_SCAN_MIN_PRICE", 50.0) if s else 50.0
        self.max_price = getattr(s, "EOD_SCAN_MAX_PRICE", 10000.0) if s else 10000.0
        self.min_volume = getattr(s, "EOD_SCAN_MIN_VOLUME", 200000) if s else 200000

    # ─── Main entry point ──────────────────────────────────────────────

    def screen_sync(self) -> List[str]:
        """
        Synchronous screening pipeline.
        Returns list of 50-70 filtered NSE symbols.
        """
        try:
            logger.info("UniverseScreener: starting lightweight scan...")

            # Step 1: Get all NSE symbols
            symbols = self._get_all_nse_symbols()
            if not symbols:
                logger.warning("UniverseScreener: no symbols found")
                return []
            logger.info(f"UniverseScreener: loaded {len(symbols)} NSE symbols")

            # Step 2: Batch download 3-month OHLCV
            data = self._batch_download_data(symbols)
            if not data:
                logger.warning("UniverseScreener: no data downloaded")
                return []
            logger.info(f"UniverseScreener: downloaded data for {len(data)} stocks")

            # Step 3: Apply filters
            filtered = self._apply_filters(data)
            if filtered.empty:
                logger.warning("UniverseScreener: no stocks passed filters")
                return []
            logger.info(f"UniverseScreener: {len(filtered)} stocks passed filters")

            # Step 4: Rank and select top N
            candidates = self._rank_and_select(filtered)
            logger.info(
                f"UniverseScreener: returning {len(candidates)} candidates"
            )
            return candidates

        except Exception as e:
            logger.error(f"UniverseScreener failed: {e}")
            return []

    # ─── Symbol fetching (3-tier) ──────────────────────────────────────

    def _get_all_nse_symbols(self) -> List[str]:
        """
        Get all NSE equity symbols using 3-tier fallback:
          Tier A: Cached JSON file (< 7 days old)
          Tier B: Fetch from NSE archives CSV
          Tier C: Expanded local fallback (alpha_universe + cache dir)
        """
        # Tier A: cached file
        cached = self._load_symbol_cache()
        if cached:
            logger.info(f"UniverseScreener: using cached symbol list ({len(cached)} symbols)")
            return cached

        # Tier B: fetch from NSE
        fetched = self._fetch_nse_master_list()
        if fetched and len(fetched) > 500:
            self._save_symbol_cache(fetched)
            logger.info(f"UniverseScreener: fetched {len(fetched)} symbols from NSE")
            return fetched

        # Tier C: expanded local fallback
        fallback = self._get_expanded_fallback_symbols()
        logger.info(f"UniverseScreener: using local fallback ({len(fallback)} symbols)")
        return fallback

    def _load_symbol_cache(self) -> Optional[List[str]]:
        """Load cached NSE symbol list if fresh (< cache_max_age_days)."""
        try:
            if not self.symbol_cache_file.exists():
                return None
            with open(self.symbol_cache_file) as f:
                data = json.load(f)
            updated = date.fromisoformat(data["updated_at"])
            if (date.today() - updated).days > self.cache_max_age_days:
                return None
            symbols = data.get("symbols", [])
            return symbols if len(symbols) > 100 else None
        except Exception:
            return None

    def _save_symbol_cache(self, symbols: List[str]) -> None:
        """Save NSE symbol list to JSON cache."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            cache_data = {
                "updated_at": date.today().isoformat(),
                "count": len(symbols),
                "symbols": symbols,
            }
            with open(self.symbol_cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save symbol cache: {e}")

    def _fetch_nse_master_list(self) -> List[str]:
        """
        Fetch complete NSE equity list from NSE archives CSV.
        Returns all equity (EQ series) symbols.
        Uses requests (with urllib fallback).
        """
        urls = [
            "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
            "https://www1.nseindia.com/content/equities/EQUITY_L.csv",
        ]
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/csv,text/plain,*/*",
        }

        for url in urls:
            try:
                text = None
                # Try requests first, then urllib
                try:
                    import requests as req
                    resp = req.get(url, headers=headers, timeout=30)
                    if resp.status_code == 200:
                        text = resp.text
                except ImportError:
                    import urllib.request
                    rq = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(rq, timeout=30) as resp:  # nosec B310 - fixed https host (NSE index CSV)
                        text = resp.read().decode("utf-8")

                if not text:
                    continue

                df = pd.read_csv(io.StringIO(text))
                # Normalize column names
                df.columns = [c.strip().upper() for c in df.columns]

                if "SYMBOL" not in df.columns:
                    continue

                # Filter to EQ series only (equities, not bonds/rights)
                if " SERIES" in df.columns or "SERIES" in df.columns:
                    series_col = "SERIES" if "SERIES" in df.columns else " SERIES"
                    df = df[df[series_col].str.strip() == "EQ"]

                symbols = df["SYMBOL"].str.strip().tolist()
                # Allow NSE canonical tradingsymbols (alphanumerics plus
                # `&` and `-` — e.g. M&M, BAJAJ-AUTO, J&KBANK). The old
                # `isalnum()` filter dropped these, which then broke the
                # chart/price lookups for those tickers.
                _ALLOWED = re.compile(r"^[A-Z0-9&\-]+$")
                symbols = [
                    s for s in symbols
                    if s and isinstance(s, str) and len(s) >= 2 and _ALLOWED.match(s)
                ]
                if symbols:
                    return sorted(set(symbols))
            except Exception as e:
                logger.debug(f"NSE fetch from {url} failed: {e}")
                continue

        return []

    def _get_expanded_fallback_symbols(self) -> List[str]:
        """
        Build expanded symbol list from all available local sources:
          1. alpha_universe.txt (~200 stocks)
          2. curated_universe.txt (~65 stocks)
          3. data/cache/ CSV filenames (~300+ stocks)
        """
        symbols = set()

        # 1. alpha_universe.txt
        for filename in ["alpha_universe.txt", "alpha_universe_fixed.txt"]:
            path = self.data_dir / filename
            if path.exists():
                try:
                    for line in path.read_text().splitlines():
                        s = line.strip().split("#")[0].strip().upper()
                        s = s.replace(".NS", "").replace(".BO", "")
                        if s and len(s) >= 2 and s[0].isalpha():
                            symbols.add(s)
                except Exception:
                    pass

        # 2. curated_universe.txt
        curated_path = self.data_dir / "curated_universe.txt"
        if curated_path.exists():
            try:
                for line in curated_path.read_text().splitlines():
                    s = line.strip().split("#")[0].strip().upper()
                    s = s.replace(".NS", "").replace(".BO", "")
                    if s and len(s) >= 2 and s[0].isalpha():
                        symbols.add(s)
            except Exception:
                pass

        # 3. Cached CSV files in data/cache/
        cache_dir = self.data_dir / "cache"
        if cache_dir.exists():
            try:
                for f in cache_dir.glob("*.csv"):
                    name = f.stem
                    # Extract symbol from patterns like "RELIANCE_NS_daily"
                    parts = name.split("_")
                    if parts:
                        sym = parts[0].upper()
                        if sym and len(sym) >= 2 and sym[0].isalpha():
                            symbols.add(sym)
            except Exception:
                pass

        return sorted(symbols)

    # ─── Data download ─────────────────────────────────────────────────

    def _batch_download_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Download 3-month daily OHLCV for all symbols via Kite Connect.
        Uses KiteDataProvider.fetch_historical_batch() with rate limiting.
        """
        try:
            from .market import get_market_data_provider
            provider = get_market_data_provider()._get_kite_provider()
        except Exception as e:
            logger.error(f"Failed to get data provider: {e}")
            return {}

        logger.info(f"UniverseScreener: fetching {len(symbols)} symbols via Kite Connect...")
        batch_data = provider.fetch_historical_batch(symbols, period=self.data_period)

        result = {}
        for symbol, df in batch_data.items():
            try:
                if df is None or df.empty:
                    continue
                df.columns = [c.lower() for c in df.columns]
                df = df.dropna(subset=["close"])
                if len(df) >= self.min_trading_days:
                    result[symbol] = df
            except Exception:
                continue

        return result

    # ─── Filtering ─────────────────────────────────────────────────────

    def _apply_filters(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Apply lightweight screening filters:
          1. Min trading days (≥20)
          2. Price range (₹50 - ₹10,000)
          3. Avg daily volume 20-day (>200K)
          4. Momentum (1mo return > 0 OR above 50-SMA OR 3mo return > -10%)
          5. Volatility cap (daily return std < 8%)
        """
        rows = []

        for symbol, df in data.items():
            try:
                if len(df) < self.min_trading_days:
                    continue

                close = df["close"].iloc[-1]

                # Filter 1: Price range
                if close < self.min_price or close > self.max_price:
                    continue

                # Filter 2: Average daily volume (20-day)
                avg_volume_20 = df["volume"].tail(20).mean()
                if avg_volume_20 < self.min_volume:
                    continue

                # Filter 3: Liquidity score (proxy for market cap)
                liquidity_score = close * avg_volume_20

                # Filter 4: Momentum (pass if ANY one of three checks is true)
                # a) 1-month return > 0
                idx_1m = min(22, len(df) - 1)
                close_1m_ago = df["close"].iloc[-idx_1m]
                return_1m = (close - close_1m_ago) / close_1m_ago if close_1m_ago > 0 else 0

                # b) Price above 50-day SMA
                sma_50 = df["close"].tail(min(50, len(df))).mean()
                above_sma50 = close > sma_50

                # c) 3-month return > -10%
                close_3m_ago = df["close"].iloc[0]
                return_3m = (close - close_3m_ago) / close_3m_ago if close_3m_ago > 0 else 0

                momentum_pass = (return_1m > 0) or above_sma50 or (return_3m > -0.10)
                if not momentum_pass:
                    continue

                # Filter 5: Volatility cap
                daily_returns = df["close"].pct_change().dropna()
                if len(daily_returns) < 10:
                    continue
                volatility = daily_returns.std()
                if volatility > self.max_volatility:
                    continue

                rows.append({
                    "symbol": symbol,
                    "close": close,
                    "avg_volume_20": avg_volume_20,
                    "liquidity_score": liquidity_score,
                    "return_1m": return_1m,
                    "return_3m": return_3m,
                    "above_sma50": above_sma50,
                    "volatility": volatility,
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    # ─── Ranking ───────────────────────────────────────────────────────

    def _rank_and_select(self, df: pd.DataFrame) -> List[str]:
        """
        Rank filtered stocks by composite score and return top N.

        Scoring weights:
          50% — Liquidity (log-scaled price × volume)
          30% — Momentum (1-month return, capped ±20%)
          20% — Trend (above 50-SMA bonus)
        """
        if df.empty:
            return []

        # Normalize liquidity (log scale to avoid mega-cap domination)
        df = df.copy()
        df["liq_norm"] = np.log1p(df["liquidity_score"])
        liq_max = df["liq_norm"].max()
        if liq_max > 0:
            df["liq_norm"] = df["liq_norm"] / liq_max

        # Normalize momentum (cap at ±20%)
        df["mom_norm"] = df["return_1m"].clip(-0.2, 0.2)
        mom_min = df["mom_norm"].min()
        mom_max = df["mom_norm"].max()
        mom_range = mom_max - mom_min
        if mom_range > 0:
            df["mom_norm"] = (df["mom_norm"] - mom_min) / mom_range
        else:
            df["mom_norm"] = 0.5

        # SMA50 bonus
        df["sma_bonus"] = df["above_sma50"].astype(float)

        # Composite score
        df["composite_score"] = (
            0.50 * df["liq_norm"]
            + 0.30 * df["mom_norm"]
            + 0.20 * df["sma_bonus"]
        )

        # Sort and select top N
        df = df.sort_values("composite_score", ascending=False)
        candidates = df["symbol"].head(self.max_candidates).tolist()
        return candidates


# ─── Standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    screener = UniverseScreener()
    candidates = screener.screen_sync()
    print(f"\n{'=' * 60}")
    print(f"UniverseScreener Results: {len(candidates)} candidates")
    print(f"{'=' * 60}")
    for i, sym in enumerate(candidates, 1):
        print(f"  {i:3d}. {sym}")
