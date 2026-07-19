"""
================================================================================
QUANT X LIVE SCREENER ENGINE
================================================================================
Real Kite Connect-based screening with 50+ scanners.

Pipeline:
  1. UniverseScreener fetches NSE symbol list (cached 1h)
  2. Kite Connect historical data fetches 6-month OHLCV
  3. compute_all_indicators() from ml/features computes 40+ indicators
  4. Per-scanner filter functions applied to summary DataFrame
  5. Results formatted to match frontend API contract

All scanner results cached for 5 minutes to avoid re-downloading.
================================================================================
"""

from .filters import (
    SCANNER_FILTERS,
    _filter_full_screening,
)
from ml.features.indicators import compute_all_indicators
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Add project root to path for ml imports.
# After PR-A4: file lives at backend/data/screener/engine.py — parents[3] is repo root.
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================================
# SCANNER MENU — complete scanner definitions for Quant X Screener
# ============================================================================
SCANNER_MENU = {
    "exchanges": {
        "N": "NSE (All NSE Stocks)",
        "B": "BSE (All BSE Stocks)",
        "S": "NSE Nifty 50",
        "J": "NSE Nifty Next 50",
        "W": "NSE Nifty 100",
        "E": "NSE Nifty 200",
        "Q": "NSE Nifty 500",
        "Z": "NSE Nifty Midcap 50",
        "F": "NSE Nifty F&O Stocks",
        "G": "NSE Sectoral Indices",
    },
    "scan_types": {
        "X": {
            "name": "Scanners",
            "description": "50+ Professional Stock Scanners",
            "submenu": {
                # Breakout Scanners (0-10)
                0: {"name": "Full Screening", "description": "All patterns, indicators & breakouts"},
                1: {"name": "Breakout (Consolidation)", "description": "Stocks breaking out of consolidation zones"},
                2: {"name": "Top Gainers", "description": "Today's top gainers (>2%)"},
                3: {"name": "Top Losers", "description": "Today's top losers (>2%)"},
                4: {"name": "Volume Breakout", "description": "Unusual volume with price breakout"},
                5: {"name": "52 Week High", "description": "Stocks at 52-week high"},
                6: {"name": "10 Day High", "description": "Stocks at 10-day high"},
                7: {"name": "52 Week Low", "description": "Stocks at 52-week low (reversal potential)"},
                8: {"name": "Volume Surge", "description": "Volume > 2.5x average"},
                9: {"name": "RSI Oversold", "description": "RSI < 30 (potential bounce)"},
                10: {"name": "RSI Overbought", "description": "RSI > 70 (momentum stocks)"},
                # Moving Average Strategies (11-15)
                11: {"name": "Short-term MA Crossover", "description": "Price crossed 20 EMA"},
                12: {"name": "Bullish Engulfing", "description": "Bullish engulfing candlestick"},
                13: {"name": "Bearish Engulfing", "description": "Bearish engulfing pattern"},
                14: {"name": "VCP (Volatility Contraction)", "description": "Mark Minervini VCP pattern"},
                15: {"name": "Bull Crossover", "description": "20 EMA crossing 50 EMA"},
                # Advanced Patterns (16-25)
                16: {"name": "IPO Base Breakout", "description": "Recent IPOs breaking out"},
                17: {"name": "Bull Momentum", "description": "Strong bullish momentum"},
                18: {"name": "ATR Trailing", "description": "ATR-based trailing stops"},
                19: {"name": "PSar Reversal", "description": "Parabolic SAR reversal signal"},
                # NOTE: scanner 20 (ORB) removed 2026-05-31 — needs intraday data
                # which we don't ingest yet. Re-introduce once tick_collector is live.
                21: {"name": "NR4 Pattern", "description": "Narrow Range 4-day pattern"},
                22: {"name": "NR7 Pattern", "description": "Narrow Range 7-day pattern"},
                # NOTE: scanner IDs 23, 24, 25 (Cup&Handle / Double Bottom / Inv H&S)
                # are now served by the v2 Pattern Scanner (services.chart_patterns)
                # via /api/screener/patterns/v2/scan with the full ML + regime + volume
                # gate pipeline. The legacy IDs were removed 2026-05-31.
                # Momentum & Trend (26-35)
                26: {"name": "MACD Crossover", "description": "MACD bullish crossover"},
                27: {"name": "MACD Bearish", "description": "MACD bearish crossover"},
                28: {"name": "Inside Bar", "description": "Inside bar pattern (NR)"},
                29: {"name": "TTM Squeeze", "description": "TTM Squeeze indicator"},
                30: {"name": "Momentum Burst", "description": "Sudden momentum increase"},
                31: {"name": "Trend Template", "description": "Mark Minervini trend template"},
                32: {"name": "Super Trend", "description": "Super Trend indicator signal"},
                33: {"name": "Pivot Breakout", "description": "Breaking above pivot levels"},
                34: {"name": "Delivery %", "description": "High delivery percentage (>50%) — institutional interest"},
                35: {"name": "Bulk Deals", "description": "Recent bulk/block deals"},
                # Smart Money & F&O (36-42)
                36: {"name": "FII Net Buyers", "description": "Stocks with FII net buying"},
                37: {"name": "DII Net Buyers", "description": "Stocks with DII net buying"},
                38: {"name": "FII+DII Positive", "description": "Combined institutional buying"},
                39: {"name": "OI Analysis", "description": "Open Interest analysis for F&O"},
                40: {"name": "Long Buildup", "description": "F&O Long buildup (price up + OI up)"},
                41: {"name": "Short Buildup", "description": "F&O Short buildup (price down + OI up)"},
                42: {"name": "Short Covering", "description": "F&O Short covering (price up + OI down)"},
                # NOTE: scanner IDs 43-47, 49-51 (chart-pattern scanners) removed
                # 2026-05-31 — replaced by v2 Pattern Scanner. Scanner 48
                # (High & Tight Flag) stays because it's a pure indicator filter,
                # not a chart-pattern detection.
                48: {"name": "High & Tight Flag", "description": "High & tight flag momentum pattern"},
                # PR-S9 — cutting-edge new scanners (52-61)
                52: {"name": "Power Setup", "description": "Above EMA200 + RSI 50-70 + MACD + Vol + ADX>22 — high-conviction swing"},
                53: {"name": "Squeeze Release", "description": "TTM was squeezing, today breaking out with volume"},
                54: {"name": "MA Stack Bullish", "description": "Price > EMA21 > SMA50 > SMA200 (textbook uptrend)"},
                55: {"name": "Pre-Breakout Coil", "description": "NR7 + above EMA21 + low volume — breakout imminent"},
                56: {"name": "Fresh Trend Start", "description": "ADX rising from <20 + price > MAs — early in new trend"},
                57: {"name": "Oversold Bounce Setup", "description": "RSI<35 in structural uptrend (above SMA200)"},
                58: {"name": "Breakout w/ Volume", "description": "52w high + 2× volume + +1.5% — Stage-2 entry"},
                59: {"name": "Pullback to EMA21", "description": "Uptrend stock pulled back to 21EMA support"},
                60: {"name": "BB Squeeze Release", "description": "Bollinger band breakout from tight range with volume"},
                61: {"name": "RS Leader", "description": "Outperforming market median by 2%+ on real volume"},
                # PR-S17 — bearish counterparts to the top bullish setups
                62: {"name": "Power Setup Short", "description": "Below EMA200 + RSI 30-50 + MACD bear + Vol + ADX>22 — high-conviction short"},
                63: {"name": "MA Stack Bearish", "description": "Price < EMA21 < SMA50 < SMA200 (textbook downtrend)"},
                64: {"name": "Fresh Downtrend Start", "description": "ADX rising from <20 + price < MAs — early in new downtrend"},
                65: {"name": "Overbought Rejection Setup", "description": "RSI>65 in structural downtrend (below SMA200)"},
                66: {"name": "Breakdown w/ Volume", "description": "52w low + 2× volume + -1.5% — Stage-4 short"},
                67: {"name": "Rally to EMA21 (Short)", "description": "Downtrend stock rallied into 21EMA resistance"},
                68: {"name": "BB Squeeze Release Short", "description": "Bollinger band breakdown from tight range with volume"},
                69: {"name": "RS Laggard", "description": "Underperforming market median by 2%+ — short the weak"},
                70: {"name": "Bear Momentum", "description": "Strong bearish momentum: -3%+, RSI<35, MACD bear, volume"},
                71: {"name": "Momentum Crash", "description": "Sudden bearish acceleration — bar size > 1.5× ATR with volume"},
                # PR-S18 — 10 institutional swing setups (verified author algorithms)
                72: {"name": "Pocket Pivot", "description": "Kacher/Morales — vol > max 10-day down-day vol, close ≥ EMA10"},
                73: {"name": "Wyckoff Spring", "description": "Penetration of support reclaimed same day on normal volume"},
                74: {"name": "Episodic Pivot", "description": "Bonde/Qullamaggie — 8%+ gap with 2× volume into 52w highs"},
                75: {"name": "Holy Grail", "description": "Raschke — ADX≥30 trend + EMA21 pullback + mild RSI"},
                76: {"name": "Coiled Spring", "description": "Crabel — BB-width 60-bar rank <25% + NR7 near 52w highs"},
                77: {"name": "Inside After Trend Day", "description": "Crabel — inside bar after >1.5×ATR trend day"},
                78: {"name": "Three White Soldiers Pullback", "description": "Nison/Minervini — 3 green closes after EMA21 dip"},
                79: {"name": "Dragonfly Doji at MA", "description": "Nison — dragonfly doji touching EMA21/SMA50 in uptrend"},
                80: {"name": "Gap Fill Reversal", "description": "Murphy — exhaustion gap closed below open with volume (bearish)"},
                81: {"name": "Weekly Pivot Reclaim", "description": "Ochoa/Varsity — close above CPR TC after prior week below BC"},
                # PR-S18 — 5 positional setups
                82: {"name": "Stage 2 Acceleration", "description": "Weinstein — close > SMA150 rising + 52w high + volume"},
                83: {"name": "CAN SLIM Base Breakout", "description": "O'Neil — flat 60-bar base + 52w high + 1.5× volume"},
                84: {"name": "Cup-Handle Volume Pocket", "description": "O'Neil — 12-33% base depth + breakout volume signature"},
                85: {"name": "PEAD Proxy", "description": "Bernard-Thomas — |return|>5% + 3× volume + uptrend (60d drift)"},
                86: {"name": "Channel Mid-Reversion", "description": "Murphy/Pring — ascending channel + RSI<40 mean-revert long"},
                # PR-S20 — per-stock F&O scanners (need NSE participant OI feed)
                87: {"name": "Long Unwinding", "description": "Price ↓ + OI ↓ — long-position covering (trend-exhaustion warning)"},
                88: {"name": "OI Spike", "description": "Single-symbol OI ≥20% change — institutional positioning footprint"},
            },
        },
        "C": {
            "name": "Nifty Prediction (AI/ML)",
            "description": "AI-powered Nifty index prediction",
            "features": [
                "HMM Regime Detector (bull/bear/sideways)",
                "LightGBM signal classifier",
                "TFT 5-bar price forecaster",
                "Support/Resistance from real pivots",
            ],
        },
        "M": {
            "name": "ML Signals",
            "description": "Machine Learning based trading signals",
            "features": [
                "RandomForest breakout meta-labeler",
                "LightGBM 3-class signal gate",
                "AI stock ranker",
                "Ensemble meta-learner",
            ],
        },
        "T": {
            "name": "Trend Forecast",
            "description": "Multi-timeframe trend forecasting",
            "features": [
                "Intraday / Short-term / Medium-term analysis",
                "RSI + MACD + SMA confluence",
                "ATR-based target estimation",
            ],
        },
    },
}


# =============================================================================
# STOCK METADATA (name + sector for ~200 NSE stocks)
# =============================================================================

NSE_STOCK_INFO: Dict[str, Dict[str, str]] = {
    # Nifty 50
    "RELIANCE": {"name": "Reliance Industries", "sector": "Energy"},
    "TCS": {"name": "Tata Consultancy Services", "sector": "IT"},
    "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking"},
    "INFY": {"name": "Infosys", "sector": "IT"},
    "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking"},
    "HINDUNILVR": {"name": "Hindustan Unilever", "sector": "FMCG"},
    "SBIN": {"name": "State Bank of India", "sector": "Banking"},
    "BHARTIARTL": {"name": "Bharti Airtel", "sector": "Telecom"},
    "KOTAKBANK": {"name": "Kotak Mahindra Bank", "sector": "Banking"},
    "ITC": {"name": "ITC Limited", "sector": "FMCG"},
    "LT": {"name": "Larsen & Toubro", "sector": "Infrastructure"},
    "AXISBANK": {"name": "Axis Bank", "sector": "Banking"},
    "ASIANPAINT": {"name": "Asian Paints", "sector": "Paints"},
    "MARUTI": {"name": "Maruti Suzuki", "sector": "Auto"},
    "TITAN": {"name": "Titan Company", "sector": "Consumer"},
    "BAJFINANCE": {"name": "Bajaj Finance", "sector": "NBFC"},
    "WIPRO": {"name": "Wipro", "sector": "IT"},
    "ONGC": {"name": "Oil & Natural Gas Corp", "sector": "Energy"},
    "NTPC": {"name": "NTPC Limited", "sector": "Power"},
    "POWERGRID": {"name": "Power Grid Corp", "sector": "Power"},
    "SUNPHARMA": {"name": "Sun Pharmaceutical", "sector": "Pharma"},
    "ULTRACEMCO": {"name": "UltraTech Cement", "sector": "Cement"},
    "TATAMOTORS": {"name": "Tata Motors", "sector": "Auto"},
    "NESTLEIND": {"name": "Nestle India", "sector": "FMCG"},
    "TECHM": {"name": "Tech Mahindra", "sector": "IT"},
    "M&M": {"name": "Mahindra & Mahindra", "sector": "Auto"},
    "HCLTECH": {"name": "HCL Technologies", "sector": "IT"},
    "BAJAJFINSV": {"name": "Bajaj Finserv", "sector": "NBFC"},
    "ADANIENT": {"name": "Adani Enterprises", "sector": "Infra"},
    "ADANIPORTS": {"name": "Adani Ports", "sector": "Ports"},
    # Nifty Next 50
    "DIVISLAB": {"name": "Divi's Laboratories", "sector": "Pharma"},
    "DRREDDY": {"name": "Dr. Reddy's Labs", "sector": "Pharma"},
    "CIPLA": {"name": "Cipla", "sector": "Pharma"},
    "GRASIM": {"name": "Grasim Industries", "sector": "Diversified"},
    "BRITANNIA": {"name": "Britannia Industries", "sector": "FMCG"},
    "HINDALCO": {"name": "Hindalco Industries", "sector": "Metals"},
    "JSWSTEEL": {"name": "JSW Steel", "sector": "Steel"},
    "TATASTEEL": {"name": "Tata Steel", "sector": "Steel"},
    "COALINDIA": {"name": "Coal India", "sector": "Mining"},
    "INDUSINDBK": {"name": "IndusInd Bank", "sector": "Banking"},
    "BPCL": {"name": "Bharat Petroleum", "sector": "Energy"},
    "EICHERMOT": {"name": "Eicher Motors", "sector": "Auto"},
    "HEROMOTOCO": {"name": "Hero MotoCorp", "sector": "Auto"},
    "BAJAJ-AUTO": {"name": "Bajaj Auto", "sector": "Auto"},
    "TATACONSUM": {"name": "Tata Consumer", "sector": "FMCG"},
    "SHRIRAMFIN": {"name": "Shriram Finance", "sector": "NBFC"},
    "APOLLOHOSP": {"name": "Apollo Hospitals", "sector": "Healthcare"},
    "LTIM": {"name": "LTIMindtree", "sector": "IT"},
    "HAL": {"name": "Hindustan Aeronautics", "sector": "Defence"},
    "BEL": {"name": "Bharat Electronics", "sector": "Defence"},
    # Midcap & Smallcap
    "TRENT": {"name": "Trent Limited", "sector": "Retail"},
    "PERSISTENT": {"name": "Persistent Systems", "sector": "IT"},
    "POLYCAB": {"name": "Polycab India", "sector": "Cables"},
    "DIXON": {"name": "Dixon Technologies", "sector": "Electronics"},
    "COFORGE": {"name": "Coforge", "sector": "IT"},
    "MUTHOOTFIN": {"name": "Muthoot Finance", "sector": "NBFC"},
    "ASTRAL": {"name": "Astral Ltd", "sector": "Pipes"},
    "PIIND": {"name": "PI Industries", "sector": "Chemicals"},
    "DEEPAKNTR": {"name": "Deepak Nitrite", "sector": "Chemicals"},
    "ANGELONE": {"name": "Angel One", "sector": "Broking"},
    "HAPPSTMNDS": {"name": "Happiest Minds", "sector": "IT"},
    "TANLA": {"name": "Tanla Platforms", "sector": "IT"},
    "ZOMATO": {"name": "Zomato", "sector": "Food Tech"},
    "DELHIVERY": {"name": "Delhivery", "sector": "Logistics"},
    "IRCTC": {"name": "IRCTC", "sector": "Travel"},
    "IRFC": {"name": "IRFC", "sector": "Finance"},
    "DLF": {"name": "DLF Limited", "sector": "Real Estate"},
    "GODREJPROP": {"name": "Godrej Properties", "sector": "Real Estate"},
    "OBEROIRLTY": {"name": "Oberoi Realty", "sector": "Real Estate"},
    "CHOLAFIN": {"name": "Cholamandalam Finance", "sector": "NBFC"},
    "MFSL": {"name": "Max Financial Services", "sector": "Insurance"},
    "SBICARD": {"name": "SBI Cards", "sector": "Finance"},
    "CANBK": {"name": "Canara Bank", "sector": "Banking"},
    "PNB": {"name": "Punjab National Bank", "sector": "Banking"},
    "BANKBARODA": {"name": "Bank of Baroda", "sector": "Banking"},
    "NHPC": {"name": "NHPC Limited", "sector": "Power"},
    "SJVN": {"name": "SJVN Limited", "sector": "Power"},
    "RECLTD": {"name": "REC Limited", "sector": "Power"},
    "PFC": {"name": "Power Finance Corp", "sector": "Power"},
    "GAIL": {"name": "GAIL India", "sector": "Gas"},
    "NMDC": {"name": "NMDC Limited", "sector": "Mining"},
    "SAIL": {"name": "Steel Authority", "sector": "Steel"},
    "VEDL": {"name": "Vedanta", "sector": "Metals"},
    "JINDALSTEL": {"name": "Jindal Steel", "sector": "Steel"},
    "LUPIN": {"name": "Lupin", "sector": "Pharma"},
    "AUROPHARMA": {"name": "Aurobindo Pharma", "sector": "Pharma"},
    "BIOCON": {"name": "Biocon", "sector": "Pharma"},
    "DABUR": {"name": "Dabur India", "sector": "FMCG"},
    "MARICO": {"name": "Marico Limited", "sector": "FMCG"},
    "COLPAL": {"name": "Colgate Palmolive", "sector": "FMCG"},
    "GODREJCP": {"name": "Godrej Consumer", "sector": "FMCG"},
    "HAVELLS": {"name": "Havells India", "sector": "Electricals"},
    "VOLTAS": {"name": "Voltas", "sector": "Consumer Durables"},
    "CROMPTON": {"name": "Crompton Greaves", "sector": "Electricals"},
    "PIDILITIND": {"name": "Pidilite Industries", "sector": "Chemicals"},
    "SRF": {"name": "SRF Limited", "sector": "Chemicals"},
    "SIEMENS": {"name": "Siemens India", "sector": "Capital Goods"},
    "ABB": {"name": "ABB India", "sector": "Capital Goods"},
    "INDIGO": {"name": "InterGlobe Aviation", "sector": "Aviation"},
    "PAGEIND": {"name": "Page Industries", "sector": "Textiles"},
    "MPHASIS": {"name": "Mphasis", "sector": "IT"},
    "KPITTECH": {"name": "KPIT Technologies", "sector": "IT"},
    "OFSS": {"name": "Oracle Financial", "sector": "IT"},
    "NAUKRI": {"name": "Info Edge", "sector": "Internet"},
}


# Filter library extracted to ``screener/filters.py`` so this module
# focuses on engine orchestration (universe → indicators → dispatch).


# =============================================================================
# LIVE SCREENER ENGINE
# =============================================================================

class LiveScreenerEngine:
    """
    Real-data screener engine.
    Uses Kite Connect + jugaad-data for market data.
    Pipeline: Kite data → compute_all_indicators → scanner filters → frontend JSON.
    """

    def __init__(self):
        from ...data.universe import UniverseScreener
        self._universe_screener = UniverseScreener()

        # Caches
        self._universe_cache: Optional[Tuple[List[str], datetime]] = None
        self._computed_cache: Optional[Tuple[pd.DataFrame, Dict[str, pd.DataFrame], datetime]] = None
        self._scanner_cache: Dict[str, Tuple[Dict, datetime]] = {}
        self._nifty_cache: Optional[Tuple[Dict, datetime]] = None

        self.CACHE_TTL = timedelta(minutes=5)
        self.UNIVERSE_CACHE_TTL = timedelta(hours=1)
        self.BATCH_SIZE = 200
        self.MIN_TRADING_DAYS = 20

        # Data provider
        self._data_source = "kite"

        self.screener_active = True
        logger.info(f"LiveScreenerEngine data source: {self._data_source}")

        # Project root: this file lives at backend/data/screener/engine.py,
        # so parents[3] is the repo root (same Path expression as ROOT_DIR above).
        self._project_root = str(ROOT_DIR)

        # Model loading via registry compat (B2 first, disk fallback).
        from pathlib import Path as _Path
        try:
            from ...ai.registry import resolve_model_file as _resolve_model_file
        except ImportError:
            _resolve_model_file = None

        def _resolve(model_name: str, filename: str):
            disk = _Path(self._project_root) / "artifacts" / "models" / filename
            if _resolve_model_file is None:
                return str(disk) if disk.exists() else None
            resolved = _resolve_model_file(model_name, filename, disk)
            return str(resolved) if resolved else None

        # BreakoutMetaLabeler — PROD, Scanner Lab confidence tag.
        self._ml_labeler = None
        try:
            from ml.features.patterns import BreakoutMetaLabeler
            model_path = _resolve("breakout_meta_labeler", "breakout_meta_labeler.pkl")
            if model_path:
                labeler = BreakoutMetaLabeler()
                labeler.load(model_path)
                if labeler.is_trained:
                    self._ml_labeler = labeler
                    logger.info("ML breakout meta-labeler loaded successfully")
        except Exception as e:
            logger.debug(f"ML meta-labeler not loaded (patterns work without it): {e}")

        # HMM regime detector — PROD.
        self._regime_detector = None
        try:
            from ml.regime_detector import MarketRegimeDetector
            regime_path = _resolve("regime_hmm", "regime_hmm.pkl")
            if regime_path:
                detector = MarketRegimeDetector()
                detector.load(regime_path)
                if detector.is_trained:
                    self._regime_detector = detector
                    logger.info("HMM regime detector loaded for screener")
        except Exception as e:
            logger.debug(f"Regime detector not loaded: {e}")

        # LGBMGate — SHADOW (screener can still surface the probability).
        self._lgbm_gate = None
        try:
            from ...ai.model_registry import LGBMGate
            lgbm_path = _resolve("lgbm_signal_gate", "lgbm_signal_gate.txt")
            if lgbm_path:
                self._lgbm_gate = LGBMGate(lgbm_path)
                logger.info("LGBMGate loaded for screener (SHADOW)")
        except Exception as e:
            logger.debug(f"LGBMGate not loaded: {e}")

    # =========================================================================
    # UNIVERSE
    # =========================================================================

    def _get_universe(self) -> List[str]:
        """Get NSE symbol universe, cached for 1 hour."""
        if self._universe_cache:
            symbols, cached_at = self._universe_cache
            if datetime.now() - cached_at < self.UNIVERSE_CACHE_TTL:
                return symbols

        symbols = self._universe_screener._get_all_nse_symbols()
        if not symbols:
            # Fallback: hardcoded top stocks
            symbols = list(NSE_STOCK_INFO.keys())

        self._universe_cache = (symbols, datetime.now())
        logger.info(f"LiveScreener: loaded {len(symbols)} symbols")
        return symbols

    # =========================================================================
    # DATA PIPELINE
    # =========================================================================

    def _get_computed_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Core data pipeline. Downloads OHLCV + computes all indicators.
        Uses Kite Connect + jugaad-data for market data.
        Returns (summary_df, per_symbol_dfs), cached for 5 minutes.
        """
        if self._computed_cache:
            summary_df, per_symbol_dfs, cached_at = self._computed_cache
            if datetime.now() - cached_at < self.CACHE_TTL and not summary_df.empty:
                return summary_df, per_symbol_dfs

        symbols = self._get_universe()
        if not symbols:
            return pd.DataFrame(), {}

        # PR-S2.1 — keep the legacy bulk pipeline at 500 symbols (its
        # synchronous fetch blocks startup pre-warm when bigger). The
        # full 2,136-symbol NSE universe is served by the streaming
        # /patterns/v2/scan/stream endpoint instead, which fans out in
        # batches without blocking. The 50+ technical screeners stay
        # on largecaps-only via this cap.
        symbols = symbols[:500]

        # Fetch via Kite admin + jugaad-data
        per_symbol_dfs, summary_rows = self._fetch_via_kite(symbols)

        if not summary_rows:
            logger.warning("LiveScreener: no data computed")
            return pd.DataFrame(), {}

        summary_df = pd.DataFrame(summary_rows)
        self._computed_cache = (summary_df, per_symbol_dfs, datetime.now())
        logger.info(f"LiveScreener: computed indicators for {len(summary_df)} stocks via {self._data_source}")
        return summary_df, per_symbol_dfs

    def _fetch_via_kite(self, symbols: List[str]) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
        """Fetch historical OHLCV from Kite Connect + compute indicators."""
        from ...data.market import get_market_data_provider
        provider = get_market_data_provider()._get_kite_provider()

        per_symbol_dfs: Dict[str, pd.DataFrame] = {}
        summary_rows: List[Dict] = []
        total = len(symbols)
        processed = 0

        # Use batch fetch from Kite provider
        batch_data = provider.fetch_historical_batch(symbols, period='6mo')

        for symbol, df in batch_data.items():
            try:
                if df is None or df.empty or len(df) < 20:
                    continue

                # Ensure lowercase columns
                df.columns = [c.lower() for c in df.columns]

                # Compute indicators
                df = compute_all_indicators(df)
                per_symbol_dfs[symbol] = df

                # Build summary row using the full extraction method
                latest = df.iloc[-1]
                row = self._extract_summary_row(symbol, df, latest)
                if row:
                    summary_rows.append(row)
                else:
                    continue
                processed += 1
            except Exception as e:
                logger.debug(f"Kite fetch failed for {symbol}: {e}")
                continue

        logger.info(f"Kite: processed {processed}/{total} symbols")
        return per_symbol_dfs, summary_rows

    def _extract_summary_row(self, symbol: str, df: pd.DataFrame, last: pd.Series) -> Optional[Dict]:
        """Wrapper around data/screener/sources.extract_summary_row (PR-A4.3)."""
        from .sources import extract_summary_row
        return extract_summary_row(symbol, df, last)

    # =========================================================================
    # SCANNER EXECUTION
    # =========================================================================

    async def run_scanner(
        self,
        scanner_id: int,
        exchange: str = "N",
        index: str = "12",
    ) -> Dict[str, Any]:
        """Run a specific scanner with real data."""
        scanner_info = SCANNER_MENU["scan_types"]["X"]["submenu"].get(scanner_id, {})

        # Check scanner result cache
        cache_key = f"scan_{exchange}_{index}_{scanner_id}"
        if cache_key in self._scanner_cache:
            cached_data, cached_at = self._scanner_cache[cache_key]
            if datetime.now() - cached_at < self.CACHE_TTL:
                return cached_data

        # Get computed data (run in thread to avoid blocking event loop)
        import asyncio
        summary_df, _per_symbol_dfs = await asyncio.to_thread(self._get_computed_data)

        if summary_df.empty:
            return {
                "success": False,
                "scanner_id": scanner_id,
                "error": "No market data available. Try again in a moment.",
                "results": [],
                "count": 0,
            }

        # Apply filter function — pattern-detection scanners (formerly 23-25,
        # 43-47, 49-51) and ORB (20) live in the v2 pipeline now; any unknown
        # ID falls back to the full screening view rather than 404'ing.
        filter_fn = SCANNER_FILTERS.get(scanner_id, _filter_full_screening)
        try:
            filtered = filter_fn(summary_df.copy())
        except Exception as e:
            logger.error(f"Scanner {scanner_id} filter error: {e}")
            filtered = pd.DataFrame()

        results = self._format_for_frontend(filtered, scanner_id)

        response = {
            "success": True,
            "scanner_id": scanner_id,
            "scanner_name": scanner_info.get("name", f"Scanner {scanner_id}"),
            "scanner_description": scanner_info.get("description", ""),
            "exchange": exchange,
            "timestamp": datetime.now().isoformat(),
            "source": "live",
            "data_provider": self._data_source,
            "results": results[:50],  # Cap at 50 results
            "count": min(len(results), 50),
        }
        self._scanner_cache[cache_key] = (response, datetime.now())
        return response

    # ================================================================
    # FORMATTING HELPERS — see data/screener/formatting.py
    # Pattern-scanner formatting (format_pattern_signal) removed along
    # with _run_pattern_scanner when the legacy chart-pattern IDs went
    # to the v2 pipeline. format_stock_result kept for full-screening
    # fallback path inside formatting.py itself.
    # ================================================================

    def _format_for_frontend(self, df: pd.DataFrame, scanner_id: int) -> List[Dict]:
        from .formatting import format_for_frontend
        return format_for_frontend(df, scanner_id, NSE_STOCK_INFO)

    # =========================================================================
    # AI / ML ENDPOINTS
    # =========================================================================

    def _fetch_index_df(self, td_symbol: str = "NIFTY 50", yf_symbol: str = "^NSEI",
                        period: str = "6mo") -> Optional[pd.DataFrame]:
        """Fetch index historical data from Kite Connect."""
        from ...data.market import get_market_data_provider
        provider = get_market_data_provider()._get_kite_provider()
        try:
            # Map the legacy symbol names to index name
            index_map = {
                "NIFTY 50": "NIFTY",
                "Nifty 50": "NIFTY",
                "NIFTY BANK": "BANKNIFTY",
                "^NSEI": "NIFTY",
                "^NSEBANK": "BANKNIFTY",
                "^INDIAVIX": "VIX"}
            index_name = index_map.get(td_symbol, index_map.get(yf_symbol, "NIFTY"))
            df = provider.get_historical_index(index_name, period)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception as e:
            logger.debug(f"Kite index fetch failed for {td_symbol}: {e}")
        return None

    @property
    def _market(self):
        """Lazy MarketAnalytics — see data/screener/market.py."""
        if not hasattr(self, "_market_cached") or self._market_cached is None:
            from .market import MarketAnalytics
            self._market_cached = MarketAnalytics(
                fetch_index_df=self._fetch_index_df,
                get_computed_data=self._get_computed_data,
                regime_detector=self._regime_detector,
                lgbm_gate=self._lgbm_gate,
                stock_info=NSE_STOCK_INFO,
                cache_ttl=self.CACHE_TTL,
            )
        return self._market_cached

    async def get_nifty_prediction(self) -> Dict[str, Any]:
        """Delegated to data/screener/market.MarketAnalytics (PR-A4.5)."""
        return await self._market.get_nifty_prediction()

    async def get_trend_forecast(self, symbol: str = "NIFTY") -> Dict[str, Any]:
        """Delegated to data/screener/market.MarketAnalytics."""
        return await self._market.get_trend_forecast(symbol)

    async def get_market_regime(self) -> Dict[str, Any]:
        """Delegated to data/screener/market.MarketAnalytics."""
        return await self._market.get_market_regime()

    async def get_trend_analysis(self) -> Dict[str, Any]:
        """Delegated to data/screener/market.MarketAnalytics."""
        return await self._market.get_trend_analysis()

    def get_all_scanners(self) -> Dict[str, Any]:
        """Return complete scanner menu and capabilities."""
        scanner_details = SCANNER_MENU["scan_types"]["X"]["submenu"]
        return {
            "total_scanners": len(scanner_details),
            "exchanges": list(SCANNER_MENU["exchanges"].keys()),
            "stock_universe": {
                "NSE": "1800+ stocks",
                "BSE": "3000+ stocks",
                "F&O": "200+ derivatives",
            },
            "categories": [
                {"id": "breakout", "name": "Breakout Scanners", "count": 8, "scanners": [0, 1, 4, 5, 6, 7, 20, 33]},
                {"id": "momentum", "name": "Momentum Scanners", "count": 7, "scanners": [2, 3, 10, 17, 26, 30, 31]},
                {"id": "volume", "name": "Volume Scanners", "count": 5, "scanners": [4, 8, 34, 35, 38]},
                {"id": "reversal", "name": "Reversal Scanners", "count": 6, "scanners": [9, 12, 19, 24, 25, 28]},
                {"id": "patterns", "name": "Chart Patterns", "count": 12, "scanners": [12, 13, 14, 21, 22, 23, 24, 25, 43, 44, 45, 46]},
                {"id": "ma_strategies", "name": "Moving Average Strategies", "count": 5, "scanners": [11, 15, 26, 27, 32]},
                {"id": "smart_money", "name": "Smart Money / Institutional", "count": 5, "scanners": [34, 35, 36, 37, 38]},
                {"id": "fo_analysis", "name": "F&O / Derivatives", "count": 4, "scanners": [39, 40, 41, 42]},
                {"id": "chart_patterns", "name": "Advanced Chart Patterns", "count": 9, "scanners": [43, 44, 45, 46, 47, 49, 50, 51]},
            ],
            "ai_ml_features": {
                "nifty_prediction": {
                    "enabled": True,
                    "models": ["HMM Regime", "LightGBM", "TFT Forecaster"],
                    "source": "live",
                },
                "ml_signals": {
                    "enabled": True,
                    "models": ["RandomForest Meta-Labeler", "LightGBM Gate", "Ensemble"],
                },
                "trend_forecast": {"enabled": True, "timeframes": ["Intraday", "Short-term", "Medium-term"]},
                "quantai_picks": {"enabled": True, "model": "Alpha cross-sectional ranker"},
            },
            "scanner_details": scanner_details,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_live_screener_instance: Optional[LiveScreenerEngine] = None


def get_live_screener() -> LiveScreenerEngine:
    """Get singleton LiveScreenerEngine instance."""
    global _live_screener_instance
    if _live_screener_instance is None:
        _live_screener_instance = LiveScreenerEngine()
        logger.info("LiveScreenerEngine initialized (Kite + jugaad-data)")
    return _live_screener_instance
