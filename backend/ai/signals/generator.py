"""
================================================================================
SWING AI SIGNAL GENERATION SERVICE
================================================================================
Generates trading signals using 6 backtested strategies:
1. Each strategy scans independently for setups
2. ML meta-labeler filters weak pattern breakouts
3. Best signal per symbol (deduplication)
4. Signals saved to Supabase for frontend display

Strategies (backtested on 419 stocks x 5 years):
- BOS_Structure: 46% WR, PF 1.35 (Break of market structure)
- Volume_Reversal: 42.6% WR, PF 1.08 (Wyckoff VPA)
- Trend_Pullback: 36.9% WR (MA pullback in uptrend)
- Reversal_Patterns: 35% WR, PF 1.04 (IHS, double bottom, cup & handle)
- Candle_Reversal: 32% WR (Candlestick at support)
- Consolidation_Breakout: 32.4% WR + ML filter (Pattern breakouts)
================================================================================
"""

from .voters import (
    make_lgbm_voter, make_qlib_voter, make_regime_voter,
    make_tft_voter,
)
from .types import EnsembleVoter, GeneratedSignal
from .persistence import save_signals, save_universe, cache_candles
from .options import OptionsSignalEngine
from .ensemble import compute_ensemble_score
from ..qlib.engine import get_qlib_engine
from ..registry import resolve_model_file
from ...ai.feature_engineering import compute_features, split_feature_sets, build_feature_row
from ...ai.model_registry import LGBMGate, TFTPredictor
from ml.regime_detector import MarketRegimeDetector, compute_regime_features
import os
import logging
import sys
from pathlib import Path
from datetime import date
from typing import Dict, List, Optional, Tuple
import pandas as pd
from ...core.config import settings
from ...data.market import get_market_data_provider

logger = logging.getLogger(__name__)

# Allow importing from repo root (ml module). After the PR-A3 move the
# file lives at backend/ai/signals/generator.py — parents[3] is repo root.
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Note: ml.scanner (scan_stock + get_all_strategies) and
# ml.features.indicators are intentionally NOT imported here. The 6
# hand-coded strategies are no longer in the alpha path (locked plan
# §3.2). They remain available for Scanner Lab via direct import.

try:
    from ...data.universe import UniverseScreener
except ImportError:
    UniverseScreener = None

try:
    from ...trading.fo.engine import FOTradingEngine, NSE_LOT_SIZES
    from ...trading.fo.instruments import InstrumentMaster
except ImportError:
    FOTradingEngine = None
    NSE_LOT_SIZES = {}
    InstrumentMaster = None

# Required model adapters. None of these are wrapped in try/except — if
# any are unimportable in production we want the service to fail fast at
# startup, not produce signals via heuristic fallbacks (locked 2026-04-19).

# In-package imports (siblings under ai/signals/).

# HMM bear-regime final confidence multiplier (Step 2 §1.12).
_BEAR_REGIME_CONFIDENCE_GATE = 0.6


class SignalGenerator:
    """
    Signal generation service.

    Model-first pipeline (locked plan §3.2): LGBM + TFT + Qlib + HMM run on
    every candidate (sentiment/Mood was removed from the ensemble 2026-06-06
    — it's a standalone engine now). A signal is emitted only when
    ``min_agreement`` voters concur with BUY AND the weighted confidence
    ≥ ``min_confidence`` AND TFT-derived risk:reward ≥ ``min_risk_reward``.

    Entry/SL/TPs are derived from TFT quantile forecasts + ATR safety
    bounds. The 6 hand-coded strategies are NOT in the alpha path
    (they remain available for Scanner Lab).
    """

    def __init__(
        self,
        supabase_client,
        min_confidence: float = 40.0,
        min_risk_reward: float = 1.5,
        min_agreement: int = 3,
        **kwargs,  # Accept and ignore legacy params (modal_endpoint, use_enhanced_ai, etc.)
    ):
        self.supabase = supabase_client
        self.min_confidence = min_confidence
        self.min_risk_reward = min_risk_reward
        # Minimum number of model voters (out of 5) that must concur
        # with BUY for a candidate to qualify. 3/5 = simple majority,
        # the safe default; can be tightened to 4/5 in admin config
        # for high-conviction-only mode.
        self.min_agreement = min_agreement

        # Strategy name → catalog ID mapping (loaded on first use)
        self._strategy_catalog_map: Optional[Dict[str, str]] = None

        # ---------------------------------------------------------------
        # All required AI models load at construction time. Missing or
        # broken artifacts raise — never degrade to heuristic defaults
        # (locked 2026-04-19: "no fallbacks, ML features ship only with
        # real trained models").
        # ---------------------------------------------------------------

        def _require(model_name: str, filename: str) -> Path:
            path = resolve_model_file(
                model_name, filename, ROOT_DIR / "artifacts" / "models" / filename,
            )
            if path is None:
                raise RuntimeError(
                    f"{model_name} artifact ({filename}) not found in registry or disk. "
                    f"Promote a trained version: "
                    f"python -m ml.training.runner --only {model_name} --promote"
                )
            return path

        # PR-A: BreakoutMetaLabeler loader removed from signal-path.
        # The Scanner Lab loads it independently via live_screener_engine.
        # SignalGenerator never called labeler.predict() — dead loader.
        self._ml_labeler = None

        # LGBM signal gate — REQUIRED PROD voter (weight 0.30).
        self._lgbm_gate = LGBMGate(
            str(_require("lgbm_signal_gate", "lgbm_signal_gate.txt"))
        )
        logger.info("LGBMGate loaded (PROD voter, weight=0.30)")

        # HMM regime detector — REQUIRED. Drives bear gate + sizing.
        detector = MarketRegimeDetector()
        detector.load(str(_require("regime_hmm", "regime_hmm.pkl")))
        if not detector.is_trained:
            raise RuntimeError(
                "regime_hmm loaded but is_trained=False"
            )
        self._regime_detector = detector
        logger.info("HMM regime detector loaded (PROD voter, weight=0.10)")

        # TFT price forecaster — REQUIRED PROD voter (weight 0.30).
        self._tft_predictor = TFTPredictor(
            str(_require("tft_swing", "tft_model.ckpt")),
            str(_require("tft_swing", "tft_config.json")),
        )
        logger.info("TFT price forecaster loaded (PROD voter, weight=0.30)")

        # Qlib alpha158 cross-sectional ranker — REQUIRED PROD voter
        # (weight 0.20). Initialises Qlib against the production NSE
        # provider directory and loads the trained booster from registry.
        self._qlib_engine = get_qlib_engine()
        if not self._qlib_engine.loaded and not self._qlib_engine.load():
            raise RuntimeError(
                "qlib_alpha158 engine failed to load. Required: "
                "(1) pyqlib installed, "
                "(2) NSE provider directory at $QLIB_PROVIDER_URI "
                "(bootstrap: scripts/data/ingest_nse_to_qlib.py), "
                "(3) qlib_alpha158 promoted in model_versions."
            )
        logger.info("QlibEngine loaded (PROD voter, weight=0.20)")

        # Sentiment ("Mood") was removed from the signal ensemble 2026-06-06
        # — it was a tiny 0.10 voter (an LLM at runtime, not real FinBERT) that
        # added more noise than edge. It now lives only as a standalone
        # on-demand engine (News Intelligence / SentimentEngine → Mood). No
        # sentiment voter is loaded here (the stale "FinBERT PROD voter" log
        # was removed — the ensemble is 4 voters: LGBM/TFT/Qlib/HMM).

        # Ensemble meta-learner — retired (Step 1 §6 / Step 2 §10).
        self._ensemble_model = None

        # F&O helpers (optional)
        self.fo_engine = FOTradingEngine() if FOTradingEngine else None
        if InstrumentMaster:
            self.instrument_master = InstrumentMaster(getattr(settings, 'FNO_INSTRUMENTS_FILE', ''))
            self.fo_symbols = set(NSE_LOT_SIZES.keys())
            if self.instrument_master.available():
                self.fo_symbols = self.instrument_master.get_fo_symbols() or self.fo_symbols
        else:
            self.instrument_master = None
            self.fo_symbols = set()

        self.fo_lot_sizes = {
            "NIFTY": 25, "BANKNIFTY": 15, "RELIANCE": 250, "TCS": 150,
            "HDFCBANK": 550, "INFY": 300, "ICICIBANK": 700, "SBIN": 750,
            "TATASTEEL": 425, "TRENT": 385, "POLYCAB": 200
        }

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    async def generate_daily_signals(self) -> List[GeneratedSignal]:
        """Main entry point — generates all signals for the day."""
        logger.info("Starting daily signal generation...")
        try:
            signals = await self.generate_intraday_signals(save=True)
            logger.info(f"Generated {len(signals)} signals for today")
            return signals
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            raise

    async def generate_eod_signals(self, signal_date: Optional[date] = None) -> List[GeneratedSignal]:
        """End-of-day scan. Signals are saved for the next trading day."""
        result = await self.run_eod_scan(signal_date=signal_date)
        return result.get("signals", [])

    async def run_eod_scan(
        self,
        signal_date: Optional[date] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, object]:
        """Run EOD scan, persist candidate universe, and generate signals."""
        candidates, source = self._load_eod_universe()
        trade_date = signal_date or date.today()
        await save_universe(
            self.supabase, candidates, trade_date, source,
            settings.EOD_SCAN_TYPE, run_id,
        )

        signals = await self.generate_intraday_signals(
            save=True,
            candidates=candidates,
            signal_date=trade_date,
        )

        return {
            "signals": signals,
            "candidate_count": len(candidates),
            "source": source,
            "scan_type": settings.EOD_SCAN_TYPE,
        }

    # ========================================================================
    # CORE SIGNAL GENERATION
    # ========================================================================

    async def generate_intraday_signals(
        self,
        save: bool = False,
        candidates: Optional[List[str]] = None,
        signal_date: Optional[date] = None,
    ) -> List[GeneratedSignal]:
        """Model-first signal pipeline (locked plan §3.2).

        For each candidate symbol, every required AI voter runs:
          - LGBMGate    → buy/hold/sell verdict probability
          - TFTPredictor → 5-day p10/p50/p90 forecast
          - QlibEngine  → cross-sectional rank (pre-fetched)
          - FinBERT     → news sentiment (pre-fetched)
          - HMM regime  → market-wide bear gate (pre-fetched)

        A signal is emitted ONLY when:
          1. ≥ ``min_agreement`` voters concur with BUY
          2. weighted ensemble confidence ≥ ``min_confidence``
          3. TFT-derived risk:reward ≥ ``min_risk_reward``

        Entry/SL/TPs are derived from TFT quantiles + ATR safety bounds.
        No strategies, no rule-based candidate detection — every emitted
        signal is the product of real model agreement.
        """
        logger.info("Starting model-first signal generation...")

        candidates = candidates or self._load_universe()
        signals: List[GeneratedSignal] = []
        provider = get_market_data_provider()

        # --- HMM Regime Detection (REQUIRED — no default-to-bull) ---
        # Regime drives the bear-gate sizing. If we can't compute it we
        # refuse the entire batch instead of serving signals at "default
        # bull" weights — that would be a heuristic fallback for a real
        # model output.
        nifty = provider.get_historical("NIFTY", period="6mo", interval="1d")
        if nifty is None or len(nifty) < 30:
            raise RuntimeError(
                f"NIFTY history unavailable (rows={0 if nifty is None else len(nifty)}) "
                "— cannot compute regime; refusing to ship signals"
            )
        nifty.columns = [c.lower() for c in nifty.columns]
        vix = provider.get_historical("VIX", period="6mo", interval="1d")
        if vix is None or len(vix) == 0:
            raise RuntimeError(
                "VIX history unavailable — required for regime features"
            )
        vix.columns = [c.lower() for c in vix.columns]
        regime_features = compute_regime_features(nifty, vix)
        regime_info = self._regime_detector.predict_regime(regime_features)
        logger.info(
            "Market regime: %s (confidence=%.2f)",
            regime_info["regime"], regime_info["confidence"],
        )

        # --- Qlib alpha158 cross-sectional ranks ---
        # rank_universe runs the trained LightGBM booster over Qlib's
        # Alpha158 features for every NSE instrument and returns one
        # row per symbol. We index by symbol so the per-candidate lookup
        # is O(1). Empty result = Qlib couldn't compute scores; refuse.
        qlib_rows = self._qlib_engine.rank_universe(instruments="nse_all")
        if not qlib_rows:
            raise RuntimeError(
                "qlib_alpha158 rank_universe returned no rows — "
                "Alpha158 fetch failed or instrument universe empty"
            )
        n_qlib = len(qlib_rows)
        qlib_score_by_symbol: Dict[str, float] = {}
        qlib_rank_by_symbol: Dict[str, int] = {}
        for r in qlib_rows:
            sym = r["symbol"]
            qlib_rank_by_symbol[sym] = int(r["qlib_rank"])
            # rank 1 = best → score 1.0; rank N = worst → score 0.0
            qlib_score_by_symbol[sym] = 1.0 - (r["qlib_rank"] - 1) / max(n_qlib - 1, 1)
        logger.info("QlibEngine ranked %d instruments", n_qlib)

        # --- Per-symbol model loop ---
        regime_label = regime_info["regime"]
        bear_active = regime_label == "bear"

        for symbol in candidates:
            try:
                # Skip when Qlib has no row — the voter would be missing,
                # which is incompatible with the no-fallbacks contract.
                qlib_score = qlib_score_by_symbol.get(symbol)
                if qlib_score is None:
                    logger.debug("Qlib has no row for %s — skipping", symbol)
                    continue

                hist = await provider.get_historical_async(
                    symbol, period="1y", interval="1d",
                )
                if hist is None or hist.empty:
                    continue
                hist = hist.copy()
                hist.columns = [c.lower() for c in hist.columns]
                hist = hist.tail(260)  # ~1y trading days

                # Append today's quote if the daily bar isn't there yet.
                try:
                    last_date = hist.index[-1].date()
                    if last_date < date.today():
                        quote = provider.get_quote(symbol)
                        if quote and quote.ltp:
                            hist.loc[pd.Timestamp(date.today())] = {
                                "open": quote.open,
                                "high": quote.high,
                                "low": quote.low,
                                "close": quote.ltp,
                                "volume": quote.volume,
                            }
                except Exception:
                    pass

                if len(hist) < 200:
                    continue
                await cache_candles(self.supabase, symbol, hist)

                # Compute features once — used by both LGBM and the
                # ATR-based level derivation below.
                feat_df = compute_features(hist)

                # LGBM inference. Per-symbol failure → skip the symbol
                # (don't synthesise a 0.0 score).
                try:
                    feat_row = build_feature_row(feat_df)
                    lgbm_feats, _ = split_feature_sets(feat_row)
                    lgbm_direction, _, lgbm_probs = self._lgbm_gate.predict(lgbm_feats)
                    lgbm_buy_prob = lgbm_probs.get("buy", 0.0) / 100.0
                except Exception as e:
                    logger.warning(
                        "LGBM inference failed for %s (%s) — skipping symbol",
                        symbol, e,
                    )
                    continue

                # TFT inference.
                try:
                    tft_result = self._tft_predictor.predict_for_stock(hist, symbol)
                except Exception as e:
                    logger.warning(
                        "TFT inference failed for %s (%s) — skipping symbol",
                        symbol, e,
                    )
                    continue
                if tft_result is None:
                    logger.warning(
                        "TFT returned None for %s (insufficient history?) — skipping",
                        symbol,
                    )
                    continue
                tft_score = float(tft_result.get("score", 0.0))
                tft_bullish = tft_result.get("direction") == "bullish"

                # --- Voter list (builders in ai/signals/voters.py) ---
                tft_direction = "bullish" if tft_bullish else "bearish"
                voters: List[EnsembleVoter] = [
                    make_lgbm_voter(lgbm_buy_prob, lgbm_direction),
                    make_tft_voter(tft_score, tft_direction),
                    make_qlib_voter(qlib_score),
                    make_regime_voter(regime_info["regime_id"], bear_active),
                ]

                # Agreement gate — at least min_agreement voters must
                # concur with BUY for the signal to be considered.
                model_agreement = sum(1 for v in voters if v.direction_agrees)
                if model_agreement < self.min_agreement:
                    logger.debug(
                        "Skip %s: only %d/%d voters agree with BUY",
                        symbol, model_agreement, len(voters),
                    )
                    continue

                # Ensemble confidence + bear-regime size multiplier.
                confidence = compute_ensemble_score(voters)
                if bear_active:
                    confidence *= _BEAR_REGIME_CONFIDENCE_GATE
                if confidence < self.min_confidence:
                    continue

                # Derive entry/SL/TP from TFT quantiles + ATR safety.
                try:
                    entry, stop_loss, target_1, target_2, target_3 = self._derive_levels(
                        hist, feat_df, tft_result,
                    )
                except ValueError as exc:
                    logger.debug(
                        "Skip %s: level derivation failed (%s)", symbol, exc,
                    )
                    continue

                risk = entry - stop_loss
                rr_ratio = (target_1 - entry) / risk
                if rr_ratio < self.min_risk_reward:
                    continue

                qlib_rank = qlib_rank_by_symbol.get(symbol)
                reasons = [
                    f"Models:{model_agreement}/{len(voters)}",
                    f"Forecast:{tft_score:.2f}",
                    f"Gate:{lgbm_direction}({lgbm_buy_prob * 100:.0f}%)",
                    f"Alpha:#{qlib_rank}/{n_qlib}" if qlib_rank else "Alpha:—",
                    f"Regime:{regime_label}",
                    "Direction:LONG",
                    "Segment:EQUITY",
                ]

                signal = GeneratedSignal(
                    symbol=symbol,
                    exchange="NSE",
                    segment="EQUITY",
                    direction="LONG",
                    confidence=round(confidence, 2),
                    entry_price=round(entry, 2),
                    stop_loss=round(stop_loss, 2),
                    target_1=round(target_1, 2),
                    target_2=round(target_2, 2),
                    target_3=round(target_3, 2),
                    risk_reward=round(rr_ratio, 2),
                    catboost_score=0.0,           # meta-labeler not in alpha (Scanner Lab only)
                    tft_score=round(tft_score, 4),
                    stockformer_score=0.0,        # legacy column, retired
                    lgbm_score=round(lgbm_buy_prob, 4),
                    model_agreement=model_agreement,
                    reasons=reasons,
                    is_premium=confidence >= 75,
                    strategy_name="ai_ensemble",
                    tft_prediction=tft_result,
                    regime_at_signal=regime_label,
                    regime_snapshot=regime_info,
                    lgbm_buy_prob=round(lgbm_buy_prob, 4),
                )
                signals.append(signal)

            except Exception as e:
                logger.warning("Signal generation failed for %s: %s", symbol, e)

        # No dedup — model-first generates exactly one signal per symbol.
        signals.sort(key=lambda x: x.confidence, reverse=True)
        if save:
            await save_signals(
                self.supabase, signals, signal_date,
                catalog_cache=self._catalog_cache(),
            )
        logger.info("Generated %d AI-ensembled signals", len(signals))
        return signals

    # ========================================================================
    # ENSEMBLE SCORING — see ai/signals/ensemble.py
    # ========================================================================

    @staticmethod
    def _derive_levels(
        hist: pd.DataFrame,
        feat_df: pd.DataFrame,
        tft_result: Dict,
    ) -> Tuple[float, float, float, float, float]:
        """Compute (entry, stop_loss, target_1, target_2, target_3) from
        TFT quantile forecasts + ATR safety bounds.

        - **entry**: most recent close.
        - **stop_loss**: ``min(p10_terminal, entry - 2.0 * ATR)`` — the
          *wider* (further from entry) of TFT's pessimistic 10th-percentile
          forecast and a 2-ATR safety stop. We want stops at least 2σ
          of recent volatility away from entry to avoid noise stop-outs
          on a 3-10 day swing hold; if TFT thinks the worst case is
          even further, give the trade that much room.
        - **target_1**: ``min(p90_terminal, entry + 4.0 * ATR)`` — TFT's
          optimistic 90th-percentile forecast capped at 4-ATR ceiling
          to avoid pricing in unrealistic moves.
        - **target_2 / target_3**: 1.5× / 2.0× extension of risk_unit
          (target_1 - entry) for trailing-target ladders.

        Raises ``ValueError`` when the derived levels are non-tradable
        (entry ≤ 0, ATR missing, SL ≥ entry, target_1 ≤ entry, or TFT
        quantiles missing). Caller skips the candidate.
        """
        last_close = float(hist["close"].iloc[-1])
        if last_close <= 0:
            raise ValueError("non-positive close price")

        atr_series = feat_df.get("atr_14")
        if atr_series is None or atr_series.empty:
            raise ValueError("ATR_14 not in feature frame")
        atr = float(atr_series.iloc[-1])
        if not (atr > 0):
            raise ValueError(f"ATR non-positive ({atr})")

        p10_list = tft_result.get("p10") or []
        p90_list = tft_result.get("p90") or []
        if not p10_list or not p90_list:
            raise ValueError("TFT p10/p90 quantile arrays missing")
        p10_term = float(p10_list[-1])
        p90_term = float(p90_list[-1])

        # Wider stop wins (further below entry). Always < entry because
        # the ATR floor (entry - 2*ATR) is strictly below entry when ATR > 0,
        # which we already enforced above.
        stop_loss = min(p10_term, last_close - 2.0 * atr)

        # Conservative target wins (closer to entry, higher hit prob).
        target_1 = min(p90_term, last_close + 4.0 * atr)
        if target_1 <= last_close:
            raise ValueError(
                f"target_1 {target_1:.2f} <= entry {last_close:.2f} "
                "(TFT p90 below current — bearish quantiles)"
            )

        risk_unit = target_1 - last_close
        target_2 = last_close + 1.5 * risk_unit
        target_3 = last_close + 2.0 * risk_unit
        return last_close, stop_loss, target_1, target_2, target_3

    # ========================================================================
    # UNIVERSE LOADING
    # ========================================================================

    def _load_universe(self) -> List[str]:
        """Load the alpha universe from disk. Refuses to substitute a
        hardcoded list when the file is missing — universe is data,
        a baked-in symbol list would silently shrink the alpha surface."""
        path = settings.ALPHA_UNIVERSE_FILE
        if not os.path.exists(path):
            raise RuntimeError(
                f"Alpha universe file not found at {path}. "
                f"Set ALPHA_UNIVERSE_FILE or seed data/nse_tiers/."
            )
        with open(path, "r", encoding="utf-8") as f:
            symbols = [s.strip().upper() for s in f if s.strip()]
        if not symbols:
            raise RuntimeError(f"Alpha universe file {path} is empty")
        return symbols[: settings.ALPHA_UNIVERSE_SIZE]

    def _load_eod_universe(self) -> Tuple[List[str], str]:
        """Load EOD universe via UniverseScreener (preferred) or
        ALPHA_UNIVERSE_FILE. Both paths are real data sources — no
        hardcoded fallback list."""
        if UniverseScreener is not None:
            screener = UniverseScreener()
            candidates = screener.screen_sync()
            if candidates and len(candidates) >= 30:
                logger.info(f"UniverseScreener returned {len(candidates)} candidates")
                return candidates, "universe_screener"
            logger.warning(
                "UniverseScreener returned %d candidates (<30) — "
                "falling back to ALPHA_UNIVERSE_FILE",
                len(candidates) if candidates else 0,
            )
        return self._load_universe(), "alpha_universe_file"

    # ========================================================================
    # PERSISTENCE — see ai/signals/persistence.py
    # ========================================================================

    def _catalog_cache(self) -> Dict[str, str]:
        """Lazy-init the catalog→UUID cache shared by save_signals,
        resolve_catalog_id, and OptionsSignalEngine."""
        if self._strategy_catalog_map is None:
            self._strategy_catalog_map = {}
        return self._strategy_catalog_map

    async def get_today_signals(
        self,
        segment: Optional[str] = None,
        direction: Optional[str] = None,
        is_premium: Optional[bool] = None
    ) -> List[Dict]:
        """Fetch today's signals from database."""
        today = date.today().isoformat()

        query = self.supabase.table("signals").select("*").eq("date", today).eq("status", "active")

        if segment:
            query = query.eq("segment", segment)
        if direction:
            query = query.eq("direction", direction)
        if is_premium is not None:
            query = query.eq("is_premium", is_premium)

        result = query.order("confidence", desc=True).execute()
        return result.data or []

    # ========================================================================
    # OPTIONS SIGNAL GENERATION (Sprint 4 — deployment-aware)
    # ========================================================================

    @property
    def _options_engine(self):
        """Lazily-built OptionsSignalEngine — see ai/signals/options.py."""
        if not hasattr(self, "_options_engine_cached") or self._options_engine_cached is None:
            self._options_engine_cached = OptionsSignalEngine(
                supabase=self.supabase,
                market_data_provider=get_market_data_provider(),
                fo_engine=FOTradingEngine,
                catalog_cache=self._catalog_cache(),
            )
        return self._options_engine_cached

    async def generate_options_signals(self, save: bool = True) -> List[GeneratedSignal]:
        """Wrapper around ai/signals/options.OptionsSignalEngine.generate_signals."""
        return await self._options_engine.generate_signals(save)

    async def monitor_options_positions(self) -> None:
        """Wrapper around ai/signals/options.OptionsSignalEngine.monitor_positions."""
        return await self._options_engine.monitor_positions()

    @staticmethod
    def _strip_exchange_suffix(symbol: str) -> str:
        """Normalize symbols to NSE ticker without suffix."""
        if symbol.endswith(".NS"):
            return symbol[:-3]
        return symbol
