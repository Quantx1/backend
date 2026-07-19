"""Market-level analytics for the screener — Nifty prediction, trend
forecasts, regime + sector breadth.

Wrapped in a class so the engine can inject the HMM detector, LGBM gate,
nifty cache, and the data-fetch callables. Pure compute happens here;
broker I/O lives behind the injected ``fetch_index_df`` and
``get_computed_data`` callables.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
import ta

logger = logging.getLogger(__name__)


class MarketAnalytics:
    """Engine-owned helper that produces market-level analytics.

    Dependencies are injected so this module has no upward imports:

        fetch_index_df:   (td_symbol, yf_symbol, period) -> DataFrame | None
        get_computed_data: () -> (summary_df, per_symbol_dfs)
        regime_detector:  ml.regime_detector.MarketRegimeDetector | None
        lgbm_gate:        services.model_registry.LGBMGate | None
        stock_info:       dict mapping symbol -> {"sector": ...}
        cache_ttl:        timedelta governing nifty-prediction freshness
    """

    def __init__(
        self,
        fetch_index_df: Callable[[str, str, str], Optional[pd.DataFrame]],
        get_computed_data: Callable[[], Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]],
        regime_detector,
        lgbm_gate,
        stock_info: Dict[str, Dict[str, str]],
        cache_ttl: timedelta,
    ):
        self._fetch_index_df = fetch_index_df
        self._get_computed_data = get_computed_data
        self._regime_detector = regime_detector
        self._lgbm_gate = lgbm_gate
        self._stock_info = stock_info
        self._cache_ttl = cache_ttl
        self._nifty_cache: Optional[Tuple[Dict, datetime]] = None

    # ──────────────────────────────────────────────────────────────────
    # Nifty 50 prediction
    # ──────────────────────────────────────────────────────────────────

    async def get_nifty_prediction(self) -> Dict[str, Any]:
        """AI/ML-powered Nifty 50 prediction using HMM regime + LGBM gate."""
        if self._nifty_cache:
            cached_data, cached_at = self._nifty_cache
            if datetime.now() - cached_at < self._cache_ttl:
                return cached_data

        try:
            df = self._fetch_index_df("NIFTY 50", "^NSEI", "1y")
            if df is None or df.empty:
                return {"error": "Could not fetch Nifty data"}

            close = df['close']
            current_level = round(float(close.iloc[-1]), 2)
            prev_close = float(close.iloc[-2]) if len(close) > 1 else current_level
            change_pct = round((current_level - prev_close) / prev_close * 100, 2)

            # Technical indicators
            rsi = ta.momentum.rsi(close, window=14)
            rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            macd_ind = ta.trend.MACD(close)
            macd_val = float(macd_ind.macd().iloc[-1])
            macd_sig = float(macd_ind.macd_signal().iloc[-1])
            sma_20 = float(close.rolling(20).mean().iloc[-1])
            sma_50 = float(close.rolling(50).mean().iloc[-1])
            sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            bb_upper = float(bb.bollinger_hband().iloc[-1])
            bb_lower = float(bb.bollinger_lband().iloc[-1])
            bb_pct = float(bb.bollinger_pband().iloc[-1]) if not pd.isna(bb.bollinger_pband().iloc[-1]) else 0.5
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], close, window=14)
            atr_val = float(atr.average_true_range().iloc[-1]) if not pd.isna(atr.average_true_range().iloc[-1]) else 0
            obv = ta.volume.OnBalanceVolumeIndicator(close, df['volume'])
            obv_val = float(obv.on_balance_volume().iloc[-1])
            vol_ratio = float(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                              ) if df['volume'].rolling(20).mean().iloc[-1] > 0 else 1.0
            ema_20 = float(close.ewm(span=20).mean().iloc[-1])
            ema_50 = float(close.ewm(span=50).mean().iloc[-1])
            body_pct = abs(float(df['close'].iloc[-1] - df['open'].iloc[-1])) / \
                current_level * 100 if current_level > 0 else 0
            wick_pct = (float(df['high'].iloc[-1] - df['low'].iloc[-1]) - abs(float(df['close'].iloc[-1] -
                        df['open'].iloc[-1]))) / current_level * 100 if current_level > 0 else 0

            # HMM regime detection
            regime_result = None
            if self._regime_detector is not None:
                try:
                    from ml.regime_detector import compute_regime_features
                    vix_df = None
                    try:
                        vix_df = self._fetch_index_df("INDIA VIX", "^INDIAVIX", "1y")
                    except Exception:
                        pass
                    regime_features = compute_regime_features(df, vix_df)
                    regime_result = self._regime_detector.predict_regime(regime_features)
                    logger.debug(f"HMM regime: {regime_result['regime']} (conf={regime_result['confidence']:.2f})")
                except Exception as e:
                    logger.debug(f"HMM regime prediction failed: {e}")

            # LGBM signal gate
            lgbm_result = None
            if self._lgbm_gate is not None:
                try:
                    vwap_diff = 0.0
                    if 'volume' in df.columns:
                        typical = (df['high'] + df['low'] + df['close']) / 3
                        cum_vol = df['volume'].cumsum()
                        cum_tp_vol = (typical * df['volume']).cumsum()
                        vwap = float((cum_tp_vol / cum_vol).iloc[-1]) if float(cum_vol.iloc[-1]) > 0 else current_level
                        vwap_diff = (current_level - vwap) / vwap * 100 if vwap > 0 else 0

                    features = {
                        "close": current_level, "rsi_14": rsi_val,
                        "macd": macd_val, "macd_signal": macd_sig,
                        "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_percent": bb_pct,
                        "ema_20": ema_20, "ema_50": ema_50, "atr_14": atr_val,
                        "volume_ratio": vol_ratio, "obv": obv_val,
                        "vwap_diff": vwap_diff, "body_pct": body_pct, "wick_pct": wick_pct,
                    }
                    direction_lgbm, confidence_lgbm, probs = self._lgbm_gate.predict(features)
                    lgbm_result = {
                        "direction": direction_lgbm,
                        "confidence": round(confidence_lgbm, 1),
                        "probabilities": {k: round(v, 1) for k, v in probs.items()},
                    }
                    logger.debug(f"LGBM signal: {direction_lgbm} ({confidence_lgbm:.1f}%)")
                except Exception as e:
                    logger.debug(f"LGBM prediction failed: {e}")

            # Combine models for final prediction
            if lgbm_result:
                direction = lgbm_result["direction"]
                if direction == "BUY":
                    final_direction = "BULLISH"
                elif direction == "SELL":
                    final_direction = "BEARISH"
                else:
                    final_direction = "NEUTRAL"
                confidence = lgbm_result["confidence"]
            elif regime_result and regime_result.get("confidence", 0) > 0.4:
                regime = regime_result["regime"]
                final_direction = "BULLISH" if regime == "bull" else "BEARISH" if regime == "bear" else "NEUTRAL"
                confidence = round(regime_result["confidence"] * 100, 1)
            else:
                bullish_signals = sum([
                    current_level > sma_50,
                    rsi_val > 50,
                    macd_val > macd_sig,
                    current_level > float(close.iloc[-5]) if len(close) > 5 else False,
                ])
                if bullish_signals >= 3:
                    final_direction = "BULLISH"
                    confidence = round(55 + bullish_signals * 5, 1)
                elif bullish_signals <= 1:
                    final_direction = "BEARISH"
                    confidence = round(55 + (4 - bullish_signals) * 5, 1)
                else:
                    final_direction = "NEUTRAL"
                    confidence = 50.0

            # Support/resistance from actual pivots
            try:
                from ml.features.indicators import (
                    compute_all_indicators, detect_support_resistance,
                )
                full_df = compute_all_indicators(df)
                support_levels, resistance_levels = detect_support_resistance(full_df)
            except Exception:
                support_levels = [round(current_level * m, 0) for m in [0.98, 0.96, 0.94]]
                resistance_levels = [round(current_level * m, 0) for m in [1.02, 1.04, 1.06]]

            models_used = []
            if regime_result:
                models_used.append("HMM Regime Detector")
            if lgbm_result:
                models_used.append("LightGBM Signal Gate")
            if not models_used:
                models_used.append("Technical Indicators")

            result = {
                "current_level": current_level,
                "change_percent": change_pct,
                "prediction": {
                    "direction": final_direction,
                    "confidence": confidence,
                    "models_used": models_used,
                },
                "regime": regime_result if regime_result else {
                    "regime": final_direction.lower() if final_direction != "NEUTRAL" else "sideways",
                    "confidence": confidence / 100,
                    "source": "technical_fallback",
                },
                "lgbm_signal": lgbm_result,
                "support_levels": support_levels[:3] if support_levels else [],
                "resistance_levels": resistance_levels[:3] if resistance_levels else [],
                "indicators": {
                    "rsi": round(rsi_val, 1),
                    "macd": round(macd_val, 2),
                    "macd_signal": round(macd_sig, 2),
                    "sma_20": round(sma_20, 2),
                    "sma_50": round(sma_50, 2),
                    "sma_200": round(sma_200, 2),
                    "bb_percent": round(bb_pct, 3),
                    "atr_14": round(atr_val, 2),
                    "volume_ratio": round(vol_ratio, 2),
                },
                "timestamp": datetime.now().isoformat(),
            }

            self._nifty_cache = (result, datetime.now())
            return result

        except Exception as e:
            logger.error(f"Nifty prediction error: {e}")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────────────
    # Trend forecast for an arbitrary symbol
    # ──────────────────────────────────────────────────────────────────

    async def get_trend_forecast(self, symbol: str = "NIFTY") -> Dict[str, Any]:
        """Real multi-timeframe trend forecast using Kite data."""
        try:
            sym_upper = symbol.upper()
            td_symbol = "NIFTY 50" if sym_upper in ("NIFTY", "NIFTY50") else sym_upper
            yf_symbol = "^NSEI" if sym_upper in ("NIFTY", "NIFTY50") else f"{sym_upper}.NS"

            df = self._fetch_index_df(td_symbol, yf_symbol, "1y")
            if df is None or df.empty:
                return {"error": f"No data for {symbol}"}

            close = df['close']
            current_price = float(close.iloc[-1])

            rsi = ta.momentum.rsi(close, window=14)
            rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            macd_ind = ta.trend.MACD(close)
            macd_val = float(macd_ind.macd().iloc[-1])
            macd_sig = float(macd_ind.macd_signal().iloc[-1])
            sma_20 = float(close.rolling(20).mean().iloc[-1])
            sma_50 = float(close.rolling(50).mean().iloc[-1])

            recent_5d = float(close.iloc[-5]) if len(close) > 5 else current_price
            intraday_trend = (
                "Bullish" if current_price > recent_5d
                else "Bearish" if current_price < recent_5d else "Sideways"
            )
            short_trend = (
                "Bullish" if current_price > sma_20 and rsi_val > 50
                else "Bearish" if current_price < sma_20 else "Sideways"
            )
            medium_trend = (
                "Bullish" if current_price > sma_50 and macd_val > macd_sig
                else "Bearish" if current_price < sma_50 else "Sideways"
            )
            atr = float((df['high'] - df['low']).rolling(14).mean().iloc[-1])

            return {
                "symbol": symbol.upper(),
                "current_price": round(current_price, 2),
                "timestamp": datetime.now().isoformat(),
                "timeframes": {
                    "intraday": {
                        "trend": intraday_trend,
                        "strength": round(min(abs(current_price - recent_5d) / current_price * 100, 1.0), 2),
                        "target": round(current_price + atr, 2),
                        "stop_loss": round(current_price - atr, 2),
                    },
                    "short_term": {
                        "trend": short_trend,
                        "strength": round(min(abs(rsi_val - 50) / 50, 1.0), 2),
                        "target": round(current_price + atr * 2, 2),
                        "stop_loss": round(current_price - atr * 1.5, 2),
                        "duration": "1-2 weeks",
                    },
                    "medium_term": {
                        "trend": medium_trend,
                        "strength": round(min(abs(current_price - sma_50) / sma_50, 1.0), 2) if sma_50 > 0 else 0,
                        "target": round(current_price + atr * 4, 2),
                        "stop_loss": round(current_price - atr * 3, 2),
                        "duration": "1-3 months",
                    },
                },
                "technical_indicators": {
                    "rsi_14": round(rsi_val, 1),
                    "macd_signal": "Bullish" if macd_val > macd_sig else "Bearish",
                    "adx": round(float(ta.trend.ADXIndicator(df['high'], df['low'], close).adx().iloc[-1]), 1),
                },
            }
        except Exception as e:
            logger.error(f"Trend forecast error for {symbol}: {e}")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────────────
    # Breadth-based regime + sector analysis
    # ──────────────────────────────────────────────────────────────────

    async def get_market_regime(self) -> Dict[str, Any]:
        """Real market regime from breadth analysis."""
        summary_df, _ = self._get_computed_data()
        if summary_df.empty:
            return {"regime": "UNKNOWN", "error": "No data available"}

        total = len(summary_df)
        above_200sma = int((summary_df['close'] > summary_df['sma_200']).sum())
        above_50sma = int((summary_df['close'] > summary_df['sma_50']).sum())
        bullish_macd = int((summary_df['macd'] > summary_df['macd_signal']).sum())

        breadth_200 = above_200sma / total * 100 if total > 0 else 50
        breadth_50 = above_50sma / total * 100 if total > 0 else 50

        if breadth_200 > 60:
            regime = "BULL"
            desc = "Broad market uptrend with expanding breadth"
        elif breadth_200 < 40:
            regime = "BEAR"
            desc = "Market in downtrend with contracting breadth"
        else:
            regime = "SIDEWAYS"
            desc = "Range-bound market with mixed signals"

        return {
            "regime": regime,
            "description": desc,
            "confidence": round(abs(breadth_200 - 50) + 50, 1),
            "breadth_200sma": round(breadth_200, 1),
            "breadth_50sma": round(breadth_50, 1),
            "bullish_macd_pct": round(bullish_macd / total * 100 if total > 0 else 50, 1),
            "stocks_analyzed": total,
            "timestamp": datetime.now().isoformat(),
        }

    async def get_trend_analysis(self) -> Dict[str, Any]:
        """Real sector-wise trend analysis."""
        summary_df, _ = self._get_computed_data()
        if summary_df.empty:
            return {"error": "No data available"}

        summary_df = summary_df.copy()
        summary_df['sector'] = summary_df['symbol'].map(
            lambda s: self._stock_info.get(s, {}).get('sector', 'Other')
        )

        total = len(summary_df)
        above_50 = int((summary_df['close'] > summary_df['sma_50']).sum())
        below_50 = total - above_50
        bull_pct = round(above_50 / total * 100) if total > 0 else 50

        sectors = {}
        for sector, group in summary_df.groupby('sector'):
            if len(group) < 3:
                continue
            sector_bullish = (group['close'] > group['sma_50']).sum()
            sector_total = len(group)
            pct = sector_bullish / sector_total * 100 if sector_total > 0 else 50
            sectors[sector] = {
                "trend": "BULLISH" if pct > 60 else "BEARISH" if pct < 40 else "NEUTRAL",
                "strength": round(pct),
                "stocks": sector_total,
            }

        return {
            "summary": {
                "bullish_stocks": above_50,
                "bearish_stocks": below_50,
                "bullish_pct": bull_pct,
                "bearish_pct": 100 - bull_pct,
                "overall_trend": "BULLISH" if bull_pct > 60 else "BEARISH" if bull_pct < 40 else "NEUTRAL",
            },
            "sectors": sectors,
            "stocks_analyzed": total,
            "timestamp": datetime.now().isoformat(),
        }
