"""Deep-dive explanation for a scanner hit (PR-S3).

When a user clicks a scanner result, this composes:
  * `why_matched` — the indicator values + thresholds that fired
  * `regime_context` — current regime, sentiment, breadth + how the
                       pattern type historically performs in this regime
  * `similar_setups` — k-NN over historical pattern feature vectors,
                       with forward-return distribution stats
  * `ai_thesis` — a 2-3 sentence paragraph from the LLM (via the
                  OpenRouter gateway) that ties it all together
                  (factual, no recommendation language)
  * `suggested_levels` — entry / stop / target derived from ATR +
                         nearest pivot supports/resistances

Honest stats > made-up confidence. The thesis is rendered without
"BUY"/"SELL" verbs because the AutoPilot/strategy stack owns the
actual trade decision (locked memory: LLMs don't gate trades).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.features.patterns import (
    BreakoutSignal,
    PatternResult,
    scan_all_patterns,
)

from ...ai.agents.response_cache import cache_get, cache_set, seconds_to_ist_eod

logger = logging.getLogger(__name__)


# Number of comparable historical setups to fetch for the k-NN context.
_HISTORICAL_SAMPLE_SIZE = 50


@dataclass
class IndicatorValue:
    """One indicator value at detection bar — name, value, threshold rule
    that fired, status."""
    name: str
    value: float
    threshold: Optional[float] = None
    operator: Optional[str] = None     # ">", "<", ">=", etc.
    fired: bool = True
    note: Optional[str] = None


@dataclass
class SuggestedLevels:
    """Entry / stop / target with how each was derived."""
    entry: float
    stop: float
    stop_basis: str                    # "atr_2x" | "pattern_low" | "swing_low"
    target1: float
    target1_basis: str                 # "pattern_height" | "next_resistance" | "rr_2x"
    target2: Optional[float] = None
    risk_reward: float = 0.0


@dataclass
class PatternExplanation:
    """The full deep-dive payload for one scanner hit."""
    symbol: str
    pattern_type: str
    last_price: float
    detected_at: str

    # Rule-engine numbers
    quality_score: float
    pattern_height_pct: float
    duration_bars: int
    candle_confirmed_touches: int

    # ML scoring
    ml_score: float
    composite_score: float

    # Context
    regime: Optional[str]
    why_matched: List[Dict[str, Any]]         # IndicatorValue dicts
    suggested: Dict[str, Any]                 # SuggestedLevels dict

    # AI narration + historical context
    ai_thesis: Optional[str] = None
    similar_setups: List[Dict[str, Any]] = field(default_factory=list)
    historical_winrate_pct: Optional[float] = None   # rolling 90d realised, NOT marketing claim

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Why-matched extraction ──────────────────────────────────────────


def _why_matched(
    bars: pd.DataFrame, pat: PatternResult, sig: BreakoutSignal,
) -> List[IndicatorValue]:
    """Extract the indicator values that drove the detection."""
    out: List[IndicatorValue] = []
    last = bars.iloc[-1]
    close = float(last["close"])
    float(last["high"])
    volume = float(last["volume"])
    vol_sma = float(bars["volume"].iloc[-20:].mean())

    # 1. Breakout level vs current price
    out.append(IndicatorValue(
        name="Close vs breakout level",
        value=round(close, 2),
        threshold=round(float(pat.breakout_level), 2),
        operator=">",
        fired=bool(close > pat.breakout_level),
        note=f"Close {'broke above' if close > pat.breakout_level else 'still below'} resistance",
    ))

    # 2. Volume confirmation
    vol_ratio = volume / vol_sma if vol_sma > 0 else 0
    out.append(IndicatorValue(
        name="Volume / SMA20",
        value=round(vol_ratio, 2),
        threshold=1.2,
        operator=">=",
        fired=bool(vol_ratio >= 1.2),
        note=f"{'Confirmed' if vol_ratio >= 1.2 else 'Thin'} volume on breakout",
    ))

    # 3. Pattern quality (engine emits 0–100; normalise to 0–1 for the UI)
    quality_norm = (
        pat.quality_score / 100.0 if pat.quality_score > 1 else pat.quality_score
    )
    out.append(IndicatorValue(
        name="Pattern quality score",
        value=round(quality_norm, 3),
        threshold=0.50,
        operator=">=",
        fired=bool(quality_norm >= 0.50),
        note=f"{pat.candle_confirmed_touches}/{max(1, pat.duration_bars // 10)} candle-confirmed touches",
    ))

    # 4. RSI (computed inline so we don't need ta dep here)
    rsi14 = _compute_rsi(bars["close"].values, 14)
    if not np.isnan(rsi14):
        out.append(IndicatorValue(
            name="RSI(14)",
            value=round(float(rsi14), 1),
            threshold=70.0 if rsi14 < 70 else 30.0,
            operator="<" if rsi14 < 70 else ">",
            fired=bool(20 < rsi14 < 80),
            note=(
                "Neutral momentum" if 40 <= rsi14 <= 60
                else "Bullish momentum" if rsi14 > 60
                else "Oversold bounce risk" if rsi14 < 30
                else "Overbought risk" if rsi14 > 70
                else "Building momentum"
            ),
        ))

    # 5. Pattern height as % of current price (gives a sense of trade size)
    if pat.support_level > 0:
        height_pct = pat.pattern_height / pat.support_level * 100
        out.append(IndicatorValue(
            name="Pattern height",
            value=round(height_pct, 2),
            note=f"{round(height_pct, 1)}% range — sizing reference",
        ))

    return out


def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Simple RSI — last value only. Returns NaN if not enough data."""
    if len(closes) < period + 1:
        return float("nan")
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ── Suggested levels (entry/stop/target) ────────────────────────────


def _suggest_levels(
    bars: pd.DataFrame, pat: PatternResult, sig: BreakoutSignal,
) -> SuggestedLevels:
    """Use BreakoutSignal levels + augment with ATR-based stop fallback."""
    closes = bars["close"].values
    highs = bars["high"].values
    lows = bars["low"].values
    n = len(bars)

    # 14-bar ATR (Wilder's smoothing — approximation)
    if n >= 15:
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        atr = float(np.mean(tr[-14:]))
    else:
        atr = max(0.01, float(np.std(closes[-min(n, 14):])))

    float(closes[-1])

    # Entry — at the breakout level (or slightly above)
    entry = float(sig.entry_price)

    # Stop — pattern_low takes priority; ATR fallback if too tight
    stop = float(sig.stop_loss)
    stop_basis = "pattern_low"
    atr_stop = entry - 2 * atr
    if abs(entry - stop) / entry < 0.005:    # < 0.5% — too tight, expand
        stop = atr_stop
        stop_basis = "atr_2x"

    # Targets — first from pattern_height projection, second 1.5× that
    target1 = float(sig.target)
    target1_basis = "pattern_height"
    target2 = entry + 1.5 * (target1 - entry) if target1 > entry else None

    risk = abs(entry - stop)
    reward = abs(target1 - entry)
    rr = round(reward / risk, 2) if risk > 1e-6 else 0.0

    return SuggestedLevels(
        entry=round(entry, 2),
        stop=round(stop, 2),
        stop_basis=stop_basis,
        target1=round(target1, 2),
        target1_basis=target1_basis,
        target2=round(target2, 2) if target2 else None,
        risk_reward=rr,
    )


# ── AI thesis (LLM via OpenRouter gateway) ──────────────────────────


def _llm_thesis(
    symbol: str, pat: PatternResult, sig: BreakoutSignal,
    levels: SuggestedLevels, regime: Optional[str], ml_score: float,
) -> Optional[str]:
    """Compose a 2-3 sentence factual paragraph via the LLM (OpenRouter
    gateway). Returns None on any failure — the UI falls back to a
    deterministic summary.

    Locked: never includes BUY/SELL recommendation language. The trade
    decision belongs to AutoPilot / the user's strategy, not the LLM.
    """
    try:
        from ...ai.agents.llm import complete_sync

        # Persistent per-(symbol, pattern, day) cache — locked cost control
        # for public scanner endpoints is in-service caching, not caps.
        cache_key = f"scanthesis:{symbol}:{pat.pattern_type}:{date.today().isoformat()}"
        hit = cache_get(cache_key)
        if hit and hit.get("thesis"):
            return hit["thesis"]

        prompt = (
            f"Write 2-3 sentences (max 50 words) explaining this chart pattern "
            f"to an experienced Indian trader. Be factual. Reference the regime + "
            f"the pattern's typical bias. Do not say 'buy' or 'sell'. Do not "
            f"recommend an action — just describe what the chart shows.\n\n"
            f"Symbol: {symbol}\n"
            f"Pattern: {pat.pattern_type.replace('_', ' ')}\n"
            f"Quality score: {pat.quality_score:.2f}/1.0\n"
            f"ML probability: {ml_score:.2f}\n"
            f"Current regime: {regime or 'unknown'}\n"
            f"Entry {levels.entry}, stop {levels.stop}, target {levels.target1} "
            f"(R:R {levels.risk_reward}:1)"
        )
        text = complete_sync(prompt, role="scanner_thesis", feature="chart_patterns_explain", temperature=0.2)
        if not text:
            return None
        # Sanity-cap length
        if len(text) > 500:
            text = text[:500].rsplit(".", 1)[0] + "."
        if text:   # never cache empty — failures must retry
            cache_set(cache_key, {"thesis": text}, ttl_seconds=seconds_to_ist_eod(),
                      surface="scanner_thesis", model="")
        return text or None
    except Exception as e:
        logger.debug("thesis failed for %s: %s", symbol, e)
        return None


def _deterministic_thesis(
    symbol: str, pat: PatternResult, levels: SuggestedLevels,
    regime: Optional[str], ml_score: float,
) -> str:
    """Fallback thesis composed from pattern + level facts. No LLM."""
    pattern_label = pat.pattern_type.replace("_", " ").title()
    regime_part = f" in a {regime} regime" if regime else ""
    quality_norm = (
        pat.quality_score / 100.0 if pat.quality_score > 1 else pat.quality_score
    )
    ml_part = (
        f" Model probability {ml_score:.0%}."
        if ml_score >= 0 else ""
    )
    return (
        f"{symbol} formed a {pattern_label}{regime_part} with quality "
        f"{quality_norm:.0%}. Pattern projects toward {levels.target1} "
        f"with stop at {levels.stop} (R:R {levels.risk_reward}:1).{ml_part}"
    )


# ── Public entry point ──────────────────────────────────────────────


def explain_symbol(
    symbol: str,
    bars: pd.DataFrame,
    *,
    regime: Optional[str] = None,
    use_llm: bool = True,
) -> Optional[PatternExplanation]:
    """Run detection + scoring + explanation for one symbol.

    Returns None if no usable pattern is found.
    """
    if bars is None or len(bars) < 100:
        return None

    try:
        signals = scan_all_patterns(bars, lookback=min(250, len(bars)))
    except Exception as e:
        logger.warning("explain_symbol scan failed for %s: %s", symbol, e)
        return None
    if not signals:
        return None

    # Pick the highest-quality signal (the scanner ranks the same way)
    sig = max(signals, key=lambda s: s.pattern.quality_score)
    pat = sig.pattern

    # ML score
    try:
        from .scanner import _get_labeler
        labeler = _get_labeler()
        ml_score = labeler.score_signal(bars, sig) if labeler else -1.0
    except Exception:
        ml_score = -1.0

    levels = _suggest_levels(bars, pat, sig)
    why = _why_matched(bars, pat, sig)

    thesis: Optional[str] = None
    if use_llm:
        thesis = _llm_thesis(symbol, pat, sig, levels, regime, ml_score)
    if not thesis:
        thesis = _deterministic_thesis(symbol, pat, levels, regime, ml_score)

    last_price = float(bars["close"].iloc[-1])
    detected_at = (
        str(bars.index[-1].date())
        if hasattr(bars.index[-1], "date") else str(bars.index[-1])
    )

    # Composite score same formula as the scanner. Quality is normalised
    # 0–100 → 0–1 because the underlying engine emits the larger range.
    quality_norm = (
        pat.quality_score / 100.0 if pat.quality_score > 1 else pat.quality_score
    )
    vol_sma = float(bars["volume"].iloc[-20:].mean())
    vol_ratio = (
        float(bars["volume"].iloc[-1]) / vol_sma if vol_sma > 0 else 0
    )
    rule_part = 0.45 * quality_norm
    ml_part = 0.40 * max(0.0, ml_score)
    vol_bonus = min(0.15, max(0.0, (vol_ratio - 1.0) / 5.0))
    composite = round(rule_part + ml_part + vol_bonus, 4)

    return PatternExplanation(
        symbol=symbol,
        pattern_type=pat.pattern_type,
        last_price=round(last_price, 2),
        detected_at=detected_at,
        quality_score=round(quality_norm, 4),
        pattern_height_pct=round(
            (pat.pattern_height / pat.support_level * 100) if pat.support_level > 0 else 0.0, 2,
        ),
        duration_bars=pat.duration_bars,
        candle_confirmed_touches=pat.candle_confirmed_touches,
        ml_score=round(ml_score, 4),
        composite_score=composite,
        regime=regime,
        why_matched=[asdict(w) for w in why],
        suggested=asdict(levels),
        ai_thesis=thesis,
        similar_setups=[],            # PR-S3.1 — populate from outcome_models
        historical_winrate_pct=None,  # PR-S3.1 — rolling 90-day realised
    )
