"""Pattern-scanner v2 — gates our chart pattern algo with ML + regime + volume.

Pipeline per symbol:
    1. Load OHLCV (daily, ~1 year)
    2. Run ml.features.patterns.scan_for_patterns()
    3. For each detected PatternResult:
        a. Drop if quality_score < MIN_QUALITY
        b. Score via BreakoutMetaLabeler.score_signal() → ml_score
        c. Drop if ml_score < MIN_ML_THRESHOLD
        d. Drop if volume_ratio on detection bar < MIN_VOLUME_RATIO
        e. Drop if pattern direction conflicts with current regime
    4. Return survivors with derived entry/stop/target + honest stats

Trade-offs encoded here:
    * High recall over high precision — let traders see ranked options.
    * ML threshold deliberately permissive (0.55) because the meta-labeler
      was trained on noisy walk-forward labels; tighter cuts (0.7+) drop
      to ~zero hits on a typical session.
    * Volume gate hard-coded at 1.2× SMA20 — empirically the line above
      which most "breakouts" are actually breakouts vs. random drift.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ml.features.patterns import (
    BreakoutMetaLabeler,
    BreakoutSignal,
    PatternResult,
    scan_all_patterns,
)


# ── Universe helpers ────────────────────────────────────────────────


def full_nse_universe() -> List[str]:
    """All ~2,136 NSE EQ symbols from data/nse_all_symbols.json.

    Falls back to load_universe('nse_all') if the JSON is unavailable
    (e.g. fresh install before the cache is regenerated).
    """
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[3] / "data" / "nse_all_symbols.json"
    try:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            syms = data.get("symbols") or []
            if syms:
                return [s.strip().upper() for s in syms if s]
    except Exception as e:
        logger.warning("full_nse_universe JSON read failed: %s", e)
    # Fallback
    try:
        from backend.ai.qlib.data_handler import load_universe
        return load_universe("nse_all")
    except Exception:
        return []


def filter_by_sector(symbols: List[str], sectors: List[str]) -> List[str]:
    """Keep only symbols whose canonical sector is in `sectors`.

    Symbols without sector metadata are dropped — explicit sector filter
    means the user wants only those tagged stocks, not unknowns.
    """
    if not sectors:
        return symbols
    try:
        from backend.ai.sector_taxonomy import sector_for_symbol
    except Exception:
        return symbols
    wanted = {s.strip() for s in sectors if s}
    return [s for s in symbols if sector_for_symbol(s) in wanted]


logger = logging.getLogger(__name__)


# ── Gate thresholds (tunable; tighter = fewer + higher-quality hits) ─
#
# Note: pattern_engine quality_score is 0–100 (NOT 0–1) per
# ml/features/patterns.py — we normalise to 0–1 internally below so the
# composite formula stays clean.

MIN_QUALITY = 0.50              # normalised quality floor (0–1 after /100)
MIN_ML_THRESHOLD = 0.35         # BreakoutMetaLabeler floor — model RF is
# noisy, scores cluster near 0.3–0.6; tighter
# cuts (0.55+) leave almost no hits.
MIN_VOLUME_RATIO = 1.0          # detection bar volume vs SMA20 (1.0 = average)
MIN_BARS_REQUIRED = 100         # min bars for pattern detection


# ── Pattern direction map ───────────────────────────────────────────
# Conservative classification: only block obvious mismatches
# (bullish patterns in confirmed bear, vice versa).

_BULLISH_PATTERNS = {
    "ascending_triangle", "cup_handle", "bull_flag", "bullish_flag",
    "double_bottom", "inverse_head_shoulders", "rounding_bottom",
    "bullish_engulfing", "morning_star", "hammer", "three_white_soldiers",
    "vcp", "bullish_pennant",
}

_BEARISH_PATTERNS = {
    "descending_triangle", "head_shoulders", "bear_flag", "bearish_flag",
    "double_top", "rounding_top", "bearish_engulfing", "evening_star",
    "shooting_star", "three_black_crows", "bearish_pennant",
}


def _is_aligned_with_regime(pattern_type: str, regime: Optional[str]) -> bool:
    """True if the pattern's bias matches the current regime, or regime
    is unknown / sideways (in which case we don't filter)."""
    if regime is None:
        return True
    regime = regime.lower()
    if regime in ("sideways", "neutral", "transition"):
        return True
    pt = pattern_type.lower()
    if regime == "bull" and pt in _BEARISH_PATTERNS:
        return False
    if regime == "bear" and pt in _BULLISH_PATTERNS:
        return False
    return True


# ── ML model singleton ──────────────────────────────────────────────

_LABELER_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "models" / "breakout_meta_labeler.pkl"
_labeler: Optional[BreakoutMetaLabeler] = None


def _get_labeler() -> Optional[BreakoutMetaLabeler]:
    """Lazy-load the BreakoutMetaLabeler. Returns None if unavailable —
    callers should fall back to rule-only scoring."""
    global _labeler
    if _labeler is not None:
        return _labeler
    try:
        lab = BreakoutMetaLabeler()
        lab.load(str(_LABELER_PATH))
        if lab.is_trained:
            _labeler = lab
            logger.info("BreakoutMetaLabeler loaded from %s", _LABELER_PATH)
            return _labeler
        logger.warning("BreakoutMetaLabeler file present but not trained")
    except Exception as e:
        logger.warning("BreakoutMetaLabeler load failed: %s", e)
    return None


# ── Output shape ────────────────────────────────────────────────────


@dataclass
class PatternMatch:
    """One scanner hit — what the UI renders per row."""
    symbol: str
    pattern_type: str
    direction: str                          # "bullish" | "bearish" | "neutral"
    quality_score: float                    # 0–1, from the rule engine
    ml_score: float                         # 0–1, from BreakoutMetaLabeler (-1 if unavailable)
    composite_score: float                  # weighted blend, used for ranking

    # Trade levels (derived from BreakoutSignal)
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float

    # Detection context
    detected_at: str                        # ISO date of detection bar
    last_price: float
    volume_ratio: float                     # current vs SMA20
    regime_at_detection: Optional[str]

    # For the deep-dive view
    pattern_height_pct: float
    duration_bars: int
    candle_confirmed_touches: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _direction_of(pattern_type: str) -> str:
    pt = pattern_type.lower()
    if pt in _BULLISH_PATTERNS:
        return "bullish"
    if pt in _BEARISH_PATTERNS:
        return "bearish"
    return "neutral"


def _to_match(
    symbol: str,
    pat: PatternResult,
    sig: BreakoutSignal,
    bars: pd.DataFrame,
    *,
    ml_score: float,
    regime: Optional[str],
) -> PatternMatch:
    last_price = float(bars["close"].iloc[-1])
    last_volume = float(bars["volume"].iloc[-1])
    vol_sma = float(bars["volume"].iloc[-20:].mean())
    vol_ratio = last_volume / vol_sma if vol_sma > 0 else 0.0
    risk = abs(sig.entry_price - sig.stop_loss)
    reward = abs(sig.target - sig.entry_price)
    rr = reward / risk if risk > 1e-6 else 0.0
    # Normalise quality to 0–1 for the composite + the wire payload so the UI
    # gets consistent percentages everywhere.
    quality_norm = pat.quality_score / 100.0 if pat.quality_score > 1 else pat.quality_score
    composite = _composite_score(quality_norm, ml_score, vol_ratio)

    return PatternMatch(
        symbol=symbol,
        pattern_type=pat.pattern_type,
        direction=_direction_of(pat.pattern_type),
        quality_score=round(quality_norm, 4),
        ml_score=round(ml_score, 4),
        composite_score=round(composite, 4),
        entry_price=round(sig.entry_price, 2),
        stop_loss=round(sig.stop_loss, 2),
        take_profit=round(sig.target, 2),
        risk_reward=round(rr, 2),
        detected_at=str(bars.index[-1].date()) if hasattr(bars.index[-1], "date") else str(bars.index[-1]),
        last_price=round(last_price, 2),
        volume_ratio=round(vol_ratio, 2),
        regime_at_detection=regime,
        pattern_height_pct=round(
            (pat.pattern_height / pat.support_level * 100) if pat.support_level > 0 else 0.0, 2,
        ),
        duration_bars=pat.duration_bars,
        candle_confirmed_touches=pat.candle_confirmed_touches,
    )


def _composite_score(quality: float, ml_score: float, vol_ratio: float) -> float:
    """Blend rule + ML + volume into a single comparable number.

    Weights chosen so a clean rule-detection alone (quality=0.9, no ML)
    still scores well, but ML agreement (>=0.7) bumps it past a noisier
    detection. Volume gate is a soft bonus, not a multiplier."""
    rule_part = 0.45 * quality
    ml_part = 0.40 * max(0.0, ml_score)   # treat -1 (unavailable) as 0
    vol_bonus = min(0.15, (vol_ratio - 1.0) / 5.0)  # +0.15 max for vol_ratio>=1.75
    return rule_part + ml_part + max(0.0, vol_bonus)


# ── Public API ──────────────────────────────────────────────────────


def scan_symbol(
    symbol: str,
    bars: pd.DataFrame,
    *,
    regime: Optional[str] = None,
    min_quality: float = MIN_QUALITY,
    min_ml: float = MIN_ML_THRESHOLD,
    min_volume_ratio: float = MIN_VOLUME_RATIO,
) -> List[PatternMatch]:
    """Run the full v2 pipeline on one symbol's bars.

    `bars` must have lowercase columns open/high/low/close/volume + a
    DatetimeIndex (sorted ascending). Returns matches sorted by
    composite_score descending — empty list if no pattern survives the
    gates.
    """
    if bars is None or len(bars) < MIN_BARS_REQUIRED:
        return []

    # 1. Detect patterns
    try:
        signals = scan_all_patterns(bars, lookback=min(250, len(bars)))
    except Exception as e:
        logger.debug("pattern scan failed for %s: %s", symbol, e)
        return []
    if not signals:
        return []

    # 2. ML scorer (may be None on a fresh install)
    labeler = _get_labeler()

    # 3. Gate each signal
    matches: List[PatternMatch] = []
    seen_types = set()
    for sig in signals:
        pat = sig.pattern
        # Normalise quality_score from 0–100 → 0–1 for consistent gating
        quality_norm = pat.quality_score / 100.0 if pat.quality_score > 1 else pat.quality_score
        if quality_norm < min_quality:
            continue

        # ML scoring — -1 means model unavailable, allow through
        ml_score = labeler.score_signal(bars, sig) if labeler else -1.0
        if ml_score >= 0 and ml_score < min_ml:
            continue

        # Volume gate — detection bar must have above-average volume
        try:
            last_vol = float(bars["volume"].iloc[-1])
            vol_sma = float(bars["volume"].iloc[-20:].mean())
            if vol_sma > 0 and last_vol / vol_sma < min_volume_ratio:
                continue
        except Exception:
            pass

        # Regime gate
        if not _is_aligned_with_regime(pat.pattern_type, regime):
            continue

        # Dedup by type — keep the best per pattern_type per symbol
        if pat.pattern_type in seen_types:
            continue
        seen_types.add(pat.pattern_type)

        matches.append(_to_match(
            symbol, pat, sig, bars,
            ml_score=ml_score, regime=regime,
        ))

    matches.sort(key=lambda m: m.composite_score, reverse=True)
    return matches


async def scan_universe_streaming(
    symbols: Sequence[str],
    *,
    bars_fetcher,
    regime: Optional[str] = None,
    max_workers: int = 8,
    batch_size: int = 12,
):
    """Async generator — yields per-symbol PatternMatch lists as they
    complete. Use for SSE so the UI gets live progress instead of waiting
    for the whole 2,136-symbol scan.

    Each yield is a tuple:
        (kind, payload)
            "match"   → (sym, [PatternMatch, …])  one symbol's hits
            "progress"→ {processed, total}        every batch
            "done"    → None                       end-of-stream sentinel

    Errors are isolated per symbol (logged at DEBUG, swallowed). The
    scanner does NOT raise mid-stream — the caller can assume the
    generator runs to completion unless cancelled.
    """
    import asyncio

    if not symbols:
        yield ("done", None)
        return

    total = len(symbols)
    processed = 0
    loop = asyncio.get_event_loop()

    def _one(sym: str):
        try:
            bars = bars_fetcher(sym)
            if bars is None or bars.empty:
                return sym, []
            return sym, scan_symbol(sym, bars, regime=regime)
        except Exception as e:
            logger.debug("scan_universe_streaming %s failed: %s", sym, e)
            return sym, []

    # Slice into batches; each batch fans out via thread pool, awaits all
    # results, yields them to the caller, then proceeds. Keeps the SSE
    # stream incremental without unbounded concurrency.
    for i in range(0, total, batch_size):
        chunk = list(symbols[i: i + batch_size])
        tasks = [loop.run_in_executor(None, _one, s) for s in chunk]
        for fut in asyncio.as_completed(tasks):
            sym, matches = await fut
            if matches:
                yield ("match", (sym, matches))
        processed += len(chunk)
        yield ("progress", {"processed": processed, "total": total})

    yield ("done", None)


def scan_universe(
    symbols: Sequence[str],
    *,
    bars_fetcher,
    regime: Optional[str] = None,
    max_workers: int = 6,
    limit: int = 50,
) -> List[PatternMatch]:
    """Scan a universe of symbols in parallel.

    `bars_fetcher(symbol) -> DataFrame | None` — caller provides the data
    accessor so the scanner stays decoupled from the (lazy-imported)
    MarketDataProvider. Failures per-symbol are swallowed; only the global
    pool size is bounded.

    Returns the top `limit` matches across all symbols, sorted by
    composite_score descending.
    """
    if not symbols:
        return []

    all_matches: List[PatternMatch] = []

    def _one(sym: str) -> List[PatternMatch]:
        try:
            bars = bars_fetcher(sym)
            if bars is None or bars.empty:
                return []
            return scan_symbol(sym, bars, regime=regime)
        except Exception as e:
            logger.debug("scan_universe %s failed: %s", sym, e)
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, s): s for s in symbols}
        for fut in as_completed(futures):
            try:
                all_matches.extend(fut.result())
            except Exception as e:
                logger.debug("scan_universe future failed for %s: %s",
                             futures[fut], e)

    all_matches.sort(key=lambda m: m.composite_score, reverse=True)
    return all_matches[:limit]
