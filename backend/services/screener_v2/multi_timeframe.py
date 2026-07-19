"""Multi-timeframe (MTF) agreement scanner (PR-S12).

The strongest setups are when MOMENTUM agrees across timeframes:
  * Daily   : trend + breakout context
  * 1-hour  : intermediate momentum
  * 15-min  : intraday confirmation

A stock that's bullish on all three is way higher conviction than one
that's bullish on daily alone (which could be a 2 PM reversal trap).

Pipeline:
  1. For each symbol, pull bars at multiple timeframes
  2. Compute simple bullish/bearish vote at each timeframe
      vote = +1 if (close > EMA21 AND RSI > 50 AND volume_ratio > 1.0)
             -1 if (close < EMA21 AND RSI < 50)
              0 otherwise
  3. Surface symbols where ALL requested timeframes vote the same direction

Returns ranked list of {symbol, direction, votes_by_tf, agreement_score}.

LOCKED: pure rule-based. No new ML model — uses indicators we already
compute. AI enrichment available via the existing /v2/explain endpoint.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# Timeframes the scanner can request — yfinance compatible
SUPPORTED_TIMEFRAMES = ("15m", "1h", "1d")


@dataclass
class TFVote:
    """One timeframe's vote for a symbol."""
    timeframe: str
    direction: str               # "bullish" | "bearish" | "neutral"
    rsi: float
    close: float
    ema21: float
    volume_ratio: float
    note: str = ""


@dataclass
class MTFMatch:
    """A multi-timeframe agreement hit."""
    symbol: str
    sector: Optional[str]
    direction: str               # the common direction across all requested tfs
    agreement_count: int         # how many tfs voted that direction
    total_timeframes: int
    last_price: float
    change_pct: float
    votes: List[Dict[str, Any]] = field(default_factory=list)
    composite_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_vote(bars: pd.DataFrame, tf: str) -> Optional[TFVote]:
    """Compute one timeframe's vote from OHLCV bars."""
    if bars is None or len(bars) < 25:
        return None

    closes = bars["close"]
    volumes = bars["volume"]
    last = bars.iloc[-1]
    close = float(last["close"])

    # 21-bar EMA
    ema21 = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])

    # RSI(14) — quick computation
    delta = closes.diff().dropna().tail(14)
    gain = float(delta[delta > 0].sum() / 14)
    loss = float(-delta[delta < 0].sum() / 14)
    if loss == 0:
        rsi = 100.0
    else:
        rsi = 100.0 - (100.0 / (1 + gain / loss))

    # Volume vs 20-bar SMA
    vol_sma = float(volumes.tail(20).mean()) if len(volumes) >= 20 else float(volumes.mean())
    vol_ratio = float(last["volume"]) / vol_sma if vol_sma > 0 else 1.0

    if close > ema21 and rsi > 50 and vol_ratio > 0.8:
        direction = "bullish"
        note = f"Close > EMA21, RSI {rsi:.0f}, Vol {vol_ratio:.1f}×"
    elif close < ema21 and rsi < 50:
        direction = "bearish"
        note = f"Close < EMA21, RSI {rsi:.0f}"
    else:
        direction = "neutral"
        note = f"RSI {rsi:.0f}, mixed"

    return TFVote(
        timeframe=tf, direction=direction,
        rsi=round(rsi, 1), close=round(close, 2),
        ema21=round(ema21, 2), volume_ratio=round(vol_ratio, 2),
        note=note,
    )


def _agreement_match(votes: List[TFVote]) -> Optional[str]:
    """Find the common direction (if any) across all valid votes."""
    if not votes:
        return None
    dirs = [v.direction for v in votes]
    if all(d == "bullish" for d in dirs):
        return "bullish"
    if all(d == "bearish" for d in dirs):
        return "bearish"
    return None


def scan_multi_timeframe(
    symbols: Sequence[str],
    *,
    timeframes: Sequence[str] = ("15m", "1h", "1d"),
    direction: Optional[str] = None,
    max_workers: int = 6,
    stock_info: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[MTFMatch]:
    """Scan symbols for multi-timeframe agreement.

    `direction`: if "bullish" or "bearish", only return matches in that
    direction. None = both.
    """
    from backend.data.market import get_market_data_provider

    bad_tfs = [t for t in timeframes if t not in SUPPORTED_TIMEFRAMES]
    if bad_tfs:
        raise ValueError(f"unsupported timeframes: {bad_tfs}")

    stock_info = stock_info or {}
    mp = get_market_data_provider()

    def _per_symbol(sym: str) -> Optional[MTFMatch]:
        votes: List[TFVote] = []
        last_close = 0.0
        change_pct = 0.0
        try:
            for tf in timeframes:
                # Period scales with timeframe — yfinance limits:
                #   15m → 60d max
                #   1h  → 730d
                #   1d  → max
                period = {"15m": "30d", "1h": "60d", "1d": "6mo"}[tf]
                df = mp.get_historical(sym, period=period, interval=tf)
                if df is None or df.empty:
                    return None
                df = df.copy()
                df.columns = [c.lower() for c in df.columns]
                v = _compute_vote(df, tf)
                if v is None:
                    return None
                votes.append(v)
                if tf == "1d":
                    last_close = v.close
                    closes = df["close"]
                    if len(closes) >= 2:
                        prev = float(closes.iloc[-2])
                        change_pct = (last_close / prev - 1) * 100 if prev > 0 else 0
        except Exception as e:
            logger.debug("mtf %s failed: %s", sym, e)
            return None

        agreed = _agreement_match(votes)
        if not agreed:
            return None

        info = stock_info.get(sym, {})
        composite = sum(1.0 for v in votes) + abs(change_pct) * 0.05
        return MTFMatch(
            symbol=sym,
            sector=info.get("sector"),
            direction=agreed,
            agreement_count=len(votes),
            total_timeframes=len(timeframes),
            last_price=last_close,
            change_pct=round(change_pct, 2),
            votes=[asdict(v) for v in votes],
            composite_score=round(composite, 3),
        )

    matches: List[MTFMatch] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_per_symbol, s): s for s in symbols}
        for fut in as_completed(futs):
            m = fut.result()
            if m is None:
                continue
            if direction and m.direction != direction:
                continue
            matches.append(m)

    matches.sort(key=lambda m: m.composite_score, reverse=True)
    return matches
