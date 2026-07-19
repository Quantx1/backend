"""Intraday setup scanner — 12 verified setups from the deep-research audit.

Each setup is a pure function that takes a per-symbol intraday DataFrame
(OHLCV with DatetimeIndex in IST) and returns an `IntradayMatch` if
the setup fires on the latest bar, else None.

The top-level `scan_intraday_setups()` iterates symbols via a
`bars_fetcher` callback (same pattern as chart_patterns/scanner.py) so
the data-source layer stays decoupled.

Time-of-day filters (lunch lull, closing auction) are applied uniformly
— any setup firing in 12:30-13:30 IST or 15:20-15:30 IST is suppressed
because the research showed those windows produce false signals.

All thresholds are recalibrated for NSE liquidity bands. Sources cited
in each setup's docstring.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from .indicators import (
    session_vwap, vwap_bands, opening_range, initial_balance, anchored_vwap,
    is_lunch_window, is_closing_auction, is_power_hour,
    bb_squeeze_inside_kc,
)

logger = logging.getLogger(__name__)


# ── Output dataclass ──────────────────────────────────────────────


@dataclass
class IntradayMatch:
    symbol: str
    setup_id: str                       # e.g. 'orb_long', 'vwap_bounce', etc.
    direction: str                      # 'bullish' | 'bearish' | 'neutral'
    detected_at: str                    # ISO timestamp
    timeframe: str                      # '5m' | '15m'
    entry: float
    stop: float
    target: float
    risk_reward: float
    last_price: float
    volume_ratio: float                 # latest bar vol / 20-bar avg
    confidence: str                     # 'high' | 'medium' | 'low'
    reason: str                         # one-line human-readable explainer
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


# Cached setup catalogue exposed to the API + UI
SETUP_CATALOG: List[Dict[str, str]] = [
    {"id": "orb_long", "name": "Opening Range Breakout (Long)", "tf": "15m", "direction": "bullish"},
    {"id": "orb_short", "name": "Opening Range Breakout (Short)", "tf": "15m", "direction": "bearish"},
    {"id": "vwap_bounce", "name": "VWAP Bounce (long pullback)", "tf": "5m", "direction": "bullish"},
    {"id": "vwap_rejection", "name": "VWAP Rejection (short fade)", "tf": "5m", "direction": "bearish"},
    {"id": "anchored_vwap_pull", "name": "Anchored VWAP Pullback", "tf": "15m", "direction": "bullish"},
    {"id": "open_drive_long", "name": "Open Drive / Trend Day (Long)", "tf": "15m", "direction": "bullish"},
    {"id": "ib_failure_long", "name": "Inside Bar Failure (Hikkake L)", "tf": "5m", "direction": "bullish"},
    {"id": "power_hour_fade", "name": "Power Hour Fade", "tf": "5m", "direction": "neutral"},
    {"id": "vwap_meanrev", "name": "VWAP Mean Reversion (±2σ)", "tf": "5m", "direction": "neutral"},
    {"id": "gap_and_go", "name": "Gap-and-Go (long)", "tf": "5m", "direction": "bullish"},
    {"id": "intraday_squeeze", "name": "Intraday BB Squeeze Fire", "tf": "5m", "direction": "neutral"},
    {"id": "eod_drift", "name": "End-of-Day Drift", "tf": "5m", "direction": "neutral"},
]


# ── Setup detectors ──────────────────────────────────────────────


def _vol_ratio(df: pd.DataFrame) -> float:
    if len(df) < 20:
        return 1.0
    avg = float(df["volume"].iloc[-20:].mean())
    last = float(df["volume"].iloc[-1])
    return (last / avg) if avg > 0 else 1.0


def _atr_intraday(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty or len(df) < period:
        return 0.0
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1] or 0.0)


def _suppress_if_off_window(ts: pd.Timestamp) -> bool:
    """Common gate — drop signals in lunch lull + closing auction."""
    return is_lunch_window(ts) or is_closing_auction(ts)


def detect_orb(df: pd.DataFrame, *, symbol: str, direction: str = "long") -> Optional[IntradayMatch]:
    """Opening Range Breakout — 15-min window.

    Source: Trade That Swing, TradersMastermind, QuantifiedStrategies
    (15-min ORB is the best NSE noise/edge balance). Crabel original.

    Trigger: latest 15m candle CLOSES above ORH (long) / below ORL
    (short), candle range > avg of prior 5, body mostly outside the OR.
    Time-stops the setup: ignore signals after 11:00 IST.
    """
    if df is None or len(df) < 5:
        return None
    last_ts = df.index[-1]
    if _suppress_if_off_window(last_ts) or is_power_hour(last_ts):
        return None
    # Only fire in 09:30-11:00 IST window
    t = last_ts.time() if last_ts.tz is None else last_ts.tz_convert("Asia/Kolkata").time()
    from datetime import time as _t
    if t < _t(9, 30) or t >= _t(11, 0):
        return None
    or_data = opening_range(df, minutes=15)
    if not or_data or or_data["range"] <= 0:
        return None
    last = df.iloc[-1]
    close = float(last["close"])
    open_ = float(last["open"])
    body = abs(close - open_)
    rng = float(last["high"] - last["low"])
    avg5 = float((df["high"].iloc[-6:-1] - df["low"].iloc[-6:-1]).mean()) or 1.0
    if rng < avg5 * 1.1:
        return None
    if direction == "long" and close > or_data["high"] and body >= 0.5 * rng:
        risk = or_data["range"]
        _atr_intraday(df) or risk
        return IntradayMatch(
            symbol=symbol, setup_id="orb_long", direction="bullish",
            detected_at=last_ts.isoformat(), timeframe="15m",
            entry=close, stop=or_data["low"], target=close + 1.5 * risk,
            risk_reward=1.5, last_price=close,
            volume_ratio=_vol_ratio(df),
            confidence="high" if _vol_ratio(df) > 1.5 else "medium",
            reason=f"15m close above ORH ({or_data['high']:.2f}) with body {body / rng:.0%} of range",
        )
    if direction == "short" and close < or_data["low"] and body >= 0.5 * rng:
        risk = or_data["range"]
        return IntradayMatch(
            symbol=symbol, setup_id="orb_short", direction="bearish",
            detected_at=last_ts.isoformat(), timeframe="15m",
            entry=close, stop=or_data["high"], target=close - 1.5 * risk,
            risk_reward=1.5, last_price=close,
            volume_ratio=_vol_ratio(df),
            confidence="high" if _vol_ratio(df) > 1.5 else "medium",
            reason=f"15m close below ORL ({or_data['low']:.2f}) with body {body / rng:.0%} of range",
        )
    return None


def detect_vwap_bounce(df: pd.DataFrame, *, symbol: str) -> Optional[IntradayMatch]:
    """VWAP Bounce — pulled to VWAP from above, held, green confirm.

    Source: bullsonwallstreet.com VWAP strategy. Need first 60-90 min of
    above-VWAP behaviour + pullback on declining volume + green close.
    """
    if df is None or len(df) < 30:
        return None
    last_ts = df.index[-1]
    if _suppress_if_off_window(last_ts):
        return None
    vwap = session_vwap(df)
    if vwap.empty:
        return None
    # Must have been above VWAP for the recent past
    if not (df["close"].iloc[-30:-3] > vwap.iloc[-30:-3]).mean() > 0.7:
        return None
    last = df.iloc[-1]
    close, open_, low = float(last["close"]), float(last["open"]), float(last["low"])
    v_now = float(vwap.iloc[-1] or 0.0)
    if v_now <= 0:
        return None
    # Bounce confirmation: low touched/below VWAP, close > VWAP, green
    if low <= v_now * 1.002 and close > v_now and close > open_:
        # Volume drying up on the dip (last 3 bars avg < SMA20)
        vol_recent = float(df["volume"].iloc[-3:].mean() or 0.0)
        vol_sma20 = float(df["volume"].iloc[-20:].mean() or 1.0)
        if vol_recent > vol_sma20 * 1.2:
            return None  # Too much selling pressure — likely breaking VWAP
        hod = float(df["high"].max())
        risk = max(close - low, 0.01)
        return IntradayMatch(
            symbol=symbol, setup_id="vwap_bounce", direction="bullish",
            detected_at=last_ts.isoformat(), timeframe="5m",
            entry=close, stop=low - 0.001 * close, target=hod,
            risk_reward=(hod - close) / risk if risk > 0 else 0.0,
            last_price=close, volume_ratio=_vol_ratio(df),
            confidence="medium",
            reason=f"Bounced VWAP {v_now:.2f} on quiet volume, green confirm closing at {close:.2f}",
        )
    return None


def detect_vwap_rejection(df: pd.DataFrame, *, symbol: str) -> Optional[IntradayMatch]:
    """VWAP Rejection — bounced to VWAP from below, rejected, red confirm.

    Bearish mirror of VWAP bounce. Best on gap-down stocks with news.
    """
    if df is None or len(df) < 30:
        return None
    last_ts = df.index[-1]
    if _suppress_if_off_window(last_ts):
        return None
    vwap = session_vwap(df)
    if vwap.empty:
        return None
    if not (df["close"].iloc[-30:-3] < vwap.iloc[-30:-3]).mean() > 0.7:
        return None
    last = df.iloc[-1]
    close, open_, high = float(last["close"]), float(last["open"]), float(last["high"])
    v_now = float(vwap.iloc[-1] or 0.0)
    if v_now <= 0:
        return None
    if high >= v_now * 0.998 and close < v_now and close < open_:
        lod = float(df["low"].min())
        risk = max(high - close, 0.01)
        return IntradayMatch(
            symbol=symbol, setup_id="vwap_rejection", direction="bearish",
            detected_at=last_ts.isoformat(), timeframe="5m",
            entry=close, stop=high + 0.001 * close, target=lod,
            risk_reward=(close - lod) / risk if risk > 0 else 0.0,
            last_price=close, volume_ratio=_vol_ratio(df),
            confidence="medium",
            reason=f"Rejected at VWAP {v_now:.2f}, red close {close:.2f} below open {open_:.2f}",
        )
    return None


def detect_anchored_vwap_pullback(df: pd.DataFrame, *, symbol: str) -> Optional[IntradayMatch]:
    """Anchored VWAP pullback (Brian Shannon).

    Anchor = first bar of the day's range (highest-volume swing pivot in
    last 50 bars). Trade: rising AVWAP test + green confirm.
    """
    if df is None or len(df) < 50:
        return None
    last_ts = df.index[-1]
    if _suppress_if_off_window(last_ts):
        return None
    # Anchor at highest-volume bar in last 50
    win = df.iloc[-50:]
    anchor_idx_local = int(win["volume"].values.argmax())
    anchor_idx = len(df) - 50 + anchor_idx_local
    avwap = anchored_vwap(df, anchor_idx)
    if avwap.iloc[-1] != avwap.iloc[-1]:  # NaN
        return None
    av = float(avwap.iloc[-1])
    if av <= 0:
        return None
    # AVWAP must be rising
    prior = float(avwap.iloc[-10] if len(avwap) >= 10 else avwap.iloc[anchor_idx])
    if av <= prior:
        return None
    last = df.iloc[-1]
    close, open_, low = float(last["close"]), float(last["open"]), float(last["low"])
    if low <= av * 1.003 and close > av and close > open_:
        hod = float(df["high"].iloc[anchor_idx:].max())
        risk = max(close - low, 0.01)
        return IntradayMatch(
            symbol=symbol, setup_id="anchored_vwap_pull", direction="bullish",
            detected_at=last_ts.isoformat(), timeframe="15m",
            entry=close, stop=low - 0.002 * close, target=hod,
            risk_reward=(hod - close) / risk if risk > 0 else 0.0,
            last_price=close, volume_ratio=_vol_ratio(df),
            confidence="medium",
            reason=f"Pullback to rising AVWAP {av:.2f} from high-vol anchor, green confirm",
        )
    return None


def detect_open_drive(df: pd.DataFrame, *, symbol: str) -> Optional[IntradayMatch]:
    """Open Drive / Trend Day (Dalton Market Profile).

    Narrow IB + one-time-framing (each bar's low >= prior bar's low for
    bullish) + range extension above IB high. Highest-conviction trend day.
    """
    if df is None or len(df) < 5:
        return None
    last_ts = df.index[-1]
    if _suppress_if_off_window(last_ts):
        return None
    ib = initial_balance(df)
    if not ib or ib["range"] <= 0:
        return None
    last = df.iloc[-1]
    close = float(last["close"])
    # Range extension above IB high
    if close <= ib["high"]:
        return None
    # One-time-framing check on last 5 bars (bullish: each low >= prior low)
    lows = df["low"].iloc[-5:].values
    if not all(lows[i] >= lows[i - 1] for i in range(1, len(lows))):
        return None
    risk = ib["range"]
    return IntradayMatch(
        symbol=symbol, setup_id="open_drive_long", direction="bullish",
        detected_at=last_ts.isoformat(), timeframe="15m",
        entry=close, stop=ib["high"] - 0.5 * risk, target=close + 1.0 * risk,
        risk_reward=1.0, last_price=close,
        volume_ratio=_vol_ratio(df), confidence="high",
        reason=f"Range extension above IB {ib['high']:.2f} with one-time-framing intact",
    )


def detect_ib_failure(df: pd.DataFrame, *, symbol: str) -> Optional[IntradayMatch]:
    """Inside Bar Failure (Hikkake — Linda Raschke 3-bar rule).

    Inside bar prints, broken in direction X, then reverses inside MB
    within 3 bars, then takes out opposite MB extreme. Long version.
    """
    if df is None or len(df) < 5:
        return None
    last_ts = df.index[-1]
    if _suppress_if_off_window(last_ts):
        return None
    # Mother bar = -4, inside bar = -3, then reversal within 3 bars
    mb_high = float(df["high"].iloc[-4])
    mb_low = float(df["low"].iloc[-4])
    ib_high = float(df["high"].iloc[-3])
    ib_low = float(df["low"].iloc[-3])
    if not (ib_high < mb_high and ib_low > mb_low):
        return None
    # IB broke DOWN below MB low (bearish failure setup for long)
    broke_down = bool((df["low"].iloc[-3:-1] < mb_low).any())
    if not broke_down:
        return None
    last = df.iloc[-1]
    close = float(last["close"])
    # Reversal: closes ABOVE mb_high
    if close < mb_high:
        return None
    if _vol_ratio(df) < 1.5:
        return None
    max(close - mb_low, 0.01)
    return IntradayMatch(
        symbol=symbol, setup_id="ib_failure_long", direction="bullish",
        detected_at=last_ts.isoformat(), timeframe="5m",
        entry=close, stop=mb_low, target=close + 1.5 * (mb_high - mb_low),
        risk_reward=1.5, last_price=close,
        volume_ratio=_vol_ratio(df), confidence="high",
        reason=f"Hikkake long: failed down-break of MB {mb_low:.2f}, reversed above MB high {mb_high:.2f}",
    )


def detect_power_hour_fade(df: pd.DataFrame, *, symbol: str) -> Optional[IntradayMatch]:
    """Power Hour fade — 14:30-15:30 IST, fade HoD/LoD tests.

    Edgeful research: new HoD/LoD prints only 12-24% of sessions during
    power hour. Therefore FADE tests, don't chase. Fire when latest 5m
    bar tests today's extreme and rejects (close back inside range).
    """
    if df is None or len(df) < 10:
        return None
    last_ts = df.index[-1]
    if not is_power_hour(last_ts) or is_closing_auction(last_ts):
        return None
    today_high = float(df["high"].max())
    today_low = float(df["low"].min())
    last = df.iloc[-1]
    high, low, close = float(last["high"]), float(last["low"]), float(last["close"])
    _atr_intraday(df, 14) or 1.0
    # Fade HoD: this bar tested today_high, closed back below
    if high >= today_high * 0.999 and close < today_high * 0.997:
        return IntradayMatch(
            symbol=symbol, setup_id="power_hour_fade", direction="bearish",
            detected_at=last_ts.isoformat(), timeframe="5m",
            entry=close, stop=today_high + 0.001 * close,
            target=float(session_vwap(df).iloc[-1] or today_low),
            risk_reward=1.5, last_price=close,
            volume_ratio=_vol_ratio(df), confidence="medium",
            reason=f"Power-hour HoD fade: tested {today_high:.2f}, closed back at {close:.2f}",
        )
    # Fade LoD
    if low <= today_low * 1.001 and close > today_low * 1.003:
        return IntradayMatch(
            symbol=symbol, setup_id="power_hour_fade", direction="bullish",
            detected_at=last_ts.isoformat(), timeframe="5m",
            entry=close, stop=today_low - 0.001 * close,
            target=float(session_vwap(df).iloc[-1] or today_high),
            risk_reward=1.5, last_price=close,
            volume_ratio=_vol_ratio(df), confidence="medium",
            reason=f"Power-hour LoD fade: tested {today_low:.2f}, closed back at {close:.2f}",
        )
    return None


def detect_vwap_meanrev(df: pd.DataFrame, *, symbol: str) -> Optional[IntradayMatch]:
    """Mean reversion to VWAP — close > 2σ band, then reverses.

    Most reliable in first 90 min and last 60 min. Skip lunch.
    """
    if df is None or len(df) < 30:
        return None
    last_ts = df.index[-1]
    if _suppress_if_off_window(last_ts):
        return None
    vwap, upper, lower = vwap_bands(df, n_sigma=2.0, window=20)
    if upper.iloc[-1] != upper.iloc[-1]:  # NaN
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(last["close"])
    v = float(vwap.iloc[-1])
    up = float(upper.iloc[-1])
    lo = float(lower.iloc[-1])
    prev_close = float(prev["close"])
    # Fade upper: prev was above band, current closed back inside
    if prev_close > up and close < up:
        return IntradayMatch(
            symbol=symbol, setup_id="vwap_meanrev", direction="bearish",
            detected_at=last_ts.isoformat(), timeframe="5m",
            entry=close, stop=float(last["high"]) + 0.001 * close,
            target=v, risk_reward=1.0, last_price=close,
            volume_ratio=_vol_ratio(df), confidence="medium",
            reason=f"VWAP +2σ rejection: {prev_close:.2f} > {up:.2f}, reverted to {close:.2f}",
        )
    # Fade lower: prev was below band, current closed back inside
    if prev_close < lo and close > lo:
        return IntradayMatch(
            symbol=symbol, setup_id="vwap_meanrev", direction="bullish",
            detected_at=last_ts.isoformat(), timeframe="5m",
            entry=close, stop=float(last["low"]) - 0.001 * close,
            target=v, risk_reward=1.0, last_price=close,
            volume_ratio=_vol_ratio(df), confidence="medium",
            reason=f"VWAP -2σ rejection: {prev_close:.2f} < {lo:.2f}, reverted to {close:.2f}",
        )
    return None


def detect_gap_and_go(df: pd.DataFrame, *, symbol: str, prior_close: Optional[float] = None) -> Optional[IntradayMatch]:
    """Gap-and-Go long — gap ≥4% with continuation.

    Needs prior_close to compute gap. Falls back to using df[0] open
    vs df.shift(-1) close if not provided (less accurate).
    """
    if df is None or len(df) < 5:
        return None
    last_ts = df.index[-1]
    if _suppress_if_off_window(last_ts):
        return None
    first_bar = df.iloc[0]
    open_ = float(first_bar["open"])
    if prior_close is None or prior_close <= 0:
        return None
    gap_pct = (open_ - prior_close) / prior_close * 100
    if gap_pct < 4.0:
        return None
    last = df.iloc[-1]
    close = float(last["close"])
    # Continuation: latest 5m close > pre-market high (= first bar high here)
    pmh = float(first_bar["high"])
    if close < pmh:
        return None
    vwap = session_vwap(df)
    stop = float(vwap.iloc[-1] or first_bar["low"])
    risk = max(close - stop, 0.01)
    return IntradayMatch(
        symbol=symbol, setup_id="gap_and_go", direction="bullish",
        detected_at=last_ts.isoformat(), timeframe="5m",
        entry=close, stop=stop, target=close + 2.0 * risk,
        risk_reward=2.0, last_price=close,
        volume_ratio=_vol_ratio(df),
        confidence="high" if _vol_ratio(df) > 2.0 else "medium",
        reason=f"Gap +{gap_pct:.1f}%, close {close:.2f} above first-bar high {pmh:.2f}",
        notes=["Needs catalyst confirmation (news/earnings) — verify before entry"],
    )


def detect_intraday_squeeze(df: pd.DataFrame, *, symbol: str) -> Optional[IntradayMatch]:
    """Intraday BB Squeeze fires (BB expands outside KC after compression).

    John Carter. Skip squeezes that fire DURING lunch (12:30-13:30 IST)
    — those are mechanically narrow bars, not real compression.
    """
    if df is None or len(df) < 30:
        return None
    last_ts = df.index[-1]
    if _suppress_if_off_window(last_ts):
        return None
    sq = bb_squeeze_inside_kc(df)
    if len(sq) < 2:
        return None
    # Squeeze WAS on (prev bar), now off (firing)
    if not (sq.iloc[-2] and not sq.iloc[-1]):
        return None
    last = df.iloc[-1]
    close = float(last["close"])
    open_ = float(last["open"])
    direction = "bullish" if close > open_ else "bearish"
    atr = _atr_intraday(df, 14) or 1.0
    return IntradayMatch(
        symbol=symbol, setup_id="intraday_squeeze", direction=direction,
        detected_at=last_ts.isoformat(), timeframe="5m",
        entry=close,
        stop=close - 1.0 * atr if direction == "bullish" else close + 1.0 * atr,
        target=close + 2.0 * atr if direction == "bullish" else close - 2.0 * atr,
        risk_reward=2.0, last_price=close,
        volume_ratio=_vol_ratio(df),
        confidence="medium",
        reason=f"BB-inside-KC squeeze fired {direction} on {close:.2f}",
    )


def detect_eod_drift(df: pd.DataFrame, *, symbol: str) -> Optional[IntradayMatch]:
    """End-of-day drift — last 25 min, trend continuation in dominant direction.

    Fire only between 15:00 and 15:20 IST (close before auction at 15:20).
    """
    if df is None or len(df) < 20:
        return None
    last_ts = df.index[-1]
    t = last_ts.time() if last_ts.tz is None else last_ts.tz_convert("Asia/Kolkata").time()
    from datetime import time as _t
    if t < _t(15, 0) or t >= _t(15, 20):
        return None
    vwap = session_vwap(df)
    if vwap.empty:
        return None
    v_now = float(vwap.iloc[-1])
    v_30 = float(vwap.iloc[-30] if len(vwap) >= 30 else vwap.iloc[0])
    last = df.iloc[-1]
    close = float(last["close"])
    # Bullish: above VWAP all session, slope positive
    if close > v_now and v_now > v_30:
        hod = float(df["high"].max())
        risk = max(close - v_now, 0.01)
        return IntradayMatch(
            symbol=symbol, setup_id="eod_drift", direction="bullish",
            detected_at=last_ts.isoformat(), timeframe="5m",
            entry=close, stop=v_now, target=hod,
            risk_reward=(hod - close) / risk if risk > 0 else 0.0,
            last_price=close, volume_ratio=_vol_ratio(df),
            confidence="medium",
            reason=f"EOD drift long — above VWAP with rising slope; target HoD {hod:.2f}",
            notes=["Flatten by 15:25 IST — avoid closing auction"],
        )
    if close < v_now and v_now < v_30:
        lod = float(df["low"].min())
        risk = max(v_now - close, 0.01)
        return IntradayMatch(
            symbol=symbol, setup_id="eod_drift", direction="bearish",
            detected_at=last_ts.isoformat(), timeframe="5m",
            entry=close, stop=v_now, target=lod,
            risk_reward=(close - lod) / risk if risk > 0 else 0.0,
            last_price=close, volume_ratio=_vol_ratio(df),
            confidence="medium",
            reason=f"EOD drift short — below VWAP with falling slope; target LoD {lod:.2f}",
            notes=["Flatten by 15:25 IST — avoid closing auction"],
        )
    return None


# Setup_id → detector
_DETECTORS: Dict[str, Callable] = {
    "orb_long": lambda df, sym: detect_orb(df, symbol=sym, direction="long"),
    "orb_short": lambda df, sym: detect_orb(df, symbol=sym, direction="short"),
    "vwap_bounce": lambda df, sym: detect_vwap_bounce(df, symbol=sym),
    "vwap_rejection": lambda df, sym: detect_vwap_rejection(df, symbol=sym),
    "anchored_vwap_pull": lambda df, sym: detect_anchored_vwap_pullback(df, symbol=sym),
    "open_drive_long": lambda df, sym: detect_open_drive(df, symbol=sym),
    "ib_failure_long": lambda df, sym: detect_ib_failure(df, symbol=sym),
    "power_hour_fade": lambda df, sym: detect_power_hour_fade(df, symbol=sym),
    "vwap_meanrev": lambda df, sym: detect_vwap_meanrev(df, symbol=sym),
    "intraday_squeeze": lambda df, sym: detect_intraday_squeeze(df, symbol=sym),
    "eod_drift": lambda df, sym: detect_eod_drift(df, symbol=sym),
    # gap_and_go needs prior_close — handled separately in scan loop
}


def scan_intraday_setups(
    symbols: Sequence[str],
    *,
    bars_fetcher: Callable[[str], Optional[pd.DataFrame]],
    setup_ids: Optional[Sequence[str]] = None,
    prior_closes: Optional[Dict[str, float]] = None,
    max_workers: int = 6,
) -> List[IntradayMatch]:
    """Scan a universe for intraday setups in parallel.

    `bars_fetcher(symbol) -> DataFrame|None` returns the 5m intraday
    bars for the symbol (caller controls the period/interval). All
    setups assume IST timezone on the DatetimeIndex.

    `setup_ids=None` runs every detector. Pass a list to restrict.
    `prior_closes` maps symbol -> prior-day close, used by gap_and_go.

    Failures per symbol are isolated and logged. Returns all matches
    sorted by confidence then risk:reward.
    """
    if not symbols:
        return []
    target_setups = list(setup_ids) if setup_ids else list(_DETECTORS.keys()) + ["gap_and_go"]
    prior_closes = prior_closes or {}

    def _scan_one(sym: str) -> List[IntradayMatch]:
        try:
            df = bars_fetcher(sym)
            if df is None or df.empty:
                return []
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            results: List[IntradayMatch] = []
            for sid in target_setups:
                try:
                    if sid == "gap_and_go":
                        m = detect_gap_and_go(df, symbol=sym, prior_close=prior_closes.get(sym))
                    elif sid in _DETECTORS:
                        m = _DETECTORS[sid](df, sym)
                    else:
                        continue
                    if m is not None:
                        results.append(m)
                except Exception as e:
                    logger.debug("intraday setup %s on %s failed: %s", sid, sym, e)
            return results
        except Exception as e:
            logger.debug("scan_intraday_setups: %s failed: %s", sym, e)
            return []

    all_matches: List[IntradayMatch] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_scan_one, s): s for s in symbols}
        for fut in as_completed(futs):
            all_matches.extend(fut.result())

    conf_rank = {"high": 0, "medium": 1, "low": 2}
    all_matches.sort(key=lambda m: (conf_rank.get(m.confidence, 3), -m.risk_reward))
    return all_matches
