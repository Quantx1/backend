"""Translate a scanner IntradayMatch into a /ws payload + a `signals` DB row.

Pure functions, no side effects — the single source of truth for how an
intraday setup becomes a user-facing signal."""
from __future__ import annotations

from typing import Dict

from .scanner import IntradayMatch

CONFIDENCE_INT = {"high": 80, "medium": 60, "low": 40}
_DIRECTION = {"bullish": "LONG", "bearish": "SHORT", "neutral": "NEUTRAL"}


def _direction(match: IntradayMatch) -> str:
    return _DIRECTION.get(match.direction, "NEUTRAL")


def match_to_ws_payload(match: IntradayMatch) -> Dict:
    """Compact payload for the INTRADAY_SIGNAL WebSocket message."""
    return {
        "symbol": match.symbol,
        "setup_id": match.setup_id,
        "direction": _direction(match),
        "confidence": CONFIDENCE_INT.get(match.confidence, 40),
        "timeframe": match.timeframe,
        "entry": match.entry,
        "stop": match.stop,
        "target": match.target,
        "risk_reward": match.risk_reward,
        "reason": match.reason,
        "detected_at": match.detected_at,
    }


def match_to_signal_row(match: IntradayMatch) -> Dict:
    """Row for ``supabase.table("signals").insert(...)`` (signal_type=intraday)."""
    return {
        "symbol": match.symbol,
        "exchange": "NSE",
        "segment": "EQUITY",
        "signal_type": "intraday",
        "direction": _direction(match),
        "confidence": CONFIDENCE_INT.get(match.confidence, 40),
        "engine_name": "Intraday",
        # setup_id lives in raw_scores below — it is NOT a top-level `signals`
        # column, and including it would make PostgREST reject the whole insert.
        "entry_price": match.entry,
        "stop_loss": match.stop,
        "target_1": match.target,
        "risk_reward": match.risk_reward,
        "reasons": [match.reason],
        "raw_scores": {"setup_id": match.setup_id, "confidence": match.confidence,
                       "volume_ratio": match.volume_ratio},
        "status": "active",
    }
