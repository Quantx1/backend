"""Per-style signal output schema for the 4-engine serving layer.

Each style engine (momentum/swing/positional/intraday) emits its own
StyleSignal subclass with style-specific fields (spec §4.7). Kept separate
from the v1 ensemble `GeneratedSignal` in types.py — this is additive.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List


class Style(str, Enum):
    MOMENTUM = "momentum"
    SWING = "swing"
    # POSITIONAL / INTRADAY added when their engines land.


@dataclass
class StyleSignal:
    """Base output every style engine produces. Levels come from the risk
    engine; the engine fills rank/percentile/confidence."""
    symbol: str
    style: Style
    rank: int
    percentile: float
    confidence: float
    direction: str
    entry_price: float
    stop_loss: float
    target: float
    risk_reward: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["style"] = self.style.value
        return d


@dataclass
class MomentumSignal(StyleSignal):
    """Momentum ranker output (spec §4.7)."""
    expected_return: float = 0.0
    top_decile_prob: float = 0.0


@dataclass
class SwingSignal(StyleSignal):
    """Swing ranker output (spec §4.7) — 10-day horizon, same meta bar as
    momentum (identical field shape keeps the hub's meta bar uniform)."""
    expected_return: float = 0.0
    top_decile_prob: float = 0.0
