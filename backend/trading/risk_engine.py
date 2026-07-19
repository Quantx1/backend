"""Per-style ATR risk engine (spec §4.6). Models emit expected_return +
confidence; this turns a reference price + ATR into entry/SL/target levels.

Separate from backend/trading/risk.py (position sizing / day-loss / exposure
limits) — different responsibility, no overlap. Pure functions, no I/O.
"""
from __future__ import annotations

from typing import Dict, Tuple

from backend.ai.signals.style_types import Style

#: style -> (stop_loss_atr_mult, take_profit_atr_mult)
RISK_PARAMS: Dict[Style, Tuple[float, float]] = {
    Style.MOMENTUM: (1.5, 3.0),
    # Swing: tighter SL/target than momentum — 10-day horizon vs 20.
    Style.SWING: (1.2, 2.4),
}


def derive_levels(
    direction: str, ref_price: float, atr: float, style: Style
) -> Tuple[float, float, float, float]:
    """Return (entry, stop_loss, target, risk_reward). BUY-only for now
    (the style engines are long-only rankers). Raises ValueError on bad input."""
    if atr <= 0:
        raise ValueError(f"atr must be > 0, got {atr}")
    if ref_price <= 0:
        raise ValueError(f"ref_price must be > 0, got {ref_price}")
    if direction != "BUY":
        raise ValueError(f"only BUY supported, got {direction}")
    if style not in RISK_PARAMS:
        raise ValueError(f"no risk params for style {style}")

    sl_mult, tp_mult = RISK_PARAMS[style]
    entry = float(ref_price)
    stop_loss = round(entry - sl_mult * atr, 2)
    target = round(entry + tp_mult * atr, 2)
    if entry <= stop_loss:
        raise ValueError("degenerate levels: entry <= stop_loss")
    risk_reward = round((target - entry) / (entry - stop_loss), 2)
    return entry, stop_loss, target, risk_reward
