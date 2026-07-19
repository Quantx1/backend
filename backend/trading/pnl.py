"""
P&L math — single source of truth for unrealized + close-side P&L.

Three call sites consumed this logic with subtle drift before:

  * ``scheduler._calculate_pnl`` (live monitoring) — pure direction
    math, no charges.
  * ``realtime.update_position_pnl`` (WebSocket push) — same pure
    math, inlined.
  * ``scheduler._close_position`` (paper close on SL/target/EOD hit)
    — gross math, ZERO charges → wrote ``net_pnl = gross``.
  * ``trades_routes.close_trade_record`` (user-triggered close) —
    applied 0.1% (EQUITY) / 0.05% (F&O) simulated charges.

So the same paper trade's ``net_pnl`` differed depending on whether
the scheduler fired the close (gross) or the user clicked Close
(net). Both write to the same ``trades`` row. This module is the
fix — both paths now get the same gross + charges + net triple.

Note on paper-trading cost models:
  ``paper_routes.py`` (the standalone paper trader) applies the cost
  on the buy + sell legs (``cost = value * 1.001``, ``proceeds =
  value * 0.999``) rather than as a flat charge on close. That's a
  separate cost model, intentionally not touched here. If we ever
  unify the two paper systems, both should converge on these
  ``compute_close_pnl`` charges and drop the price-side adjustment.
"""

from __future__ import annotations

from typing import Tuple, TypedDict

# Simulated charge rates applied on close. EQUITY rate covers the
# basket of STT (sell-side 0.025% delivery), brokerage (~0.03%),
# exchange + clearing (~0.003%), GST (18% on brokerage), SEBI fee,
# stamp duty (buy-side ~0.015%) — round-trip totals ~0.10% which is
# what we charge once at close. F&O is roughly half (no STT on
# delivery, smaller brokerage cap), giving ~0.05%.
EQUITY_CHARGE_RATE: float = 0.001
FNO_CHARGE_RATE: float = 0.0005


class ClosePnL(TypedDict):
    gross_pnl: float
    charges: float
    net_pnl: float
    pnl_percent: float


def compute_unrealized_pnl(
    direction: str,
    average_price: float,
    current_price: float,
    quantity: float,
) -> Tuple[float, float]:
    """Compute (pnl, pnl_percent) for an open position.

    LONG profits when current rises above average; SHORT profits when
    current drops below average. ``pnl_percent`` is gross %, computed
    against capital deployed (``average_price * quantity``) — not
    against current value, so a 50% move on either side still reads
    as ±50%.
    """
    direction_up = direction == "LONG"
    pnl = (current_price - average_price) * quantity if direction_up else (average_price - current_price) * quantity
    capital = average_price * quantity
    pnl_percent = (pnl / capital) * 100 if capital > 0 else 0.0
    return pnl, pnl_percent


def compute_close_pnl(
    direction: str,
    entry_price: float,
    exit_price: float,
    quantity: float,
    segment: str,
) -> ClosePnL:
    """Compute gross + simulated charges + net P&L for a close.

    Charges are flat rates on ``position_value`` (entry × qty).
    EQUITY: 0.1% (round-trip-effective). F&O: 0.05%. Anything else
    (e.g., FUTURES) gets 0% — extend here when those costs land.

    ``pnl_percent`` is reported on net P&L vs position value, matching
    what the trades table stores and what the route returned before.
    """
    direction_up = direction == "LONG"
    gross = (exit_price - entry_price) * quantity if direction_up else (entry_price - exit_price) * quantity
    position_value = abs(entry_price * quantity)

    if segment == "EQUITY":
        rate = EQUITY_CHARGE_RATE
    elif segment in ("FUTURES", "OPTIONS"):
        rate = FNO_CHARGE_RATE
    else:
        rate = 0.0
    charges = position_value * rate
    net = gross - charges
    pct = (net / position_value) * 100 if position_value > 0 else 0.0

    return {
        "gross_pnl": gross,
        "charges": charges,
        "net_pnl": net,
        "pnl_percent": pct,
    }
