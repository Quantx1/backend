"""Strategy adjustment engine.

O.2 (2026-05-31) — Sensibull's most-loved feature, adapted for Quant X.
Given an OPEN multi-leg position + current market state, suggests the
canonical adjustment (roll, hedge, defend, scale-out) so the user
isn't staring at a 75%-loss position without a plan.

Rule-based (no LLM gating — per project lock `project_agents_decision_2026_05_10`).
Each adjustment fires when the position state matches a specific
distress pattern documented in McMillan / tastytrade / Sensibull.

Inputs (caller fetches via `paper_options_executor.mark_to_market`):
    position_row    — paper_option_positions row with current_value, pnl
    legs            — paper_option_legs rows with current_delta etc.
    spot            — underlying spot
    vix             — India VIX value
    dte             — days to expiry

Returns a list of AdjustmentSuggestion (most relevant first).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AdjustmentSuggestion:
    """One adjustment idea for a struggling position."""
    action: str                       # 'roll' | 'hedge' | 'defend' | 'close' | 'scale_in'
    name: str                         # short label
    urgency: str                      # 'critical' | 'recommended' | 'optional'
    rationale: str                    # one-line why-now
    steps: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    risk_notes: List[str] = field(default_factory=list)
    source_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "name": self.name,
            "urgency": self.urgency,
            "rationale": self.rationale,
            "steps": self.steps,
            "expected_outcome": self.expected_outcome,
            "risk_notes": self.risk_notes,
            "source_label": self.source_label,
        }


def _pnl_pct(position: Dict[str, Any]) -> Optional[float]:
    entry = position.get("net_premium")
    upnl = position.get("unrealized_pnl")
    if not entry or upnl is None:
        return None
    base = abs(float(entry)) or 1.0
    return float(upnl) / base * 100


def _is_short_premium(legs: List[Dict[str, Any]]) -> bool:
    """Heuristic: more SELL legs than BUY legs → short-premium strategy."""
    sells = sum(1 for L in legs if str(L.get("side", "")).upper() == "SELL")
    buys = sum(1 for L in legs if str(L.get("side", "")).upper() == "BUY")
    return sells > buys


def _tested_side(spot: float, legs: List[Dict[str, Any]]) -> Optional[str]:
    """Which side of a 2-sided structure is the trade tested at?

    Returns 'call' if spot near/above short call strikes, 'put' if near/
    below short puts, None if balanced.
    """
    short_calls = [float(L["strike"]) for L in legs
                   if str(L.get("side", "")).upper() == "SELL"
                   and str(L.get("option_type", "")).upper() == "CE"]
    short_puts = [float(L["strike"]) for L in legs
                  if str(L.get("side", "")).upper() == "SELL"
                  and str(L.get("option_type", "")).upper() == "PE"]
    if short_calls and spot >= min(short_calls) * 0.98:
        return "call"
    if short_puts and spot <= max(short_puts) * 1.02:
        return "put"
    return None


# ── Adjustment generators ────────────────────────────────────────


def _adj_take_profit_50(position, legs, spot, vix, dte):
    """Standard tastytrade rule: short premium → take profit at 50%."""
    pnl_pct = _pnl_pct(position)
    if not _is_short_premium(legs):
        return None
    # For short premium, "profit" means current_value moved toward 0.
    # pnl_pct > 0 means we've collected — fire at >=50% of credit captured.
    if pnl_pct is None or pnl_pct < 50:
        return None
    return AdjustmentSuggestion(
        action="close",
        name="Take Profit 50%",
        urgency="recommended",
        rationale=f"Captured {pnl_pct:.0f}% of max credit — risk/reward inverted from here.",
        steps=[
            "Close ALL legs at market",
            "Free margin for the next setup",
        ],
        expected_outcome="Lock in profit; eliminate residual gamma + assignment risk",
        risk_notes=["Holding longer = risking gains for marginal extra credit"],
        source_label="tastytrade — 50% profit-take canonical rule",
    )


def _adj_stop_loss_200(position, legs, spot, vix, dte):
    """Short premium loss has reached 200% of credit — close out."""
    pnl_pct = _pnl_pct(position)
    if not _is_short_premium(legs):
        return None
    if pnl_pct is None or pnl_pct > -200:
        return None
    return AdjustmentSuggestion(
        action="close",
        name="Stop Loss Hit (-200%)",
        urgency="critical",
        rationale=f"Position at {pnl_pct:.0f}% of credit — past the standard 200% stop.",
        steps=[
            "Close all legs at market immediately",
            "Do NOT roll further — accept the loss",
        ],
        expected_outcome="Cap downside; preserve margin",
        risk_notes=["Rolling losers tends to compound losses — discipline wins"],
        source_label="tastytrade — 200% stop on short-premium",
    )


def _adj_roll_tested_side(position, legs, spot, vix, dte):
    """Tested-side roll — when one side of a 2-sided structure is breached,
    roll the tested side further OTM to give it room."""
    tested = _tested_side(spot, legs)
    if tested is None or dte is None or dte < 5:
        return None
    pnl_pct = _pnl_pct(position)
    if pnl_pct is None or pnl_pct > -50:
        return None     # don't roll a winner
    name = f"Roll Tested {tested.upper()} Side"
    side_label = "call up" if tested == "call" else "put down"
    return AdjustmentSuggestion(
        action="roll",
        name=name,
        urgency="recommended",
        rationale=f"Spot {spot:.0f} testing short {tested} strikes — roll {side_label} to give it room.",
        steps=[
            f"Buy back the tested short {tested}",
            f"Sell a new short {tested} ~3-5% further OTM in the same expiry",
            "Keep the untested side unchanged",
        ],
        expected_outcome="Wider profit zone; collect extra credit; reduce immediate assignment risk",
        risk_notes=[
            "Adds total credit but widens BE on the tested side",
            "Adjust only if your directional thesis hasn't changed",
        ],
        source_label="tastytrade — tested-side roll on iron condor / strangle",
    )


def _adj_roll_to_next_week(position, legs, spot, vix, dte):
    """When DTE drops below 3 on a short-premium position with
    unfavourable Greeks, roll the whole structure to next weekly."""
    if not _is_short_premium(legs):
        return None
    if dte is None or dte > 3 or dte < 0:
        return None
    return AdjustmentSuggestion(
        action="roll",
        name="Roll to Next Weekly Expiry",
        urgency="recommended",
        rationale=f"Only {dte} DTE remaining — gamma risk rising fast on short-premium structure.",
        steps=[
            "Close current legs at market",
            "Reopen identical structure in next-week expiry",
            "Lock in any credit gained from the roll",
        ],
        expected_outcome="Reset theta clock; cap pin/assignment risk near expiry",
        risk_notes=[
            "Roll cost may eat 30-50% of accumulated profit — accept the trade-off",
            "Don't roll into binary events (RBI/FOMC week)",
        ],
        source_label="Sensibull / Optionalpha — 21-DTE / 3-DTE roll rules",
    )


def _adj_add_protective_wing(position, legs, spot, vix, dte):
    """Naked short straddle/strangle in elevated VIX → add protective
    wings to convert to defined-risk."""
    has_long_protection = any(
        str(L.get("side", "")).upper() == "BUY" for L in legs
    )
    if has_long_protection:
        return None
    if not _is_short_premium(legs):
        return None
    if vix is None or vix < 18:
        return None
    pnl_pct = _pnl_pct(position)
    return AdjustmentSuggestion(
        action="hedge",
        name="Add Protective Wings",
        urgency="critical" if (pnl_pct is not None and pnl_pct < -50) else "recommended",
        rationale=f"Naked short premium in VIX {vix:.1f} elevated regime — convert to defined-risk.",
        steps=[
            f"Buy OTM put ~3-5% below spot (≈{round(spot * 0.96, 0)})",
            f"Buy OTM call ~3-5% above spot (≈{round(spot * 1.04, 0)})",
            "Reduces max loss from unlimited to defined",
        ],
        expected_outcome="Cap tail-risk at the cost of some upside premium",
        risk_notes=[
            "Wings cost premium — reduces credit but worth it in stressed regime",
            "Critical hedge if VIX is RISING, not just elevated",
        ],
        source_label="McMillan — naked-to-defined-risk conversion",
    )


def _adj_scale_out_winner(position, legs, spot, vix, dte):
    """Winner at 25% — scale out half to lock-in, ride remaining."""
    pnl_pct = _pnl_pct(position)
    if pnl_pct is None or pnl_pct < 25 or pnl_pct > 50:
        return None
    if not _is_short_premium(legs):
        return None
    return AdjustmentSuggestion(
        action="scale_in",
        name="Scale Out 50% at +25%",
        urgency="optional",
        rationale=f"Up {pnl_pct:.0f}% — bank half, let the rest run to 50% TP.",
        steps=[
            "Close half the lots in each leg",
            "Trail remaining half with same stop discipline",
        ],
        expected_outcome="Lock in guaranteed profit while keeping upside",
        risk_notes=["Reduces theta capture rate on remaining half"],
        source_label="tastytrade — scale-out winners",
    )


def _adj_defend_with_calendar(position, legs, spot, vix, dte):
    """Vertical spread getting tested → add a calendar at the same strike
    in next month to extend duration cheaply."""
    sells_count = sum(1 for L in legs if str(L.get("side", "")).upper() == "SELL")
    buys_count = sum(1 for L in legs if str(L.get("side", "")).upper() == "BUY")
    if sells_count != 1 or buys_count != 1:
        return None
    pnl_pct = _pnl_pct(position)
    if pnl_pct is None or pnl_pct > -30:
        return None
    return AdjustmentSuggestion(
        action="defend",
        name="Add Calendar Hedge",
        urgency="recommended",
        rationale=f"Vertical at {pnl_pct:.0f}% — add far-month leg to extend duration without doubling debit.",
        steps=[
            "Buy a longer-dated long option at the same strike as your existing long",
            "Creates a diagonal/calendar overlay",
            "Doesn't change immediate P&L much but extends time",
        ],
        expected_outcome="Buys time for thesis to play out without compounding risk",
        risk_notes=["Costs additional debit", "Adds long-vega exposure"],
        source_label="McMillan — calendar overlay defense",
    )


# ── Entry point ─────────────────────────────────────────────────


def suggest_adjustments(
    position_row: Dict[str, Any],
    legs: List[Dict[str, Any]],
    *,
    spot: float,
    vix: Optional[float] = None,
) -> List[AdjustmentSuggestion]:
    """Run all rule-based adjusters; return the ones that fire ranked
    by urgency (critical → recommended → optional)."""
    from datetime import date
    dte = None
    if position_row.get("expiry_date"):
        try:
            exp = date.fromisoformat(str(position_row["expiry_date"])[:10])
            dte = max(0, (exp - date.today()).days)
        except Exception:
            pass

    out: List[AdjustmentSuggestion] = []
    for fn in (
        _adj_stop_loss_200,
        _adj_add_protective_wing,
        _adj_take_profit_50,
        _adj_roll_tested_side,
        _adj_roll_to_next_week,
        _adj_defend_with_calendar,
        _adj_scale_out_winner,
    ):
        try:
            s = fn(position_row, legs, spot, vix, dte)
            if s is not None:
                out.append(s)
        except Exception:
            continue

    urgency_rank = {"critical": 0, "recommended": 1, "optional": 2}
    out.sort(key=lambda s: urgency_rank.get(s.urgency, 3))
    return out
