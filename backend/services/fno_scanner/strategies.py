"""F&O strategy suggestions — fires structured trade ideas based on the
current option-chain snapshot + India VIX regime.

Algorithms sourced from:
  - Volatility Box backtests (IV Rank > 50 short-premium uplift +8.6pp):
      https://volatilitybox.com/research/iv-rank-vs-iv-percentile/
  - tastytrade-school 16-delta short strangle defaults
  - Zerodha Varsity straddle/strangle/calendar chapters
  - OptionsTradingIQ Iron Condor success-rate backtest (~77.6% on SPY 16Δ)
  - ORATS calendar-spread IV-contango backtest (Medium)
  - IIFL/Bajaj AMC India VIX regime bands

Output is DESCRIPTIVE (suggested setup, capital req, why now) — not a
trade order. AutoPilot routing (Phase 3) is a separate decision tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .lot_sizes import lot_value
from .snapshot import IndexSnapshot


@dataclass
class StrategySuggestion:
    name: str                         # short label
    bias: str                         # bullish | bearish | neutral
    confidence: str                   # high | medium | low
    rationale: str                    # one-line why-now
    margin_estimate_inr: Optional[int] = None
    suggested_legs: List[str] = field(default_factory=list)
    risk_notes: List[str] = field(default_factory=list)
    source_label: str = ""            # which research-rule fired this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bias": self.bias,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "margin_estimate_inr": self.margin_estimate_inr,
            "suggested_legs": self.suggested_legs,
            "risk_notes": self.risk_notes,
            "source_label": self.source_label,
        }


# ── VIX regime classification (verified standard) ──────────────────


def classify_vix_regime(vix: Optional[float]) -> str:
    """India VIX regime bands.

    Sources:
      - https://www.bajajamc.com/knowledge-centre/india-vix-index
      - https://www.ifmcinstitute.com/vix-index-india/
      - Volatility Box IC research (sweet-spot 13-22)
    """
    if vix is None:
        return "unknown"
    if vix < 13:
        return "complacent"        # too cheap to sell — favor long-premium
    if vix < 18:
        return "normal"            # 16-delta condors / strangles work
    if vix < 22:
        return "elevated"          # premium-sell sweet spot (richest credits)
    return "stressed"              # defined-risk only, reduce size


_REGIME_GUIDANCE: Dict[str, Dict[str, str]] = {
    "complacent": {
        "long_premium": "Favor — debit spreads, calendars, long straddles ahead of events.",
        "short_premium": "Avoid — credits too thin to justify tail risk.",
    },
    "normal": {
        "long_premium": "OK situational — calendars around events.",
        "short_premium": "Favor — 16-delta iron condors / short strangles, 30-45 DTE.",
    },
    "elevated": {
        "long_premium": "Avoid — paying rich premium without a vol-spike catalyst.",
        "short_premium": "Sweet spot — 12-16 delta shorts capture the richest credits.",
    },
    "stressed": {
        "long_premium": "Tail-protection: long puts / put spreads.",
        "short_premium": "Defined-risk only (condors, NOT naked straddles); 10-delta shorts, half size.",
    },
}


# ── Strategy suggesters ───────────────────────────────────────────


def _suggest_iron_condor(snap: IndexSnapshot, vix: Optional[float]) -> Optional[StrategySuggestion]:
    """16-delta iron condor — works when VIX is in the 15-22 sweet spot.
    16Δ ≈ ±1σ; structure ~70-80% win-rate per OptionsTradingIQ backtest."""
    if vix is None or vix < 13 or vix > 22:
        return None
    margin = int(lot_value(snap.symbol, snap.spot) * 0.05)   # ~5% of notional
    return StrategySuggestion(
        name="Iron Condor",
        bias="neutral",
        confidence="high" if 15 <= vix <= 20 else "medium",
        rationale=f"VIX {vix:.1f} in sweet spot for premium selling; PCR {snap.pcr_oi:.2f}.",
        margin_estimate_inr=margin or None,
        suggested_legs=[
            f"Short ~16Δ Call (≈{round(snap.spot * 1.02, 0)} strike)",
            f"Short ~16Δ Put  (≈{round(snap.spot * 0.98, 0)} strike)",
            "Long wings 5Δ further OTM",
        ],
        risk_notes=[
            "Take profit at 50% of max credit",
            "Stop at 200% of credit received",
            "Close at 21 DTE if not at target",
        ],
        source_label="OptionsTradingIQ 16Δ backtest (77.6% WR on SPY)",
    )


def _suggest_short_strangle(
        snap: IndexSnapshot,
        vix: Optional[float],
        iv_rank: Optional[float]) -> Optional[StrategySuggestion]:
    """Short strangle when IV Rank > 50, ADX/range-bound, VIX moderate-elevated.
    Higher payout than condor but unlimited risk — flag prominently."""
    if vix is None or vix < 15 or vix > 22:
        return None
    if iv_rank is not None and iv_rank < 50:
        return None
    margin = int(lot_value(snap.symbol, snap.spot) * 0.08)
    return StrategySuggestion(
        name="Short Strangle",
        bias="neutral",
        confidence="medium",
        rationale=(
            f"VIX {vix:.1f} elevated; "
            + (f"IV Rank {iv_rank:.0f} ≥ 50 supports premium selling." if iv_rank is not None else "elevated vol regime.")
        ),
        margin_estimate_inr=margin or None,
        suggested_legs=[
            f"Short 16Δ Call (≈{round(snap.spot * 1.02, 0)})",
            f"Short 16Δ Put  (≈{round(snap.spot * 0.98, 0)})",
        ],
        risk_notes=[
            "Unlimited risk — strict per-trade SL non-negotiable",
            "TP 25-50% of credit; SL 150-200%",
            "Avoid through event days (RBI / FOMC / Budget)",
        ],
        source_label="Sensibull / 5paisa strangle defaults",
    )


def _suggest_long_calendar(snap: IndexSnapshot, vix: Optional[float]) -> Optional[StrategySuggestion]:
    """Long calendar — sell near-month, buy far-month at same strike.
    Best when IV term structure is in backwardation (near IV > far IV)
    OR right after an event-driven near-IV crush."""
    if vix is None:
        return None
    # Use IV skew + ATM IV as a (very rough) backwardation proxy when
    # we don't have a far-month chain to compare against.
    if snap.iv_atm is None:
        return None
    margin = int(lot_value(snap.symbol, snap.spot) * 0.015)
    return StrategySuggestion(
        name="ATM Calendar (long vega)",
        bias="neutral",
        confidence="medium" if vix < 18 else "low",
        rationale=(
            f"ATM IV {snap.iv_atm:.1%}; "
            "best fired the morning AFTER an event when near-month IV crushes."
        ),
        margin_estimate_inr=margin or None,
        suggested_legs=[
            f"Sell weekly ATM ({round(snap.spot, 0)})",
            f"Buy next-month ATM ({round(snap.spot, 0)})",
        ],
        risk_notes=[
            "Long vega — wins if vol expands or stays put",
            "Degrades fast in trending markets",
        ],
        source_label="ORATS calendar-spread IV-contango backtest",
    )


def _suggest_long_premium_debit(snap: IndexSnapshot, vix: Optional[float]) -> Optional[StrategySuggestion]:
    """Long premium (debit spreads / long straddle) when VIX is low —
    cheap options + directional thesis from PCR extreme."""
    if vix is None or vix >= 15:
        return None
    bias = "neutral"
    legs: List[str] = []
    rationale = f"VIX {vix:.1f} below 15 — premium is cheap. "
    if snap.pcr_tag == "extreme_bullish":
        bias = "bullish"
        rationale += f"PCR {snap.pcr_oi:.2f} contrarian-bullish."
        legs = [
            f"Long ATM Call ({round(snap.spot, 0)})",
            f"Short OTM Call ({round(snap.spot * 1.02, 0)})",
            "= bull call spread, defined risk",
        ]
    elif snap.pcr_tag == "extreme_bearish":
        bias = "bearish"
        rationale += f"PCR {snap.pcr_oi:.2f} contrarian-bearish."
        legs = [
            f"Long ATM Put ({round(snap.spot, 0)})",
            f"Short OTM Put ({round(snap.spot * 0.98, 0)})",
            "= bear put spread, defined risk",
        ]
    else:
        bias = "neutral"
        rationale += "Long straddle / ATM if range break expected."
        legs = [
            f"Long ATM Call ({round(snap.spot, 0)})",
            f"Long ATM Put ({round(snap.spot, 0)})",
        ]
    margin = int(lot_value(snap.symbol, snap.spot) * 0.02)
    return StrategySuggestion(
        name="Long Premium (debit)",
        bias=bias,
        confidence="medium",
        rationale=rationale,
        margin_estimate_inr=margin or None,
        suggested_legs=legs,
        risk_notes=[
            "Defined risk = debit paid",
            "Time decay works against; size for the move-or-die thesis",
        ],
        source_label="Volatility Box low-IV long-premium rule",
    )


def _suggest_gamma_scalping(
        snap: IndexSnapshot,
        vix: Optional[float],
        iv_rank: Optional[float]) -> Optional[StrategySuggestion]:
    """Delta-neutral gamma scalping — buy a straddle/strangle, hedge with
    the underlying future when delta drifts. Profits when RV > IV after
    round-trip costs.

    Source: Volatility Box gamma-scalping research; MenthorQ delta-hedging
    guide; Profitmart writeup. Retail-unfriendly below ₹10L capital;
    requires intraday rebalancing 4-8× per day or every 30-60 min.

    Hard gates (per research):
      - IV Rank < 30 (cheap options — long-vol thesis)
      - Capital advisory: ₹10L+ recommended; we expose lot value so user
        decides
    """
    if vix is None or vix >= 18:
        return None       # Need cheap vol to justify the structure
    if iv_rank is not None and iv_rank >= 30:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    if notional <= 0:
        return None
    iv_text = f"{iv_rank:.0f}" if iv_rank is not None else "low"
    return StrategySuggestion(
        name="Gamma Scalping (advanced)",
        bias="neutral",
        confidence="low",          # complexity + cost burden is high
        rationale=(
            f"IV Rank {iv_text}, "
            f"VIX {vix:.1f} — long-vol thesis viable. Profits if RV beats IV."
        ),
        margin_estimate_inr=int(notional * 0.10),   # straddle + hedge buffer
        suggested_legs=[
            f"Long ATM Straddle (≈{round(snap.spot, 0)})",
            "Δ-hedge with index future at delta band ±10-20 per leg",
            "Rebalance every 30-60 min (retail) / 4-8×/day (market-maker)",
        ],
        risk_notes=[
            "₹10L+ capital recommended — round-trip costs eat retail edges",
            "RV must beat IV after STT+brokerage (~0.05-0.10% per hedge cycle)",
            "Bleeds theta in quiet markets — kill switch if realized vol stays below entry IV",
            "Requires active intraday attention — not set-and-forget",
        ],
        source_label="Volatility Box / MenthorQ gamma-scalping research",
    )


def _suggest_cvd_divergence_short(snap: IndexSnapshot, vix: Optional[float]) -> Optional[StrategySuggestion]:
    """Cumulative Volume Delta (CVD) divergence — bearish.

    CRITICAL NSE caveat: standard NSE feeds do NOT expose true bid/ask
    aggression. We approximate with the tick-rule (uptick=buy, downtick=
    sell), which weakens the signal. Limit to most-liquid F&O names.

    Source: Bookmap CVD guide; GoCharting delta-divergence documentation.
    """
    # Without intraday tick data we can't FIRE this signal from the
    # snapshot alone — surface as an INFORMATIONAL setup template so
    # the user knows the pattern + when to apply it manually, with the
    # hard caveat that CVD on NSE is approximation-only.
    if snap.iv_skew is None or snap.iv_skew < 0.01:
        return None       # Need elevated put skew to corroborate
    return StrategySuggestion(
        name="CVD Divergence Watch (informational)",
        bias="bearish",
        confidence="low",
        rationale=(
            f"IV skew +{snap.iv_skew * 100:.1f}% — puts bid up relative to calls."
            " Pair with intraday CVD on 5-min bars to confirm bearish divergence."
        ),
        suggested_legs=[
            "Watch 5-min chart: price prints higher high, CVD prints lower high → short",
            "Stop above the divergence high; target = VWAP or prior swing low",
        ],
        risk_notes=[
            "NSE tick-rule CVD is APPROXIMATION (no L1 aggressor flag) — weaker signal",
            "Restrict to most-liquid F&O names (>10k trades/day)",
            "Skip on event days (RBI/FOMC/Budget) — pure flow signals get overwhelmed",
        ],
        source_label="Bookmap CVD; GoCharting delta-divergence (NSE caveat applies)",
    )


def _suggest_max_pain_pull(snap: IndexSnapshot) -> Optional[StrategySuggestion]:
    """When |spot - MaxPain| > 1.5% with ≥2 DTE, options market structure
    biases price back toward Max Pain by expiry. ~55-65% accuracy within
    ±100pts on Nifty per StockMojo / Varsity historical observation."""
    if not snap.pull_to_max_pain_signal or snap.max_pain is None:
        return None
    direction = "down" if snap.spot > snap.max_pain else "up"
    bias = "bearish" if direction == "down" else "bullish"
    return StrategySuggestion(
        name="Max Pain Pull",
        bias=bias,
        confidence="medium",
        rationale=(
            f"Spot {snap.spot:.0f} is {abs(snap.max_pain_distance_pct):.1f}% "
            f"{'above' if direction == 'down' else 'below'} Max Pain "
            f"{snap.max_pain:.0f} with {snap.days_to_expiry} DTE."
        ),
        suggested_legs=[
            f"Bias trade toward Max Pain ({snap.max_pain:.0f})",
            "Pair with a short premium structure (condor / strangle) for theta tailwind",
        ],
        risk_notes=[
            "Accuracy drops sharply on event days — skip into RBI/FOMC/Budget",
            "Pull effect strongest in the last 2-3 sessions before expiry",
        ],
        source_label="Zerodha Varsity Max Pain + PCR chapter; StockMojo Nifty MP study",
    )


# ────────────────────────────────────────────────────────────────────
# O.1 (2026-05-31) — Strategy template library expansion 7 → 30.
# 23 additional structures sourced from:
#   - McMillan "Options as a Strategic Investment" (5th ed.)
#   - Sensibull strategy templates library
#   - tastytrade-school directional + neutral templates
#   - OptionsTradingIQ + Volatility Box backtests
# Each template fires only when its preconditions match the snapshot —
# the same rationale gate prevents users seeing "Long Strangle suggested"
# in a regime where it would lose money to theta.
# ────────────────────────────────────────────────────────────────────


def _suggest_bull_call_spread(snap, vix, iv_rank):
    """Long ATM call + Short OTM call — bullish, defined risk, low cost."""
    if snap.pcr_tag not in ("extreme_bullish", "bullish"):
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Bull Call Spread",
        bias="bullish",
        confidence="high" if snap.pcr_tag == "extreme_bullish" else "medium",
        rationale=f"PCR {snap.pcr_oi:.2f} ({snap.pcr_tag}) — defined-risk bullish.",
        margin_estimate_inr=int(notional * 0.015) or None,
        suggested_legs=[
            f"Long ATM Call (≈{round(snap.spot, 0)})",
            f"Short OTM Call (≈{round(snap.spot * 1.02, 0)})",
        ],
        risk_notes=["Max loss = net debit", "Time decay works against you"],
        source_label="McMillan ch. 9 — Bull Call Spread",
    )


def _suggest_bear_call_spread(snap, vix, iv_rank):
    """Short OTM call + Long further-OTM call — bearish, credit, defined risk."""
    if snap.pcr_tag not in ("bearish", "extreme_bearish"):
        return None
    if vix is None or vix < 14:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Bear Call Spread",
        bias="bearish",
        confidence="medium",
        rationale=f"PCR {snap.pcr_oi:.2f} bearish + VIX {vix:.1f} elevated — credit collect.",
        margin_estimate_inr=int(notional * 0.04) or None,
        suggested_legs=[
            f"Short OTM Call (≈{round(snap.spot * 1.01, 0)})",
            f"Long further-OTM Call (≈{round(snap.spot * 1.03, 0)})",
        ],
        risk_notes=["Max profit = credit", "Take profit at 50% of credit"],
        source_label="McMillan ch. 9 — Bear Call Spread",
    )


def _suggest_bull_put_spread(snap, vix, iv_rank):
    """Short OTM put + Long further-OTM put — bullish, credit, defined risk."""
    if snap.pcr_tag not in ("bullish", "extreme_bullish"):
        return None
    if vix is None or vix < 14:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Bull Put Spread",
        bias="bullish",
        confidence="medium",
        rationale=f"PCR {snap.pcr_oi:.2f} bullish + VIX {vix:.1f} — sell puts with floor.",
        margin_estimate_inr=int(notional * 0.04) or None,
        suggested_legs=[
            f"Short OTM Put (≈{round(snap.spot * 0.99, 0)})",
            f"Long further-OTM Put (≈{round(snap.spot * 0.97, 0)})",
        ],
        risk_notes=["Max profit = credit", "TP at 50% / SL at 200%"],
        source_label="tastytrade — Bull Put Spread",
    )


def _suggest_bear_put_spread(snap, vix, iv_rank):
    """Long ATM put + Short OTM put — bearish, defined risk, low cost."""
    if snap.pcr_tag not in ("bearish", "extreme_bearish"):
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Bear Put Spread",
        bias="bearish",
        confidence="medium",
        rationale=f"PCR {snap.pcr_oi:.2f} ({snap.pcr_tag}) — defined-risk bearish.",
        margin_estimate_inr=int(notional * 0.015) or None,
        suggested_legs=[
            f"Long ATM Put (≈{round(snap.spot, 0)})",
            f"Short OTM Put (≈{round(snap.spot * 0.98, 0)})",
        ],
        risk_notes=["Max loss = net debit", "Time decay works against you"],
        source_label="McMillan ch. 9 — Bear Put Spread",
    )


def _suggest_protective_put(snap, vix, iv_rank):
    """Long underlying + Long ATM Put — insurance for held positions."""
    if vix is None or vix < 15:
        return None  # only insure when vol is elevated enough to matter
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Protective Put",
        bias="bullish",
        confidence="medium",
        rationale=f"VIX {vix:.1f} elevated — buy downside insurance on long index exposure.",
        margin_estimate_inr=int(notional * 0.02) or None,
        suggested_legs=[
            "Hold long index/futures",
            f"Long ATM Put (≈{round(snap.spot, 0)})",
        ],
        risk_notes=["Premium = insurance cost", "Best around event days (RBI/FOMC/Budget)"],
        source_label="McMillan — Protective Put",
    )


def _suggest_covered_call(snap, vix, iv_rank):
    """Long underlying + Short OTM call — income on held long positions."""
    if vix is None or vix < 13:
        return None
    if iv_rank is not None and iv_rank < 40:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Covered Call",
        bias="neutral",
        confidence="medium",
        rationale=f"VIX {vix:.1f} + IV Rank acceptable — collect call premium on long.",
        margin_estimate_inr=int(notional * 0.95) or None,    # full underlying
        suggested_legs=[
            "Hold long index/futures",
            f"Short OTM Call (≈{round(snap.spot * 1.03, 0)})",
        ],
        risk_notes=["Caps upside at strike", "Assignment risk near expiry"],
        source_label="McMillan — Covered Call",
    )


def _suggest_cash_secured_put(snap, vix, iv_rank):
    """Short OTM put backed by cash — get paid to buy lower."""
    if vix is None or vix < 14:
        return None
    if snap.pcr_tag not in ("bullish", "extreme_bullish", "normal"):
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Cash-Secured Put",
        bias="bullish",
        confidence="medium",
        rationale=f"VIX {vix:.1f} — collect put premium, willing to be long at strike.",
        margin_estimate_inr=int(notional * 0.20) or None,
        suggested_legs=[
            f"Short OTM Put (≈{round(snap.spot * 0.97, 0)})",
            "Hold cash for assignment",
        ],
        risk_notes=["Max loss = strike - premium (large)", "Stop at 200% of credit"],
        source_label="tastytrade — Cash-Secured Put",
    )


def _suggest_long_straddle(snap, vix, iv_rank):
    """Long ATM call + Long ATM put — bet on big move either direction."""
    if vix is None or vix > 18:
        return None  # too expensive in high vol
    if iv_rank is not None and iv_rank > 40:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Long Straddle",
        bias="neutral",
        confidence="medium",
        rationale=f"VIX {vix:.1f} low + IV Rank low — cheap premium, big-move thesis.",
        margin_estimate_inr=int(notional * 0.04) or None,
        suggested_legs=[
            f"Long ATM Call (≈{round(snap.spot, 0)})",
            f"Long ATM Put (≈{round(snap.spot, 0)})",
        ],
        risk_notes=["Time decay accelerates near expiry", "Need 2+ DTE for movement"],
        source_label="Zerodha Varsity — Long Straddle",
    )


def _suggest_long_strangle(snap, vix, iv_rank):
    """Long OTM call + Long OTM put — cheaper than straddle, needs bigger move."""
    if vix is None or vix > 18:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Long Strangle",
        bias="neutral",
        confidence="low",
        rationale=f"VIX {vix:.1f} low — cheap bet on a big move outside wings.",
        margin_estimate_inr=int(notional * 0.025) or None,
        suggested_legs=[
            f"Long OTM Call (≈{round(snap.spot * 1.02, 0)})",
            f"Long OTM Put (≈{round(snap.spot * 0.98, 0)})",
        ],
        risk_notes=["Needs >2% move to profit", "Time decay brutal under 7 DTE"],
        source_label="McMillan — Long Strangle",
    )


def _suggest_long_iron_butterfly(snap, vix, iv_rank):
    """Long Iron Butterfly — defined risk neutral, premium buyer."""
    if vix is None or vix > 18:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Long Iron Butterfly",
        bias="neutral",
        confidence="low",
        rationale=f"VIX {vix:.1f} — defined-risk volatility bet around event.",
        margin_estimate_inr=int(notional * 0.02) or None,
        suggested_legs=[
            f"Long OTM Call (≈{round(snap.spot * 1.02, 0)})",
            f"Short ATM Call (≈{round(snap.spot, 0)})",
            f"Short ATM Put (≈{round(snap.spot, 0)})",
            f"Long OTM Put (≈{round(snap.spot * 0.98, 0)})",
        ],
        risk_notes=["Profit only on large move beyond wings", "TP 50% of debit"],
        source_label="McMillan — Long Iron Butterfly",
    )


def _suggest_reverse_iron_condor(snap, vix, iv_rank):
    """Reverse Iron Condor — defined risk vol-buy with wider profit zone."""
    if vix is None or vix > 16:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Reverse Iron Condor",
        bias="neutral",
        confidence="low",
        rationale=f"VIX {vix:.1f} cheap — pay debit for wide vol-expansion bet.",
        margin_estimate_inr=int(notional * 0.025) or None,
        suggested_legs=[
            f"Long OTM Put (≈{round(snap.spot * 0.97, 0)})",
            f"Short further-OTM Put (≈{round(snap.spot * 0.94, 0)})",
            f"Long OTM Call (≈{round(snap.spot * 1.03, 0)})",
            f"Short further-OTM Call (≈{round(snap.spot * 1.06, 0)})",
        ],
        risk_notes=["Profit if price moves beyond either wing", "Defined risk = debit"],
        source_label="tastytrade — Reverse Iron Condor",
    )


def _suggest_jade_lizard(snap, vix, iv_rank):
    """Short OTM Call Spread + Short OTM Put — credit, NO upside risk if
    total credit > call spread width. Pure premium harvester."""
    if vix is None or vix < 15 or vix > 22:
        return None
    if iv_rank is not None and iv_rank < 50:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Jade Lizard",
        bias="neutral",
        confidence="medium",
        rationale=f"VIX {vix:.1f} sweet spot + IV Rank ≥50 — no upside risk if credit > spread.",
        margin_estimate_inr=int(notional * 0.06) or None,
        suggested_legs=[
            f"Short OTM Call (≈{round(snap.spot * 1.02, 0)})",
            f"Long further-OTM Call (≈{round(snap.spot * 1.04, 0)})",
            f"Short OTM Put (≈{round(snap.spot * 0.97, 0)})",
        ],
        risk_notes=["Ensure call spread width < total credit (no upside risk)", "Downside risk uncapped below put strike"],
        source_label="tastytrade — Jade Lizard",
    )


def _suggest_big_lizard(snap, vix, iv_rank):
    """Big Lizard — Jade Lizard's straddled cousin. Sell ATM straddle +
    Long OTM call to cap upside. Higher credit, defined upside risk."""
    if vix is None or vix < 16 or vix > 22:
        return None
    if iv_rank is not None and iv_rank < 60:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Big Lizard",
        bias="neutral",
        confidence="low",
        rationale=f"VIX {vix:.1f} elevated + IV Rank ≥60 — higher-credit Jade variant.",
        margin_estimate_inr=int(notional * 0.08) or None,
        suggested_legs=[
            f"Short ATM Call (≈{round(snap.spot, 0)})",
            f"Long OTM Call (≈{round(snap.spot * 1.02, 0)})",
            f"Short ATM Put (≈{round(snap.spot, 0)})",
        ],
        risk_notes=["No upside risk if credit > call spread width", "Wide downside risk"],
        source_label="tastytrade — Big Lizard",
    )


def _suggest_broken_wing_butterfly(snap, vix, iv_rank):
    """Broken-Wing Butterfly — skewed butterfly with no risk on one side."""
    if vix is None or vix < 14:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    bias = "bullish" if snap.pcr_tag in ("bullish", "extreme_bullish") else "bearish"
    return StrategySuggestion(
        name="Broken-Wing Butterfly",
        bias=bias,
        confidence="medium",
        rationale=f"Skewed butterfly aligning with PCR {snap.pcr_oi:.2f} bias.",
        margin_estimate_inr=int(notional * 0.025) or None,
        suggested_legs=[
            f"Long ATM Call (≈{round(snap.spot, 0)})",
            f"Short 2× OTM Call (≈{round(snap.spot * 1.02, 0)})",
            f"Long further-OTM Call (≈{round(snap.spot * 1.05, 0)})",
        ],
        risk_notes=["No risk on one side if structured properly", "Pin risk at short strike"],
        source_label="OptionsTradingIQ — BWB",
    )


def _suggest_ratio_spread(snap, vix, iv_rank):
    """1 Long + 2 Short OTM — directional bet with small credit."""
    if vix is None or vix < 16:
        return None
    if iv_rank is not None and iv_rank < 50:
        return None
    bias = "bullish" if snap.pcr_tag in ("bullish", "extreme_bullish") else "bearish"
    notional = lot_value(snap.symbol, snap.spot)
    side = "Call" if bias == "bullish" else "Put"
    return StrategySuggestion(
        name=f"{side} Ratio Spread",
        bias=bias,
        confidence="low",
        rationale=f"PCR {snap.pcr_oi:.2f} {bias} + IV Rank ≥50 — collect credit with directional bias.",
        margin_estimate_inr=int(notional * 0.10) or None,
        suggested_legs=[
            f"Long ATM {side} (≈{round(snap.spot, 0)})",
            f"Short 2× OTM {side} (≈{round(snap.spot * (1.02 if bias == 'bullish' else 0.98), 0)})",
        ],
        risk_notes=["Unlimited risk beyond short strikes", "Best on view of capped move"],
        source_label="McMillan — Ratio Spread",
    )


def _suggest_diagonal_spread(snap, vix, iv_rank):
    """Diagonal — sell near-month + buy far-month at different strikes."""
    if vix is None or vix > 20:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    bias = "bullish" if snap.pcr_tag in ("bullish", "extreme_bullish") else "neutral"
    return StrategySuggestion(
        name="Diagonal Spread",
        bias=bias,
        confidence="low",
        rationale=f"VIX {vix:.1f} moderate — calendar + vertical combo for theta + delta.",
        margin_estimate_inr=int(notional * 0.025) or None,
        suggested_legs=[
            f"Sell weekly ATM (≈{round(snap.spot, 0)})",
            f"Buy monthly OTM (≈{round(snap.spot * 1.02, 0)})",
        ],
        risk_notes=["Wins on stable-to-trending price", "Far leg holds long vega"],
        source_label="McMillan — Diagonal Spread",
    )


def _suggest_double_calendar(snap, vix, iv_rank):
    """Two calendar spreads at OTM call + OTM put — wider profit zone."""
    if vix is None or vix > 16:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Double Calendar",
        bias="neutral",
        confidence="low",
        rationale=f"VIX {vix:.1f} low + neutral bias — wide-zone vega/theta combo.",
        margin_estimate_inr=int(notional * 0.03) or None,
        suggested_legs=[
            f"Sell weekly OTM Call ({round(snap.spot * 1.02, 0)}) / Buy monthly same strike",
            f"Sell weekly OTM Put ({round(snap.spot * 0.98, 0)}) / Buy monthly same strike",
        ],
        risk_notes=["Two wing profits if price drifts between strikes", "Defined risk = total debit"],
        source_label="ORATS — Double Calendar",
    )


def _suggest_christmas_tree(snap, vix, iv_rank):
    """Christmas Tree butterfly — 1-3-2 ratio for specific price target."""
    if vix is None or vix > 18:
        return None
    bias = "bullish" if snap.pcr_tag in ("bullish", "extreme_bullish") else "bearish"
    notional = lot_value(snap.symbol, snap.spot)
    side = "Call" if bias == "bullish" else "Put"
    return StrategySuggestion(
        name=f"{side} Christmas Tree",
        bias=bias,
        confidence="low",
        rationale="Pinpoint move to specific strike with low debit.",
        margin_estimate_inr=int(notional * 0.015) or None,
        suggested_legs=[
            f"Long 1× ATM {side}",
            f"Short 3× OTM {side}",
            f"Long 2× further-OTM {side}",
        ],
        risk_notes=["Profit zone is narrow", "Pin risk on short strikes"],
        source_label="McMillan — Christmas Tree",
    )


def _suggest_synthetic_long(snap, vix, iv_rank):
    """Synthetic Long Stock = Long Call + Short Put (same strike)."""
    if snap.pcr_tag not in ("bullish", "extreme_bullish"):
        return None
    if vix is None or vix < 14:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Synthetic Long",
        bias="bullish",
        confidence="low",
        rationale="PCR bullish — equivalent to long index with options margin.",
        margin_estimate_inr=int(notional * 0.10) or None,
        suggested_legs=[
            f"Long ATM Call (≈{round(snap.spot, 0)})",
            f"Short ATM Put (≈{round(snap.spot, 0)})",
        ],
        risk_notes=["P&L identical to long index", "Short put = unlimited downside"],
        source_label="McMillan — Synthetic Long",
    )


def _suggest_synthetic_short(snap, vix, iv_rank):
    """Synthetic Short Stock = Short Call + Long Put (same strike)."""
    if snap.pcr_tag not in ("bearish", "extreme_bearish"):
        return None
    if vix is None or vix < 14:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Synthetic Short",
        bias="bearish",
        confidence="low",
        rationale="PCR bearish — short the index synthetically.",
        margin_estimate_inr=int(notional * 0.10) or None,
        suggested_legs=[
            f"Short ATM Call (≈{round(snap.spot, 0)})",
            f"Long ATM Put (≈{round(snap.spot, 0)})",
        ],
        risk_notes=["P&L identical to short index", "Short call = unlimited upside"],
        source_label="McMillan — Synthetic Short",
    )


def _suggest_long_strap(snap, vix, iv_rank):
    """Long Strap = 2 Calls + 1 Put — bullish vol bet."""
    if vix is None or vix > 16:
        return None
    if snap.pcr_tag not in ("bullish", "extreme_bullish"):
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Long Strap",
        bias="bullish",
        confidence="low",
        rationale=f"PCR bullish + low VIX {vix:.1f} — leveraged bullish vol bet.",
        margin_estimate_inr=int(notional * 0.05) or None,
        suggested_legs=[
            f"Long 2× ATM Call (≈{round(snap.spot, 0)})",
            f"Long 1× ATM Put (≈{round(snap.spot, 0)})",
        ],
        risk_notes=["Profits more on UP-move than DOWN", "Time decay 3× the straddle"],
        source_label="McMillan — Long Strap",
    )


def _suggest_collar(snap, vix, iv_rank):
    """Collar = Long stock + Long OTM put + Short OTM call (zero-cost
    hedge for held positions)."""
    if vix is None or vix < 15:
        return None
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Zero-Cost Collar",
        bias="neutral",
        confidence="medium",
        rationale=f"VIX {vix:.1f} elevated — hedge long index without paying premium.",
        margin_estimate_inr=int(notional * 0.95) or None,
        suggested_legs=[
            "Hold long index/futures",
            f"Long OTM Put (≈{round(snap.spot * 0.97, 0)})",
            f"Short OTM Call (≈{round(snap.spot * 1.03, 0)})",
        ],
        risk_notes=["Caps both upside and downside", "Best for held long positions"],
        source_label="McMillan — Collar",
    )


def _suggest_box_spread(snap, vix, iv_rank):
    """Box Spread = interest-rate equivalent. Mostly used by institutions."""
    notional = lot_value(snap.symbol, snap.spot)
    return StrategySuggestion(
        name="Box Spread (interest-rate)",
        bias="neutral",
        confidence="low",
        rationale="Synthetic risk-free interest rate trade — institutional only.",
        margin_estimate_inr=int(notional * 0.05) or None,
        suggested_legs=[
            f"Bull Call Spread at ({round(snap.spot * 0.99, 0)}, {round(snap.spot * 1.01, 0)})",
            f"Bear Put Spread at ({round(snap.spot * 0.99, 0)}, {round(snap.spot * 1.01, 0)})",
        ],
        risk_notes=["Profit = risk-free rate × time", "Requires very tight execution"],
        source_label="McMillan — Box Spread",
    )


def suggest_strategies(
    snap: IndexSnapshot,
    vix: Optional[float] = None,
    iv_rank: Optional[float] = None,
) -> List[StrategySuggestion]:
    """Run all rule-based suggesters; return the ones that fire, sorted by
    confidence then bias-clarity. Always include a VIX-regime guidance
    note as the first element so the user sees the lens we're applying.
    """
    regime = classify_vix_regime(vix)
    guidance = _REGIME_GUIDANCE.get(regime, {})
    out: List[StrategySuggestion] = []

    # Top: regime guidance (descriptive, not a trade)
    out.append(StrategySuggestion(
        name=f"VIX Regime: {regime}",
        bias="neutral",
        confidence="high",
        rationale=(
            f"India VIX {vix:.1f}." if vix is not None else "India VIX unavailable."
        ) + (
            f" Long premium: {guidance.get('long_premium', 'n/a')} "
            f"Short premium: {guidance.get('short_premium', 'n/a')}"
            if guidance else ""
        ),
        source_label="Bajaj AMC / IFMC India VIX regime bands",
    ))

    for fn in (
        # Original 7 (PR-S19/S20)
        lambda: _suggest_iron_condor(snap, vix),
        lambda: _suggest_short_strangle(snap, vix, iv_rank),
        lambda: _suggest_long_calendar(snap, vix),
        lambda: _suggest_long_premium_debit(snap, vix),
        lambda: _suggest_max_pain_pull(snap),
        lambda: _suggest_gamma_scalping(snap, vix, iv_rank),
        lambda: _suggest_cvd_divergence_short(snap, vix),
        # O.1 new 23 templates
        lambda: _suggest_bull_call_spread(snap, vix, iv_rank),
        lambda: _suggest_bear_call_spread(snap, vix, iv_rank),
        lambda: _suggest_bull_put_spread(snap, vix, iv_rank),
        lambda: _suggest_bear_put_spread(snap, vix, iv_rank),
        lambda: _suggest_protective_put(snap, vix, iv_rank),
        lambda: _suggest_covered_call(snap, vix, iv_rank),
        lambda: _suggest_cash_secured_put(snap, vix, iv_rank),
        lambda: _suggest_long_straddle(snap, vix, iv_rank),
        lambda: _suggest_long_strangle(snap, vix, iv_rank),
        lambda: _suggest_long_iron_butterfly(snap, vix, iv_rank),
        lambda: _suggest_reverse_iron_condor(snap, vix, iv_rank),
        lambda: _suggest_jade_lizard(snap, vix, iv_rank),
        lambda: _suggest_big_lizard(snap, vix, iv_rank),
        lambda: _suggest_broken_wing_butterfly(snap, vix, iv_rank),
        lambda: _suggest_ratio_spread(snap, vix, iv_rank),
        lambda: _suggest_diagonal_spread(snap, vix, iv_rank),
        lambda: _suggest_double_calendar(snap, vix, iv_rank),
        lambda: _suggest_christmas_tree(snap, vix, iv_rank),
        lambda: _suggest_synthetic_long(snap, vix, iv_rank),
        lambda: _suggest_synthetic_short(snap, vix, iv_rank),
        lambda: _suggest_long_strap(snap, vix, iv_rank),
        lambda: _suggest_collar(snap, vix, iv_rank),
        lambda: _suggest_box_spread(snap, vix, iv_rank),
    ):
        try:
            s = fn()
            if s is not None:
                out.append(s)
        except Exception:
            continue

    # Sort: keep regime guidance first, then by confidence descending
    conf_rank = {"high": 0, "medium": 1, "low": 2}
    head, tail = out[:1], out[1:]
    tail.sort(key=lambda s: (conf_rank.get(s.confidence, 3), s.name))
    return head + tail
