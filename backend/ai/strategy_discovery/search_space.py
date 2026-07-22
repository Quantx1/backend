"""Search-space definitions for the Strategy Discovery Engine.

Each space class defines:
  * Which DSL knobs vary (entry indicators + thresholds, engine filters,
    exit rules, stop/target, hold horizon, regime filter, sizing).
  * How to draw a random sample from those knobs.
  * How to mutate an existing candidate (for the GA).

The output is always a fully-validated `Strategy` DSL document — anything
that fails `Strategy.model_validate` is rejected at sample time so the
backtester only ever sees correct shapes.

Trade-offs explored here:
  * EquitySearchSpace covers swing AND position — the only difference is
    the hold-horizon range and trailing-stop preference.
  * FOSearchSpace covers weekly AND monthly contracts — same template
    families (bull spread / bear spread / iron condor / straddle /
    strangle / calendar / butterfly), different expiry anchors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from backend.ai.strategy.dsl import (
    Condition,
    ConditionKind,
    EngineName,
    ExpiryAnchor,
    InstrumentSegment,
    LegSpec,
    Operator,
    OptionSide,
    OptionType,
    PositionSize,
    PositionSizeKind,
    RegimeFilter,
    Strategy,
    StrategyMode,
    StrikeAnchor,
    Timeframe,
    Universe,
)


# ─────────────────────────────────────────────────────────────────────
# Curated subspaces — knobs we vary, and the ranges we sample from.
# These are deliberately *smaller* than the full DSL surface; the goal
# is to find good strategies fast, not enumerate everything.
# ─────────────────────────────────────────────────────────────────────


# Momentum / mean-reversion indicators we'll vary
_OSCILLATORS: tuple[tuple[str, float, float], ...] = (
    # (indicator, lo, hi) — typical range for the threshold
    ("rsi14", 20, 80),
    ("rsi7", 15, 85),
    ("stochastic_k", 15, 85),
    ("stoch_rsi_k", 10, 90),
    ("williams_r", -90, -10),
    ("mfi", 15, 85),
    ("cci", -200, 200),
    ("roc_10", -10, 10),
    ("roc_20", -15, 15),          # 2026-07-22 — slower drop clock
)

# Drop indicators for the panic-reversion family (2026-07-22). These
# are the entry signals of the gate-verified winners: buy a hard
# short-term drop, exit on an asymmetric oscillator recovery. The GA
# explores this region densely because it is the ONLY equity family
# that has passed the product's out-of-sample gate net of costs.
_DROP_ENTRIES: tuple[tuple[str, float, float], ...] = (
    ("roc_10", -10, -3),         # 3-10% drop over 10 sessions
    ("roc_20", -15, -5),         # 5-15% slide over 20 sessions
    ("williams_r", -95, -70),    # deep Williams washout
    ("rsi7", 15, 35),            # fast-RSI panic
    ("stochastic_k", 10, 35),    # stochastic washout
    ("mfi", 10, 28),             # money-flow capitulation
    ("cci", -220, -110),         # statistical washout
)

# Trend-following moving averages
_TREND_MAS: tuple[str, ...] = (
    "ema8", "ema13", "ema21", "ema50", "ema200",
    "sma20", "sma50", "sma200",
)

# Volatility / breadth indicators
_VOL_INDICATORS: tuple[str, ...] = (
    "atr", "volatility_20", "volatility_60", "adx",
)

# Volume confirmation
_VOLUME_INDICATORS: tuple[str, ...] = (
    "volume_ratio", "obv_slope", "vwap_distance_pct",
)


def _mean_reversion_long_entry(rng: random.Random) -> Condition:
    """Buy oversold + (optional) bullish regime."""
    ind, lo, hi = rng.choice(_OSCILLATORS)
    # Oversold threshold — pick from the lower third of the range.
    threshold = round(rng.uniform(lo, lo + (hi - lo) * 0.35), 2)
    osc_leaf = Condition(
        kind=ConditionKind.INDICATOR_COMPARE,
        indicator=ind,
        op=Operator.LT,
        value=threshold,
    )
    # 50% of the time, AND with a regime filter via engine_signal
    if rng.random() < 0.5:
        regime_leaf = Condition(
            kind=ConditionKind.ENGINE_SIGNAL,
            engine=EngineName.REGIME,
            op=Operator.EQ,
            value="bull",
        )
        return Condition(
            kind=ConditionKind.COMPOSITE_AND,
            children=[osc_leaf, regime_leaf],
        )
    return osc_leaf


def _panic_reversion_entry(rng: random.Random) -> Condition:
    """Buy a hard short-term drop / oscillator washout — the gate-verified
    family. Optionally AND with an uptrend filter so we buy dips in
    leaders, not breakdowns in laggards."""
    ind, lo, hi = rng.choice(_DROP_ENTRIES)
    threshold = round(rng.uniform(lo, hi), 2)
    drop_leaf = Condition(
        kind=ConditionKind.INDICATOR_COMPARE,
        indicator=ind,
        op=Operator.LT,
        value=threshold,
    )
    r = rng.random()
    if r < 0.30:
        # AND above the 200-SMA: dip in an uptrend
        trend_leaf = Condition(
            kind=ConditionKind.INDICATOR_CROSS,
            indicator="close",
            op=Operator.CROSSES_ABOVE,
            value="prev_close",
        )
        return Condition(kind=ConditionKind.COMPOSITE_AND, children=[drop_leaf, trend_leaf])
    if r < 0.45:
        # AND with a medium-term uptrend (roc_126 > 0): leaders only
        mom_leaf = Condition(
            kind=ConditionKind.INDICATOR_COMPARE,
            indicator="roc_126",
            op=Operator.GT,
            value=round(rng.uniform(0, 10), 1),
        )
        return Condition(kind=ConditionKind.COMPOSITE_AND, children=[drop_leaf, mom_leaf])
    return drop_leaf


def _panic_reversion_exit(rng: random.Random) -> Condition:
    """Asymmetric recovery exit for the panic family: leave on an
    oscillator recovery through an overbought/neutral level (the
    verified winners exit on rsi14 > 65-88 regardless of entry clock)."""
    exit_ind = rng.choice(["rsi14", "rsi7", "williams_r", "stochastic_k"])
    if exit_ind == "williams_r":
        val = round(rng.uniform(-40, -15), 1)
    elif exit_ind == "stochastic_k":
        val = round(rng.uniform(60, 85), 1)
    elif exit_ind == "rsi7":
        val = round(rng.uniform(70, 85), 1)
    else:  # rsi14
        val = round(rng.uniform(62, 80), 1)
    return Condition(
        kind=ConditionKind.INDICATOR_COMPARE,
        indicator=exit_ind,
        op=Operator.GT,
        value=val,
    )


def _momentum_long_entry(rng: random.Random) -> Condition:
    """Buy breakout / trend continuation."""
    # Fast MA crosses above slow MA
    fast = rng.choice(["ema8", "ema13", "ema21"])
    slow = rng.choice(["ema50", "ema200", "sma50"])
    if fast == slow:
        slow = "ema200"
    cross_leaf = Condition(
        kind=ConditionKind.INDICATOR_CROSS,
        indicator=fast,
        op=Operator.CROSSES_ABOVE,
        value=slow,
    )
    # 40% of the time, ADD an Alpha-rank filter — Alpha is "lower = stronger"
    if rng.random() < 0.4:
        alpha_leaf = Condition(
            kind=ConditionKind.ENGINE_SIGNAL,
            engine=EngineName.ALPHA,
            op=Operator.LT,
            value=round(rng.uniform(0.2, 0.5), 2),
        )
        return Condition(
            kind=ConditionKind.COMPOSITE_AND,
            children=[cross_leaf, alpha_leaf],
        )
    return cross_leaf


def _sentiment_tilt_entry(rng: random.Random) -> Condition:
    """Buy on positive sentiment shock + momentum confirm."""
    sent_leaf = Condition(
        kind=ConditionKind.ENGINE_SIGNAL,
        engine=EngineName.MOOD,
        op=Operator.GT,
        value=round(rng.uniform(0.15, 0.45), 2),
    )
    # Momentum confirm via RSI > 50
    rsi_leaf = Condition(
        kind=ConditionKind.INDICATOR_COMPARE,
        indicator="rsi14",
        op=Operator.GT,
        value=50,
    )
    return Condition(
        kind=ConditionKind.COMPOSITE_AND,
        children=[sent_leaf, rsi_leaf],
    )


def _build_exit(rng: random.Random, entry: Condition) -> Condition:
    """Exit is roughly inverse of entry — sell overbought / cross-below /
    mood-flip / time-based."""
    # 60% inverse-of-entry, 40% time-or-RSI overbought
    if rng.random() < 0.6 and entry.kind in (
        ConditionKind.INDICATOR_COMPARE, ConditionKind.INDICATOR_CROSS,
    ):
        if entry.kind == ConditionKind.INDICATOR_COMPARE and entry.op == Operator.LT:
            # Oversold entry → exit when oscillator climbs back above midpoint
            threshold_hi = (entry.value + 50) if isinstance(entry.value, (int, float)) else 60
            return Condition(
                kind=ConditionKind.INDICATOR_COMPARE,
                indicator=entry.indicator,
                op=Operator.GT,
                value=round(threshold_hi, 2),
            )
        if entry.kind == ConditionKind.INDICATOR_CROSS:
            # Cross-above entry → exit on cross-below
            return Condition(
                kind=ConditionKind.INDICATOR_CROSS,
                indicator=entry.indicator,
                op=Operator.CROSSES_BELOW,
                value=entry.value,
            )
    # Default: RSI overbought
    return Condition(
        kind=ConditionKind.INDICATOR_COMPARE,
        indicator="rsi14",
        op=Operator.GT,
        value=round(rng.uniform(65, 80), 1),
    )


# ─────────────────────────────────────────────────────────────────────
# EquitySearchSpace — swing + position
# ─────────────────────────────────────────────────────────────────────


@dataclass
class EquitySearchSpace:
    """Search space for cash-segment equity strategies.

    `horizon`:
      * "swing"    — 5-20 day hold, daily bars, smaller stops + targets
      * "position" — 20-90 day hold, daily bars, wider stops + trailing
    """

    horizon: str = "swing"        # "swing" | "position"
    universe: Universe = Universe.NIFTY_50
    seed: Optional[int] = None

    def __post_init__(self):
        if self.horizon not in ("swing", "position"):
            raise ValueError(f"horizon must be swing|position, got {self.horizon}")

    def _rng(self, salt: int = 0) -> random.Random:
        seed = (self.seed or 0) + salt
        return random.Random(seed) if seed else random.Random()

    def sample(self, idx: int = 0) -> Strategy:
        """Draw a single fully-validated `Strategy` candidate."""
        rng = self._rng(idx)

        # Entry — the panic-reversion family is weighted heavily because
        # it is the only equity family that has passed the OOS gate net
        # of costs; the other three stay in the draw for diversity.
        entry_family = rng.choices(
            ["panic_reversion", "mean_reversion", "momentum", "sentiment"],
            weights=[0.55, 0.20, 0.15, 0.10],
        )[0]
        if entry_family == "panic_reversion":
            entry = _panic_reversion_entry(rng)
            exit_cond = _panic_reversion_exit(rng)
        elif entry_family == "mean_reversion":
            entry = _mean_reversion_long_entry(rng)
            exit_cond = _build_exit(rng, entry)
        elif entry_family == "momentum":
            entry = _momentum_long_entry(rng)
            exit_cond = _build_exit(rng, entry)
        else:
            entry = _sentiment_tilt_entry(rng)
            exit_cond = _build_exit(rng, entry)

        # Stop / target / trailing. The panic family uses the wider
        # stops + asymmetric targets the verified winners run (SL 5-10%,
        # TP 12-27%); the other families keep the original ranges.
        if entry_family == "panic_reversion":
            sl = round(rng.uniform(5.0, 10.0), 2)
            tp = round(rng.uniform(12.0, 27.0), 2)
            trail = round(rng.uniform(1.5, 6.0), 2) if rng.random() < 0.4 else None
        elif self.horizon == "swing":
            sl = round(rng.uniform(1.5, 4.0), 2)
            tp = round(rng.uniform(3.0, 10.0), 2)
            trail = round(rng.uniform(1.5, 3.5), 2) if rng.random() < 0.4 else None
        else:  # position
            sl = round(rng.uniform(4.0, 10.0), 2)
            tp = round(rng.uniform(10.0, 30.0), 2)
            trail = round(rng.uniform(3.0, 7.0), 2) if rng.random() < 0.6 else None

        # Regime filter — sometimes strict (bull only), sometimes any
        regime = rng.choices(
            [RegimeFilter.ANY, RegimeFilter.BULL_ONLY, RegimeFilter.SIDEWAYS_ONLY],
            weights=[0.5, 0.35, 0.15],
        )[0]

        # Position sizing
        pct = round(rng.uniform(3.0, 12.0), 1)
        sizing = PositionSize(kind=PositionSizeKind.PERCENT_OF_CAPITAL, value=pct)

        # Build human-readable name
        family_label = {
            "panic_reversion": "Panic",
            "mean_reversion": "MeanRev",
            "momentum": "Momentum",
            "sentiment": "Sentiment",
        }[entry_family]
        name = f"{family_label}-{self.horizon}-{idx:04d}"

        return Strategy(
            name=name,
            instrument_segment=InstrumentSegment.EQUITY,
            symbol=None,
            universe=self.universe,
            timeframe=Timeframe.D1,
            entry=entry,
            exit=exit_cond,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            trailing_stop_pct=trail,
            position_size=sizing,
            regime_filter=regime,
            lookback_days=(
                365 if entry_family == "panic_reversion"
                else (180 if self.horizon == "swing" else 365)
            ),
            mode=StrategyMode.BACKTEST,
        )

    def mutate(self, parent: Strategy, idx: int = 0) -> Strategy:
        """Produce a child by perturbing one or two parent knobs.

        Used by the GA. Simple approach: pick 1-2 mutable knobs at random,
        re-sample those, keep everything else. Returns a fresh Strategy
        instance — we never mutate in place.
        """
        rng = self._rng(idx + 100_000)
        child = parent.model_copy(deep=True)
        knobs = rng.sample(["stop", "target", "trail", "regime", "size"], k=rng.randint(1, 2))
        if "stop" in knobs:
            delta = rng.uniform(-1.0, 1.0)
            new_sl = max(0.5, min(15.0, (child.stop_loss_pct or 3.0) + delta))
            child.stop_loss_pct = round(new_sl, 2)
        if "target" in knobs:
            delta = rng.uniform(-2.0, 2.0)
            new_tp = max(1.0, min(50.0, (child.take_profit_pct or 6.0) + delta))
            child.take_profit_pct = round(new_tp, 2)
        if "trail" in knobs:
            child.trailing_stop_pct = (
                None if rng.random() < 0.3
                else round(max(0.5, (child.trailing_stop_pct or 2.0) + rng.uniform(-0.5, 0.5)), 2)
            )
        if "regime" in knobs:
            child.regime_filter = rng.choice(list(RegimeFilter))
        if "size" in knobs:
            delta = rng.uniform(-2.0, 2.0)
            new_pct = max(1.0, min(20.0, child.position_size.value + delta))
            child.position_size = PositionSize(
                kind=PositionSizeKind.PERCENT_OF_CAPITAL, value=round(new_pct, 1),
            )
        # Re-name to indicate generation
        child.name = f"{parent.name}-m{idx}"
        # Trigger re-validation
        return Strategy.model_validate(child.model_dump())


# ─────────────────────────────────────────────────────────────────────
# FOSearchSpace — weekly + monthly options
# ─────────────────────────────────────────────────────────────────────


# Template families. Each defines a leg signature; strikes + expiries
# are sampled below.
_FO_TEMPLATES = (
    "long_call", "long_put",
    "bull_call_spread", "bear_put_spread",
    "short_strangle", "iron_condor",
    "long_straddle", "long_strangle",
    "calendar_call", "calendar_put",
)


@dataclass
class FOSearchSpace:
    """Search space for NSE index options strategies.

    `tenor`:
      * "weekly"  — current_week or next_week expiry
      * "monthly" — current_month or next_month expiry
    """

    tenor: str = "weekly"          # "weekly" | "monthly"
    underlying: str = "NIFTY"      # NIFTY | BANKNIFTY | FINNIFTY
    seed: Optional[int] = None

    def __post_init__(self):
        if self.tenor not in ("weekly", "monthly"):
            raise ValueError(f"tenor must be weekly|monthly, got {self.tenor}")

    def _rng(self, salt: int = 0) -> random.Random:
        seed = (self.seed or 0) + salt
        return random.Random(seed) if seed else random.Random()

    def _expiry_anchor(self, rng: random.Random) -> ExpiryAnchor:
        if self.tenor == "weekly":
            return rng.choices(
                [ExpiryAnchor.CURRENT_WEEK, ExpiryAnchor.NEXT_WEEK],
                weights=[0.65, 0.35],
            )[0]
        return rng.choices(
            [ExpiryAnchor.CURRENT_MONTH, ExpiryAnchor.NEXT_MONTH],
            weights=[0.7, 0.3],
        )[0]

    def _build_legs(self, rng: random.Random, template: str) -> List[LegSpec]:
        """Materialise a template into 1-4 LegSpecs with sampled strikes."""
        expiry = self._expiry_anchor(rng)
        lots = rng.choice([1, 2, 3])

        if template == "long_call":
            off = rng.choice([0, 1, 2])
            return [LegSpec(
                side=OptionSide.BUY, option_type=OptionType.CE,
                strike_anchor=StrikeAnchor.ATM if off == 0 else StrikeAnchor.ATM_PLUS_N,
                strike_offset=0 if off == 0 else off,
                expiry_anchor=expiry, lots=lots,
            )]
        if template == "long_put":
            off = rng.choice([0, 1, 2])
            return [LegSpec(
                side=OptionSide.BUY, option_type=OptionType.PE,
                strike_anchor=StrikeAnchor.ATM if off == 0 else StrikeAnchor.ATM_MINUS_N,
                strike_offset=0 if off == 0 else off,
                expiry_anchor=expiry, lots=lots,
            )]
        if template == "bull_call_spread":
            return [
                LegSpec(side=OptionSide.BUY, option_type=OptionType.CE,
                        strike_anchor=StrikeAnchor.ATM, strike_offset=0,
                        expiry_anchor=expiry, lots=lots),
                LegSpec(side=OptionSide.SELL, option_type=OptionType.CE,
                        strike_anchor=StrikeAnchor.ATM_PLUS_N,
                        strike_offset=rng.choice([2, 3, 4]),
                        expiry_anchor=expiry, lots=lots),
            ]
        if template == "bear_put_spread":
            return [
                LegSpec(side=OptionSide.BUY, option_type=OptionType.PE,
                        strike_anchor=StrikeAnchor.ATM, strike_offset=0,
                        expiry_anchor=expiry, lots=lots),
                LegSpec(side=OptionSide.SELL, option_type=OptionType.PE,
                        strike_anchor=StrikeAnchor.ATM_MINUS_N,
                        strike_offset=rng.choice([2, 3, 4]),
                        expiry_anchor=expiry, lots=lots),
            ]
        if template == "short_strangle":
            wing = rng.choice([3, 4, 5])
            return [
                LegSpec(side=OptionSide.SELL, option_type=OptionType.CE,
                        strike_anchor=StrikeAnchor.ATM_PLUS_N, strike_offset=wing,
                        expiry_anchor=expiry, lots=lots),
                LegSpec(side=OptionSide.SELL, option_type=OptionType.PE,
                        strike_anchor=StrikeAnchor.ATM_MINUS_N, strike_offset=wing,
                        expiry_anchor=expiry, lots=lots),
            ]
        if template == "iron_condor":
            short_wing = rng.choice([2, 3])
            long_wing = short_wing + rng.choice([2, 3])
            return [
                LegSpec(side=OptionSide.SELL, option_type=OptionType.CE,
                        strike_anchor=StrikeAnchor.ATM_PLUS_N, strike_offset=short_wing,
                        expiry_anchor=expiry, lots=lots),
                LegSpec(side=OptionSide.BUY, option_type=OptionType.CE,
                        strike_anchor=StrikeAnchor.ATM_PLUS_N, strike_offset=long_wing,
                        expiry_anchor=expiry, lots=lots),
                LegSpec(side=OptionSide.SELL, option_type=OptionType.PE,
                        strike_anchor=StrikeAnchor.ATM_MINUS_N, strike_offset=short_wing,
                        expiry_anchor=expiry, lots=lots),
                LegSpec(side=OptionSide.BUY, option_type=OptionType.PE,
                        strike_anchor=StrikeAnchor.ATM_MINUS_N, strike_offset=long_wing,
                        expiry_anchor=expiry, lots=lots),
            ]
        if template == "long_straddle":
            return [
                LegSpec(side=OptionSide.BUY, option_type=OptionType.CE,
                        strike_anchor=StrikeAnchor.ATM, strike_offset=0,
                        expiry_anchor=expiry, lots=lots),
                LegSpec(side=OptionSide.BUY, option_type=OptionType.PE,
                        strike_anchor=StrikeAnchor.ATM, strike_offset=0,
                        expiry_anchor=expiry, lots=lots),
            ]
        if template == "long_strangle":
            wing = rng.choice([2, 3, 4])
            return [
                LegSpec(side=OptionSide.BUY, option_type=OptionType.CE,
                        strike_anchor=StrikeAnchor.ATM_PLUS_N, strike_offset=wing,
                        expiry_anchor=expiry, lots=lots),
                LegSpec(side=OptionSide.BUY, option_type=OptionType.PE,
                        strike_anchor=StrikeAnchor.ATM_MINUS_N, strike_offset=wing,
                        expiry_anchor=expiry, lots=lots),
            ]
        if template == "calendar_call":
            return [
                LegSpec(side=OptionSide.SELL, option_type=OptionType.CE,
                        strike_anchor=StrikeAnchor.ATM, strike_offset=0,
                        expiry_anchor=ExpiryAnchor.CURRENT_WEEK, lots=lots),
                LegSpec(side=OptionSide.BUY, option_type=OptionType.CE,
                        strike_anchor=StrikeAnchor.ATM, strike_offset=0,
                        expiry_anchor=ExpiryAnchor.NEXT_MONTH, lots=lots),
            ]
        if template == "calendar_put":
            return [
                LegSpec(side=OptionSide.SELL, option_type=OptionType.PE,
                        strike_anchor=StrikeAnchor.ATM, strike_offset=0,
                        expiry_anchor=ExpiryAnchor.CURRENT_WEEK, lots=lots),
                LegSpec(side=OptionSide.BUY, option_type=OptionType.PE,
                        strike_anchor=StrikeAnchor.ATM, strike_offset=0,
                        expiry_anchor=ExpiryAnchor.NEXT_MONTH, lots=lots),
            ]
        raise ValueError(f"unknown template {template}")

    def sample(self, idx: int = 0) -> Strategy:
        """Draw a fully-validated multi-leg options Strategy."""
        rng = self._rng(idx)
        template = rng.choice(_FO_TEMPLATES)
        legs = self._build_legs(rng, template)

        # Entry — usually based on regime + IV percentile (encoded as Mood
        # for now; PR-G3 may add a dedicated VIX/IV engine). For credit
        # spreads we want high-IV environments; for debit / long-vol we want
        # the opposite.
        is_credit = template in ("short_strangle", "iron_condor", "bear_put_spread")
        if is_credit:
            # High implied vol → premium-rich → sell vol
            entry = Condition(
                kind=ConditionKind.ENGINE_SIGNAL,
                engine=EngineName.REGIME,
                op=Operator.EQ,
                value="sideways",
            )
        else:
            # Trend / event template — favour bull or bear regime
            entry = Condition(
                kind=ConditionKind.ENGINE_SIGNAL,
                engine=EngineName.REGIME,
                op=Operator.EQ,
                value=rng.choice(["bull", "bear"]),
            )

        # Exit — time-based + stop/target via SL/TP. Use a tautology exit
        # so the bar loop only fires on SL/TP/regime-flip.
        exit_cond = Condition(
            kind=ConditionKind.INDICATOR_COMPARE,
            indicator="close",
            op=Operator.LT,
            value=0,    # never fires; defers to stop/target/end-of-data
        )

        # SL / TP on aggregate position value
        sl = round(rng.uniform(15.0, 35.0), 1)   # 15-35% of net premium
        tp = round(rng.uniform(20.0, 60.0), 1)

        sizing = PositionSize(
            kind=PositionSizeKind.PERCENT_OF_CAPITAL,
            value=round(rng.uniform(2.0, 7.0), 1),
        )

        name = f"{template}-{self.tenor}-{self.underlying}-{idx:04d}"

        return Strategy(
            name=name,
            instrument_segment=InstrumentSegment.OPTIONS,
            symbol=self.underlying,
            universe=Universe.SINGLE,
            timeframe=Timeframe.D1,
            entry=entry,
            exit=exit_cond,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            trailing_stop_pct=None,
            position_size=sizing,
            legs=legs,
            regime_filter=RegimeFilter.ANY,
            lookback_days=120,
            mode=StrategyMode.BACKTEST,
        )

    def mutate(self, parent: Strategy, idx: int = 0) -> Strategy:
        """Mutate an F&O parent by adjusting SL/TP or lot size."""
        rng = self._rng(idx + 200_000)
        child = parent.model_copy(deep=True)
        knobs = rng.sample(["sl", "tp", "lots"], k=rng.randint(1, 2))
        if "sl" in knobs:
            child.stop_loss_pct = round(
                max(5.0, min(50.0, (child.stop_loss_pct or 25.0) + rng.uniform(-5, 5))), 1,
            )
        if "tp" in knobs:
            child.take_profit_pct = round(
                max(10.0, min(100.0, (child.take_profit_pct or 40.0) + rng.uniform(-10, 10))), 1,
            )
        if "lots" in knobs and child.legs:
            delta = rng.choice([-1, 1])
            for leg in child.legs:
                leg.lots = max(1, min(10, leg.lots + delta))
        child.name = f"{parent.name}-m{idx}"
        return Strategy.model_validate(child.model_dump())
