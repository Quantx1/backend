"""
Strategy DSL — Pydantic schema per v2 design spec § 7.1.

A Strategy is a constrained JSON document. The DSL deliberately rejects
arbitrary code: every indicator must come from the closed registry, every
engine reference must be in the whitelist, every operator must be from
the enum. This is the safety boundary that lets users author strategies
via the Studio NL→DSL agent without exposing dynamic code execution.

Round-trip guarantee: ``Strategy.model_dump_json()`` must round-trip
through ``Strategy.model_validate_json()`` identically. We store the
DSL in the ``user_strategies.dsl`` JSONB column with no transformation.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────
# Enums — every string field that takes a fixed set of values goes
# through an Enum so Pydantic rejects typos / injection attempts at
# validation time.
# ─────────────────────────────────────────────────────────────────────


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class Universe(str, Enum):
    """How wide the strategy fishes. ``single`` requires Strategy.symbol."""
    SINGLE = "single"
    NIFTY_50 = "nifty50"
    NIFTY_100 = "nifty100"
    NIFTY_500 = "nifty500"
    SECTOR_IT = "sector:IT"
    SECTOR_BANK = "sector:BANK"
    SECTOR_AUTO = "sector:AUTO"
    SECTOR_PHARMA = "sector:PHARMA"
    SECTOR_FMCG = "sector:FMCG"
    SECTOR_METAL = "sector:METAL"
    SECTOR_ENERGY = "sector:ENERGY"
    SECTOR_INFRA = "sector:INFRA"


class StrategyMode(str, Enum):
    """Strategy execution mode — drives the state machine in registry.py."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class RegimeFilter(str, Enum):
    BULL_ONLY = "bull_only"
    BEAR_ONLY = "bear_only"
    SIDEWAYS_ONLY = "sideways_only"
    ANY = "any"


class ConditionKind(str, Enum):
    """The shape of a single rule. Composite nodes nest children."""
    INDICATOR_COMPARE = "indicator_compare"
    INDICATOR_CROSS = "indicator_cross"
    ENGINE_SIGNAL = "engine_signal"
    COMPOSITE_AND = "composite_and"
    COMPOSITE_OR = "composite_or"


class Operator(str, Enum):
    LT = "<"
    GT = ">"
    LTE = "<="
    GTE = ">="
    EQ = "=="
    NE = "!="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    BETWEEN = "between"
    OUTSIDE = "outside"


class EngineName(str, Enum):
    """3 engines whose models are PROD-promoted and actually emit signals.

    Trim history:
      - 2026-05-25 (PR-M cut): ``Horizon`` removed (TimesFM dropped from v1)
      - 2026-05-25 (engines cleanup): ``Pulse``, ``Vision``, ``Verdict``
        removed. They had no PROD model behind them so engine_signal
        conditions would never fire — better to fail-loud at validate time
        than silently no-op at runtime.

    What stays:
      - Alpha   → Qlib alpha158 v4 (cross-sectional ranker, lower = stronger)
      - Mood    → FinBERT-India v1 (sentiment_5d_mean in [-1, 1])
      - Regime  → HMM regime_hmm v20 ('bull' | 'sideways' | 'bear')

    Re-add when their backing model reaches PROD post-v1.
    AutoPilot is the executor, not a signal source — intentionally absent.
    """
    ALPHA = "Alpha"
    MOOD = "Mood"
    REGIME = "Regime"


class PositionSizeKind(str, Enum):
    PERCENT_OF_CAPITAL = "percent_of_capital"
    FIXED_QTY = "fixed_qty"
    RISK_BASED = "risk_based"


# ─────────────────────────────────────────────────────────────────────
# Multi-leg options support (PR-J)
# ─────────────────────────────────────────────────────────────────────


class InstrumentSegment(str, Enum):
    """Top-level instrument family.

    ``EQUITY`` (default) means single-instrument cash-segment trades.
    ``OPTIONS`` means a multi-leg options position — Strategy.legs becomes
    required, ``stop_loss_pct``/``take_profit_pct`` then apply to aggregate
    position value (not per-leg).
    """
    EQUITY = "EQUITY"
    OPTIONS = "OPTIONS"


class OptionSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OptionType(str, Enum):
    CE = "CE"
    PE = "PE"


class StrikeAnchor(str, Enum):
    """How a leg's strike is selected relative to spot.

    ``ATM``        — at-the-money (closest interval to spot).
    ``ATM_PLUS_N`` — ATM + N * strike_interval (use ``strike_offset`` field).
    ``ATM_MINUS_N``— ATM − N * strike_interval.
    ``OTM_DELTA``  — strike whose absolute delta ≈ ``strike_offset`` (e.g. 0.25).
    ``PCT_OFFSET`` — strike at spot × (1 + strike_offset/100).
    """
    ATM = "ATM"
    ATM_PLUS_N = "ATM+N"
    ATM_MINUS_N = "ATM-N"
    OTM_DELTA = "OTM_DELTA"
    PCT_OFFSET = "PCT_OFFSET"


class ExpiryAnchor(str, Enum):
    """Symbolic expiry — resolved to a concrete date at entry time."""
    CURRENT_WEEK = "current_week"
    NEXT_WEEK = "next_week"
    CURRENT_MONTH = "current_month"
    NEXT_MONTH = "next_month"


class LegSpec(BaseModel):
    """One leg of a multi-leg options position.

    Declarative — strike is expressed as an anchor + offset (e.g. ATM+2)
    rather than an absolute price, so the same template works at any
    spot level. The leg resolver materializes (anchor, offset, expiry)
    into concrete (strike, expiry_date) at order placement.
    """
    side: OptionSide
    option_type: OptionType
    strike_anchor: StrikeAnchor
    strike_offset: float = Field(default=0, description=(
        "Interpretation depends on strike_anchor: "
        "ATM+N / ATM-N → integer N (number of strike intervals away from ATM). "
        "OTM_DELTA → absolute delta target in (0, 0.5). "
        "PCT_OFFSET → percent above spot (negative = below). "
        "ATM → ignored, must be 0."
    ))
    expiry: ExpiryAnchor = ExpiryAnchor.CURRENT_WEEK
    qty_lots: int = Field(default=1, ge=1, le=100)

    @model_validator(mode="after")
    def _offset_matches_anchor(self):
        a = self.strike_anchor
        o = self.strike_offset
        if a == StrikeAnchor.ATM:
            if o != 0:
                raise ValueError("strike_anchor=ATM requires strike_offset=0")
        elif a in (StrikeAnchor.ATM_PLUS_N, StrikeAnchor.ATM_MINUS_N):
            if o <= 0 or o != int(o):
                raise ValueError(
                    f"strike_anchor={a.value} requires a positive integer "
                    f"strike_offset (got {o})"
                )
            if o > 20:
                raise ValueError(
                    f"strike_offset {o} too far OTM — max 20 intervals "
                    f"(prevents unrealistic templates)"
                )
        elif a == StrikeAnchor.OTM_DELTA:
            if not (0 < o < 0.5):
                raise ValueError(
                    f"strike_anchor=OTM_DELTA requires strike_offset in (0, 0.5), "
                    f"got {o}"
                )
        elif a == StrikeAnchor.PCT_OFFSET:
            if not (-50 < o < 50) or o == 0:
                raise ValueError(
                    f"strike_anchor=PCT_OFFSET requires strike_offset in "
                    f"(-50, 50)\\{{0}}, got {o}"
                )
        return self


# ─────────────────────────────────────────────────────────────────────
# Indicator registry — the closed set. Adding a new indicator means:
#   (1) add to INDICATOR_REGISTRY here, and
#   (2) implement it in indicators.py
# Anything not in this set is rejected at validate time, preventing
# user-strategy injection of arbitrary attribute names.
# ─────────────────────────────────────────────────────────────────────

INDICATOR_REGISTRY: tuple[str, ...] = (
    # Momentum
    "rsi7", "rsi9", "rsi14",
    "stochastic_k", "stochastic_d",
    "stoch_rsi_k", "stoch_rsi_d",            # PR-FEATURES — StochRSI for sharper extremes
    "williams_r",
    "mfi",
    "cci",
    "roc_10", "roc_20",                       # PR-FEATURES — rate-of-change momentum
    # Trend
    "ema5", "ema8", "ema13", "ema21", "ema50", "ema100", "ema200",
    "sma10", "sma20", "sma50", "sma100", "sma200",
    "macd", "macd_signal", "macd_hist",
    "adx",
    "di_plus", "di_minus",                    # PR-FEATURES — directional strength components
    "supertrend",
    "psar",
    # Volatility / bands
    "atr",
    "bbands_upper", "bbands_middle", "bbands_lower",
    "volatility_20", "volatility_60",         # PR-FEATURES — realized vol short + long
    "volatility_regime",                       # PR-FEATURES — 0/1/2 = low/normal/high
    # Volume / flow
    "vwap",
    "obv",
    "obv_slope",                              # PR-FEATURES — accumulation vs distribution
    "volume_sma20",
    "volume_ratio",                           # PR-FEATURES — current vol / SMA20
    "volume_delta_20",                        # PR-FEATURES — signed volume sum
    "vwap_distance_pct",                      # PR-FEATURES — % distance from VWAP
    # Session features (PR-FEATURES — timestamp-derived)
    "minutes_since_open",
    "session_progress",
    "is_first_hour",
    "is_last_hour",
    # Price refs
    "close", "open", "high", "low",
    "prev_close", "prev_high", "prev_low",
    # Pivots (classic floor-trader, prior-bar derived)
    "pivot_point", "pivot_r1", "pivot_s1", "pivot_r2", "pivot_s2", "pivot_r3", "pivot_s3",
    # Donchian channel (prior-N-bar extremes — turtle breakouts)
    "donchian_high_20", "donchian_low_20", "donchian_high_55", "donchian_low_55",
    # Candle patterns (boolean — value 1=present, 0=absent)
    "pattern_doji", "pattern_hammer", "pattern_inverted_hammer",
    "pattern_bullish_engulfing", "pattern_bearish_engulfing",
    "pattern_morning_star", "pattern_evening_star",
    "pattern_bullish_harami", "pattern_bearish_harami",
    "pattern_three_white_soldiers", "pattern_three_black_crows",
)


# ─────────────────────────────────────────────────────────────────────
# Condition — recursive DSL node
# ─────────────────────────────────────────────────────────────────────


class Condition(BaseModel):
    """Single rule node. Recursive via children for composite_and/or."""

    kind: ConditionKind
    indicator: Optional[str] = None
    op: Optional[Operator] = None
    value: Optional[Union[float, int, str, List[float]]] = None
    engine: Optional[EngineName] = None
    children: Optional[List["Condition"]] = None

    @field_validator("indicator")
    @classmethod
    def _indicator_in_registry(cls, v):
        if v is None:
            return v
        if v not in INDICATOR_REGISTRY:
            raise ValueError(
                f"unknown indicator '{v}'. Allowed: {INDICATOR_REGISTRY}"
            )
        return v

    @model_validator(mode="after")
    def _shape_matches_kind(self):
        kind = self.kind
        if kind in (ConditionKind.COMPOSITE_AND, ConditionKind.COMPOSITE_OR):
            if not self.children or len(self.children) < 2:
                raise ValueError(
                    f"{kind.value} requires >= 2 children, got "
                    f"{len(self.children) if self.children else 0}",
                )
            if any([self.indicator, self.op, self.value, self.engine]):
                raise ValueError(
                    f"{kind.value} must not set indicator/op/value/engine — "
                    f"those belong to leaf children",
                )
            return self
        # Leaf nodes: children must be absent.
        if self.children:
            raise ValueError(
                f"kind {kind.value} is a leaf — must not have children",
            )
        if kind == ConditionKind.INDICATOR_COMPARE:
            if self.indicator is None or self.op is None or self.value is None:
                raise ValueError(
                    "indicator_compare requires indicator + op + value",
                )
        elif kind == ConditionKind.INDICATOR_CROSS:
            if self.indicator is None or self.op is None or self.value is None:
                raise ValueError(
                    "indicator_cross requires indicator + op + value "
                    "(value is the second indicator name as a string)",
                )
            if self.op not in (Operator.CROSSES_ABOVE, Operator.CROSSES_BELOW):
                raise ValueError(
                    f"indicator_cross op must be crosses_above|crosses_below, "
                    f"got {self.op.value}",
                )
            if not isinstance(self.value, str) or self.value not in INDICATOR_REGISTRY:
                raise ValueError(
                    f"indicator_cross value must be an indicator name from "
                    f"the registry, got '{self.value}'",
                )
        elif kind == ConditionKind.ENGINE_SIGNAL:
            if self.engine is None or self.op is None or self.value is None:
                raise ValueError(
                    "engine_signal requires engine + op + value",
                )
        # Validate between/outside values are 2-element numeric arrays.
        if self.op in (Operator.BETWEEN, Operator.OUTSIDE):
            v = self.value
            if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)):
                raise ValueError(
                    f"op {self.op.value} requires value=[lo, hi] of two numbers",
                )
            lo, hi = v
            if lo >= hi:
                raise ValueError(
                    f"op {self.op.value} requires lo < hi, got [{lo}, {hi}]",
                )
        return self


Condition.model_rebuild()


# ─────────────────────────────────────────────────────────────────────
# Position sizing
# ─────────────────────────────────────────────────────────────────────


class PositionSize(BaseModel):
    """How big the trade should be on each entry."""
    kind: PositionSizeKind
    value: float = Field(gt=0)

    @model_validator(mode="after")
    def _value_bounds(self):
        if self.kind == PositionSizeKind.PERCENT_OF_CAPITAL:
            if not 0 < self.value <= 100:
                raise ValueError(
                    f"percent_of_capital value must be in (0, 100], got {self.value}",
                )
        elif self.kind == PositionSizeKind.RISK_BASED:
            if not 0 < self.value <= 5:
                raise ValueError(
                    f"risk_based value (% of capital per trade) must be in "
                    f"(0, 5], got {self.value}",
                )
        return self


# ─────────────────────────────────────────────────────────────────────
# Strategy — the root document
# ─────────────────────────────────────────────────────────────────────


class Strategy(BaseModel):
    """The root DSL document. Stored verbatim in user_strategies.dsl JSONB.

    Supports two segments via ``instrument_segment``:

    * ``EQUITY`` (default) — single instrument, cash segment. ``legs`` must
      be empty. Existing equity templates require no changes.
    * ``OPTIONS`` — multi-leg index/stock options. ``legs`` must contain
      1-4 entries. Stop-loss / take-profit apply to the aggregate position
      value (sum of leg mark prices × side), not to any single leg.
    """

    name: str = Field(min_length=1, max_length=120)
    instrument_segment: InstrumentSegment = InstrumentSegment.EQUITY
    symbol: Optional[str] = Field(default=None, max_length=20)
    universe: Universe
    timeframe: Timeframe
    entry: Condition
    exit: Condition
    stop_loss_pct: Optional[float] = Field(default=None, gt=0, lt=100)
    take_profit_pct: Optional[float] = Field(default=None, gt=0, lt=1000)
    trailing_stop_pct: Optional[float] = Field(default=None, gt=0, lt=100)
    position_size: PositionSize
    legs: Optional[List[LegSpec]] = Field(default=None, description=(
        "Required when instrument_segment=OPTIONS. 1-4 legs. For EQUITY, "
        "must be None or empty."
    ))
    regime_filter: RegimeFilter = RegimeFilter.ANY
    lookback_days: int = Field(default=90, ge=10, le=730)
    mode: StrategyMode = StrategyMode.BACKTEST
    # Auto square-off — force-exit any open intraday position at this IST
    # clock time (e.g. "15:09"), the uTrade-style end-of-day flatten. Intraday
    # only; daily+ strategies already settle on the 15:30 close tick.
    square_off_time: Optional[str] = Field(
        default=None,
        description="IST HH:MM to force-exit open intraday positions, e.g. '15:09'.",
    )

    @model_validator(mode="after")
    def _single_universe_requires_symbol(self):
        if self.universe == Universe.SINGLE and not self.symbol:
            raise ValueError(
                "universe='single' requires symbol to be set",
            )
        if self.universe != Universe.SINGLE and self.symbol:
            # symbol is ignored for non-single universes; strip to avoid misleading
            object.__setattr__(self, "symbol", None)
        return self

    @model_validator(mode="after")
    def _intraday_timeframe_must_have_stops(self):
        intraday = self.timeframe in (
            Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1,
        )
        if intraday and self.stop_loss_pct is None:
            raise ValueError(
                f"intraday timeframe {self.timeframe.value} requires stop_loss_pct — "
                f"never run an intraday strategy without a hard stop",
            )
        return self

    @model_validator(mode="after")
    def _validate_square_off_time(self):
        if self.square_off_time is None:
            return self
        import re as _re
        m = _re.fullmatch(r"([01]?\d|2[0-3]):([0-5]?\d)", self.square_off_time.strip())
        if not m:
            raise ValueError(
                f"square_off_time must be IST HH:MM (00:00-23:59), got "
                f"'{self.square_off_time}'",
            )
        # Normalize to zero-padded HH:MM ("9:9" → "09:09").
        object.__setattr__(self, "square_off_time", f"{int(m.group(1)):02d}:{int(m.group(2)):02d}")
        # Square-off is an intraday concept; daily+ strategies settle on close.
        intraday = self.timeframe in (
            Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1,
        )
        if not intraday:
            raise ValueError(
                "square_off_time only applies to intraday timeframes "
                "(1m/5m/15m/30m/1h) — daily+ strategies settle on the close",
            )
        return self

    @model_validator(mode="after")
    def _legs_match_segment(self):
        if self.instrument_segment == InstrumentSegment.OPTIONS:
            if not self.legs:
                raise ValueError(
                    "instrument_segment=OPTIONS requires legs (1-4 entries)",
                )
            if len(self.legs) > 4:
                raise ValueError(
                    f"OPTIONS strategy supports at most 4 legs, got {len(self.legs)}",
                )
            # OPTIONS strategies must target a known underlier — index
            # options are single-underlying instruments.
            if self.universe != Universe.SINGLE:
                raise ValueError(
                    "OPTIONS strategies require universe='single' and a "
                    "symbol (NIFTY / BANKNIFTY / FINNIFTY / etc.) — they "
                    "trade a specific underlying",
                )
            # HIGH #7 (2026-05-31) — Calendar spreads (multi-expiry legs)
            # NOW SUPPORTED. Previously this validator hard-blocked the
            # 8+ multi-expiry templates we ship in fno_scanner.strategies
            # (Calendar, Diagonal, Double Calendar, Christmas Tree, etc.).
            # The pricing layer + payoff calculator handle mixed expiries
            # correctly; the only restriction was here in the validator.
            #
            # Guardrail: cap to 4 distinct expiry anchors per strategy
            # so we don't ship arbitrary-leg ratio bombs.
            anchors = {leg.expiry for leg in self.legs}
            if len(anchors) > 4:
                raise ValueError(
                    f"at most 4 distinct expiry anchors per multi-leg strategy "
                    f"(got {len(anchors)}: {sorted(a.value for a in anchors)})",
                )
        else:  # EQUITY
            if self.legs:
                raise ValueError(
                    "instrument_segment=EQUITY must not set legs — those "
                    "are only for OPTIONS strategies",
                )
        return self


__all__ = [
    "Strategy",
    "Condition",
    "PositionSize",
    "LegSpec",
    "Timeframe",
    "Universe",
    "StrategyMode",
    "RegimeFilter",
    "ConditionKind",
    "Operator",
    "EngineName",
    "PositionSizeKind",
    "InstrumentSegment",
    "OptionSide",
    "OptionType",
    "StrikeAnchor",
    "ExpiryAnchor",
    "INDICATOR_REGISTRY",
]
