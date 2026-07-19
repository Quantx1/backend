"""
================================================================================
QUANT X - CUTTING-EDGE RISK MANAGEMENT ENGINE
================================================================================
5-Layer Risk Management for Maximum Returns with Minimum Drawdown
Handles: Equity, Futures, Options with proper margin & lot calculations
================================================================================
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================


class RiskLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class Segment(Enum):
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class RiskProfile:
    name: str
    risk_per_trade: float  # % of capital
    max_positions: int
    min_confidence: float
    max_sector_exposure: float  # % of capital
    max_daily_loss: float  # % of capital
    max_weekly_loss: float
    max_monthly_loss: float
    position_size_factor: float
    allow_pyramiding: bool
    trailing_sl_enabled: bool


@dataclass
class MarketCondition:
    vix: float
    nifty_change: float
    fii_net: float
    advance_decline_ratio: float
    pcr: float  # Put-Call Ratio


@dataclass
class Signal:
    symbol: str
    segment: Segment
    direction: Direction
    confidence: float
    entry_price: float
    stop_loss: float
    target: float
    lot_size: int = 1
    expiry: Optional[date] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None  # CE or PE


@dataclass
class PositionSize:
    quantity: int
    lots: int
    position_value: float
    margin_required: float
    risk_amount: float
    risk_percent: float
    approved: bool
    rejection_reason: Optional[str] = None

# ============================================================================
# RISK PROFILES
# ============================================================================


RISK_PROFILES = {
    "conservative": RiskProfile(
        name="Conservative",
        risk_per_trade=2.0,
        max_positions=3,
        min_confidence=20,
        max_sector_exposure=30,
        max_daily_loss=3.0,
        max_weekly_loss=5.0,
        max_monthly_loss=10.0,
        position_size_factor=0.7,
        allow_pyramiding=False,
        trailing_sl_enabled=True
    ),
    "moderate": RiskProfile(
        name="Moderate",
        risk_per_trade=3.0,
        max_positions=5,
        min_confidence=15,
        max_sector_exposure=40,
        max_daily_loss=5.0,
        max_weekly_loss=8.0,
        max_monthly_loss=15.0,
        position_size_factor=1.0,
        allow_pyramiding=True,
        trailing_sl_enabled=True
    ),
    "aggressive": RiskProfile(
        name="Aggressive",
        risk_per_trade=5.0,
        max_positions=8,
        min_confidence=10,
        max_sector_exposure=50,
        max_daily_loss=7.0,
        max_weekly_loss=12.0,
        max_monthly_loss=20.0,
        position_size_factor=1.3,
        allow_pyramiding=True,
        trailing_sl_enabled=False
    )
}

# ============================================================================
# F&O LOT SIZES (NSE - Updated Jan 2025)
# ============================================================================

FO_LOT_SIZES = {
    # Index
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "MIDCPNIFTY": 50,

    # Stocks (sample - actual list has 180+ stocks)
    "RELIANCE": 250,
    "TCS": 150,
    "HDFCBANK": 550,
    "INFY": 300,
    "ICICIBANK": 700,
    "BHARTIARTL": 475,
    "SBIN": 750,
    "KOTAKBANK": 400,
    "LT": 150,
    "AXISBANK": 600,
    "TATASTEEL": 550,
    "JSWSTEEL": 300,
    "HINDALCO": 1075,
    "VEDL": 1700,
    "TATAMOTORS": 575,
    "MARUTI": 50,
    "M&M": 350,
    "BAJFINANCE": 125,
    "BAJAJFINSV": 125,
    "TITAN": 200,
    "TRENT": 200,
    "POLYCAB": 100,
    "PERSISTENT": 100,
    "DIXON": 75,
    "TATAELXSI": 75,
    "ABB": 125,
    "SIEMENS": 75,
    "HAL": 150,
    "BEL": 3250,
    "ADANIENT": 250,
    "ADANIPORTS": 500,
    "DLF": 825,
    "GODREJPROP": 275,
    # Add more as needed
}

# ============================================================================
# MARGIN REQUIREMENTS (Approximate)
# ============================================================================

MARGIN_REQUIREMENTS = {
    Segment.EQUITY: {
        "delivery": 1.0,  # 100% for CNC
        "intraday": 0.2   # 20% for MIS
    },
    Segment.FUTURES: {
        "nrml": 0.15,     # ~15% SPAN + Exposure
        "mis": 0.075      # ~7.5% for intraday
    },
    Segment.OPTIONS: {
        "buy": 1.0,       # Premium only
        "sell": 0.20      # ~20% for writing
    }
}

# ============================================================================
# RISK MANAGEMENT ENGINE
# ============================================================================


class RiskManagementEngine:
    """
    5-Layer Risk Management System

    Layer 1: Signal Level - Confidence, Model Agreement
    Layer 2: Position Level - Size, Risk per trade
    Layer 3: Portfolio Level - Max positions, Sector exposure, Correlation
    Layer 4: Market Level - VIX, Trend, Circuit breakers
    Layer 5: System Level - Daily/Weekly/Monthly loss limits
    """

    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.cache = {}

    # ==========================================================================
    # LAYER 1: SIGNAL LEVEL CHECKS
    # ==========================================================================

    def check_signal_quality(
        self,
        signal: Signal,
        profile: RiskProfile
    ) -> Tuple[bool, str]:
        """
        Check if signal meets quality thresholds
        """
        # Check confidence
        if signal.confidence < profile.min_confidence:
            return False, f"Confidence {signal.confidence}% below minimum {profile.min_confidence}%"

        # Check risk:reward
        # OPTIONS selling strategies (straddle/strangle) may have sl=0 (risk managed via margin)
        if signal.segment == Segment.OPTIONS and signal.stop_loss == 0:
            pass  # skip R:R and SL distance checks for options with no explicit SL
        else:
            if signal.direction == Direction.LONG:
                risk = signal.entry_price - signal.stop_loss
                reward = signal.target - signal.entry_price
            else:
                risk = signal.stop_loss - signal.entry_price
                reward = signal.entry_price - signal.target

            if risk <= 0:
                return False, "Invalid stop loss - no risk defined"

            rr_ratio = reward / risk
            if rr_ratio < 1.5:
                return False, f"Risk:Reward {rr_ratio:.2f} below minimum 1.5"

            # Check stop loss distance (max 5% for equity, 3% for F&O)
            sl_percent = abs(signal.entry_price - signal.stop_loss) / max(signal.entry_price, 1) * 100
            max_sl = 5.0 if signal.segment == Segment.EQUITY else 3.0

            if sl_percent > max_sl:
                return False, f"Stop loss {sl_percent:.2f}% too wide (max {max_sl}%)"

        return True, "Signal quality check passed"

    # ==========================================================================
    # LAYER 2: POSITION SIZING
    # ==========================================================================

    def calculate_position_size(
        self,
        signal: Signal,
        capital: float,
        profile: RiskProfile,
        available_margin: Optional[float] = None
    ) -> PositionSize:
        """
        Calculate optimal position size based on risk

        Formula: Position Size = (Capital × Risk%) / (Entry - SL)
        Adjusted for: F&O lot sizes, Margin requirements, Risk profile
        """
        # Calculate risk amount
        risk_percent = profile.risk_per_trade * profile.position_size_factor
        risk_amount = capital * (risk_percent / 100)

        # Calculate risk per unit
        if signal.direction == Direction.LONG:
            risk_per_unit = signal.entry_price - signal.stop_loss
        else:
            risk_per_unit = signal.stop_loss - signal.entry_price

        if risk_per_unit <= 0:
            return PositionSize(
                quantity=0, lots=0, position_value=0, margin_required=0,
                risk_amount=0, risk_percent=0, approved=False,
                rejection_reason="Invalid stop loss"
            )

        # Calculate base quantity
        base_quantity = int(risk_amount / risk_per_unit)

        # Adjust for F&O lot sizes
        if signal.segment in [Segment.FUTURES, Segment.OPTIONS]:
            lot_size = FO_LOT_SIZES.get(signal.symbol, signal.lot_size)
            lots = max(1, base_quantity // lot_size)
            quantity = lots * lot_size
        else:
            lots = 1
            quantity = base_quantity

        if quantity < 1:
            return PositionSize(
                quantity=0, lots=0, position_value=0, margin_required=0,
                risk_amount=0, risk_percent=0, approved=False,
                rejection_reason="Position size too small for given risk"
            )

        # Calculate position value and margin
        position_value = quantity * signal.entry_price

        if signal.segment == Segment.EQUITY:
            margin_required = position_value * MARGIN_REQUIREMENTS[Segment.EQUITY]["delivery"]
        elif signal.segment == Segment.FUTURES:
            margin_required = position_value * MARGIN_REQUIREMENTS[Segment.FUTURES]["nrml"]
        else:  # OPTIONS
            if signal.direction == Direction.LONG:
                margin_required = position_value  # Premium only for buying
            else:
                margin_required = position_value * MARGIN_REQUIREMENTS[Segment.OPTIONS]["sell"]

        # Check against available margin
        if available_margin and margin_required > available_margin:
            # Reduce lots to fit margin
            if signal.segment in [Segment.FUTURES, Segment.OPTIONS]:
                max_lots = int(available_margin / (lot_size * signal.entry_price *
                               MARGIN_REQUIREMENTS[signal.segment]["nrml" if signal.segment == Segment.FUTURES else "buy"]))
                if max_lots < 1:
                    return PositionSize(
                        quantity=0, lots=0, position_value=0, margin_required=margin_required,
                        risk_amount=0, risk_percent=0, approved=False,
                        rejection_reason=f"Insufficient margin. Required: ₹{margin_required:,.0f}"
                    )
                lots = max_lots
                quantity = lots * lot_size
                position_value = quantity * signal.entry_price
                margin_required = available_margin
            else:
                quantity = int(available_margin / signal.entry_price)
                position_value = quantity * signal.entry_price
                margin_required = position_value

        # Recalculate actual risk
        actual_risk = quantity * risk_per_unit
        actual_risk_percent = (actual_risk / capital) * 100

        return PositionSize(
            quantity=quantity,
            lots=lots,
            position_value=position_value,
            margin_required=margin_required,
            risk_amount=actual_risk,
            risk_percent=actual_risk_percent,
            approved=True
        )

    # ==========================================================================
    # LAYER 3: PORTFOLIO LEVEL CHECKS
    # ==========================================================================

    async def check_portfolio_limits(
        self,
        user_id: str,
        signal: Signal,
        profile: RiskProfile
    ) -> Tuple[bool, str]:
        """
        Check portfolio-level constraints
        """
        # Get current positions
        result = self.supabase.table("positions").select("*").eq(
            "user_id", user_id
        ).eq("is_active", True).execute()

        positions = result.data or []

        # Check max positions
        if len(positions) >= profile.max_positions:
            return False, f"Maximum positions ({profile.max_positions}) reached"

        # Check if already have position in this symbol
        existing = [p for p in positions if p["symbol"] == signal.symbol]
        if existing and not profile.allow_pyramiding:
            return False, f"Already have position in {signal.symbol}"

        # Check sector exposure (simplified - group by first letter)
        # In production, use proper sector mapping
        symbol_sector = signal.symbol[0]  # Simplified
        sector_exposure = sum(
            p["position_value"] for p in positions
            if p["symbol"][0] == symbol_sector
        )

        # Get user capital
        profile_result = self.supabase.table("user_profiles").select(
            "capital"
        ).eq("id", user_id).single().execute()

        capital = profile_result.data["capital"]
        sector_percent = (sector_exposure / capital) * 100 if capital > 0 else 0

        if sector_percent >= profile.max_sector_exposure:
            return False, f"Sector exposure {sector_percent:.1f}% exceeds limit {profile.max_sector_exposure}%"

        # Check correlation (simplified - same direction positions)
        same_direction = [p for p in positions if p["direction"] == signal.direction.value]
        if len(same_direction) >= profile.max_positions * 0.8:
            return False, "Too many positions in same direction - diversify"

        return True, "Portfolio checks passed"

    # ==========================================================================
    # LAYER 4: MARKET LEVEL CHECKS
    # ==========================================================================

    def check_market_conditions(
        self,
        market: MarketCondition,
        signal: Signal
    ) -> Tuple[bool, str, float]:
        """
        Check market conditions and adjust position size

        Returns: (approved, message, size_multiplier)

        VIX gate uses the canonical ``classify_risk_band`` so this
        engine's "block at EXTREME" rule matches what the dashboard
        widget tells the user. Previously: local cutoffs blocked at
        >30 and reduced 50% at >25 — but ``classify_risk_band`` calls
        VIX >=25 EXTREME (whose canonical recommendation is "Stop all
        new trades"). At VIX 27, the dashboard said stop, this engine
        let half-sized trades through. Now consistent.
        """
        from .market_overview import classify_risk_band

        multiplier = 1.0
        warnings = []

        # VIX-based adjustments via canonical bands.
        band = classify_risk_band(market.vix)
        if band == "EXTREME":
            return False, f"VIX {market.vix:.1f} - EXTREME - No new trades", 0
        if band == "HIGH":
            multiplier *= 0.5
            warnings.append(f"VIX {market.vix:.1f} - HIGH - Position reduced 50%")
        elif band == "MODERATE":
            multiplier *= 0.75
            warnings.append(f"VIX {market.vix:.1f} - MODERATE - Position reduced 25%")

        # Gap opening check
        if abs(market.nifty_change) > 2:
            multiplier *= 0.5
            warnings.append(f"Nifty gap {market.nifty_change:.1f}% - Wait for stabilization")

        # FII flow check
        if market.fii_net < -2000 and signal.direction == Direction.LONG:
            multiplier *= 0.75
            warnings.append("Heavy FII selling - Reduced long exposure")
        elif market.fii_net > 2000 and signal.direction == Direction.SHORT:
            multiplier *= 0.75
            warnings.append("Heavy FII buying - Reduced short exposure")

        # Breadth check
        if market.advance_decline_ratio < 0.5 and signal.direction == Direction.LONG:
            warnings.append("Weak market breadth - Consider waiting")

        # PCR check for options
        if signal.segment == Segment.OPTIONS:
            if market.pcr > 1.5:
                warnings.append("High PCR - Market may be oversold")
            elif market.pcr < 0.7:
                warnings.append("Low PCR - Market may be overbought")

        message = " | ".join(warnings) if warnings else "Market conditions favorable"
        return True, message, multiplier

    # ==========================================================================
    # LAYER 5: SYSTEM LEVEL CHECKS
    # ==========================================================================

    async def check_loss_limits(
        self,
        user_id: str,
        profile: RiskProfile
    ) -> Tuple[bool, str]:
        """
        Check daily/weekly/monthly loss limits
        """
        # Get user capital
        profile_result = self.supabase.table("user_profiles").select(
            "capital"
        ).eq("id", user_id).single().execute()
        capital = profile_result.data["capital"]

        today = date.today().isoformat()
        week_start = (date.today() - timedelta(days=date.today().weekday())).isoformat()
        month_start = date.today().replace(day=1).isoformat()

        # Get closed trades P&L
        trades = self.supabase.table("trades").select(
            "net_pnl, closed_at"
        ).eq("user_id", user_id).eq("status", "closed").execute()

        today_pnl = sum(
            t["net_pnl"] for t in trades.data
            if t["closed_at"] and t["closed_at"][:10] == today
        )

        week_pnl = sum(
            t["net_pnl"] for t in trades.data
            if t["closed_at"] and t["closed_at"][:10] >= week_start
        )

        month_pnl = sum(
            t["net_pnl"] for t in trades.data
            if t["closed_at"] and t["closed_at"][:10] >= month_start
        )

        # Check limits
        daily_loss_percent = abs(min(0, today_pnl)) / capital * 100
        if daily_loss_percent >= profile.max_daily_loss:
            return False, f"Daily loss limit ({profile.max_daily_loss}%) reached. Trading paused."

        weekly_loss_percent = abs(min(0, week_pnl)) / capital * 100
        if weekly_loss_percent >= profile.max_weekly_loss:
            return False, f"Weekly loss limit ({profile.max_weekly_loss}%) reached. Trading paused."

        monthly_loss_percent = abs(min(0, month_pnl)) / capital * 100
        if monthly_loss_percent >= profile.max_monthly_loss:
            return False, f"Monthly loss limit ({profile.max_monthly_loss}%) reached. Trading paused."

        return True, "Loss limits check passed"

    # ==========================================================================
    # MASTER RISK CHECK
    # ==========================================================================

    async def evaluate_trade(
        self,
        user_id: str,
        signal: Signal,
        capital: float,
        available_margin: float,
        profile_name: str = "moderate",
        market: Optional[MarketCondition] = None
    ) -> Dict:
        """
        Run all 5 layers of risk checks and return decision
        """
        profile = RISK_PROFILES.get(profile_name, RISK_PROFILES["moderate"])

        result = {
            "approved": False,
            "signal": {
                "symbol": signal.symbol,
                "direction": signal.direction.value,
                "segment": signal.segment.value,
                "confidence": signal.confidence
            },
            "checks": {},
            "position_size": None,
            "warnings": [],
            "rejection_reason": None
        }

        # Layer 1: Signal Quality
        passed, message = self.check_signal_quality(signal, profile)
        result["checks"]["signal_quality"] = {"passed": passed, "message": message}
        if not passed:
            result["rejection_reason"] = message
            return result

        # Layer 2: Position Sizing
        position = self.calculate_position_size(signal, capital, profile, available_margin)
        result["position_size"] = {
            "quantity": position.quantity,
            "lots": position.lots,
            "position_value": position.position_value,
            "margin_required": position.margin_required,
            "risk_amount": position.risk_amount,
            "risk_percent": position.risk_percent
        }
        result["checks"]["position_size"] = {
            "passed": position.approved,
            "message": position.rejection_reason or "Position size calculated"
        }
        if not position.approved:
            result["rejection_reason"] = position.rejection_reason
            return result

        # Layer 3: Portfolio Limits
        passed, message = await self.check_portfolio_limits(user_id, signal, profile)
        result["checks"]["portfolio_limits"] = {"passed": passed, "message": message}
        if not passed:
            result["rejection_reason"] = message
            return result

        # Layer 4: Market Conditions
        if market:
            passed, message, multiplier = self.check_market_conditions(market, signal)
            result["checks"]["market_conditions"] = {
                "passed": passed,
                "message": message,
                "size_multiplier": multiplier
            }
            if not passed:
                result["rejection_reason"] = message
                return result

            if multiplier < 1.0:
                result["warnings"].append(message)
                # Adjust position size
                result["position_size"]["quantity"] = int(position.quantity * multiplier)
                if signal.segment in [Segment.FUTURES, Segment.OPTIONS]:
                    lot_size = FO_LOT_SIZES.get(signal.symbol, signal.lot_size)
                    result["position_size"]["lots"] = max(1, result["position_size"]["quantity"] // lot_size)
                    result["position_size"]["quantity"] = result["position_size"]["lots"] * lot_size

        # Layer 5: Loss Limits
        passed, message = await self.check_loss_limits(user_id, profile)
        result["checks"]["loss_limits"] = {"passed": passed, "message": message}
        if not passed:
            result["rejection_reason"] = message
            return result

        # All checks passed
        result["approved"] = True
        return result

    # ==========================================================================
    # TRAILING STOP LOSS
    # ==========================================================================

    def calculate_trailing_sl(
        self,
        entry_price: float,
        current_price: float,
        initial_sl: float,
        direction: Direction,
        atr: float = None
    ) -> float:
        """
        Calculate trailing stop loss based on profit

        Rules:
        - Move SL to breakeven after 1R profit
        - Trail by 50% of further gains
        """
        if direction == Direction.LONG:
            initial_risk = entry_price - initial_sl
            current_profit = current_price - entry_price

            if current_profit <= 0:
                return initial_sl

            # After 1R profit, move to breakeven
            if current_profit >= initial_risk:
                breakeven_sl = entry_price + (entry_price * 0.001)  # Small buffer

                # Trail by 50% of gains beyond 1R
                extra_profit = current_profit - initial_risk
                trailing_sl = breakeven_sl + (extra_profit * 0.5)

                return max(initial_sl, trailing_sl)

            return initial_sl

        else:  # SHORT
            initial_risk = initial_sl - entry_price
            current_profit = entry_price - current_price

            if current_profit <= 0:
                return initial_sl

            if current_profit >= initial_risk:
                breakeven_sl = entry_price - (entry_price * 0.001)
                extra_profit = current_profit - initial_risk
                trailing_sl = breakeven_sl - (extra_profit * 0.5)

                return min(initial_sl, trailing_sl)

            return initial_sl

    # ==========================================================================
    # PR 132 — VIX overlay + Kelly + portfolio VaR for AutoPilot (F4)
    # ==========================================================================
    #
    # The Step 1 §F4 deterministic VIX ladder applied as a multiplier on
    # AutoPilot's gross equity exposure. Returns the *equity exposure cap*
    # (0..1). Cash absorbs the residual.
    #
    # Source of truth:
    #   <15  → 100% equity
    #   15-18 → 85%
    #   18-22 → 70%
    #   22-27 → 50%
    #   27-35 → 30%
    #   >35  → 15%

    @staticmethod
    def vix_exposure_cap(vix_level: float) -> float:
        """Map current India VIX level → max gross equity weight."""
        try:
            v = float(vix_level)
        except (TypeError, ValueError):
            return 0.70  # safe default when VIX is unknown
        if v < 15:
            return 1.00
        if v < 18:
            return 0.85
        if v < 22:
            return 0.70
        if v < 27:
            return 0.50
        if v < 35:
            return 0.30
        return 0.15

    @staticmethod
    def kelly_fraction(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        *,
        cap: float = 0.25,
    ) -> float:
        """Kelly criterion position fraction.

        f* = W − (1 − W) / R   where R = avg_win / avg_loss

        Capped at ``cap`` (default 25%) per Step 1 §F4. Returns 0 for
        negative-edge strategies so AutoPilot stays out of bad regimes.
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0
        try:
            w = max(0.0, min(1.0, float(win_rate)))
            r = float(avg_win) / float(avg_loss)
            f = w - (1.0 - w) / r
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
        if f <= 0:
            return 0.0
        # Half-Kelly is the practical default; the 25% cap above gives a
        # hard ceiling regardless of the underlying edge.
        return min(cap, f * 0.5)

    @staticmethod
    def portfolio_var_95(
        weights: Dict[str, float],
        cov_matrix: Dict[str, Dict[str, float]],
        capital: float,
    ) -> float:
        """1-day 95% parametric VaR for the proposed weight vector.

        Returns absolute ₹ loss at the 5th percentile assuming Gaussian
        returns. AutoPilot rejects any rebalance whose VaR exceeds
        ``RiskProfile.max_daily_loss_pct × capital``.

        cov_matrix is keyed by symbol → symbol → covariance(daily).
        """
        symbols = list(weights.keys())
        if not symbols:
            return 0.0
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            return 0.0
        w = np.array([weights[s] for s in symbols], dtype=float)
        cov = np.array(
            [[cov_matrix.get(a, {}).get(b, 0.0) for b in symbols] for a in symbols],
            dtype=float,
        )
        try:
            port_var_daily = float(w @ cov @ w)
            port_std_daily = max(0.0, port_var_daily) ** 0.5
        except Exception:
            return 0.0
        # 95% one-tailed parametric VaR = 1.645 σ; multiply by capital.
        return 1.645 * port_std_daily * float(capital)

    def apply_autopilot_overlays(
        self,
        target_weights: Dict[str, float],
        *,
        vix_level: float,
        regime: str,
        capital: float,
        cov_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        max_daily_loss_pct: float = 2.0,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Apply VIX overlay + bear-regime halving + VaR cap to RL output.

        Returns ``(adjusted_weights, diagnostics)``. Diagnostics carry
        the cap multipliers used so AutoPilot can render "AI moved 20% to
        cash because VIX spiked to 22" in the dashboard.
        """
        cap = self.vix_exposure_cap(vix_level)
        bear_scale = 0.5 if str(regime).lower() == "bear" else 1.0
        gross = sum(max(0.0, w) for w in target_weights.values())
        scale = 1.0
        if gross > 0:
            scale = min(1.0, (cap * bear_scale) / gross)
        adjusted = {k: max(0.0, v) * scale for k, v in target_weights.items()}

        var_loss = 0.0
        var_blocked = False
        if cov_matrix:
            var_loss = self.portfolio_var_95(adjusted, cov_matrix, capital)
            var_limit = capital * (max_daily_loss_pct / 100.0)
            if var_loss > var_limit and var_limit > 0:
                # Scale down uniformly until VaR fits the per-day limit.
                reduce = var_limit / var_loss
                adjusted = {k: v * reduce for k, v in adjusted.items()}
                var_blocked = True
                var_loss *= reduce

        return adjusted, {
            "vix_level": float(vix_level),
            "vix_exposure_cap": cap,
            "bear_scale": bear_scale,
            "applied_scale": scale,
            "var_95_inr": var_loss,
            "var_capped": var_blocked,
        }

    # ==========================================================================
    # RL EARLY-EXIT CONSULT (HOLD / EXIT / TIGHTEN on open positions)
    # ==========================================================================

    async def enforce_stops_for_user(self, user_id: str) -> Dict[str, int]:
        """Consult the narrow RL exit agent on a user's open equity positions.

        Wired live by the AutoPilot Supervisor's INTRADAY window. Scope is
        deliberately narrow (memory project_rl_exit_enabled_2026_05_25):
        HOLD / EXIT / TIGHTEN on ALREADY-OPEN positions only.

        Safety invariants:
          * Hard SL/target stay AUTHORITATIVE — they live in the scheduler's
            ``monitor_positions`` 5-min loop, which is the SOLE closer.
          * This method never places an order. EXIT/TIGHTEN only RATCHET the
            stop (writing the field ``monitor_positions`` reads — ``trades.stop_loss``),
            so the authoritative loop performs any close on its next tick.
            That makes double-execution structurally impossible.
          * Active only when ENABLE_RL_EXIT and a trained Q-table loads (one
            ships at artifacts/rl/q_table.json). Any position the agent rates
            HOLD is a no-op; hard SL/target remain authoritative throughout.

        Returns ``{exits_emitted, alerts_fired}`` (alerts_fired = tightens).
        """
        try:
            from ..ai.exit_engine.rl_exit_scaffold import (
                ENABLE_RL_EXIT, compute_rl_state, get_rl_exit_agent,
            )
        except Exception:
            return {"exits_emitted": 0, "alerts_fired": 0}

        if not ENABLE_RL_EXIT:
            return {"exits_emitted": 0, "alerts_fired": 0}
        agent = get_rl_exit_agent()
        if not agent.is_loaded:
            # No trained Q-table on disk — no-op (hard SL/target still apply).
            return {"exits_emitted": 0, "alerts_fired": 0}

        try:
            rows = (
                self.supabase.table("positions")
                .select("*, trades(stop_loss, target)")
                .eq("user_id", user_id)
                .eq("is_active", True)
                .eq("segment", "EQUITY")
                .limit(200)
                .execute()
                .data
                or []
            )
        except Exception as exc:
            logger.warning("enforce_stops_for_user: position read failed: %s", exc)
            return {"exits_emitted": 0, "alerts_fired": 0}

        exits = 0
        tightens = 0
        for pos in rows:
            try:
                acted = await self._rl_consult_position(pos, agent, compute_rl_state)
                if acted == "EXIT":
                    exits += 1
                elif acted == "TIGHTEN":
                    tightens += 1
            except Exception as exc:  # noqa: BLE001 — never let one position abort the sweep
                logger.debug("rl exit consult failed for position %s: %s", pos.get("id"), exc)
        return {"exits_emitted": exits, "alerts_fired": tightens}

    async def _rl_consult_position(self, pos, agent, compute_rl_state) -> Optional[str]:
        """Build RL state for one position, decide, and ratchet the stop.

        Returns the action taken ('EXIT'/'TIGHTEN') or None for HOLD/skip.
        """
        entry = float(pos.get("average_price") or pos.get("entry_price") or 0)
        if entry <= 0:
            return None
        trade = pos.get("trades") or {}
        if isinstance(trade, list):
            trade = trade[0] if trade else {}
        sl = float(trade.get("stop_loss") or pos.get("stop_loss") or 0)
        target = float(trade.get("target") or pos.get("target") or 0)
        direction = (pos.get("direction") or "LONG").upper()

        current = await self._latest_price(pos.get("symbol"))
        if current is None or current <= 0:
            return None

        # bars_held from opened_at (1 trading bar ≈ 1 day for swing positions).
        bars_held = 1
        opened = pos.get("opened_at") or pos.get("created_at")
        if opened:
            try:
                od = datetime.fromisoformat(str(opened).replace("Z", "+00:00"))
                bars_held = max(1, (datetime.now(od.tzinfo) - od).days)
            except Exception:
                bars_held = 1

        peak = max(entry, current, float(pos.get("peak_price") or 0) or current)
        state = compute_rl_state(
            entry_price=entry, current_price=current, bars_held=bars_held,
            max_hold_bars=40, sl=sl, target=target, trailing_active=False,
            peak_price=peak, price_history=[entry, current],
        )
        action = agent.decide(state)
        if action == "HOLD":
            return None

        # EXIT → ratchet stop to current (authoritative loop closes next tick).
        # TIGHTEN → ratchet stop halfway from current stop toward current price.
        if direction == "LONG":
            new_sl = current if action == "EXIT" else (sl + (current - sl) * 0.5 if sl else current)
            if sl and new_sl <= sl:
                return None  # never loosen
        else:  # SHORT
            new_sl = current if action == "EXIT" else (sl - (sl - current) * 0.5 if sl else current)
            if sl and new_sl >= sl:
                return None
        self._ratchet_stop(pos, round(new_sl, 2))
        logger.info(
            "RL exit %s %s: stop %s -> %s (user=%s)",
            action, pos.get("symbol"), sl, round(new_sl, 2), pos.get("user_id"),
        )
        return action

    def _ratchet_stop(self, pos, new_sl: float) -> None:
        """Write the new stop to BOTH the position and its linked trade.

        ``monitor_positions`` reads the hard stop from ``trades.stop_loss``,
        so the trade row MUST be updated for the authoritative loop to honour
        an RL exit; the position row is kept in sync for UI consistency.
        """
        try:
            self.supabase.table("positions").update(
                {"stop_loss": new_sl}
            ).eq("id", pos["id"]).execute()
        except Exception as exc:
            logger.debug("ratchet position stop failed: %s", exc)
        trade_id = pos.get("trade_id")
        if trade_id:
            try:
                self.supabase.table("trades").update(
                    {"stop_loss": new_sl}
                ).eq("id", trade_id).execute()
            except Exception as exc:
                logger.debug("ratchet trade stop failed: %s", exc)

    async def _latest_price(self, symbol: Optional[str]) -> Optional[float]:
        if not symbol:
            return None
        try:
            from ..data.market import get_market_data_provider
            provider = get_market_data_provider()
            q = provider.get_quote(symbol)
            return float(q) if q else None
        except Exception:
            return None


# ============================================================================
# F&O SPECIFIC CALCULATIONS
# ============================================================================

class FOCalculator:
    """
    Futures & Options specific calculations
    """

    @staticmethod
    def get_lot_size(symbol: str) -> int:
        """Get F&O lot size for symbol"""
        return FO_LOT_SIZES.get(symbol, 1)

    @staticmethod
    def calculate_futures_margin(
        symbol: str,
        price: float,
        lots: int = 1,
        is_intraday: bool = False
    ) -> Dict:
        """
        Calculate futures margin requirement
        """
        lot_size = FO_LOT_SIZES.get(symbol, 1)
        quantity = lots * lot_size
        contract_value = quantity * price

        margin_rate = 0.075 if is_intraday else 0.15
        margin_required = contract_value * margin_rate

        return {
            "symbol": symbol,
            "lot_size": lot_size,
            "lots": lots,
            "quantity": quantity,
            "contract_value": contract_value,
            "margin_required": margin_required,
            "margin_type": "MIS" if is_intraday else "NRML"
        }

    @staticmethod
    def calculate_options_premium(
        symbol: str,
        strike: float,
        premium: float,
        lots: int = 1,
        is_buy: bool = True
    ) -> Dict:
        """
        Calculate options premium and margin
        """
        lot_size = FO_LOT_SIZES.get(symbol, 1)
        quantity = lots * lot_size
        total_premium = quantity * premium

        if is_buy:
            margin_required = total_premium  # Only premium for buying
        else:
            # Selling requires higher margin
            margin_required = total_premium * 5  # Approximate

        return {
            "symbol": symbol,
            "strike": strike,
            "lot_size": lot_size,
            "lots": lots,
            "quantity": quantity,
            "premium_per_share": premium,
            "total_premium": total_premium,
            "margin_required": margin_required,
            "type": "BUY" if is_buy else "SELL"
        }

    @staticmethod
    def select_option_strike(
        spot_price: float,
        direction: Direction,
        strikes_available: List[float],
        otm_distance: int = 1  # Number of strikes OTM
    ) -> Tuple[float, str]:
        """
        Select appropriate option strike

        For LONG direction: Buy PUT (for short via options)
        For SHORT direction: Buy CALL (for long via options)

        Wait, that's inverse. Let me correct:
        - To go LONG on stock: Buy CALL or Sell PUT
        - To go SHORT on stock: Buy PUT or Sell CALL

        We prefer buying for limited risk.
        """
        # Sort strikes
        strikes = sorted(strikes_available)

        # Find ATM strike
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        atm_index = strikes.index(atm_strike)

        if direction == Direction.LONG:
            # Buy CALL, slightly OTM
            target_index = min(atm_index + otm_distance, len(strikes) - 1)
            return strikes[target_index], "CE"
        else:
            # Buy PUT, slightly OTM
            target_index = max(atm_index - otm_distance, 0)
            return strikes[target_index], "PE"

    @staticmethod
    def get_next_expiry(symbol: str = "NIFTY") -> date:
        """
        Get next F&O expiry date.

        Rules:
        - Index options: Weekly (Thursday)
        - Stock options: Monthly (last Thursday)

        ``today`` and the post-close hour check both use IST so this
        works the same on a UTC-hosted server. Previously we used
        ``datetime.now().hour`` which is server-local — on UTC at
        12:00 (= 17:30 IST, market closed) the check would read
        ``hour=12 < 15`` and return Thursday today, when the correct
        next expiry is the following Thursday.
        """
        from .market_calendar import IST

        now_ist = datetime.now(IST)
        today_ist = now_ist.date()

        # Find next Thursday
        days_until_thursday = (3 - today_ist.weekday()) % 7
        if days_until_thursday == 0 and now_ist.hour >= 15:
            # It's Thursday and market has closed — roll to next week.
            days_until_thursday = 7

        next_thursday = today_ist + timedelta(days=days_until_thursday)

        # Note: should also skip NSE holidays via market_calendar's
        # next_trading_day when wired into a real call site.

        return next_thursday


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def example_usage():
    """Example of using the risk management engine"""

    # Initialize
    from supabase import create_client
    supabase = create_client("url", "key")
    risk_engine = RiskManagementEngine(supabase)

    # Create signal
    signal = Signal(
        symbol="TRENT",
        segment=Segment.EQUITY,
        direction=Direction.LONG,
        confidence=82,
        entry_price=2765,
        stop_loss=2680,
        target=2930
    )

    # Create market condition
    market = MarketCondition(
        vix=14.5,
        nifty_change=0.45,
        fii_net=2450,
        advance_decline_ratio=1.8,
        pcr=1.1
    )

    # Evaluate trade
    result = await risk_engine.evaluate_trade(
        user_id="user-123",
        signal=signal,
        capital=500000,
        available_margin=400000,
        profile_name="moderate",
        market=market
    )

    print(f"Trade Approved: {result['approved']}")
    print(f"Position Size: {result['position_size']}")
    print(f"Checks: {result['checks']}")

    # F&O Example
    fo_calc = FOCalculator()

    # Futures margin
    futures_margin = fo_calc.calculate_futures_margin(
        symbol="NIFTY",
        price=21850,
        lots=2
    )
    print(f"Futures Margin: {futures_margin}")

    # Options strike selection
    strikes = [21500, 21600, 21700, 21800, 21900, 22000, 22100, 22200]
    strike, option_type = fo_calc.select_option_strike(
        spot_price=21850,
        direction=Direction.LONG,
        strikes_available=strikes
    )
    print(f"Selected Strike: {strike} {option_type}")


if __name__ == "__main__":
    asyncio.run(example_usage())
