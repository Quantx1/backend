"""
================================================================================
QUANT X F&O TRADING ENGINE
================================================================================
Complete Options & Futures Trading with:
- NSE Lot Sizes
- Margin Calculations
- Greeks (Delta, Gamma, Theta, Vega)
- Expiry Management
- Option Chain Analysis
- Smart Strike Selection
================================================================================
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ============================================================================
# NSE LOT SIZES (Updated Jan 2025)
# ============================================================================

NSE_LOT_SIZES = {
    # Index
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "MIDCPNIFTY": 50,

    # Large Cap Stocks
    "RELIANCE": 250,
    "TCS": 150,
    "HDFCBANK": 550,
    "INFY": 300,
    "ICICIBANK": 700,
    "BHARTIARTL": 475,
    "SBIN": 750,
    "KOTAKBANK": 400,
    "LT": 150,
    "AXISBANK": 625,
    "HINDUNILVR": 300,
    "ITC": 1600,
    "BAJFINANCE": 125,
    "MARUTI": 100,
    "ASIANPAINT": 200,
    "TITAN": 375,
    "TATAMOTORS": 1425,
    "SUNPHARMA": 700,
    "WIPRO": 1500,
    "HCLTECH": 350,
    "ADANIENT": 500,
    "ADANIPORTS": 625,
    "POWERGRID": 2700,
    "NTPC": 2850,
    "ONGC": 3850,
    "COALINDIA": 2100,
    "TATASTEEL": 425,
    "JSWSTEEL": 500,
    "HINDALCO": 1400,
    "INDUSINDBK": 450,
    "TECHM": 600,
    "M&M": 350,
    "DRREDDY": 125,
    "DIVISLAB": 150,
    "CIPLA": 650,
    "EICHERMOT": 175,
    "BAJAJ-AUTO": 250,
    "HEROMOTOCO": 300,
    "TRENT": 385,
    "POLYCAB": 200,
    "PERSISTENT": 200,
    "DIXON": 175,
    "VEDL": 2300,
    "TATAPOWER": 2025,
    "DLF": 825,
    "GODREJPROP": 500,
    "ABB": 125,
    "SIEMENS": 150,
    "HAL": 150,
    "BEL": 3250,
}

# ============================================================================
# MARGIN REQUIREMENTS (Approximate %)
# ============================================================================

MARGIN_REQUIREMENTS = {
    "NIFTY": {"futures": 12, "options_buy": 100, "options_sell": 15},
    "BANKNIFTY": {"futures": 15, "options_buy": 100, "options_sell": 18},
    "STOCKS": {"futures": 20, "options_buy": 100, "options_sell": 25},
}

# ============================================================================
# DATA CLASSES
# ============================================================================


class OptionType(Enum):
    CALL = "CE"
    PUT = "PE"


class InstrumentType(Enum):
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


@dataclass
class OptionContract:
    symbol: str
    strike: float
    option_type: OptionType
    expiry: date
    lot_size: int
    ltp: float
    bid: float
    ask: float
    oi: int
    oi_change: int
    volume: int
    iv: float
    delta: float = 0
    gamma: float = 0
    theta: float = 0
    vega: float = 0


@dataclass
class FuturesContract:
    symbol: str
    expiry: date
    lot_size: int
    ltp: float
    bid: float
    ask: float
    oi: int
    basis: float  # Premium/Discount to spot


@dataclass
class FOTrade:
    symbol: str
    instrument_type: InstrumentType
    direction: str  # LONG or SHORT

    # Contract details
    strike: Optional[float] = None
    option_type: Optional[OptionType] = None
    expiry: Optional[date] = None

    # Position
    lots: int = 1
    lot_size: int = 1
    quantity: int = 1

    # Prices
    entry_price: float = 0
    stop_loss: float = 0
    target: float = 0

    # Margin
    margin_required: float = 0
    premium_paid: float = 0

    # Greeks
    delta: float = 0
    theta: float = 0

# ============================================================================
# BLACK-SCHOLES MODEL FOR OPTIONS PRICING
# ============================================================================


class BlackScholes:
    """
    Black-Scholes Option Pricing Model
    Calculate theoretical prices and Greeks
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1"""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2"""
        if T <= 0 or sigma <= 0:
            return 0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Call option price"""
        if T <= 0:
            return max(0, S - K)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Put option price"""
        if T <= 0:
            return max(0, K - S)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> float:
        """Calculate Delta"""
        if T <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        if option_type == OptionType.CALL:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Gamma"""
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> float:
        """Calculate Theta (per day)"""
        if T <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)

        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        if option_type == OptionType.CALL:
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)

        return (term1 + term2) / 365  # Per day

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Vega"""
        if T <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in IV

    @staticmethod
    def implied_volatility(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        tolerance: float = 0.0001,
        max_iterations: int = 100
    ) -> float:
        """Calculate Implied Volatility using Newton-Raphson"""
        sigma = 0.3  # Initial guess

        for _ in range(max_iterations):
            if option_type == OptionType.CALL:
                price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                price = BlackScholes.put_price(S, K, T, r, sigma)

            vega = BlackScholes.vega(S, K, T, r, sigma) * 100

            if abs(vega) < 1e-10:
                break

            diff = market_price - price
            if abs(diff) < tolerance:
                return sigma

            sigma = sigma + diff / vega
            sigma = max(0.01, min(sigma, 5.0))  # Clamp between 1% and 500%

        return sigma

# ============================================================================
# F&O TRADING ENGINE
# ============================================================================


class FOTradingEngine:
    """
    Complete F&O Trading Engine
    Handles Options and Futures with proper risk management
    """

    def __init__(self, risk_free_rate: float = 0.07):
        self.risk_free_rate = risk_free_rate
        self.bs = BlackScholes()

    def get_lot_size(self, symbol: str) -> int:
        """Get NSE lot size for symbol"""
        return NSE_LOT_SIZES.get(symbol.upper(), 1)

    def get_expiry_dates(self, symbol: str, instrument_type: InstrumentType) -> List[date]:
        """Get available expiry dates"""
        today = date.today()
        expiries = []

        if instrument_type == InstrumentType.FUTURES:
            # Futures are monthly expiries (last Thursday) for all symbols
            for month_offset in range(3):
                year = today.year
                month = today.month + month_offset
                if month > 12:
                    month -= 12
                    year += 1
                last_day = date(year, month + 1, 1) - \
                    timedelta(days=1) if month < 12 else date(year + 1, 1, 1) - timedelta(days=1)
                while last_day.weekday() != 3:  # Thursday
                    last_day -= timedelta(days=1)
                if last_day > today:
                    expiries.append(last_day)
        else:
            if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                # Weekly expiries (Thursday for Nifty, Wednesday for BankNifty)
                expiry_day = 3 if symbol in ["NIFTY", "FINNIFTY"] else 2
                current = today
                while len(expiries) < 4:
                    days_ahead = expiry_day - current.weekday()
                    if days_ahead <= 0:
                        days_ahead += 7
                    next_expiry = current + timedelta(days=days_ahead)
                    if next_expiry > today:
                        expiries.append(next_expiry)
                    current = next_expiry + timedelta(days=1)
            else:
                # Monthly expiries (last Thursday of month)
                for month_offset in range(3):
                    year = today.year
                    month = today.month + month_offset
                    if month > 12:
                        month -= 12
                        year += 1
                    last_day = date(year, month + 1, 1) - \
                        timedelta(days=1) if month < 12 else date(year + 1, 1, 1) - timedelta(days=1)
                    while last_day.weekday() != 3:  # Thursday
                        last_day -= timedelta(days=1)
                    if last_day > today:
                        expiries.append(last_day)

        return expiries

    def calculate_margin(
        self,
        symbol: str,
        instrument_type: InstrumentType,
        option_type: Optional[OptionType],
        strike: float,
        spot_price: float,
        lots: int,
        is_sell: bool = False
    ) -> Dict:
        """Calculate margin required for F&O position"""

        lot_size = self.get_lot_size(symbol)
        quantity = lots * lot_size

        if instrument_type == InstrumentType.FUTURES:
            # Futures margin
            if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                margin_pct = MARGIN_REQUIREMENTS.get(symbol, MARGIN_REQUIREMENTS["NIFTY"])["futures"]
            else:
                margin_pct = MARGIN_REQUIREMENTS["STOCKS"]["futures"]

            margin = spot_price * quantity * (margin_pct / 100)

            return {
                "initial_margin": margin,
                "exposure_margin": margin * 0.3,
                "total_margin": margin * 1.3,
                "lot_size": lot_size,
                "quantity": quantity
            }

        else:  # OPTIONS
            if is_sell:  # Option selling requires margin
                if symbol in ["NIFTY", "BANKNIFTY"]:
                    margin_pct = MARGIN_REQUIREMENTS.get(symbol, MARGIN_REQUIREMENTS["NIFTY"])["options_sell"]
                else:
                    margin_pct = MARGIN_REQUIREMENTS["STOCKS"]["options_sell"]

                margin = spot_price * quantity * (margin_pct / 100)

                return {
                    "initial_margin": margin,
                    "exposure_margin": margin * 0.2,
                    "total_margin": margin * 1.2,
                    "lot_size": lot_size,
                    "quantity": quantity,
                    "is_margin_trade": True
                }
            else:  # Option buying - only premium
                # Premium is the max loss
                return {
                    "initial_margin": 0,
                    "exposure_margin": 0,
                    "total_margin": 0,  # Only premium paid
                    "lot_size": lot_size,
                    "quantity": quantity,
                    "is_margin_trade": False
                }

    def select_strike(
        self,
        spot_price: float,
        direction: str,  # LONG or SHORT (of underlying)
        signal_confidence: float,
        days_to_expiry: int
    ) -> Tuple[float, OptionType]:
        """
        Smart strike selection based on signal and market conditions

        For LONG underlying: Buy PUT (for hedging) or Sell PUT (bullish)
        For SHORT underlying: Buy CALL (for hedging) or Buy PUT (bearish directional)
        """

        # Round to nearest strike
        if spot_price > 10000:  # Index
            strike_interval = 50
        elif spot_price > 1000:
            strike_interval = 50
        elif spot_price > 500:
            strike_interval = 10
        else:
            strike_interval = 5

        atm_strike = round(spot_price / strike_interval) * strike_interval

        if direction == "LONG":
            # Bullish - Use PUT options
            if signal_confidence >= 80:
                # High confidence - Buy slightly ITM PUT for protection or Sell OTM PUT
                strike = atm_strike - (2 * strike_interval)  # OTM PUT to sell
                option_type = OptionType.PUT
            else:
                # Lower confidence - Buy ATM PUT for protection
                strike = atm_strike
                option_type = OptionType.PUT
        else:  # SHORT
            # Bearish - Use PUT options for directional trade
            if signal_confidence >= 80:
                # High confidence - Buy ITM PUT
                strike = atm_strike + (2 * strike_interval)  # ITM PUT
                option_type = OptionType.PUT
            else:
                # Lower confidence - Buy ATM PUT
                strike = atm_strike
                option_type = OptionType.PUT

        return strike, option_type

    def calculate_greeks(
        self,
        spot_price: float,
        strike: float,
        days_to_expiry: int,
        iv: float,
        option_type: OptionType
    ) -> Dict:
        """Calculate all Greeks for an option"""

        T = days_to_expiry / 365
        r = self.risk_free_rate
        sigma = iv / 100 if iv > 1 else iv

        delta = self.bs.delta(spot_price, strike, T, r, sigma, option_type)
        gamma = self.bs.gamma(spot_price, strike, T, r, sigma)
        theta = self.bs.theta(spot_price, strike, T, r, sigma, option_type)
        vega = self.bs.vega(spot_price, strike, T, r, sigma)

        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 2),
            "vega": round(vega, 2)
        }

    def create_fo_trade(
        self,
        symbol: str,
        direction: str,
        signal_confidence: float,
        spot_price: float,
        stop_loss_pct: float,
        target_pct: float,
        capital: float,
        risk_per_trade_pct: float,
        preferred_instrument: str = "put_options"  # put_options, futures, both
    ) -> Optional[FOTrade]:
        """
        Create F&O trade from signal

        For SHORT signals:
        - put_options: Buy PUT options (limited risk)
        - futures: Short futures (unlimited risk, higher margin)
        """

        lot_size = self.get_lot_size(symbol)
        expiries = self.get_expiry_dates(symbol, InstrumentType.OPTIONS)

        if not expiries:
            logger.error(f"No expiries found for {symbol}")
            return None

        # Select nearest expiry with at least 5 days
        selected_expiry = None
        for expiry in expiries:
            days_to_expiry = (expiry - date.today()).days
            if days_to_expiry >= 5:
                selected_expiry = expiry
                break

        if not selected_expiry:
            selected_expiry = expiries[0]

        days_to_expiry = (selected_expiry - date.today()).days

        # Calculate risk amount
        risk_amount = capital * (risk_per_trade_pct / 100)

        if preferred_instrument == "futures":
            # FUTURES TRADE
            margin_info = self.calculate_margin(
                symbol, InstrumentType.FUTURES, None, 0, spot_price, 1
            )

            # Calculate lots based on risk
            sl_distance = spot_price * (stop_loss_pct / 100)
            risk_per_lot = sl_distance * lot_size
            lots = max(1, int(risk_amount / risk_per_lot))

            # Check margin
            total_margin = margin_info["total_margin"] * lots
            if total_margin > capital * 0.8:  # Max 80% capital as margin
                lots = max(1, int((capital * 0.8) / margin_info["total_margin"]))

            if direction == "LONG":
                entry = spot_price
                sl = entry * (1 - stop_loss_pct / 100)
                target = entry * (1 + target_pct / 100)
            else:  # SHORT
                entry = spot_price
                sl = entry * (1 + stop_loss_pct / 100)
                target = entry * (1 - target_pct / 100)

            return FOTrade(
                symbol=symbol,
                instrument_type=InstrumentType.FUTURES,
                direction=direction,
                expiry=selected_expiry,
                lots=lots,
                lot_size=lot_size,
                quantity=lots * lot_size,
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target=round(target, 2),
                margin_required=round(total_margin, 2)
            )

        else:  # OPTIONS
            # Select strike
            strike, option_type = self.select_strike(
                spot_price, direction, signal_confidence, days_to_expiry
            )

            # Estimate option premium (simplified)
            iv = 0.20  # Assume 20% IV, should fetch real IV
            T = days_to_expiry / 365

            if option_type == OptionType.PUT:
                premium = self.bs.put_price(spot_price, strike, T, self.risk_free_rate, iv)
            else:
                premium = self.bs.call_price(spot_price, strike, T, self.risk_free_rate, iv)

            # Calculate lots
            premium_per_lot = premium * lot_size
            lots = max(1, int(risk_amount / premium_per_lot))

            # Calculate Greeks
            greeks = self.calculate_greeks(spot_price, strike, days_to_expiry, iv * 100, option_type)

            # For PUT buying (bearish directional trade)
            # Entry = current premium
            # SL = premium drops 50%
            # Target = premium doubles or more

            entry = premium
            sl = premium * 0.5  # Max loss is premium, but we exit at 50% loss
            target = premium * 2.0  # Target 100% return on premium

            return FOTrade(
                symbol=symbol,
                instrument_type=InstrumentType.OPTIONS,
                direction="LONG",  # We're buying the option
                strike=strike,
                option_type=option_type,
                expiry=selected_expiry,
                lots=lots,
                lot_size=lot_size,
                quantity=lots * lot_size,
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target=round(target, 2),
                margin_required=0,  # Option buying = premium only
                premium_paid=round(entry * lots * lot_size, 2),
                delta=greeks["delta"],
                theta=greeks["theta"]
            )

    def get_option_chain(self, symbol: str, expiry: date, spot_price: float,
                         base_iv: float = 0.15) -> List[OptionContract]:
        """
        Generate option chain (in production, fetch from broker API).
        base_iv: ATM implied volatility as decimal (e.g., 0.26 for 26%).
        """
        lot_size = self.get_lot_size(symbol)

        # Determine strike interval
        if spot_price > 10000:
            strike_interval = 50
        elif spot_price > 1000:
            strike_interval = 50
        else:
            strike_interval = 10

        atm_strike = round(spot_price / strike_interval) * strike_interval

        chain = []
        days_to_expiry = max(1, (expiry - date.today()).days)
        T = days_to_expiry / 365

        # Generate strikes from -10 to +10 around ATM
        for i in range(-10, 11):
            strike = atm_strike + (i * strike_interval)

            # Skip invalid strikes
            if strike <= 0:
                continue

            # IV smile: base_iv at ATM, increasing for OTM
            iv = base_iv + abs(i) * 0.005

            # Calculate prices and Greeks
            call_price = self.bs.call_price(spot_price, strike, T, self.risk_free_rate, iv)
            put_price = self.bs.put_price(spot_price, strike, T, self.risk_free_rate, iv)

            call_delta = self.bs.delta(spot_price, strike, T, self.risk_free_rate, iv, OptionType.CALL)
            put_delta = self.bs.delta(spot_price, strike, T, self.risk_free_rate, iv, OptionType.PUT)

            # Add CALL
            chain.append(OptionContract(
                symbol=symbol,
                strike=strike,
                option_type=OptionType.CALL,
                expiry=expiry,
                lot_size=lot_size,
                ltp=round(call_price, 2),
                bid=round(call_price * 0.99, 2),
                ask=round(call_price * 1.01, 2),
                oi=np.random.randint(10000, 500000),
                oi_change=np.random.randint(-50000, 50000),
                volume=np.random.randint(1000, 100000),
                iv=round(iv * 100, 2),
                delta=round(call_delta, 4)
            ))

            # Add PUT
            chain.append(OptionContract(
                symbol=symbol,
                strike=strike,
                option_type=OptionType.PUT,
                expiry=expiry,
                lot_size=lot_size,
                ltp=round(put_price, 2),
                bid=round(put_price * 0.99, 2),
                ask=round(put_price * 1.01, 2),
                oi=np.random.randint(10000, 500000),
                oi_change=np.random.randint(-50000, 50000),
                volume=np.random.randint(1000, 100000),
                iv=round(iv * 100, 2),
                delta=round(put_delta, 4)
            ))

        return chain


# ============================================================================
# RISK MANAGER FOR F&O
# ============================================================================

class FORiskManager:
    """
    F&O Specific Risk Management
    """

    def __init__(self, capital: float, risk_profile: str = "moderate"):
        self.capital = capital
        self.risk_profile = risk_profile

        # Risk limits based on profile
        self.limits = {
            "conservative": {
                "max_fo_allocation": 0.20,  # 20% of capital
                "max_single_trade": 0.03,   # 3% per trade
                "max_options_positions": 2,
                "max_futures_positions": 1,
                "min_days_to_expiry": 7,
                "prefer_buying": True,
            },
            "moderate": {
                "max_fo_allocation": 0.40,
                "max_single_trade": 0.05,
                "max_options_positions": 4,
                "max_futures_positions": 2,
                "min_days_to_expiry": 5,
                "prefer_buying": True,
            },
            "aggressive": {
                "max_fo_allocation": 0.60,
                "max_single_trade": 0.08,
                "max_options_positions": 6,
                "max_futures_positions": 3,
                "min_days_to_expiry": 3,
                "prefer_buying": False,
            }
        }

    def get_limit(self, key: str):
        return self.limits[self.risk_profile].get(key)

    def validate_trade(
        self,
        trade: FOTrade,
        current_fo_positions: int,
        current_fo_margin: float
    ) -> Tuple[bool, str]:
        """Validate F&O trade against risk limits"""

        limits = self.limits[self.risk_profile]

        # Check max F&O allocation
        max_fo_capital = self.capital * limits["max_fo_allocation"]
        trade_value = trade.margin_required or trade.premium_paid

        if current_fo_margin + trade_value > max_fo_capital:
            return False, f"F&O allocation limit exceeded. Max: {max_fo_capital}, Current: {current_fo_margin}, New: {trade_value}"

        # Check position limits
        if trade.instrument_type == InstrumentType.OPTIONS:
            if current_fo_positions >= limits["max_options_positions"]:
                return False, f"Max options positions ({limits['max_options_positions']}) reached"
        else:
            if current_fo_positions >= limits["max_futures_positions"]:
                return False, f"Max futures positions ({limits['max_futures_positions']}) reached"

        # Check expiry
        if trade.expiry:
            days_to_expiry = (trade.expiry - date.today()).days
            if days_to_expiry < limits["min_days_to_expiry"]:
                return False, f"Too close to expiry. Min: {limits['min_days_to_expiry']} days"

        # Check single trade risk
        max_trade_value = self.capital * limits["max_single_trade"]
        if trade_value > max_trade_value:
            return False, f"Trade value too high. Max: {max_trade_value}"

        return True, "Trade validated"

    def calculate_position_size(
        self,
        signal_confidence: float,
        volatility: float,
        days_to_expiry: int
    ) -> float:
        """Calculate position size factor based on conditions"""

        # Base size
        size_factor = 1.0

        # Adjust for confidence
        if signal_confidence >= 80:
            size_factor *= 1.2
        elif signal_confidence >= 70:
            size_factor *= 1.0
        else:
            size_factor *= 0.8

        # Adjust for volatility (VIX)
        if volatility > 25:
            size_factor *= 0.5
        elif volatility > 20:
            size_factor *= 0.7
        elif volatility < 15:
            size_factor *= 1.1

        # Adjust for time to expiry
        if days_to_expiry < 5:
            size_factor *= 0.6  # Gamma risk
        elif days_to_expiry < 10:
            size_factor *= 0.8

        return min(size_factor, 1.5)  # Cap at 1.5x


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize engine
    engine = FOTradingEngine()
    risk_manager = FORiskManager(capital=500000, risk_profile="moderate")

    # Example: Create SHORT trade using PUT options
    trade = engine.create_fo_trade(
        symbol="TRENT",
        direction="SHORT",
        signal_confidence=75,
        spot_price=2800,
        stop_loss_pct=3,
        target_pct=6,
        capital=500000,
        risk_per_trade_pct=3,
        preferred_instrument="put_options"
    )

    if trade:
        print(f"\n{'=' * 60}")
        print("F&O TRADE CREATED")
        print(f"{'=' * 60}")
        print(f"Symbol: {trade.symbol}")
        print(f"Type: {trade.instrument_type.value}")
        print(f"Option: {trade.strike} {trade.option_type.value if trade.option_type else 'N/A'}")
        print(f"Expiry: {trade.expiry}")
        print(f"Lots: {trade.lots} x {trade.lot_size} = {trade.quantity} qty")
        print(f"Entry: ₹{trade.entry_price}")
        print(f"Stop Loss: ₹{trade.stop_loss}")
        print(f"Target: ₹{trade.target}")
        print(f"Premium: ₹{trade.premium_paid}")
        print(f"Delta: {trade.delta}")
        print(f"Theta: {trade.theta}")

        # Validate
        valid, message = risk_manager.validate_trade(trade, 0, 0)
        print(f"\nValidation: {message}")
