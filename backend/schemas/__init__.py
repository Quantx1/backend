"""
Quant X API Schemas
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, date
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class Segment(str, Enum):
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


class TradingMode(str, Enum):
    SIGNAL_ONLY = "signal_only"
    SEMI_AUTO = "semi_auto"
    FULL_AUTO = "full_auto"


class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class TradeStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SignalStatus(str, Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    TARGET_HIT = "target_hit"
    SL_HIT = "sl_hit"
    EXPIRED = "expired"


class BillingPeriod(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


# ============================================================================
# AUTH SCHEMAS
# ============================================================================

class UserSignup(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2)
    phone: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_at: int
    user: Dict[str, Any]


class PasswordReset(BaseModel):
    email: EmailStr


# ============================================================================
# USER SCHEMAS
# ============================================================================

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    capital: Optional[float] = Field(None, ge=10000)
    risk_profile: Optional[RiskProfile] = None
    trading_mode: Optional[TradingMode] = None
    max_positions: Optional[int] = Field(None, ge=1, le=20)
    risk_per_trade: Optional[float] = Field(None, ge=0.5, le=10)
    fo_enabled: Optional[bool] = None
    preferred_option_type: Optional[str] = None
    daily_loss_limit: Optional[float] = Field(None, ge=1, le=20)
    weekly_loss_limit: Optional[float] = Field(None, ge=1, le=30)
    monthly_loss_limit: Optional[float] = Field(None, ge=1, le=50)
    trailing_sl_enabled: Optional[bool] = None
    notifications_enabled: Optional[bool] = None
    telegram_chat_id: Optional[str] = None


class UserProfile(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    phone: Optional[str]
    capital: float
    risk_profile: str
    trading_mode: str
    subscription_status: str
    subscription_plan_id: Optional[str]
    total_trades: int
    winning_trades: int
    total_pnl: float
    broker_connected: bool
    broker_name: Optional[str]
    fo_enabled: bool
    paper_trading_started_at: Optional[datetime] = None
    live_trading_whitelisted: Optional[bool] = None
    kill_switch_active: Optional[bool] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# SIGNAL SCHEMAS
# ============================================================================

class SignalBase(BaseModel):
    symbol: str
    exchange: str = "NSE"
    segment: Segment = Segment.EQUITY
    direction: Direction
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: Optional[float] = None
    target_3: Optional[float] = None
    confidence: float = Field(..., ge=0, le=100)


class SignalResponse(SignalBase):
    id: str
    date: date
    status: SignalStatus
    risk_reward: Optional[float]
    expiry_date: Optional[date] = None
    strike_price: Optional[float] = None
    option_type: Optional[str] = None
    lot_size: Optional[int] = None
    is_premium: bool
    catboost_score: Optional[float]       # DB legacy name → ML meta-labeler (RandomForest) score
    tft_score: Optional[float]            # DB legacy name → TFT price-forecast bullish score
    stockformer_score: Optional[float]    # DB legacy name → raw strategy confidence
    model_agreement: int
    reasons: List[str]
    strategy_names: Optional[List[str]] = None
    tft_prediction: Optional[dict] = None
    generated_at: datetime

    class Config:
        from_attributes = True


class SignalsListResponse(BaseModel):
    date: str
    total: int
    long_signals: List[SignalResponse]
    short_signals: List[SignalResponse]
    all_signals: List[SignalResponse]


# ============================================================================
# TRADE SCHEMAS
# ============================================================================

class ExecuteTrade(BaseModel):
    signal_id: str
    quantity: Optional[int] = None
    lots: Optional[int] = None
    custom_sl: Optional[float] = None
    custom_target: Optional[float] = None


class CloseTrade(BaseModel):
    exit_price: Optional[float] = None
    reason: str = "manual"


class TradeResponse(BaseModel):
    id: str
    user_id: str
    signal_id: Optional[str]
    symbol: str
    exchange: str
    segment: str
    direction: str
    quantity: int
    lots: int
    entry_price: float
    average_price: Optional[float]
    stop_loss: float
    target: float
    exit_price: Optional[float]
    status: str
    execution_mode: Optional[str] = None
    broker_order_id: Optional[str] = None
    gross_pnl: Optional[float]
    net_pnl: Optional[float]
    pnl_percent: Optional[float]
    exit_reason: Optional[str]
    created_at: datetime
    executed_at: Optional[datetime]
    closed_at: Optional[datetime]

    class Config:
        from_attributes = True


# ============================================================================
# POSITION SCHEMAS
# ============================================================================

class PositionResponse(BaseModel):
    id: str
    symbol: str
    exchange: str
    segment: str
    direction: str
    quantity: int
    lots: int
    average_price: float
    current_price: Optional[float]
    stop_loss: float
    target: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    margin_used: Optional[float]
    is_active: bool
    opened_at: datetime

    class Config:
        from_attributes = True


class PositionUpdate(BaseModel):
    stop_loss: Optional[float] = None
    target: Optional[float] = None


# ============================================================================
# PORTFOLIO SCHEMAS
# ============================================================================

class PortfolioSummary(BaseModel):
    capital: float
    deployed: float
    available: float
    margin_used: float
    unrealized_pnl: float
    positions: List[PositionResponse]
    equity_positions: List[PositionResponse]
    fo_positions: List[PositionResponse]


class PortfolioHistory(BaseModel):
    date: str
    day_pnl: float
    day_pnl_percent: float
    cumulative_pnl: float
    trades_taken: int
    win_rate: float


class PerformanceMetrics(BaseModel):
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_pnl: float
    best_trade: float
    worst_trade: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None


# ============================================================================
# PAYMENT SCHEMAS
# ============================================================================

class CreateOrder(BaseModel):
    plan_id: str
    billing_period: BillingPeriod


class VerifyPayment(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class PaymentResponse(BaseModel):
    order_id: str
    amount: int
    currency: str
    key_id: str


# ============================================================================
# BROKER SCHEMAS
# ============================================================================

class BrokerConnect(BaseModel):
    broker_name: str = Field(..., pattern="^(zerodha|angelone|upstox)$")
    api_key: str
    api_secret: Optional[str] = None
    client_id: Optional[str] = None
    totp_secret: Optional[str] = None
    access_token: Optional[str] = None
    redirect_url: Optional[str] = None


class BrokerStatus(BaseModel):
    connected: bool
    broker_name: Optional[str]
    last_sync: Optional[datetime]


# ============================================================================
# MARKET DATA SCHEMAS
# ============================================================================

class MarketStatus(BaseModel):
    timestamp: str
    is_open: bool
    is_trading_day: bool
    market_hours: str


class MarketData(BaseModel):
    date: str
    nifty_close: float
    nifty_change_percent: float
    banknifty_close: Optional[float]
    vix_close: float
    market_trend: str
    risk_level: str
    fii_cash: Optional[float]
    dii_cash: Optional[float]


class RiskAssessment(BaseModel):
    vix: float
    risk_level: str
    recommendation: str
    nifty_change: float
    fii_net: float
    market_trend: str
    circuit_breaker: bool


# ============================================================================
# NOTIFICATION SCHEMAS
# ============================================================================

class NotificationResponse(BaseModel):
    id: str
    type: str
    priority: str
    title: str
    message: str
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# WATCHLIST SCHEMAS
# ============================================================================

class WatchlistAdd(BaseModel):
    symbol: str
    segment: Segment = Segment.EQUITY
    alert_price_above: Optional[float] = None
    alert_price_below: Optional[float] = None


# PR 112 — partial-update payload for an existing watchlist row.
# All fields optional so a client can flip just one threshold or
# toggle alert_enabled without re-stating the rest.
class WatchlistUpdate(BaseModel):
    alert_price_above: Optional[float] = None
    alert_price_below: Optional[float] = None
    alert_enabled: Optional[bool] = None
    notes: Optional[str] = None


class WatchlistResponse(BaseModel):
    id: str
    symbol: str
    segment: str
    alert_enabled: bool
    added_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# STATS SCHEMAS
# ============================================================================

class UserStats(BaseModel):
    capital: float
    total_pnl: float
    total_trades: int
    winning_trades: int
    win_rate: float
    open_positions: int
    unrealized_pnl: float
    today_pnl: float
    week_pnl: float
    subscription_status: str


class DashboardOverview(BaseModel):
    stats: UserStats
    recent_signals: List[SignalResponse]
    active_positions: List[PositionResponse]
    market_status: MarketStatus
    notifications_count: int


# ============================================================================
# ASSISTANT SCHEMAS
# ============================================================================

# The AssistantChat*/DomainGuard request-response schemas were deleted with
# the legacy finance-assistant brain (2026-07-11 chat unification). Only the
# usage meter survives — it reports the Copilot's daily credit window.
class AssistantUsage(BaseModel):
    tier: Literal["free", "pro", "elite"]
    credits_limit: int
    credits_used: int
    credits_remaining: int
    reset_at: str


class AssistantUsageResponse(BaseModel):
    usage: AssistantUsage


# ============================================================================
# GENERIC RESPONSES
# ============================================================================

class SuccessResponse(BaseModel):
    success: bool = True
    message: str = "Operation completed successfully"


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
