"""Trading layer — order execution, risk, P&L, F&O engine.

Public API (filled in incrementally as PR-A5 lands)::

    from backend.trading import (
        compute_unrealized_pnl, compute_close_pnl, ClosePnL,
        # more added in subsequent commits
    )

See ``docs/superpowers/plans/2026-05-25-backend-structural-restructure.md``.
"""

from .autopilot_service import AutoPilotService, RebalanceDecision
from .eligibility import LiveTradeEligibility, check_live_trade_eligibility
from .execution import TradeExecutionService
from .fo import FOTradingEngine, InstrumentMaster, NSE_LOT_SIZES
from .pnl import (
    ClosePnL,
    EQUITY_CHARGE_RATE,
    FNO_CHARGE_RATE,
    compute_close_pnl,
    compute_unrealized_pnl,
)
from .risk import (
    Direction,
    FOCalculator,
    MarketCondition,
    PositionSize,
    RISK_PROFILES,
    RiskLevel,
    RiskManagementEngine,
    RiskProfile,
    Segment,
    Signal,
)

__all__ = [
    # pnl
    "ClosePnL", "EQUITY_CHARGE_RATE", "FNO_CHARGE_RATE",
    "compute_close_pnl", "compute_unrealized_pnl",
    # risk
    "Direction", "FOCalculator", "MarketCondition", "PositionSize",
    "RISK_PROFILES", "RiskLevel", "RiskManagementEngine",
    "RiskProfile", "Segment", "Signal",
    # eligibility
    "LiveTradeEligibility", "check_live_trade_eligibility",
    # execution
    "TradeExecutionService",
    # autopilot
    "AutoPilotService", "RebalanceDecision",
    # F&O package
    "FOTradingEngine", "InstrumentMaster", "NSE_LOT_SIZES",
]
