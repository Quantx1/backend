"""Intraday scanner — 5m/15m setups on per-symbol intraday bars.

P2 (2026-05-31). Parallel pipeline to services/chart_patterns/ but for
intraday-only setups (ORB, VWAP Bounce/Rejection, Anchored VWAP,
Opening Drive, Inside Bar Failure, Power Hour Fade, Mean Reversion to
VWAP, Gap-and-Go vs Gap-and-Trap, Intraday Squeeze, ORB Failure,
EOD Drift, CVD Divergence Watch).

Why a separate module: the equity SCANNER_FILTERS engine works off the
daily summary_df (one row per symbol with EOD indicators). Intraday
setups need per-bar indicators (VWAP, opening range, intraday RSI)
computed on the fly from fresh 5m/15m bars. Sharing the daily engine
would either bloat summary_df with intraday columns (wasteful for EOD
scanners) or run the daily pipeline on intraday bars (wrong window).

Sources verified per the deep-research audit (see screener/filters.py
for the per-setup docstring with cited author / paper / URL).
"""

from .indicators import (
    session_vwap,
    vwap_bands,
    opening_range,
    initial_balance,
    anchored_vwap,
    bb_squeeze_inside_kc,
    cumulative_delta_tickrule,
    is_lunch_window,
    is_power_hour,
    is_closing_auction,
)
from .scanner import (
    IntradayMatch,
    scan_intraday_setups,
    SETUP_CATALOG,
)

__all__ = [
    "session_vwap",
    "vwap_bands",
    "opening_range",
    "initial_balance",
    "anchored_vwap",
    "bb_squeeze_inside_kc",
    "cumulative_delta_tickrule",
    "is_lunch_window",
    "is_power_hour",
    "is_closing_auction",
    "IntradayMatch",
    "scan_intraday_setups",
    "SETUP_CATALOG",
]
