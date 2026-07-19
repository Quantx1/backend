"""
NSE trading-calendar helpers — pure utilities, no I/O state.

These were inline `_is_trading_day` / `_get_next_trading_day` /
`_is_market_hours` methods on ``SchedulerService``. Extracted into a
module so other services (auto-trader, paper-trading lifecycle, EOD
report generators) can reuse them without depending on the scheduler.

The primary path delegates to ``MarketDataProvider`` (Kite Connect when
configured, jugaad-data otherwise). The fallback heuristics (weekend
check + hardcoded NSE holiday list) only fire when the provider is
unavailable — this is operational resilience, not the kind of
heuristic-as-AI-output the no-fallbacks rule forbids.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# Canonical India Standard Time. ~12 modules across the codebase have
# their own ``IST = timezone(timedelta(hours=5, minutes=30))`` line; new
# code should import this one instead. NSE doesn't observe DST, so the
# fixed +05:30 offset is correct year-round.
IST = timezone(timedelta(hours=5, minutes=30))


# Hardcoded NSE holidays — fallback only when MarketDataProvider is down.
# Update annually from https://www.nseindia.com/resources/exchange-communication-holidays
_NSE_HOLIDAYS_FALLBACK: set[date] = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Ram Navami
    date(2026, 3, 31),   # Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Id
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali Balipratipada
    date(2026, 11, 24),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
}


# NSE regular session: 09:15 → 15:30 IST. Pre-open auction (09:00–09:15)
# and post-close (15:40–16:00) are intentionally excluded — most jobs
# care about regular-session hours.
_MARKET_OPEN = time(9, 15)
_MARKET_CLOSE = time(15, 30)


async def is_trading_day(check_date: Optional[date] = None) -> bool:
    """Return True when ``check_date`` (default: today) is an NSE trading day."""
    target = check_date or date.today()
    try:
        from .market import get_market_data_provider
        provider = get_market_data_provider()
        return provider.is_trading_day(target)
    except Exception as exc:
        logger.warning("Market data provider unavailable, using calendar fallback: %s", exc)
        if target.weekday() >= 5:    # Saturday/Sunday
            return False
        return target not in _NSE_HOLIDAYS_FALLBACK


def next_trading_day(after: Optional[date] = None) -> date:
    """Return the next trading day strictly after ``after`` (default: today)."""
    cursor = (after or date.today()) + timedelta(days=1)
    try:
        from .market import get_market_data_provider
        provider = get_market_data_provider()
        while not provider.is_trading_day(cursor):
            cursor += timedelta(days=1)
        return cursor
    except Exception as exc:
        logger.warning("Market data provider unavailable, using calendar fallback: %s", exc)
        # Fallback: skip weekends + hardcoded holidays.
        while cursor.weekday() >= 5 or cursor in _NSE_HOLIDAYS_FALLBACK:
            cursor += timedelta(days=1)
        return cursor


def is_market_open() -> bool:
    """Return True during NSE regular session (09:15–15:30 IST, Mon-Fri).

    Uses ``datetime.now(IST)`` in the fallback so the comparison is
    timezone-correct on any server (Railway/Vercel/Render typically run
    UTC). Previously this used ``datetime.now()`` which evaluated the
    9:15–15:30 window against server-local time — UTC server fallback
    would say "open" 14:45–21:00 IST and "closed" during actual session.
    """
    try:
        from .market import get_market_data_provider
        provider = get_market_data_provider()
        return provider.is_market_open()
    except Exception as exc:
        logger.warning("Market data provider unavailable, using session fallback: %s", exc)
        now_ist = datetime.now(IST)
        if now_ist.weekday() >= 5:
            return False
        return _MARKET_OPEN <= now_ist.time() <= _MARKET_CLOSE
