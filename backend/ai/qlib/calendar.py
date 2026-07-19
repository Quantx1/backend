"""
NSE trading calendar — canonical source of "is this a trading day?" +
ordered calendar for Qlib's binary data format.

Primary source: ``pandas_market_calendars['NSE']`` (tz Asia/Calcutta,
maintains the National Stock Exchange of India session calendar
including all declared holidays and special sessions). Secondary source
for validation: the locally-maintained ``data/nse_holidays_<year>.json``.

Qlib's binary format needs one ordered list of trading-day dates with
no gaps — ``build_qlib_calendar()`` produces that list.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_nse_calendar():
    """Return the ``pandas_market_calendars`` NSE calendar. Raises
    ImportError if the library isn't installed (dev error — it's a
    hard dependency of this module)."""
    import pandas_market_calendars as mcal
    return mcal.get_calendar("NSE")


def nse_sessions(start: str, end: Optional[str] = None) -> pd.DatetimeIndex:
    """Return every NSE trading day in ``[start, end]`` inclusive.

    ``end`` defaults to today + 30 days so forward labels don't lose data.
    """
    cal = get_nse_calendar()
    end = end or (date.today() + timedelta(days=30)).isoformat()
    schedule = cal.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index.normalize().unique()).sort_values()


def is_trading_day(d: date) -> bool:
    """True if ``d`` is an NSE session."""
    cal = get_nse_calendar()
    ts = pd.Timestamp(d)
    sched = cal.schedule(start_date=ts, end_date=ts)
    return len(sched) > 0


def build_qlib_calendar(start: str = "2010-01-01") -> List[str]:
    """Return every NSE trading day as ``YYYY-MM-DD`` strings from
    ``start`` through today + 30 days. Written to
    ``<provider_uri>/calendars/day.txt`` during Qlib ingestion."""
    return [d.strftime("%Y-%m-%d") for d in nse_sessions(start)]
