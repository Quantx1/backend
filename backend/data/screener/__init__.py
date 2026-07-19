"""Live screener — scanner orchestrator + 50+ filter definitions.

Public API (filled in as PR-A4 tasks land)::

    from backend.data.screener import (
        LiveScreenerEngine, get_live_screener,
        # filter helpers as needed
    )

See ``docs/superpowers/plans/2026-05-25-backend-structural-restructure.md``.
"""

from . import filters  # noqa: F401
from .engine import (
    LiveScreenerEngine,
    NSE_STOCK_INFO,
    SCANNER_MENU,
    get_live_screener,
)

__all__: list[str] = [
    "LiveScreenerEngine",
    "NSE_STOCK_INFO",
    "SCANNER_MENU",
    "get_live_screener",
    "filters",
]
