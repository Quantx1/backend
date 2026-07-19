"""
Real Qlib, NSE-adapted.

This package is the F2/F3/F5 alpha source. At runtime:

    from backend.ai.qlib import get_qlib_engine, load_universe

    engine = get_qlib_engine()          # qlib.init + booster load
    if engine.loaded:
        rows = engine.rank_universe(instruments="nse_all")
        # [{"symbol", "trade_date", "qlib_rank", "qlib_score_raw"}, ...]

Data-flow overview::

    yfinance/jugaad  →  scripts/data/ingest_nse_to_qlib.py  →  ~/.qlib/qlib_data/nse_data/
                                                         │
                                                         ├── calendars/day.txt (NSE)
                                                         ├── instruments/nifty50|100|250|500|nse_all.txt
                                                         └── features/<sym>/{open,high,low,close,volume,factor,vwap}.day.bin
                                                                │
                                                                ▼
    scripts/train/train_qlib_alpha158.py  →  Alpha158 + LGBModel  →  artifacts/models/qlib_alpha158/
                                                                │
                                                                ▼
                       scripts/train/upload_existing_models_to_b2.py  →  B2 registry
                                                                │
                                                                ▼
                       scheduler.qlib_nightly_rank (15:40 IST)  →  alpha_scores table

Public API — ``calendar.py`` + ``dump_bin.py`` + ``data_handler.py`` are
used only by the ingestion script. Runtime callers use ``engine.py`` and
the ``ranking.py`` helper.
"""

from .calendar import build_qlib_calendar, is_trading_day, nse_sessions
from .data_handler import (
    NSE_TIER_FILES,
    load_history,
    load_history_many,
    load_universe,
    tier_membership,
)
from .engine import QlibEngine, get_qlib_engine
from .ranking import rank_cross_section

__all__ = [
    "NSE_TIER_FILES",
    "QlibEngine",
    "build_qlib_calendar",
    "get_qlib_engine",
    "is_trading_day",
    "load_history",
    "load_history_many",
    "load_universe",
    "nse_sessions",
    "rank_cross_section",
    "tier_membership",
]
