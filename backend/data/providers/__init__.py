"""Market-data providers — Kite Connect (primary) + yfinance (fallback).

Public API::

    from backend.data.providers import (
        KiteDataProvider, get_kite_data_provider, auto_refresh_kite_token,
        YFinanceProvider, get_yfinance_provider,
    )
"""
from .kite import (
    INDEX_NAME_MAP,
    KiteAdminClient,
    KiteDataProvider,
    auto_refresh_kite_token,
    get_kite_admin_client,
    get_kite_data_provider,
)
from .yfinance import (
    INDEX_YF_MAP,
    YFinanceProvider,
    get_yfinance_provider,
)

__all__ = [
    # kite
    "INDEX_NAME_MAP", "KiteAdminClient", "KiteDataProvider",
    "auto_refresh_kite_token", "get_kite_admin_client",
    "get_kite_data_provider",
    # yfinance
    "INDEX_YF_MAP", "YFinanceProvider", "get_yfinance_provider",
]
