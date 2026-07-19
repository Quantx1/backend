"""Pure mapper: screener.in get_fundamentals() dict -> fundamentals_history row.

Honest-empty — missing fields become None (never fabricated). The screener.in
fetch lives in data/fundamentals/screener_in.py; this module only maps."""
from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _num(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _growth(g: Dict, prefix: str):
    """screener.in growth is stored period-suffixed (e.g. ``sales_growth_3_years``);
    pick a representative period (3yr → 5yr → ttm → 1yr → 10yr), honest-empty."""
    for suffix in ("3_years", "5_years", "ttm", "1_year", "10_years"):
        v = g.get(f"{prefix}_{suffix}")
        if v is not None:
            return _num(v)
    return _num(g.get(prefix))  # bare key fallback


def map_fundamentals_row(symbol: str, data: Optional[Dict], snapshot_date: str) -> Optional[Dict]:
    """Map a screener.in get_fundamentals() result to a fundamentals_history row.
    Returns None when there's no data at all (honest-empty)."""
    if not data:
        return None
    f = data.get("fundamentals") or {}
    ph = data.get("promoter_holding") or {}
    g = data.get("growth") or {}
    return {
        "snapshot_date": snapshot_date, "symbol": symbol,
        "pe": _num(f.get("pe")), "roe": _num(f.get("roe")), "roce": _num(f.get("roce")),
        "market_cap_cr": _num(f.get("market_cap_cr")),
        "book_value": _num(f.get("book_value")),
        "dividend_yield": _num(f.get("dividend_yield")),
        "current_price": _num(f.get("current_price")),
        # eps + debt_to_equity: screener.in's headline ratios don't expose these
        # reliably — reserved (honest-empty) until a deeper source (Bharat-SM-Data).
        "debt_to_equity": _num(f.get("debt_to_equity")),
        "eps": _num(f.get("eps")),
        "sales_growth": _growth(g, "sales_growth"),
        "profit_growth": _growth(g, "profit_growth"),
        "promoter_pct": _num(ph.get("promoter_pct")),
        "source": data.get("source") or "screener.in",
    }
