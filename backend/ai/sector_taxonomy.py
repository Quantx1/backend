"""NSE symbol → canonical sector taxonomy.

Generic helper used by F5 AI Portfolio (sector caps in optimizer) and F7
Portfolio Doctor (concentration risk by sector). This is pure metadata
lookup — NOT the deprecated F10 rotation feature.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


CANONICAL_SECTORS: List[str] = [
    "Banking",
    "IT",
    "Auto",
    "FMCG",
    "Pharma",
    "Metals",
    "Energy",
    "Financial Services",
    "Capital Goods",
    "Realty",
    "Consumer",
]


RAW_TO_CANONICAL: Dict[str, str] = {
    "Banking": "Banking",
    "PSU Bank": "Banking",
    "Private Bank": "Banking",
    "IT": "IT",
    "Internet": "IT",
    "Auto": "Auto",
    "FMCG": "FMCG",
    "Food Tech": "FMCG",
    "Pharma": "Pharma",
    "Healthcare": "Pharma",
    "Metals": "Metals",
    "Steel": "Metals",
    "Mining": "Metals",
    "Energy": "Energy",
    "Power": "Energy",
    "Gas": "Energy",
    "NBFC": "Financial Services",
    "Finance": "Financial Services",
    "Insurance": "Financial Services",
    "Broking": "Financial Services",
    "Capital Goods": "Capital Goods",
    "Infrastructure": "Capital Goods",
    "Infra": "Capital Goods",
    "Defence": "Capital Goods",
    "Electricals": "Capital Goods",
    "Electronics": "Capital Goods",
    "Cement": "Capital Goods",
    "Ports": "Capital Goods",
    "Real Estate": "Realty",
    "Consumer": "Consumer",
    "Consumer Durables": "Consumer",
    "Paints": "Consumer",
    "Chemicals": "Consumer",
    "Retail": "Consumer",
    "Textiles": "Consumer",
    "Telecom": "Consumer",
    "Aviation": "Consumer",
    "Logistics": "Consumer",
    "Travel": "Consumer",
    "Pipes": "Consumer",
    "Cables": "Consumer",
    "Diversified": "Consumer",
}


def map_to_canonical(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return RAW_TO_CANONICAL.get(raw.strip())


def _load_stock_info() -> Dict[str, Dict[str, str]]:
    try:
        from ..data.screener.engine import NSE_STOCK_INFO
        return NSE_STOCK_INFO
    except Exception as exc:
        logger.warning("NSE_STOCK_INFO unavailable: %s", exc)
        return {}


def sector_for_symbol(symbol: str) -> Optional[str]:
    """Canonical sector for a symbol, or None if unmapped."""
    info = _load_stock_info().get(symbol.upper(), {})
    return map_to_canonical(info.get("sector"))
