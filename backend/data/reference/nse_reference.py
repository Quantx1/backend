"""NSE reference mappers (pure) + lazy nselib fetchers.

Mappers turn raw nselib/NSE rows into clean Supabase rows; fetchers are thin
lazy wrappers (nselib imported inside the call) so the module imports without
nselib installed and the names are easy to adjust after verifying the install."""
from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

_SPLIT_HINTS = ("split", "sub-division", "face value")
_BONUS_HINTS = ("bonus",)
_DIV_HINTS = ("dividend",)
_RIGHTS_HINTS = ("rights",)
_BUYBACK_HINTS = ("buy back", "buyback")


def _clean(v) -> str:
    return str(v).strip() if v is not None else ""


def map_equity_master_rows(df) -> List[Dict]:
    """Map an NSE EQUITY_L-style DataFrame (columns may have leading spaces)
    to `instruments` rows. Tolerant of the column-name spacing nselib returns."""
    def col(row, *names):
        for n in names:
            for k in row.index:
                if k.strip().upper() == n.strip().upper():
                    return row[k]
        return None
    rows: List[Dict] = []
    for _, r in df.iterrows():
        symbol = _clean(col(r, "SYMBOL"))
        if not symbol:
            continue
        rows.append({
            "symbol": symbol, "exchange": "NSE", "instrument_type": "EQ",
            "isin": _clean(col(r, "ISIN NUMBER")) or None,
            "series": _clean(col(r, "SERIES")) or None,
            "name": _clean(col(r, "NAME OF COMPANY")) or None,
            "face_value": _to_num(col(r, "FACE VALUE")),
            "listing_date": _to_date(col(r, "DATE OF LISTING")),
            "status": "active", "source": "nselib",
        })
    return rows


def _classify_action(purpose: str) -> str:
    p = (purpose or "").lower()
    if any(h in p for h in _BONUS_HINTS):
        return "bonus"
    if any(h in p for h in _RIGHTS_HINTS):
        return "rights"
    if any(h in p for h in _BUYBACK_HINTS):
        return "buyback"
    if any(h in p for h in _SPLIT_HINTS):
        return "split"
    if any(h in p for h in _DIV_HINTS):
        return "dividend"
    return "other"


def map_corporate_action_rows(raw) -> List[Dict]:
    """Map nselib corporate-action dicts to `corporate_actions` rows."""
    rows: List[Dict] = []
    for r in raw or []:
        symbol = _clean(r.get("symbol") or r.get("Symbol"))
        ex_date = _clean(r.get("exDate") or r.get("ex_date") or r.get("Ex Date"))
        purpose = _clean(r.get("purpose") or r.get("Purpose"))
        if not symbol or not ex_date:
            continue
        rows.append({
            "symbol": symbol, "ex_date": ex_date,
            "action_type": _classify_action(purpose),
            "details": {"purpose": purpose}, "source": "nselib",
        })
    return rows


def _to_num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _to_date(v):
    s = _clean(v)
    return s or None


# ── NSE index constituents ────────────────────────────────────────────────
# Public NSE archive CSVs (Company Name, Industry, Symbol, Series, ISIN Code).
# Slugs are inconsistent (_list vs list, stray underscores) so each was probed
# against the live archive before being listed here. Category drives the
# broad/sectoral grouping the UI exposes (the table itself stores only
# memberships). "F&O STOCKS" is a synthetic index built from nselib's
# derivatives-eligible list (no CSV).
NSE_INDEX_ARCHIVE = "https://nsearchives.nseindia.com/content/indices/"

INDEX_CSV_MAP: Dict[str, "tuple"] = {
    # ── broad market ──
    "NIFTY 50": ("ind_nifty50list.csv", "broad"),
    "NIFTY NEXT 50": ("ind_niftynext50list.csv", "broad"),
    "NIFTY 100": ("ind_nifty100list.csv", "broad"),
    "NIFTY 200": ("ind_nifty200list.csv", "broad"),
    "NIFTY 500": ("ind_nifty500list.csv", "broad"),
    "NIFTY MIDCAP 50": ("ind_niftymidcap50list.csv", "broad"),
    "NIFTY MIDCAP 100": ("ind_niftymidcap100list.csv", "broad"),
    "NIFTY MIDCAP 150": ("ind_niftymidcap150list.csv", "broad"),
    "NIFTY SMALLCAP 50": ("ind_niftysmallcap50list.csv", "broad"),
    "NIFTY SMALLCAP 100": ("ind_niftysmallcap100list.csv", "broad"),
    "NIFTY SMALLCAP 250": ("ind_niftysmallcap250list.csv", "broad"),
    "NIFTY MICROCAP 250": ("ind_niftymicrocap250_list.csv", "broad"),
    "NIFTY TOTAL MARKET": ("ind_niftytotalmarket_list.csv", "broad"),
    "NIFTY LARGEMIDCAP 250": ("ind_niftylargemidcap250list.csv", "broad"),
    "NIFTY MIDSMALLCAP 400": ("ind_niftymidsmallcap400list.csv", "broad"),
    # ── sectoral / thematic ──
    "NIFTY BANK": ("ind_niftybanklist.csv", "sectoral"),
    "NIFTY AUTO": ("ind_niftyautolist.csv", "sectoral"),
    "NIFTY IT": ("ind_niftyitlist.csv", "sectoral"),
    "NIFTY FMCG": ("ind_niftyfmcglist.csv", "sectoral"),
    "NIFTY PHARMA": ("ind_niftypharmalist.csv", "sectoral"),
    "NIFTY METAL": ("ind_niftymetallist.csv", "sectoral"),
    "NIFTY MEDIA": ("ind_niftymedialist.csv", "sectoral"),
    "NIFTY REALTY": ("ind_niftyrealtylist.csv", "sectoral"),
    "NIFTY ENERGY": ("ind_niftyenergylist.csv", "sectoral"),
    "NIFTY PSU BANK": ("ind_niftypsubanklist.csv", "sectoral"),
    "NIFTY PRIVATE BANK": ("ind_nifty_privatebanklist.csv", "sectoral"),
    "NIFTY FINANCIAL SERVICES": ("ind_niftyfinancelist.csv", "sectoral"),
    "NIFTY HEALTHCARE": ("ind_niftyhealthcarelist.csv", "sectoral"),
    "NIFTY CONSUMER DURABLES": ("ind_niftyconsumerdurableslist.csv", "sectoral"),
    "NIFTY OIL GAS": ("ind_niftyoilgaslist.csv", "sectoral"),
    "NIFTY INFRA": ("ind_niftyinfralist.csv", "sectoral"),
    "NIFTY COMMODITIES": ("ind_niftycommoditieslist.csv", "sectoral"),
    "NIFTY CONSUMPTION": ("ind_niftyconsumptionlist.csv", "sectoral"),
}

# Source index whose Industry column is most authoritative for a stock's
# sector (widest coverage first). Used to derive instruments.sector.
_SECTOR_SOURCE_PRIORITY = ("NIFTY TOTAL MARKET", "NIFTY 500", "NIFTY 200", "NIFTY 100")

FNO_INDEX_NAME = "F&O STOCKS"

_NSE_CSV_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/csv,application/csv,*/*",
    "Referer": "https://www.nseindia.com/market-data/live-market-indices",
}


def _icol(row, *names):
    for n in names:
        for k in row.index:
            if str(k).strip().lower() == n.strip().lower():
                return row[k]
    return None


def map_index_constituent_rows(df, index_name: str) -> List[Dict]:
    """Map an NSE index CSV (Company Name, Industry, Symbol, ISIN Code) to
    `index_constituents` rows. Pure + symbol/industry tolerant."""
    if df is None or len(df) == 0:
        return []
    rows: List[Dict] = []
    for _, r in df.iterrows():
        sym = _clean(_icol(r, "Symbol", "SYMBOL")).upper()
        if not sym or sym.lower() in ("none", "nan"):
            continue
        rows.append({
            "index_name": index_name,
            "symbol": sym,
            "weight": None,
            "industry": _clean(_icol(r, "Industry")) or None,
            "source": "nseindices",
        })
    return rows


def build_sector_map(constituents_by_index: Dict[str, List[Dict]]) -> Dict[str, str]:
    """Derive {symbol -> sector} from index constituents, preferring the
    broadest/most-authoritative index's Industry classification."""
    sector: Dict[str, str] = {}
    ordered = list(_SECTOR_SOURCE_PRIORITY) + [
        k for k in constituents_by_index if k not in _SECTOR_SOURCE_PRIORITY
    ]
    for idx in ordered:
        for row in constituents_by_index.get(idx, []):
            ind = row.get("industry")
            sym = row.get("symbol")
            if sym and ind and sym not in sector:
                sector[sym] = ind
    return sector


# mcap tier from broad-index membership (most-inclusive tier wins via setdefault).
_MCAP_TIERS = (
    ("Large Cap", "NIFTY 100"),
    ("Mid Cap", "NIFTY MIDCAP 150"),
    ("Small Cap", "NIFTY SMALLCAP 250"),
    ("Micro Cap", "NIFTY MICROCAP 250"),
)


def build_mcap_map(constituents_by_index: Dict[str, List[Dict]]) -> Dict[str, str]:
    """Derive {symbol -> mcap tier} from broad-index membership. A symbol in
    NIFTY 100 is Large Cap; the first (most-inclusive) tier wins."""
    out: Dict[str, str] = {}
    for tier, idx in _MCAP_TIERS:
        for row in constituents_by_index.get(idx, []):
            out.setdefault(row["symbol"], tier)
    return out


# --- lazy fetchers (verify names against installed nselib before prod use) ---
def fetch_equity_master():
    """Return the NSE equity master DataFrame (lazy nselib)."""
    from nselib import capital_market
    return capital_market.equity_list()


def fetch_index_constituents_csv(filename: str, *, timeout: int = 20, retries: int = 3):
    """Download an NSE index constituent CSV -> DataFrame. Honest-empty on
    failure (never fabricates membership)."""
    import io
    import time

    import pandas as pd
    import requests

    url = NSE_INDEX_ARCHIVE + filename
    for i in range(retries):
        try:
            r = requests.get(url, headers=_NSE_CSV_HEADERS, timeout=timeout)
            first = r.text.splitlines()[0] if r.text else ""
            if r.status_code == 200 and "Symbol" in first:
                return pd.read_csv(io.StringIO(r.text))
        except Exception as e:
            logger.debug("index csv fetch failed %s: %s", filename, e)
        time.sleep(1.0 + i)
    return pd.DataFrame()


def fetch_fno_stock_symbols() -> List[str]:
    """Derivatives-eligible single-stock underlyings (lazy nselib)."""
    from nselib import capital_market
    df = capital_market.fno_equity_list()
    out: List[str] = []
    for _, r in df.iterrows():
        s = _clean(r.get("symbol") if hasattr(r, "get") else _icol(r, "symbol", "underlying"))
        s = (s or "").upper()
        if s and s.lower() not in ("none", "nan"):
            out.append(s)
    return sorted(set(out))


def fetch_corporate_actions(from_date: str, to_date: str):
    """Return nselib corporate actions for a date range (lazy)."""
    from nselib import capital_market
    return capital_market.corporate_action(from_date=from_date, to_date=to_date)
