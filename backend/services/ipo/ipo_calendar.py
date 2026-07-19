"""
IPO calendar (Phase 4, 2026-07-12).

Serves the primary-market IPO calendar — upcoming issues, currently-open issues
with their live subscription multiple, price band, dates and status — from NSE's
OWN public issues API (nseindia.com/api/all-upcoming-issues + ipo-current-issue).
Cached-live (like the F&O snapshot): a short in-process cache, no DB table, always
reasonably fresh, and HONEST-EMPTY when NSE is unreachable — never fabricated.

GMP (grey-market premium) is deliberately NOT sourced: it is unofficial grey-
market data NSE does not publish, and scraping it for paying users runs into the
Path-A data-licensing rule. The honest answer is "we don't provide GMP" rather
than a scraped number of unknown provenance.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_BASE = "https://www.nseindia.com"
_UPCOMING = f"{_BASE}/api/all-upcoming-issues?category=ipo"
_CURRENT = f"{_BASE}/api/ipo-current-issue"
_HEADERS = {
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "referer": f"{_BASE}/market-data/all-upcoming-issues-ipo",
}

_CACHE_TTL_S = 1800  # 30 min — IPO data moves slowly
_cache: Dict[str, Any] = {"ts": 0.0, "data": None}


def _num(text: Any) -> Optional[float]:
    if text is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return float(m.group()) if m else None


def _parse_price_band(s: Any) -> Dict[str, Optional[float]]:
    """'Rs.203 to Rs.214' → {low: 203, high: 214}. Single value → both equal."""
    if not s:
        return {"low": None, "high": None}
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(s).replace(",", ""))]
    if not nums:
        return {"low": None, "high": None}
    return {"low": min(nums), "high": max(nums)}


def _parse_date(s: Any) -> Optional[str]:
    """'09-Jul-2026' → '2026-07-09' (ISO). None when unparseable."""
    if not s:
        return None
    for fmt in ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s).strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _norm_status(s: Any) -> str:
    v = str(s or "").strip().lower()
    if v in ("active", "open"):
        return "open"
    if v in ("forthcoming", "upcoming"):
        return "upcoming"
    if v in ("closed", "close"):
        return "closed"
    return v or "unknown"


def _shape(raw: Dict[str, Any]) -> Dict[str, Any]:
    band = _parse_price_band(raw.get("issuePrice"))
    sub_x = _num(raw.get("noOfTime"))  # subscription multiple, only on current issues
    return {
        "symbol": (raw.get("symbol") or "").strip() or None,
        "company": (raw.get("companyName") or "").strip() or None,
        "price_band_low": band["low"],
        "price_band_high": band["high"],
        "price_band": (raw.get("issuePrice") or "").replace("Rs.", "₹").strip() or None,
        "open_date": _parse_date(raw.get("issueStartDate")),
        "close_date": _parse_date(raw.get("issueEndDate")),
        "status": _norm_status(raw.get("status")),
        "series": (raw.get("series") or "").strip() or None,
        "subscription_x": round(sub_x, 2) if sub_x is not None else None,
        # GMP intentionally absent — see module docstring.
    }


def _fetch_raw() -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Fetch both NSE issue feeds behind a session (NSE gates the api on a
    homepage cookie). Returns None on any failure — the caller honest-empties."""
    try:
        import requests
    except Exception:  # noqa: BLE001
        return None
    try:
        s = requests.Session()
        s.get(_BASE, headers=_HEADERS, timeout=10)  # prime cookies
        up = s.get(_UPCOMING, headers=_HEADERS, timeout=10)
        cur = s.get(_CURRENT, headers=_HEADERS, timeout=10)
        upcoming = up.json() if up.status_code == 200 else []
        current = cur.json() if cur.status_code == 200 else []
        if not isinstance(upcoming, list):
            upcoming = []
        if not isinstance(current, list):
            current = []
        return {"upcoming": upcoming, "current": current}
    except Exception as e:  # noqa: BLE001
        logger.debug("NSE IPO fetch failed: %s", e)
        return None


def fetch_ipo_calendar(*, force: bool = False) -> Dict[str, Any]:
    """Return the IPO calendar: {available, as_of, open: [...], upcoming: [...]}.

    ``open`` = issues currently accepting bids (with a live subscription multiple);
    ``upcoming`` = announced-but-not-yet-open. Honest-empty (available=False,
    empty lists) when NSE is unreachable."""
    now = time.monotonic()
    if not force and _cache["data"] is not None and (now - _cache["ts"]) < _CACHE_TTL_S:
        return _cache["data"]

    raw = _fetch_raw()
    if raw is None:
        out = {"available": False, "as_of": None, "open": [], "upcoming": [],
                "note": "IPO calendar source (NSE) is unavailable right now."}
        # brief negative cache so a blip self-heals
        _cache["ts"] = now
        _cache["data"] = out
        return out

    # Current-issue feed carries the live subscription multiple → treat as "open".
    open_syms = set()
    open_list: List[Dict[str, Any]] = []
    for r in raw["current"]:
        shaped = _shape(r)
        if shaped["symbol"]:
            open_syms.add(shaped["symbol"])
        open_list.append(shaped)

    upcoming_list: List[Dict[str, Any]] = []
    for r in raw["upcoming"]:
        shaped = _shape(r)
        # De-dup: an issue in the current feed shouldn't also show as upcoming.
        if shaped["symbol"] and shaped["symbol"] in open_syms:
            continue
        # NSE tags open issues 'Active' in the upcoming feed too — bucket by status.
        (open_list if shaped["status"] == "open" else upcoming_list).append(shaped)

    out = {
        "available": bool(open_list or upcoming_list),
        "as_of": datetime.utcnow().isoformat() + "Z",
        "open": open_list,
        "upcoming": upcoming_list,
    }
    _cache["ts"] = now
    _cache["data"] = out
    return out
