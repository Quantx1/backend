"""screener.in fundamentals scraper for the Portfolio Doctor (F7).

Fetches the public company page (consolidated financials preferred) and parses
the headline ratios, compounded growth ranges, pros/cons, and promoter holding.
Peers are loaded by screener via a separate async request and are intentionally
left best-effort-empty here (reliability over completeness).

No synthetic data (no-fallbacks lock ``project_no_fallbacks_no_refunds``): on
any failure this returns ``available=False`` with empty payloads + a
``last_error``, so the Doctor agents see honest-empty rather than fabricated
fundamentals. Cached per symbol (fundamentals move slowly).

The audit found ``run_finrobot_doctor`` was called with no fundamentals, so the
agents graded ROE/Debt/peers from empty JSON. This module is the source that
fills that gap.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_BASE = "https://www.screener.in/company"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0 Safari/537.36"
)
# Split TTLs: real fundamentals move slowly (6h); a transient failure (timeout /
# 403 / 5xx) is cached only briefly so a screener.in blip self-heals on the next
# request instead of pinning honest-empty for hours, process-wide.
_TTL_OK = 6 * 3600.0
_TTL_FAIL = 60.0
_cache: Dict[str, tuple] = {}  # sym -> (ts, payload, ttl)

# Process-wide rate budget against screener.in (an unauthenticated public site
# with no API contract). The Doctor can fan out one scrape per position; without
# this, a 30-stock portfolio bursts ~30-60 GETs from one IP and trips throttling.
_FETCH_LOCK = threading.Lock()
_MIN_INTERVAL_S = 0.4  # >=400ms between requests -> <=2.5 req/s, gentle
_last_fetch_ts = [0.0]


def _throttle() -> None:
    """Space out screener.in requests. Holds the lock only for the short gate
    (the inter-request sleep), never during the network call itself."""
    with _FETCH_LOCK:
        wait = _MIN_INTERVAL_S - (time.monotonic() - _last_fetch_ts[0])
        if wait > 0:
            time.sleep(wait)
        _last_fetch_ts[0] = time.monotonic()


# screener top-ratio label → our key. High/Low is skipped (range, not a scalar).
_RATIO_KEYS = {
    "Market Cap": "market_cap_cr",
    "Current Price": "current_price",
    "Stock P/E": "pe",
    "Book Value": "book_value",
    "Dividend Yield": "dividend_yield",
    "ROCE": "roce",
    "ROE": "roe",
    "Face Value": "face_value",
}

# ranges-table title → key prefix
_RANGE_KEYS = {
    "Compounded Sales Growth": "sales_growth",
    "Compounded Profit Growth": "profit_growth",
    "Return on Equity": "roe_hist",
}


def _num(text: Any) -> Optional[float]:
    """Parse '₹17,64,237Cr.' / '22.7' / '8.91%' / '15%' → float (or None)."""
    if text is None:
        return None
    s = str(text).replace(",", "").replace("₹", "").replace("%", "").strip()
    s = re.sub(r"[A-Za-z.]+$", "", s).strip()  # drop trailing 'Cr.', 'Cr', etc.
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def _fetch_html(symbol: str) -> Optional[str]:
    import requests

    sym = symbol.upper().strip().replace(".NS", "")
    headers = {"user-agent": _UA, "accept": "text/html"}
    for path in (f"{_BASE}/{sym}/consolidated/", f"{_BASE}/{sym}/"):
        try:
            _throttle()
            r = requests.get(path, headers=headers, timeout=15)
            if r.status_code == 200 and "top-ratios" in r.text:
                return r.text
        except Exception as e:  # network / DNS / timeout — try next, then degrade
            logger.debug("screener fetch %s failed: %s", path, e)
    return None


def _parse(html: str) -> Dict[str, Any]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # ── headline ratios ──
    fundamentals: Dict[str, Any] = {}
    ul = soup.select_one("#top-ratios")
    if ul:
        for li in ul.select("li"):
            name_el = li.select_one(".name")
            if not name_el:
                continue
            key = _RATIO_KEYS.get(name_el.get_text(strip=True))
            if not key:
                continue
            val_el = li.select_one(".value") or li
            fundamentals[key] = _num(val_el.get_text(" ", strip=True))

    # ── compounded growth + ROE history ──
    growth: Dict[str, Any] = {}
    for tbl in soup.select("table.ranges-table"):
        head = tbl.select_one("th")
        prefix = _RANGE_KEYS.get(head.get_text(strip=True) if head else "")
        if not prefix:
            continue
        for tr in tbl.select("tr")[1:]:
            cells = [c.get_text(strip=True) for c in tr.select("td")]
            if len(cells) >= 2:
                period = cells[0].replace(":", "").strip().lower().replace(" ", "_")
                growth[f"{prefix}_{period}"] = _num(cells[1])

    # ── qualitative pros / cons ──
    pros = [" ".join(li.get_text(strip=True).split()) for li in soup.select(".pros li")]
    cons = [" ".join(li.get_text(strip=True).split()) for li in soup.select(".cons li")]

    # ── promoter holding (latest period from the shareholding table) ──
    promoter_holding: Dict[str, Any] = {}
    sh = soup.select_one("#shareholding")
    if sh:
        for tr in sh.select("table tr"):
            cells = [c.get_text(strip=True) for c in tr.select("td,th")]
            if cells and "promoter" in cells[0].lower():
                latest = _num(cells[-1])
                if latest is not None:
                    promoter_holding = {
                        "promoter_pct": latest,
                        "periods": len(cells) - 1,
                    }
                    break

    return {
        "fundamentals": fundamentals,
        "growth": growth,
        "pros": pros,
        "cons": cons,
        "promoter_holding": promoter_holding,
        "peers": [],  # screener loads peers async; best-effort omitted for reliability
    }


def get_fundamentals(symbol: str) -> Dict[str, Any]:
    """Fundamentals for an NSE symbol from screener.in.

    Returns a dict with ``available`` (bool), ``source``, ``fundamentals``,
    ``growth``, ``pros``, ``cons``, ``promoter_holding``, ``peers``,
    ``last_error``. ``available=False`` + empty payloads on any failure —
    never synthetic.
    """
    sym = symbol.upper().strip().replace(".NS", "")
    now = time.monotonic()
    hit = _cache.get(sym)
    if hit and now - hit[0] < hit[2]:  # hit = (ts, payload, ttl)
        return hit[1]

    out: Dict[str, Any] = {
        "symbol": sym,
        "available": False,
        "source": "unavailable",
        "fundamentals": {},
        "growth": {},
        "pros": [],
        "cons": [],
        "promoter_holding": {},
        "peers": [],
        "last_error": None,
    }
    try:
        html = _fetch_html(sym)
        if not html:
            out["last_error"] = "screener.in page unavailable"
        else:
            parsed = _parse(html)
            out.update(parsed)
            has_data = bool(parsed["fundamentals"])
            out["available"] = has_data
            out["source"] = "screener.in" if has_data else "unavailable"
            if not has_data:
                out["last_error"] = "page fetched but no top-ratios parsed"
    except ImportError as e:
        out["last_error"] = f"missing dependency: {e}"
    except Exception as e:
        out["last_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    if out["last_error"]:
        logger.debug("get_fundamentals(%s): %s", sym, out["last_error"])

    # Real data caches long; a transient/unavailable result caches briefly so a
    # screener.in blip self-heals instead of poisoning the symbol for 6h.
    ttl = _TTL_OK if out["available"] else _TTL_FAIL
    _cache[sym] = (now, out, ttl)
    return out


def to_doctor_inputs(symbol: str) -> Dict[str, Any]:
    """Shape screener.in fundamentals into ``run_finrobot_doctor`` kwargs.

    Returns ``{fundamentals, promoter_holding, peers}`` — fundamentals folds in
    the growth ranges + pros/cons so the CoT agents grade real numbers. All
    empty when screener is unavailable (honest-empty)."""
    d = get_fundamentals(symbol)
    fundamentals: Dict[str, Any] = dict(d["fundamentals"])
    fundamentals.update(d["growth"])
    if d["pros"]:
        fundamentals["pros"] = d["pros"]
    if d["cons"]:
        fundamentals["cons"] = d["cons"]
    fundamentals["_source"] = d["source"]
    return {
        "fundamentals": fundamentals if d["available"] else {},
        "promoter_holding": d["promoter_holding"],
        "peers": d["peers"],
    }
