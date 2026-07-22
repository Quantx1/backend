"""Big deals + corporate actions — NSE EOD-published disclosure reports.

SEBI lane: bulk deals, block deals and corporate actions are DISCLOSURE
reports NSE itself publishes on its website after close (same lane as the
FII/DII provisional numbers we already serve) — public information, shown
with an explicit "EOD · published" label. This is NOT live exchange data;
nothing here needs the broker gate. A SEBI professional still signs off on
the whole EOD-published lane before paid launch (existing go-live item).

Sourced via nselib (internal data plumbing), cached 1h in-process,
honest-empty per section on any failure.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_CACHE: Dict[str, tuple] = {}
_TTL_S = 3600


def _num(x: Any) -> float:
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return 0.0


def _deal_rows(df: Any, deal_type: str) -> List[Dict[str, Any]]:
    if df is None or not hasattr(df, "iterrows") or df.empty:
        return []
    out: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        qty = _num(r.get("QuantityTraded"))
        price = _num(r.get("TradePrice/Wght.Avg.Price"))
        if qty <= 0 or price <= 0:
            continue
        out.append({
            "date": str(r.get("Date") or "").strip(),
            "symbol": str(r.get("Symbol") or "").strip(),
            "client": str(r.get("ClientName") or "").strip().title()[:48],
            "side": str(r.get("Buy/Sell") or "").strip().upper(),
            "qty": int(qty),
            "price": round(price, 2),
            "value_cr": round(qty * price / 1e7, 2),
            "type": deal_type,
        })
    return out


def big_deals(limit: int = 14) -> Dict[str, Any]:
    """Largest bulk + block deals of the last few sessions (by ₹ value) +
    upcoming corporate actions for F&O names. Cached 1h."""
    hit = _CACHE.get("d")
    if hit and (time.monotonic() - hit[0]) < _TTL_S:
        return hit[1]

    out: Dict[str, Any] = {
        "deals": [], "corporate_actions": [],
        "label": "NSE · EOD published disclosures",
    }
    end = date.today()
    start = end - timedelta(days=5)
    fmt = "%d-%m-%Y"

    deals: List[Dict[str, Any]] = []
    try:
        from nselib import capital_market as cm
        try:
            deals += _deal_rows(
                cm.bulk_deal_data(from_date=start.strftime(fmt), to_date=end.strftime(fmt)),
                "bulk",
            )
        except Exception as e:
            logger.debug("bulk deals fetch failed: %s", e)
        try:
            deals += _deal_rows(
                cm.block_deals_data(from_date=start.strftime(fmt), to_date=end.strftime(fmt)),
                "block",
            )
        except Exception as e:
            logger.debug("block deals fetch failed: %s", e)

        deals.sort(key=lambda d: d["value_cr"], reverse=True)
        out["deals"] = deals[:limit]

        # Corporate actions (ex-dates) — F&O names only to keep signal density.
        try:
            ca = cm.corporate_actions_for_equity(
                from_date=end.strftime(fmt),
                to_date=(end + timedelta(days=10)).strftime(fmt),
                fno_only=True,
            )
            if ca is not None and hasattr(ca, "iterrows") and not ca.empty:
                cols = {c.lower().replace(" ", ""): c for c in ca.columns}
                sym_c = cols.get("symbol")
                sub_c = cols.get("subject") or cols.get("purpose")
                ex_c = cols.get("exdate")
                rows = []
                for _, r in ca.iterrows():
                    if not sym_c or not sub_c:
                        break
                    rows.append({
                        "symbol": str(r.get(sym_c) or "").strip(),
                        "subject": str(r.get(sub_c) or "").strip()[:80],
                        "ex_date": str(r.get(ex_c) or "").strip() if ex_c else None,
                    })
                out["corporate_actions"] = [r for r in rows if r["symbol"]][:10]
        except Exception as e:
            logger.debug("corporate actions fetch failed: %s", e)
    except Exception as e:  # noqa: BLE001
        logger.warning("big_deals failed: %s", e)

    if out["deals"] or out["corporate_actions"]:
        _CACHE["d"] = (time.monotonic(), out)
    return out
