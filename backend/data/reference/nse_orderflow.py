"""Pure NSE order-flow mappers (DataFrame->rows) + lazy nselib fetchers.

Mappers are column-name-tolerant and honest-empty; fetchers lazy-import nselib
(verify names against the install — see plan header)."""
from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def _iso_date(v, fallback=None):
    """Normalize a raw NSE date (e.g. '06-Jun-2026' / '08-06-2026') to ISO
    'YYYY-MM-DD' for the Postgres DATE columns. day-first; fallback on failure
    (never let a bad date string reach the DB and silently fail the upsert)."""
    s = str(v or "").strip()
    if not s:
        return fallback
    try:
        import pandas as pd
        d = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if d is None or pd.isna(d):
            return fallback
        return d.date().isoformat()
    except Exception:
        return fallback


_PARTICIPANTS = {"client": "client", "pro": "pro", "fii": "fii", "dii": "dii",
                 "fii/fpi": "fii", "fpi": "fii"}


def _i(v):
    try:
        return int(float(str(v).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


def _n(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _col(row, *names):
    for n in names:
        key = n.strip().lower()
        for k in row.index:
            if str(k).strip().lower() == key:
                return row[k]
    return None


def _sum(row, *names):
    total, seen = 0, False
    for n in names:
        v = _i(_col(row, n))
        if v is not None:
            total += v
            seen = True
    return total if seen else None


def map_participant_oi_rows(df, trade_date: str) -> List[Dict]:
    """NSE fao_participant_oi DataFrame -> participant_oi_eod rows (sum index+stock)."""
    if df is None or len(df) == 0:
        return []
    rows: List[Dict] = []
    for _, r in df.iterrows():
        ptype = str(_col(r, "Client Type", "ClientType") or "").strip().lower()
        participant = _PARTICIPANTS.get(ptype)
        if not participant:
            continue
        rows.append({
            "date": trade_date, "participant": participant,
            "fut_long": _sum(r, "Future Index Long", "Future Stock Long"),
            "fut_short": _sum(r, "Future Index Short", "Future Stock Short"),
            "opt_call_long": _sum(r, "Option Index Call Long", "Option Stock Call Long"),
            "opt_call_short": _sum(r, "Option Index Call Short", "Option Stock Call Short"),
            "opt_put_long": _sum(r, "Option Index Put Long", "Option Stock Put Long"),
            "opt_put_short": _sum(r, "Option Index Put Short", "Option Stock Put Short"),
            "source": "nselib",
        })
    return rows


def map_fii_dii_rows(df, trade_date: str) -> List[Dict]:
    """FII/DII cash activity -> a single fii_dii_flow_eod CASH row."""
    if df is None or len(df) == 0:
        return []
    out = {"date": trade_date, "segment": "CASH", "source": "nselib"}
    for _, r in df.iterrows():
        cat = str(_col(r, "category", "Category") or "").strip().lower()
        pre = "fii" if "fii" in cat or "fpi" in cat else ("dii" if "dii" in cat else None)
        if not pre:
            continue
        out[f"{pre}_buy"] = _n(_col(r, "buyValue", "Buy Value"))
        out[f"{pre}_sell"] = _n(_col(r, "sellValue", "Sell Value"))
        out[f"{pre}_net"] = _n(_col(r, "netValue", "Net Value"))
    return [out] if len(out) > 3 else []


def map_bulk_block_rows(df, deal_type: str = "BULK") -> List[Dict]:
    # NOTE: PK excludes `price` (date,symbol,deal_type,client_name,buy_sell,qty),
    # so two same-side deals of identical qty at different prices collapse to one
    # (last price wins) — acceptable for EOD; add `price` to the PK if fidelity needed.
    if df is None or len(df) == 0:
        return []
    rows: List[Dict] = []
    for _, r in df.iterrows():
        symbol = str(_col(r, "Symbol", "symbol") or "").strip()
        date = _iso_date(_col(r, "Date", "date"))  # NSE dates are DD-Mon-YYYY -> ISO
        if not symbol or not date:
            continue
        rows.append({
            "date": date, "symbol": symbol, "deal_type": deal_type,
            "client_name": str(_col(r, "Client Name", "clientName") or "").strip(),
            "buy_sell": str(_col(r, "Buy/Sell", "buySell") or "").strip().upper(),
            "qty": _i(_col(r, "QuantityTraded", "Quantity Traded", "quantity")) or 0,
            "price": _n(_col(r, "TradePrice/Wght.Avg.Price", "Trade Price / Wght. Avg. Price", "Trade Price", "price")),
            "source": "nselib",
        })
    return rows


def map_short_selling_rows(df, trade_date: str) -> List[Dict]:
    if df is None or len(df) == 0:
        return []
    rows: List[Dict] = []
    for _, r in df.iterrows():
        symbol = str(_col(r, "Symbol", "symbol") or "").strip()
        if not symbol:
            continue
        date = _iso_date(_col(r, "Date", "date"), fallback=trade_date)  # DD-Mon-YYYY -> ISO
        rows.append({"date": date, "symbol": symbol,
                     "qty": _i(_col(r, "Quantity", "qty")), "source": "nselib"})
    return rows


def map_fno_ban_symbols(symbols, trade_date: str) -> List[Dict]:
    return [{"date": trade_date, "symbol": str(s).strip(), "source": "nselib"}
            for s in (symbols or []) if str(s).strip()]


# --- lazy fetchers (verify names at install) ---
def fetch_participant_oi(trade_date: str):
    from nselib import derivatives
    return derivatives.participant_wise_open_interest(trade_date)


def fetch_fii_dii():
    """FII/DII cash-segment provisional figures. nselib 2.5.1 dropped its
    fii_dii_trading_activity(), so fetch NSE's public JSON directly with a browser
    session (hit the homepage first for cookies, then the API). Returns a DataFrame
    (columns category/buyValue/sellValue/netValue) so map_fii_dii_rows can parse it."""
    import requests
    import pandas as pd
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json", "Accept-Language": "en-US,en;q=0.9",
    })
    s.get("https://www.nseindia.com", timeout=12)          # seed cookies
    r = s.get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=12)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def _widen_range(from_date: str, to_date: str, lookback_days: int = 7):
    """nselib bulk/block/short require to_date > from_date. Callers often pass the
    same day (the common 'today' case), which nselib rejects with 'to_date should
    greater than from_date'. Widen from_date back so the range is valid; the
    mappers keep each row's own date, so the extra days store correctly (idempotent)."""
    import datetime as _dt
    try:
        f = _dt.datetime.strptime(from_date, "%d-%m-%Y").date()
        t = _dt.datetime.strptime(to_date, "%d-%m-%Y").date()
        if f >= t:
            f = t - _dt.timedelta(days=lookback_days)
        return f.strftime("%d-%m-%Y"), t.strftime("%d-%m-%Y")
    except Exception:
        return from_date, to_date


def fetch_bulk_deals(from_date: str, to_date: str):
    from nselib import capital_market
    from_date, to_date = _widen_range(from_date, to_date)
    return capital_market.bulk_deal_data(from_date=from_date, to_date=to_date)


def fetch_block_deals(from_date: str, to_date: str):
    from nselib import capital_market
    from_date, to_date = _widen_range(from_date, to_date)
    return capital_market.block_deals_data(from_date=from_date, to_date=to_date)


def fetch_short_selling(from_date: str, to_date: str):
    from nselib import capital_market
    from_date, to_date = _widen_range(from_date, to_date)
    return capital_market.short_selling_data(from_date=from_date, to_date=to_date)


def fetch_fno_ban(trade_date: str):
    from nselib import derivatives
    return derivatives.fno_security_in_ban_period(trade_date)
