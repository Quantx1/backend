"""Pure NSE derivatives EOD mappers (F&O bhavcopy -> rows) + metrics + lazy fetch.

ONE F&O bhavcopy DataFrame yields option rows + futures rows; metrics (PCR,
max-pain) are computed per (symbol, expiry) from the option rows. Column-name
tolerant + honest-empty. Lazy nselib import (verify names at install)."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)

_OPTION_TYPES = {"CE", "PE"}


def _num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _int(v):
    n = _num(v)
    return int(n) if n is not None else None


def _col(row, *names):
    for n in names:
        key = n.strip().lower()
        for k in row.index:
            if str(k).strip().lower() == key:
                return row[k]
    return None


def _iso(v, fallback=None):
    s = str(v or "").strip()
    if not s:
        return fallback
    try:
        import warnings
        import pandas as pd
        # dayfirst handles DD-MM/DD-MMM; suppress the cosmetic UserWarning pandas
        # raises when the value is actually unambiguous ISO (the real UDiFF format).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = pd.to_datetime(s, dayfirst=True, errors="coerce")
        return d.date().isoformat() if d is not None and not pd.isna(d) else fallback
    except Exception:
        return fallback


def _instr_type(row) -> str:
    return str(_col(row, "FinInstrmTp", "InstrumentType", "instrument") or "").strip().upper()


def map_fno_options_rows(df, trade_date: str) -> List[Dict]:
    """Option contracts from an F&O bhavcopy -> options_chain_eod rows."""
    if df is None or len(df) == 0:
        return []
    rows: List[Dict] = []
    for _, r in df.iterrows():
        opt = str(_col(r, "OptnTp", "OptionType", "option_type") or "").strip().upper()
        if opt not in _OPTION_TYPES:
            continue
        symbol = str(_col(r, "TckrSymb", "Symbol", "symbol") or "").strip()
        expiry = _iso(_col(r, "XpryDt", "ExpiryDate", "expiry"))
        strike = _num(_col(r, "StrkPric", "StrikePrice", "strike"))
        if not symbol or not expiry or strike is None:
            continue
        rows.append({
            "date": trade_date, "symbol": symbol, "expiry": expiry,
            "strike": strike, "option_type": opt,
            "oi": _int(_col(r, "OpnIntrst", "OpenInterest", "oi")),
            "oi_change": _int(_col(r, "ChngInOpnIntrst", "ChangeInOI", "oi_change")),
            "volume": _int(_col(r, "TtlTradgVol", "Volume", "volume")),
            "ltp": _num(_col(r, "ClsPric", "ClosePrice", "ltp", "close")),
            "source": "nselib",
        })
    return rows


def map_fno_futures_rows(df, trade_date: str) -> List[Dict]:
    """Future contracts from an F&O bhavcopy -> futures_eod rows."""
    if df is None or len(df) == 0:
        return []
    rows: List[Dict] = []
    for _, r in df.iterrows():
        itype = _instr_type(r)
        if not (itype.startswith("FUT") or itype in ("STF", "IDF")):
            continue
        symbol = str(_col(r, "TckrSymb", "Symbol", "symbol") or "").strip()
        expiry = _iso(_col(r, "XpryDt", "ExpiryDate", "expiry"))
        if not symbol or not expiry:
            continue
        rows.append({
            "date": trade_date, "symbol": symbol, "expiry": expiry,
            "open": _num(_col(r, "OpnPric", "OpenPrice", "open")),
            "high": _num(_col(r, "HghPric", "HighPrice", "high")),
            "low": _num(_col(r, "LwPric", "LowPrice", "low")),
            "close": _num(_col(r, "ClsPric", "ClosePrice", "close")),
            "oi": _int(_col(r, "OpnIntrst", "OpenInterest", "oi")),
            "oi_change": _int(_col(r, "ChngInOpnIntrst", "ChangeInOI", "oi_change")),
            "volume": _int(_col(r, "TtlTradgVol", "Volume", "volume")),
            "source": "nselib",
        })
    return rows


def build_derivatives_metrics(option_rows: List[Dict]) -> List[Dict]:
    """Compute PCR (OI & volume), max-pain (max-total-OI strike), and CE/PE OI
    totals per (symbol, expiry) from option_chain_eod rows. Pure."""
    if not option_rows:
        return []
    # group -> (ce_oi, pe_oi, ce_vol, pe_vol, oi_by_strike)
    g: Dict[tuple, dict] = defaultdict(
        lambda: {"ce_oi": 0, "pe_oi": 0, "ce_vol": 0, "pe_vol": 0, "by_strike": defaultdict(int)})
    for r in option_rows:
        key = (r["date"], r["symbol"], r["expiry"])
        oi = r.get("oi") or 0
        vol = r.get("volume") or 0
        b = g[key]
        b["by_strike"][r["strike"]] += oi
        if r["option_type"] == "CE":
            b["ce_oi"] += oi
            b["ce_vol"] += vol
        else:
            b["pe_oi"] += oi
            b["pe_vol"] += vol
    out: List[Dict] = []
    for (date, symbol, expiry), b in g.items():
        # NOTE: this is the MAX-OI strike (a max-pain *proxy*), not the textbook
        # min-aggregate-writer-payout max-pain. Good enough for an EOD signal;
        # downstream consumers should treat it as a proxy, not exact max-pain.
        max_pain = max(b["by_strike"], key=b["by_strike"].get) if b["by_strike"] else None
        out.append({
            "date": date, "symbol": symbol, "expiry": expiry,
            "pcr_oi": round(b["pe_oi"] / b["ce_oi"], 4) if b["ce_oi"] > 0 else None,
            "pcr_volume": round(b["pe_vol"] / b["ce_vol"], 4) if b["ce_vol"] > 0 else None,
            "max_pain": max_pain,
            "total_ce_oi": b["ce_oi"], "total_pe_oi": b["pe_oi"],
            "source": "nselib",
        })
    return out


# --- lazy fetcher (verify name at install) ---
def fetch_fno_bhavcopy(trade_date: str):
    """F&O bhavcopy DataFrame for a trade date (DD-MM-YYYY). Lazy nselib."""
    from nselib import derivatives
    return derivatives.fno_bhav_copy(trade_date)
