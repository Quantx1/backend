"""Earnings Preview — pre-event grounded agent.

HONESTY: this codebase has no persisted earnings calendar. The ONLY next-date
source is a live yfinance Ticker.calendar probe (best-effort). Everything else
is real: ATM IV / expected move to expiry + IV rank (F&O names), 1-month
run-up, RS vs NIFTY, sector. Honest-empty (no date -> drivers=[]) — never
fabricate an event date.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_WINDOW_DAYS = 60
_CACHE: Dict[str, tuple] = {}     # sym -> (ts, facts) — yfinance+chain probes are slow
_TTL_S = 1800


def _expected_move_pct(iv_atm: Optional[float], days_to_expiry: Optional[int]) -> Optional[float]:
    """1-sigma expected move to expiry, % of spot. iv_atm is DECIMAL (0.18 = 18%). Pure."""
    if not iv_atm or iv_atm <= 0 or not days_to_expiry or days_to_expiry <= 0:
        return None
    return round(iv_atm * math.sqrt(days_to_expiry / 365.0) * 100, 2)


def _run_up_pct(closes: List[float], window: int = 20) -> Optional[float]:
    """% change over the last `window` bars. Pure."""
    if not closes or len(closes) <= window or not closes[-1 - window]:
        return None
    return round((closes[-1] / closes[-1 - window] - 1) * 100, 2)


def assemble_facts(symbol: str) -> Dict[str, Any]:
    sym = symbol.strip().upper()
    hit = _CACHE.get(sym)
    if hit and (time.monotonic() - hit[0]) < _TTL_S:
        return hit[1]
    facts: Dict[str, Any] = {"symbol": sym}

    try:
        from ...ai.earnings.calendar import next_earnings_date
        d = next_earnings_date(sym, window_days=_WINDOW_DAYS)
        if d is not None:
            facts["earnings"] = {"announce_date": d.isoformat(),
                                 "days_to_earnings": (d - date.today()).days,
                                 "source": "yfinance_calendar"}
    except Exception as e:  # noqa: BLE001
        logger.debug("earnings_preview date probe failed %s: %s", sym, e)

    if facts.get("earnings"):
        try:
            from ...core.database import get_supabase_admin
            rows = (get_supabase_admin().table("earnings_predictions")
                    .select("beat_prob,confidence")
                    .eq("symbol", sym).eq("announce_date", facts["earnings"]["announce_date"])
                    .limit(1).execute().data or [])
            if rows and rows[0].get("beat_prob") is not None:
                facts["prediction"] = {"beat_prob": float(rows[0]["beat_prob"]),
                                       "confidence": rows[0].get("confidence")}
        except Exception:  # noqa: BLE001
            pass

    try:
        from ..fno_scanner.snapshot import fetch_index_snapshot
        from ..fno_scanner.iv_store import iv_rank_percentile
        snap = fetch_index_snapshot(sym)
        if snap is not None and snap.iv_atm and snap.spot:
            vol = iv_rank_percentile(sym, snap.iv_atm)
            facts["volatility"] = {
                "atm_iv_pct": round(snap.iv_atm * 100, 1),
                "iv_rank": vol.get("iv_rank"), "iv_percentile": vol.get("iv_percentile"),
                "expected_move_pct_to_expiry": _expected_move_pct(snap.iv_atm, snap.days_to_expiry),
                "expiry": snap.expiry, "days_to_expiry": snap.days_to_expiry,
            }
    except Exception as e:  # noqa: BLE001
        logger.debug("earnings_preview vol facts failed %s: %s", sym, e)

    try:
        from ...data.market import get_market_data_provider
        df = get_market_data_provider().get_historical(sym, period="3mo", interval="1d")
        if df is not None and len(df):
            df.columns = [c.lower() for c in df.columns]
            ru = _run_up_pct([float(c) for c in df["close"].tolist() if c == c])
            if ru is not None:
                facts["run_up"] = {"pct_1m": ru}
    except Exception as e:  # noqa: BLE001
        logger.debug("earnings_preview run-up failed %s: %s", sym, e)
    try:
        from ..scanners.relative_strength import symbol_rs
        rs = symbol_rs(sym)
        if rs.get("rs_20d") is not None or rs.get("rs_50d") is not None:
            facts["relative_strength"] = {"rs_20d": rs.get("rs_20d"),
                                          "rs_50d": rs.get("rs_50d"),
                                          "outperforming": rs.get("outperforming")}
    except Exception:  # noqa: BLE001
        pass

    try:
        from ...core.database import get_supabase_admin
        row = (get_supabase_admin().table("instruments").select("sector,name")
               .eq("symbol", sym).eq("instrument_type", "EQ").limit(1).execute().data or [])
        if row:
            facts["sector"] = row[0].get("sector")
            facts["name"] = row[0].get("name")
    except Exception:  # noqa: BLE001
        pass

    _CACHE[sym] = (time.monotonic(), facts)
    return facts


def _drivers(facts: Dict[str, Any]) -> List[str]:
    """Deterministic bullets — [] when no confirmed date (honest-empty). Pure."""
    e = facts.get("earnings") or {}
    if not e.get("announce_date"):
        return []
    out = [f"Earnings on {e['announce_date']} — in {e['days_to_earnings']} day(s) (consensus calendar, best-effort)."]
    p = facts.get("prediction") or {}
    if p.get("beat_prob") is not None:
        out.append(f"EarningsScout beat probability: {round(p['beat_prob'] * 100)}%.")
    v = facts.get("volatility") or {}
    if v.get("expected_move_pct_to_expiry") is not None:
        out.append(f"Options imply a ±{v['expected_move_pct_to_expiry']}% move by {v.get('expiry')} "
                   f"(ATM IV {v.get('atm_iv_pct')}%).")
    if v.get("iv_rank") is not None:
        out.append(f"IV Rank {v['iv_rank']:.0f} — "
                   f"{'rich' if v['iv_rank'] >= 70 else 'cheap' if v['iv_rank'] <= 30 else 'mid-range'} "
                   "option premium into the event.")
    r = facts.get("run_up") or {}
    if r.get("pct_1m") is not None:
        out.append(f"Stock has {'run up' if r['pct_1m'] >= 0 else 'fallen'} {abs(r['pct_1m'])}% "
                   "over the past month into the event.")
    rs = facts.get("relative_strength") or {}
    if rs.get("rs_20d") is not None:
        out.append(f"{'Outperforming' if rs.get('outperforming') else 'Lagging'} NIFTY by "
                   f"{abs(rs['rs_20d'])}% over ~1 month.")
    return out


def preview(symbol: str, *, use_llm: bool = False, user_id: Optional[str] = None) -> Dict[str, Any]:
    """{symbol, facts, drivers, narrative}. Drivers deterministic; narrative
    grounded + cached per (symbol, day). Honest-empty without a date."""
    sym = symbol.strip().upper()
    facts = assemble_facts(sym)
    drivers = _drivers(facts)
    narrative = None
    if use_llm and drivers:
        from ...ai.agents.grounded import grounded_reason
        dte = (facts.get("earnings") or {}).get("days_to_earnings")
        narrative = grounded_reason(
            facts,
            f"{sym} reports earnings in {dte} day(s). Given the implied move, IV rank, "
            "recent run-up and relative strength, what should a trader watch into the event?",
            cache_key=f"earnprev:{sym}:{date.today().isoformat()}", user_id=user_id)
    return {"symbol": sym, "facts": facts, "drivers": drivers, "narrative": narrative}
