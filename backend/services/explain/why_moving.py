"""'Why is X moving today?' — the flagship grounded agent (#highest-value).

Assembles REAL drivers deterministically (price action, volume vs 20-day avg,
futures OI build-up, relative strength vs NIFTY, market regime), then a free
reasoning model narrates over them (cached per symbol/day). The deterministic
`drivers` are always returned, so the surface is useful even when the LLM is
off — and the grounding keeps the narrative cheap + high-quality.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def assemble_facts(symbol: str) -> Dict[str, Any]:
    """Gather the real, current drivers for a symbol. Best-effort per factor."""
    sym = symbol.strip().upper()
    facts: Dict[str, Any] = {"symbol": sym}

    # price / volume / change + relative strength vs NIFTY
    try:
        from ...data.market import get_market_data_provider
        mp = get_market_data_provider()
        q = mp.get_quote(sym) or {}
        chg = getattr(q, "change_percent", None)
        facts["price"] = {
            "ltp": getattr(q, "ltp", None),
            "change_pct": round(chg, 2) if chg is not None else None,
        }
        df = mp.get_historical(sym, period="1mo", interval="1d")
        cur_v = getattr(q, "volume", None)
        if df is not None and len(df):
            df.columns = [c.lower() for c in df.columns]
            if "volume" in df.columns:
                avg_v = float(df["volume"].tail(20).mean())
                cur_v = cur_v or float(df["volume"].iloc[-1])
                if avg_v and cur_v:
                    facts["volume"] = {"today": round(cur_v), "avg_20d": round(avg_v),
                                       "x_avg": round(cur_v / avg_v, 2)}
        nifty = mp.get_quote("NIFTY") or {}
        nchg = getattr(nifty, "change_percent", None)
        if chg is not None and nchg is not None:
            facts["relative_strength"] = {
                "stock_chg_pct": round(chg, 2), "nifty_chg_pct": round(nchg, 2),
                "vs_nifty_pct": round(chg - nchg, 2), "outperforming": chg > nchg,
            }
    except Exception as e:
        logger.debug("why_moving price facts failed for %s: %s", sym, e)

    # sector / name
    try:
        from ...core.database import get_supabase_admin
        sb = get_supabase_admin()
        row = (sb.table("instruments").select("sector,name")
               .eq("symbol", sym).eq("instrument_type", "EQ").limit(1).execute().data or [])
        if row:
            facts["sector"] = row[0].get("sector")
            facts["name"] = row[0].get("name")
    except Exception:
        pass

    # futures OI build-up (real now that change_pct is fixed)
    try:
        from ...data.screener.nse_data import get_nse_data
        for r in (get_nse_data().get_participant_oi().get("data") or []):
            if str(r.get("symbol", "")).upper() != sym:
                continue
            oc, pc = r.get("oi_change_pct"), r.get("change_pct")
            tag = None
            if pc is not None and oc is not None:
                tag = (("long_buildup" if pc > 0 else "short_buildup") if oc > 0
                       else ("short_covering" if pc > 0 else "long_unwinding"))
            facts["futures_oi"] = {"oi_change_pct": oc, "price_chg_pct": pc, "buildup": tag}
            break
    except Exception:
        pass

    # market regime
    try:
        from ...core.database import get_supabase_admin
        sb = get_supabase_admin()
        rrow = (sb.table("regime_history").select("regime,vix")
                .order("detected_at", desc=True).limit(1).execute().data or [])
        if rrow:
            facts["regime"] = {"market": rrow[0].get("regime"), "vix": rrow[0].get("vix")}
    except Exception:
        pass

    return facts


def _drivers(facts: Dict[str, Any]) -> List[str]:
    """Deterministic plain bullet drivers — always available, 0 tokens."""
    out: List[str] = []
    p = facts.get("price") or {}
    if p.get("change_pct") is not None:
        out.append(f"Price {'+' if p['change_pct'] >= 0 else ''}{p['change_pct']}% today.")
    v = facts.get("volume") or {}
    if v.get("x_avg"):
        out.append(f"Volume {v['x_avg']}× the 20-day average.")
    rs = facts.get("relative_strength") or {}
    if rs.get("vs_nifty_pct") is not None:
        out.append(f"{'Outperforming' if rs['outperforming'] else 'Lagging'} NIFTY by "
                   f"{abs(rs['vs_nifty_pct'])}% today.")
    oi = facts.get("futures_oi") or {}
    if oi.get("buildup"):
        out.append(f"Futures OI: {oi['buildup'].replace('_', ' ')} "
                   f"({oi.get('oi_change_pct')}% ΔOI).")
    rg = facts.get("regime") or {}
    if rg.get("market"):
        tail = f" (VIX {rg['vix']})" if rg.get("vix") else ""
        out.append(f"Market regime: {rg['market']}{tail}.")
    return out


def explain_move(symbol: str, *, use_llm: bool = True, user_id: Optional[str] = None) -> Dict[str, Any]:
    """{symbol, facts, drivers, narrative}. Drivers deterministic; narrative is
    the grounded agent, cached per symbol/day."""
    sym = symbol.strip().upper()
    facts = assemble_facts(sym)
    drivers = _drivers(facts)
    narrative = None
    if use_llm and drivers:
        from ...ai.agents.grounded import grounded_reason
        narrative = grounded_reason(
            facts, f"Why is {sym} moving today? What are the main drivers?",
            cache_key=f"whymove:{sym}:{date.today().isoformat()}", user_id=user_id)
    return {"symbol": sym, "facts": facts, "drivers": drivers, "narrative": narrative}
