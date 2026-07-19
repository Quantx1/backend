"""Watchlist Daily Digest — per-user 'what changed today' grounded agent.

The watchlist analogue of `why_moving`/`market_explainer`: assemble REAL
per-symbol facts deterministically by REUSING existing helpers (one batch
quote call, volume vs 20-day average via the candles read-through, recent
signals, firing live alerts, current regime), build per-symbol `bullets` +
a deterministic `summary` line that are ALWAYS returned (0 tokens), then
OPTIONALLY narrate with the grounded reasoner cached per
(user, watchlist-hash, day) — the key is USER-SCOPED because the watchlist
is. Honest-empty ({items: [], summary: None}) when the watchlist is empty.

NOT related to ai/digest/ (the Telegram/WhatsApp channel brief).
"""
from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MOVE_PCT = 2.0    # "moved today" threshold for the summary line
VOL_X = 1.5       # "unusual volume" bullet threshold (live_alerts uses 3.0 for ALERTS)
MAX_VOLUME_LOOKUPS = 30   # bound per-symbol get_historical fan-out


def _canon(sym: str) -> str:
    s = (sym or "").upper().strip()
    return s[:-3] if s.endswith(".NS") else s


def assemble_facts(user_id: str, *, cap: Optional[int] = None) -> Dict[str, Any]:
    """Real, current facts for every watchlist symbol. Best-effort per factor;
    {} when the watchlist is empty (honest-empty)."""
    from ...core.database import get_supabase_admin
    sb = get_supabase_admin()

    try:
        rows = (sb.table("watchlist").select("symbol, added_at")
                .eq("user_id", user_id).order("added_at", desc=False)
                .limit(100).execute().data or [])
    except Exception as e:  # noqa: BLE001
        logger.debug("watchlist_digest watchlist pull failed %s: %s", user_id, e)
        rows = []
    symbols: List[str] = []
    for r in rows:
        s = _canon(r.get("symbol") or "")
        if s and s not in symbols:
            symbols.append(s)
    if cap is not None:
        symbols = symbols[:cap]
    if not symbols:
        return {}

    facts: Dict[str, Any] = {"symbols": symbols,
                             "per_symbol": {s: {} for s in symbols}}

    quotes: Dict[str, Any] = {}
    try:
        from ...data.market import get_market_data_provider
        mp = get_market_data_provider()
        quotes = {_canon(k): v for k, v in (mp.get_quotes_batch(symbols) or {}).items()}
        for sym in symbols:
            q = quotes.get(sym)
            if q is None:
                continue
            chg = getattr(q, "change_percent", None)
            facts["per_symbol"][sym]["price"] = {
                "ltp": float(q.ltp) if getattr(q, "ltp", None) else None,
                "change_pct": round(float(chg), 2) if chg is not None else None,
            }
        for sym in symbols[:MAX_VOLUME_LOOKUPS]:
            try:
                df = mp.get_historical(sym, period="1mo", interval="1d")
                q = quotes.get(sym)
                cur_v = float(q.volume) if (q is not None and getattr(q, "volume", None)) else None
                if df is not None and len(df):
                    df.columns = [c.lower() for c in df.columns]
                    if "volume" in df.columns:
                        avg_v = float(df["volume"].tail(20).mean())
                        cur_v = cur_v or float(df["volume"].iloc[-1])
                        if avg_v and cur_v:
                            facts["per_symbol"][sym]["volume"] = {
                                "x_avg": round(cur_v / avg_v, 2)}
            except Exception:  # noqa: BLE001
                continue
    except Exception as e:  # noqa: BLE001
        logger.debug("watchlist_digest quote facts failed: %s", e)

    try:
        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        srows = (sb.table("signals")
                 .select("symbol, direction, confidence, status, created_at")
                 .in_("symbol", symbols).gte("created_at", since)
                 .order("created_at", desc=True).execute().data or [])
        for r in srows:
            slot = facts["per_symbol"].get(_canon(r.get("symbol") or ""))
            if slot is not None and "signal" not in slot:  # newest-first → keep latest
                slot["signal"] = {"direction": r.get("direction"),
                                  "confidence": r.get("confidence"),
                                  "status": r.get("status"),
                                  "created_at": r.get("created_at")}
    except Exception as e:  # noqa: BLE001
        logger.debug("watchlist_digest signal facts failed: %s", e)

    try:
        from .live_alerts import scan_live_alerts
        for a in scan_live_alerts(limit=200):
            slot = facts["per_symbol"].get(_canon(a.get("symbol") or ""))
            if slot is not None:
                slot.setdefault("alerts", []).append(
                    {"type": a.get("type"), "message": a.get("message")})
    except Exception as e:  # noqa: BLE001
        logger.debug("watchlist_digest alert facts failed: %s", e)

    try:
        rrow = (sb.table("regime_history").select("regime,vix")
                .order("detected_at", desc=True).limit(1).execute().data or [])
        if rrow and rrow[0].get("regime"):
            facts["regime"] = {"market": rrow[0].get("regime"),
                               "vix": rrow[0].get("vix")}
    except Exception:  # noqa: BLE001
        pass

    return facts


def build_symbol_bullets(sym: str, f: Dict[str, Any]) -> List[str]:
    """Deterministic per-symbol bullets — always available, 0 tokens. Pure."""
    out: List[str] = []
    p = f.get("price") or {}
    if p.get("change_pct") is not None:
        out.append(f"{'+' if p['change_pct'] >= 0 else ''}{p['change_pct']}% today.")
    v = f.get("volume") or {}
    if v.get("x_avg") and v["x_avg"] >= VOL_X:
        out.append(f"Volume {v['x_avg']}× the 20-day average.")
    s = f.get("signal") or {}
    if s.get("direction"):
        conf = ""
        if s.get("confidence") is not None:
            try:
                conf = f" ({round(float(s['confidence']) * 100)}% conf)"
            except (TypeError, ValueError):
                pass
        out.append(f"Active {s['direction']} signal{conf}.")
    for a in f.get("alerts") or []:
        if a.get("message"):
            out.append(f"Alert: {a['message']}")
    return out


def build_summary(facts: Dict[str, Any]) -> Optional[str]:
    """Deterministic 'what changed today' line. None when facts empty. Pure."""
    per = facts.get("per_symbol") or {}
    if not per:
        return None
    total = len(per)
    movers = [s for s, f in per.items()
              if abs((f.get("price") or {}).get("change_pct") or 0) >= MOVE_PCT]
    spikes = [s for s, f in per.items()
              if ((f.get("volume") or {}).get("x_avg") or 0) >= VOL_X]
    signals = [s for s, f in per.items() if (f.get("signal") or {}).get("direction")]
    alerts = [s for s, f in per.items() if f.get("alerts")]
    parts: List[str] = []
    if movers:
        parts.append(f"{len(movers)} of {total} moved ≥{MOVE_PCT:g}% "
                     f"({', '.join(movers[:3])})")
    if spikes:
        parts.append(f"unusual volume in {', '.join(spikes[:3])}")
    if signals:
        parts.append(f"active signals on {', '.join(signals[:3])}")
    if alerts:
        parts.append(f"alerts firing on {', '.join(alerts[:3])}")
    rg = facts.get("regime") or {}
    if rg.get("market"):
        parts.append(f"regime {rg['market']}")
    if not parts:
        return f"No significant changes across your {total} watchlist symbols today."
    return "Today: " + "; ".join(parts) + "."


def digest(user_id: str, *, use_llm: bool = False,
           cap: Optional[int] = None) -> Dict[str, Any]:
    """{items, summary, narrative, count}. Bullets + summary deterministic and
    ALWAYS returned (0 tokens); narrative is the grounded reasoner, cached per
    (user, watchlist-hash, day) — USER-SCOPED key. Honest-empty when the
    watchlist is empty."""
    facts = assemble_facts(user_id, cap=cap)
    per = facts.get("per_symbol") or {}
    items = [{"symbol": s, "bullets": build_symbol_bullets(s, per.get(s) or {})}
             for s in (facts.get("symbols") or [])]
    summary = build_summary(facts)
    narrative: Optional[str] = None
    if use_llm and items:
        from ...ai.agents.grounded import grounded_reason
        # Non-security digest: a short cache key over the sorted symbol set.
        # usedforsecurity=False marks intent and satisfies bandit B324 (SHA1).
        sym_hash = hashlib.sha1(",".join(sorted(per)).encode(), usedforsecurity=False).hexdigest()[:10]
        narrative = grounded_reason(
            facts,
            "Give this trader a 3-4 sentence daily digest of their watchlist: "
            "what changed today, which symbols need attention first, and why.",
            cache_key=f"wldigest:{user_id}:{sym_hash}:{date.today().isoformat()}",
            user_id=user_id)
    return {"items": items, "summary": summary, "narrative": narrative,
            "count": len(items)}
