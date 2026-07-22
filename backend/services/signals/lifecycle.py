"""Signal lifecycle engine (2026-07-21) — signals decay, trigger, and close.

A signal is NOT a static row: every session it either fills, hits its stop,
hits its target, or ages toward expiry. This module walks every open
(active / triggered) row in ``public.signals`` against REAL daily bars and
applies the honest state machine:

    active     --entry traded in a later session-->        triggered
    triggered  --bar low  <= stop   (LONG)      -->        stop_loss_hit
    triggered  --bar high >= target (LONG)      -->        target_hit
    active|triggered --today > valid_until      -->        expired

Rules that keep it honest:
  * No look-ahead: the book is generated AT the close of trade_date, so
    fills are only evaluated from the NEXT session onward.
  * Conservative same-bar tie: if one bar spans both stop and target, the
    STOP wins (we never award a win that may not have happened).
  * Outcomes: target/stop closes record result + actual_return from the
    level, not the close. Expiry of a TRIGGERED signal marks to the last
    close; expiry of a never-filled signal records no result.

Runs as the 16:15 IST scheduler job (after the 15:55 book refresh) and is
idempotent — a rerun sees the already-transitioned statuses and does
nothing new. Bars come through the market provider's candle read-through
(EOD settled data).
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _bars_since(symbol: str, start: str) -> List[Dict[str, Any]]:
    """Daily bars for `symbol` strictly AFTER `start` (YYYY-MM-DD), oldest
    first: [{date, high, low, close}]. Empty on any failure (honest no-op)."""
    try:
        from ...data.market import get_market_data_provider
        df = get_market_data_provider().get_historical(symbol, period="3mo", interval="1d")
        if df is None or len(df) == 0:
            return []
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        out: List[Dict[str, Any]] = []
        for idx, r in df.iterrows():
            d = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            if d <= start:
                continue
            try:
                out.append({
                    "date": d,
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                })
            except Exception:  # noqa: BLE001 — skip malformed bar
                continue
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug("lifecycle bars failed for %s: %s", symbol, e)
        return []


def evaluate_signal_row(
    row: Dict[str, Any],
    bars: List[Dict[str, Any]],
    today: Optional[date] = None,
) -> Optional[Dict[str, Any]]:
    """Pure state machine for ONE signal row against its post-signal bars.

    Returns the update dict (status/triggered_at/closed_at/result/
    actual_return) or None when nothing changed. Testable without a DB.
    """
    status = row.get("status")
    if status not in ("active", "triggered"):
        return None
    entry = _f(row.get("entry_price"))
    stop = _f(row.get("stop_loss"))
    target = _f(row.get("target_1"))
    if entry is None:
        return None
    is_long = (row.get("direction") or "LONG").upper() != "SHORT"
    today = today or date.today()

    update: Dict[str, Any] = {}
    triggered = status == "triggered"

    for b in bars:
        if not triggered:
            # Fill check: the entry level traded inside this bar's range.
            if b["low"] <= entry <= b["high"]:
                triggered = True
                update["status"] = "triggered"
                update["triggered_at"] = f"{b['date']}T00:00:00+00:00"
            else:
                continue  # not filled yet — nothing else can happen this bar
        # From the fill bar onward: conservative stop-first, then target.
        if stop is not None and ((is_long and b["low"] <= stop) or (not is_long and b["high"] >= stop)):
            update.update({
                "status": "stop_loss_hit",
                "closed_at": f"{b['date']}T00:00:00+00:00",
                "result": "loss",
                "actual_return": round(((stop - entry) / entry) * (1 if is_long else -1), 4),
            })
            return update
        if target is not None and ((is_long and b["high"] >= target) or (not is_long and b["low"] <= target)):
            update.update({
                "status": "target_hit",
                "closed_at": f"{b['date']}T00:00:00+00:00",
                "result": "win",
                "actual_return": round(((target - entry) / entry) * (1 if is_long else -1), 4),
            })
            return update

    # Still open — expiry check against valid_until.
    vu = str(row.get("valid_until") or "")[:10]
    if vu and str(today) > vu:
        update["status"] = "expired"
        update["closed_at"] = datetime.utcnow().isoformat()
        if triggered and bars:
            last = bars[-1]["close"]
            ret = round(((last - entry) / entry) * (1 if is_long else -1), 4)
            update["actual_return"] = ret
            update["result"] = "win" if ret > 0 else "loss"
        return update

    return update or None


def _f(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def evaluate_signal_lifecycle(
    supabase: Any = None,
    bars_fn: Callable[[str, str], List[Dict[str, Any]]] = _bars_since,
) -> Dict[str, int]:
    """Walk every open signal and persist its transitions. Returns counts."""
    if supabase is None:
        from ...core.database import get_supabase_admin
        supabase = get_supabase_admin()

    try:
        rows = (
            supabase.table("signals")
            .select("id,symbol,direction,entry_price,stop_loss,target_1,date,status,valid_until")
            .in_("status", ["active", "triggered"])
            .limit(500)
            .execute()
        ).data or []
    except Exception as e:  # noqa: BLE001
        logger.warning("lifecycle: open-signal fetch failed: %s", e)
        return {"checked": 0}

    counts: Dict[str, int] = {"checked": len(rows), "triggered": 0, "target_hit": 0,
                              "stop_loss_hit": 0, "expired": 0}
    bars_cache: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        sym = row.get("symbol")
        start = str(row.get("date") or "")[:10]
        if not sym or not start:
            continue
        if sym not in bars_cache:
            bars_cache[sym] = bars_fn(sym, start)
        # bars are fetched once per symbol from the earliest possible start —
        # re-slice per row so a symbol with two signals evaluates each fairly.
        bars = [b for b in bars_cache[sym] if b["date"] > start]
        update = evaluate_signal_row(row, bars)
        if not update:
            continue
        try:
            supabase.table("signals").update(update).eq("id", row["id"]).execute()
            final = update.get("status")
            if final in counts:
                counts[final] += 1
            elif final == "triggered":
                counts["triggered"] += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("lifecycle: update failed for %s: %s", row.get("id"), e)

    logger.info("signal lifecycle: %s", counts)
    return counts
