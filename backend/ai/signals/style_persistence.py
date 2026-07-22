"""Supabase persistence for the style-engine paper-trading window.

Tables (2026_07_07_pr_style_signals_paper_window.sql):

    style_signals          — daily persisted top-book per engine
                             PK (engine, trade_date, symbol)
    style_signal_outcomes  — matured H-bar forward returns per signal row
                             PK (engine, trade_date, symbol)

Mirrors ``persistence.py`` conventions: supabase-py only (no SQLAlchemy),
best-effort writes (log + swallow, never raise into the cron loop), callers
pass the admin client (scheduler: ``self.supabase``; API routes:
``get_supabase_admin()``). A ``None`` client resolves lazily so pure helpers
stay import-light.

Public API::

    save_style_signals(engine, trade_date, signals, status, forecast_degraded, supabase=None) -> int
    fetch_unmatured_dates(engine, horizon_days, supabase=None) -> list[date]
    fetch_signal_rows(engine, trade_date, supabase=None) -> list[dict]
    save_style_outcomes(rows, supabase=None) -> int
    fetch_outcomes(engine, supabase=None) -> list[dict]
    fetch_signal_dates(engine, supabase=None) -> list[date]
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Callable, List

logger = logging.getLogger(__name__)

# PostgREST caps a single select at max-rows (default 1000) — reads paginate.
_PAGE = 1000


def _client(supabase: Any = None):
    """Return the given client, else the process admin client (lazy import —
    keeps this module importable without the FastAPI app)."""
    if supabase is not None:
        return supabase
    from backend.api.app import get_supabase_admin  # noqa: PLC0415
    return get_supabase_admin()


def _iso(d: Any) -> str:
    """date/datetime/ISO-string -> 'YYYY-MM-DD'."""
    if isinstance(d, (date, datetime)):
        return d.isoformat()[:10]
    return str(d)[:10]


def _to_date(v: Any) -> date:
    return v if isinstance(v, date) and not isinstance(v, datetime) \
        else date.fromisoformat(str(v)[:10])


def _fetch_all(page_fn: Callable[[int, int], list]) -> List[dict]:
    """Drain a paginated PostgREST select. ``page_fn(offset, limit)`` returns
    one page of rows (already ``.execute().data``)."""
    out: List[dict] = []
    offset = 0
    while True:
        page = page_fn(offset, _PAGE) or []
        out.extend(page)
        if len(page) < _PAGE:
            return out
        offset += _PAGE


# ────────────────────────────────────────────────────────────────────
# Writes
# ────────────────────────────────────────────────────────────────────


def save_style_signals(
    engine: str,
    trade_date: Any,
    signals: List[Any],
    status: str,
    forecast_degraded: bool,
    supabase: Any = None,
) -> int:
    """Upsert the day's top-book for one engine into ``style_signals``.

    Idempotent on the (engine, trade_date, symbol) PK — a same-day rerun
    overwrites. Best-effort: logs and returns 0 on any failure; NEVER raises
    (the 15:55 cron must keep its per-engine isolation + JSON snapshot).
    Returns the number of rows written.
    """
    try:
        rows = []
        for s in signals:
            d = s.to_dict() if hasattr(s, "to_dict") else dict(s)
            rows.append({
                "engine": engine,
                "trade_date": _iso(trade_date),
                "symbol": d["symbol"],
                "rank": int(d.get("rank") or 0),
                "percentile": float(d.get("percentile") or 0.0),
                "confidence": float(d.get("confidence") or 0.0),
                "direction": d.get("direction") or "BUY",
                "entry_price": float(d.get("entry_price") or 0.0),
                "stop_loss": float(d.get("stop_loss") or 0.0),
                "target": float(d.get("target") or 0.0),
                "risk_reward": float(d.get("risk_reward") or 0.0),
                "expected_return": float(d.get("expected_return") or 0.0),
                "top_decile_prob": float(d.get("top_decile_prob") or 0.0),
                "status": status,
                "forecast_degraded": bool(forecast_degraded),
                "generated_at": datetime.utcnow().isoformat(),
            })
        if not rows:
            return 0
        sb = _client(supabase)
        sb.table("style_signals").upsert(
            rows, on_conflict="engine,trade_date,symbol",
        ).execute()
        return len(rows)
    except Exception as exc:  # noqa: BLE001 — best-effort by contract
        logger.warning("save_style_signals(%s) failed: %s", engine, exc)
        return 0


# Calendar-day validity per engine — horizon TRADING bars padded to calendar
# (momentum 20 bars ≈ 4 weeks, swing 10 bars ≈ 2 weeks).
_VALID_CALENDAR_DAYS = {"momentum": 30, "swing": 16}

# Top-N of each book that Free-tier users can see (rest is_premium).
_FREE_TOP_N = 3


def sync_signals_table(
    engine: str,
    trade_date: Any,
    signals: List[Any],
    supabase: Any = None,
) -> int:
    """Bridge the day's style-engine book into the LEGACY ``signals`` table.

    The signals PAGE (and its detail view, Counterpoint debate, history,
    alerts) reads ``public.signals`` — which stopped receiving rows when the
    v1 ensemble pipeline was retired (last write 2026-05-29). This bridge
    keeps that whole surface alive with the REAL v2 engine output: the same
    book that goes to ``style_signals`` / the JSON snapshot, mapped onto the
    legacy schema. No fabrication — every number is engine output.

    Semantics: a daily book. Writing today's book (a) expires this engine's
    previous active rows, (b) deletes any same-day rows (idempotent rerun),
    (c) inserts the fresh book (status=active, valid_until = trade_date +
    calendar padding of the engine horizon). Best-effort by contract: logs
    and returns 0 on failure, never raises into the cron.
    """
    try:
        t_iso = _iso(trade_date)
        now_iso = datetime.utcnow().isoformat()
        valid_days = _VALID_CALENDAR_DAYS.get(engine, 21)
        valid_until = (_to_date(trade_date) + timedelta(days=valid_days)).isoformat()

        # Regime context — best-effort enrichment for the detail page.
        regime = None
        try:
            from backend.services.regime.refresh import current_regime  # noqa: PLC0415
            regime = (current_regime() or {}).get("regime")
        except Exception:  # noqa: BLE001
            pass

        rows = []
        for s in signals:
            d = s.to_dict() if hasattr(s, "to_dict") else dict(s)
            rank = int(d.get("rank") or 0)
            direction = "LONG" if (d.get("direction") or "BUY").upper() in ("BUY", "LONG") else "SHORT"
            exp_ret = float(d.get("expected_return") or 0.0)
            pctile = float(d.get("percentile") or 0.0)
            reasons = list(d.get("reasons") or [])
            reasons.insert(0, f"{engine.capitalize()} engine rank #{rank} ({pctile:.0f}th percentile of the universe)")
            explanation = (
                f"{engine.capitalize()} engine ranked {d['symbol']} #{rank} in its "
                f"universe ({pctile:.0f}th percentile) with an expected "
                f"{'{:+.1f}'.format(exp_ret * 100)}% move over the model horizon. "
                f"Entry/stop/target come from the ATR risk engine "
                f"(risk:reward {float(d.get('risk_reward') or 0):.1f}). "
                "Analysis from a walk-forward-validated model — not investment advice."
            )
            rows.append({
                "symbol": d["symbol"],
                "date": t_iso,
                "signal_type": engine,
                "segment": "EQUITY",
                "exchange": "NSE",
                "direction": direction,
                "entry_price": float(d.get("entry_price") or 0.0),
                "stop_loss": float(d.get("stop_loss") or 0.0),
                "target_1": float(d.get("target") or 0.0),
                "risk_reward": float(d.get("risk_reward") or 0.0),
                "confidence": float(d.get("confidence") or 0.0),
                "expected_return": exp_ret,
                "status": "active",
                "valid_from": now_iso,
                "valid_until": valid_until,
                "generated_at": now_iso,
                "is_premium": rank > _FREE_TOP_N,
                "reasons": reasons,
                "explanation_text": explanation,
                "regime_at_signal": regime,
            })
        if not rows:
            return 0

        sb = _client(supabase)
        # (a) expire this engine's previous active book(s)
        sb.table("signals").update({"status": "expired"}) \
            .eq("signal_type", engine).in_("status", ["active", "triggered"]) \
            .lt("date", t_iso).execute()
        # (b) idempotent same-day rerun
        sb.table("signals").delete() \
            .eq("signal_type", engine).eq("date", t_iso).execute()
        # (c) fresh book
        sb.table("signals").insert(rows).execute()
        return len(rows)
    except Exception as exc:  # noqa: BLE001 — best-effort by contract
        logger.warning("sync_signals_table(%s) failed: %s", engine, exc)
        return 0


def save_style_outcomes(rows: List[dict], supabase: Any = None) -> int:
    """Upsert matured outcome rows into ``style_signal_outcomes``.

    Idempotent on the PK. Best-effort: logs and returns 0 on failure.
    """
    if not rows:
        return 0
    try:
        payload = []
        for r in rows:
            r = dict(r)
            r["trade_date"] = _iso(r["trade_date"])
            payload.append(r)
        sb = _client(supabase)
        sb.table("style_signal_outcomes").upsert(
            payload, on_conflict="engine,trade_date,symbol",
        ).execute()
        return len(payload)
    except Exception as exc:  # noqa: BLE001 — best-effort by contract
        logger.warning("save_style_outcomes failed: %s", exc)
        return 0


# ────────────────────────────────────────────────────────────────────
# Reads
# ────────────────────────────────────────────────────────────────────


def fetch_unmatured_dates(
    engine: str,
    horizon_days: int,
    supabase: Any = None,
) -> List[date]:
    """Distinct ``style_signals`` trade_dates with NO outcome rows yet.

    Cheap calendar pre-filter only: a date needs ``horizon_days`` TRADING
    bars after it to mature, which is always >= ``horizon_days`` calendar
    days — dates younger than that are dropped here without touching the
    price panel. The exact bars-after check is the caller's job (the
    scheduler holds the panel).
    """
    try:
        sb = _client(supabase)
        sig = _fetch_all(lambda off, lim: (
            sb.table("style_signals").select("trade_date")
            .eq("engine", engine).order("trade_date")
            .range(off, off + lim - 1).execute().data))
        done = _fetch_all(lambda off, lim: (
            sb.table("style_signal_outcomes").select("trade_date")
            .eq("engine", engine).order("trade_date")
            .range(off, off + lim - 1).execute().data))
    except Exception as exc:  # noqa: BLE001 — honest-empty (table may not exist yet)
        logger.warning("fetch_unmatured_dates(%s) failed: %s", engine, exc)
        return []
    matured = {_iso(r["trade_date"]) for r in done}
    candidates = sorted({_iso(r["trade_date"]) for r in sig} - matured)
    cutoff = date.today() - timedelta(days=int(horizon_days))
    return [_to_date(d) for d in candidates if _to_date(d) <= cutoff]


def fetch_signal_rows(
    engine: str,
    trade_date: Any,
    supabase: Any = None,
) -> List[dict]:
    """symbol + rank of the persisted top-book for one (engine, trade_date)."""
    try:
        sb = _client(supabase)
        res = (
            sb.table("style_signals").select("symbol, rank")
            .eq("engine", engine).eq("trade_date", _iso(trade_date))
            .order("rank").execute()
        )
        return res.data or []
    except Exception as exc:  # noqa: BLE001 — honest-empty
        logger.warning("fetch_signal_rows(%s, %s) failed: %s", engine, trade_date, exc)
        return []


def fetch_outcomes(engine: str, supabase: Any = None) -> List[dict]:
    """All matured outcome rows for one engine (paper-window API read)."""
    try:
        sb = _client(supabase)
        return _fetch_all(lambda off, lim: (
            sb.table("style_signal_outcomes")
            .select("trade_date, symbol, rank, fwd_return_h, bench_fwd_return_h, excess_h, horizon_days")
            .eq("engine", engine).order("trade_date")
            .range(off, off + lim - 1).execute().data))
    except Exception as exc:  # noqa: BLE001 — honest-empty
        logger.warning("fetch_outcomes(%s) failed: %s", engine, exc)
        return []


def fetch_signal_dates(engine: str, supabase: Any = None) -> List[date]:
    """Distinct trade_dates with a persisted top-book for one engine."""
    try:
        sb = _client(supabase)
        rows = _fetch_all(lambda off, lim: (
            sb.table("style_signals").select("trade_date")
            .eq("engine", engine).order("trade_date")
            .range(off, off + lim - 1).execute().data))
    except Exception as exc:  # noqa: BLE001 — honest-empty
        logger.warning("fetch_signal_dates(%s) failed: %s", engine, exc)
        return []
    return sorted({_to_date(r["trade_date"]) for r in rows})
