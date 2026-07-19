"""
F12 daily digest content builder.

Two shapes:

    build_morning_brief(user_id, supabase_client) -> DigestPayload
    build_evening_summary(user_id, supabase_client) -> DigestPayload

The morning brief runs pre-market (7:30 IST) and answers "what do I
need to know before the bell." The evening summary runs post-close
(17:30 IST) and answers "what moved, what changed in my book, what's
tomorrow."

Each payload is a short plain-text body (≤ 1000 chars) that renders
identically on Telegram and WhatsApp. We deliberately avoid markdown
because WhatsApp strips most of it and Telegram + WhatsApp can't be
given the same text if we lean on provider-specific formatting.

Shape:
  * Deterministic header (regime, index levels)           ← market-wide
  * Per-user personalization block (positions, triggers)  ← user-specific
  * Optional LLM prose layer for the intro line

The no-fallbacks rule applies to ML features. Prose generation is a
presentation concern: if the LLM is down we still assemble a templated
brief so the channel doesn't go silent — that's product UX, not a
model substitution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional
from ...data.market_calendar import IST

logger = logging.getLogger(__name__)


MAX_BODY_CHARS = 1000  # hard ceiling — WhatsApp templates cap at 1024.


# ============================================================================
# Types
# ============================================================================


@dataclass
class DigestPayload:
    user_id: str
    kind: str              # 'morning' | 'evening'
    body: str              # plain-text, channel-agnostic
    stats: Dict[str, Any] = field(default_factory=dict)
    source: str = "template"  # 'template' | 'llm'


# ============================================================================
# Shared market-side fetchers — cheap, called once per scheduler run
# ============================================================================


def _fetch_current_regime(sb) -> Optional[Dict[str, Any]]:
    try:
        r = (
            sb.table("regime_history")
            .select("regime, prob_bull, prob_sideways, prob_bear, vix, nifty_close, detected_at")
            .order("detected_at", desc=True)
            .limit(1)
            .execute()
        )
        return (r.data or [None])[0]
    except Exception as exc:
        logger.debug("regime lookup failed: %s", exc)
        return None


def _fetch_today_signals(sb, since_iso: str) -> List[Dict[str, Any]]:
    try:
        r = (
            sb.table("signals")
            .select("id, symbol, direction, entry_price, target_1, stop_loss, confidence, status")
            .gte("generated_at", since_iso)
            .in_("status", ["active", "pending", "triggered", "target_hit", "stop_loss_hit", "sl_hit"])
            .order("confidence", desc=True)
            .limit(10)
            .execute()
        )
        return r.data or []
    except Exception as exc:
        logger.debug("signals lookup failed: %s", exc)
        return []


def _fetch_nifty_close(sb) -> Optional[Dict[str, Any]]:
    """Latest recorded Nifty close + % change. Pulls from regime_history
    since that's where we persist the daily close alongside the regime
    snapshot."""
    try:
        r = (
            sb.table("regime_history")
            .select("nifty_close, detected_at")
            .order("detected_at", desc=True)
            .limit(2)
            .execute()
        )
        rows = r.data or []
        if len(rows) < 2 or not rows[0].get("nifty_close") or not rows[1].get("nifty_close"):
            return rows[0] if rows else None
        curr = float(rows[0]["nifty_close"])
        prev = float(rows[1]["nifty_close"])
        change_pct = ((curr - prev) / prev) * 100 if prev else 0.0
        return {
            "close": curr,
            "change_pct": round(change_pct, 2),
            "as_of": rows[0].get("detected_at"),
        }
    except Exception as exc:
        logger.debug("nifty close lookup failed: %s", exc)
        return None


def _fetch_open_positions(sb, user_id: str) -> List[Dict[str, Any]]:
    try:
        r = (
            sb.table("positions")
            # positions cols are average_price / unrealized_pnl_percent / is_active;
            # alias back to the keys the digest template expects.
            .select("symbol, direction, entry_price:average_price, current_price, "
                    "unrealized_pnl, unrealized_pnl_percentage:unrealized_pnl_percent")
            .eq("user_id", user_id)
            .eq("is_active", True)
            .limit(50)
            .execute()
        )
        return r.data or []
    except Exception as exc:
        logger.debug("positions lookup failed %s: %s", user_id, exc)
        return []


def _fetch_closed_today(sb, user_id: str, since_iso: str) -> List[Dict[str, Any]]:
    try:
        r = (
            sb.table("trades")
            .select("symbol, status, net_pnl, pnl_percent, closed_at")
            .eq("user_id", user_id)
            .eq("status", "closed")
            .gte("closed_at", since_iso)
            .limit(50)
            .execute()
        )
        return r.data or []
    except Exception as exc:
        logger.debug("closed-today lookup failed %s: %s", user_id, exc)
        return []


# ============================================================================
# Formatting
# ============================================================================


def _regime_line(regime: Optional[Dict[str, Any]]) -> str:
    if not regime:
        return "Regime — regime snapshot not available yet."
    label = str(regime.get("regime") or "").capitalize() or "Sideways"
    vix = regime.get("vix")
    vix_bit = f" (VIX {float(vix):.1f})" if vix is not None else ""
    # Playbook hint varies with regime.
    r = (regime.get("regime") or "").lower()
    hint = (
        "sizing up allowed; momentum setups preferred" if r == "bull"
        else "halve position size; defensive tilt" if r == "bear"
        else "mean-reversion works, breakouts don't; tighten SLs"
    )
    return f"Regime: {label}{vix_bit} — {hint}."


def _signals_summary(signals: List[Dict[str, Any]]) -> str:
    if not signals:
        return "No active Forecast signals this morning — stay patient."
    top = signals[0]
    sym = top.get("symbol") or "—"
    direction = top.get("direction") or "LONG"
    entry = top.get("entry_price") or 0
    target = top.get("target_1") or 0
    sl = top.get("stop_loss") or 0
    conf = top.get("confidence")
    # signals.confidence is already a 0-100 percent (range ~12-91), not a 0-1 fraction.
    conf_bit = f" ({int(conf)}% conf)" if conf is not None else ""
    rest = len(signals) - 1
    rest_bit = f" + {rest} more" if rest > 0 else ""
    return (
        f"Top signal: {sym} {direction}{conf_bit} — "
        f"entry ₹{entry:.2f}, target ₹{target:.2f}, SL ₹{sl:.2f}{rest_bit}."
    )


def _positions_summary(positions: List[Dict[str, Any]]) -> str:
    if not positions:
        return "You have no open positions."
    total_pnl = sum(float(p.get("unrealized_pnl") or 0) for p in positions)
    n = len(positions)
    sign = "+" if total_pnl >= 0 else ""
    return (
        f"{n} open position{'s' if n != 1 else ''}, unrealized "
        f"{sign}₹{total_pnl:,.0f}."
    )


def _closed_today_summary(closed: List[Dict[str, Any]]) -> str:
    if not closed:
        return "No trades closed today."
    total = sum(float(c.get("net_pnl") or 0) for c in closed)
    wins = sum(1 for c in closed if float(c.get("net_pnl") or 0) > 0)
    n = len(closed)
    sign = "+" if total >= 0 else ""
    return (
        f"{n} trade{'s' if n != 1 else ''} closed today "
        f"({wins} win{'s' if wins != 1 else ''}), net {sign}₹{total:,.0f}."
    )


def _nifty_line(nifty: Optional[Dict[str, Any]], kind: str) -> str:
    if not nifty or nifty.get("close") is None:
        return ""
    close = float(nifty["close"])
    cp = nifty.get("change_pct")
    if cp is None:
        return f"Nifty {'close' if kind == 'evening' else 'prev close'}: {close:,.0f}."
    sign = "+" if cp >= 0 else ""
    return f"Nifty {'closed' if kind == 'evening' else 'prev close'}: {close:,.0f} ({sign}{cp:.2f}%)."


def _truncate(body: str, n: int = MAX_BODY_CHARS) -> str:
    if len(body) <= n:
        return body
    return body[: n - 1].rstrip() + "…"


# ============================================================================
# LLM prose layer — optional intro line
# ============================================================================


async def _llm_intro(stats: Dict[str, Any], kind: str) -> Optional[str]:
    """One-sentence conversational opener. Skipped when the LLM is
    unavailable — the rest of the body is deterministic so the digest
    still ships without it."""
    try:
        from ..agents.llm import LLM
    except Exception:
        return None
    llm = LLM()
    if not llm.enabled:
        return None

    horizon = "pre-market" if kind == "morning" else "post-close"
    prompt = f"""Write ONE short sentence (≤18 words, plain text, no emoji,
no markdown) to open a {horizon} daily brief for an Indian retail trader.
Tone: confident, direct, conversational — not a weather report.

Do NOT mention specific AI model names (TFT, Qlib, FinBERT, HMM,
LightGBM, FinRL, XGBoost, Chronos, TimesFM, LSTM). Use only the public
engine names if referencing any: Forecast, Alpha, Regime,
Intraday, Trajectory, Mood, EarningsScout.

Stats:
{stats}

Return the sentence only — no preamble, no surrounding punctuation
other than the terminal period."""
    try:
        text = await llm.complete(prompt, temperature=0.5, top_p=0.9)
        text = (text or "").strip().strip('"').strip()
        if not text or len(text) < 12 or len(text) > 220:
            return None
        if "I cannot" in text[:60].lower():
            return None
        return text
    except Exception as exc:
        logger.debug("LLM digest intro failed: %s", exc)
        return None


# ============================================================================
# Public builders
# ============================================================================


async def build_morning_brief(
    *,
    user_id: str,
    supabase_client,
    market_cache: Optional[Dict[str, Any]] = None,
) -> DigestPayload:
    """Build the pre-market brief for one user. ``market_cache`` lets the
    fan-out pass shared market data (regime, signals, nifty) in to avoid
    re-querying per user."""
    mc = market_cache or {}
    regime = mc.get("regime") or _fetch_current_regime(supabase_client)
    # Signals generated in the last 16h catch both EOD-scan (from 15:45
    # yesterday) and any pre-market-scan output (8:30 IST today).
    since = (datetime.now(IST) - timedelta(hours=16)).astimezone(timezone.utc).isoformat()
    signals = mc.get("signals") if "signals" in mc else _fetch_today_signals(supabase_client, since)
    nifty = mc.get("nifty") if "nifty" in mc else _fetch_nifty_close(supabase_client)

    positions = _fetch_open_positions(supabase_client, user_id)

    stats = {
        "regime": regime,
        "signals_count": len(signals),
        "top_signal": signals[0] if signals else None,
        "positions_count": len(positions),
        "nifty": nifty,
    }

    intro = await _llm_intro(stats, "morning")
    source = "llm" if intro else "template"

    lines: List[str] = []
    lines.append(intro or "Pre-market brief — here's your board for today.")
    lines.append(_regime_line(regime))
    nl = _nifty_line(nifty, "morning")
    if nl:
        lines.append(nl)
    lines.append(_signals_summary(signals))
    if positions:
        lines.append(_positions_summary(positions))

    body = _truncate("\n".join(line for line in lines if line))
    return DigestPayload(
        user_id=user_id, kind="morning", body=body, stats=stats, source=source,
    )


async def build_evening_summary(
    *,
    user_id: str,
    supabase_client,
    market_cache: Optional[Dict[str, Any]] = None,
) -> DigestPayload:
    """Build the post-close summary for one user."""
    mc = market_cache or {}
    regime = mc.get("regime") or _fetch_current_regime(supabase_client)
    nifty = mc.get("nifty") if "nifty" in mc else _fetch_nifty_close(supabase_client)

    # Today's window — IST calendar day.
    today_start_ist = datetime.combine(date.today(), time(0, 0), IST)
    since_utc = today_start_ist.astimezone(timezone.utc).isoformat()
    closed = _fetch_closed_today(supabase_client, user_id, since_utc)
    positions = _fetch_open_positions(supabase_client, user_id)

    stats = {
        "regime": regime,
        "nifty": nifty,
        "closed_today_count": len(closed),
        "open_positions_count": len(positions),
    }

    intro = await _llm_intro(stats, "evening")
    source = "llm" if intro else "template"

    lines: List[str] = []
    lines.append(intro or "Market closed — here's your day.")
    nl = _nifty_line(nifty, "evening")
    if nl:
        lines.append(nl)
    lines.append(_regime_line(regime))
    lines.append(_closed_today_summary(closed))
    if positions:
        lines.append(_positions_summary(positions))

    body = _truncate("\n".join(line for line in lines if line))
    return DigestPayload(
        user_id=user_id, kind="evening", body=body, stats=stats, source=source,
    )


__all__ = [
    "DigestPayload",
    "build_morning_brief",
    "build_evening_summary",
    "IST",
    "MAX_BODY_CHARS",
]
