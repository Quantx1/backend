"""Copilot action proposer — turns a natural-language request + page context into
structured, reviewable ACTION proposals (Cursor-style "agent" mode).

PROPOSES ONLY — it never executes anything. Execution happens client-side against
the EXISTING gated endpoints (watchlist / screener / broker order), so the
server-side gates (global-halt → kill-switch → tier → broker → validation) remain
the single source of authority. This module just maps intent → a typed proposal
the user explicitly confirms.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Allowlist — any kind not in here is dropped. place_order is flagged danger and,
# on the client, routes to the fully-gated /broker/order flow.
_ALLOWED = {
    "watchlist_add", "watchlist_remove", "run_screen",
    "place_order", "create_strategy_draft",
}

_SCHEMA = (
    '{"actions": [{'
    '"kind": "watchlist_add|watchlist_remove|run_screen|place_order|create_strategy_draft", '
    '"title": "<short imperative label>", '
    '"summary": "<one line describing what it will do>", '
    '"args": { ...kind-specific... }}]} '
    "— empty array when the message is a question/analysis, not an action. Max 3. "
    'args by kind: watchlist_add / watchlist_remove {"symbol":"RELIANCE"}; '
    'run_screen {"query":"oversold largecaps in an uptrend"}; '
    'place_order {"symbol":"RELIANCE","side":"BUY"|"SELL","quantity":10,'
    '"order_type":"MARKET"|"LIMIT","price":<number, only for LIMIT>,"product":"CNC"|"MIS"}; '
    'create_strategy_draft {"prompt":"buy when RSI below 30, sell above 70"}.'
)

_SYSTEM = (
    "You convert a trader's request into concrete, reviewable ACTION proposals for "
    "an Indian-equities (NSE/BSE) app. Propose ONLY actions the user is actually "
    "asking to take right now; return an EMPTY array for questions, analysis or "
    "chit-chat. NEVER invent a ticker — use the symbol in context or one the user "
    "explicitly named. You only PROPOSE; the user confirms before anything runs. "
    "Return JSON only."
)

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9_&.\-]{0,18}$")


def _clean_symbol(v: Any) -> Optional[str]:
    if not isinstance(v, str):
        return None
    s = v.strip().upper()
    return s if _SYMBOL_RE.match(s) else None


def _validate(kind: str, args: Dict[str, Any], ctx_symbol: Optional[str]) -> Optional[Dict[str, Any]]:
    """Clean + bound a proposal's args. Returns None to DROP an invalid proposal —
    the agent must never surface a half-formed or unsafe action."""
    args = args or {}
    if kind in ("watchlist_add", "watchlist_remove"):
        sym = _clean_symbol(args.get("symbol")) or ctx_symbol
        return {"symbol": sym} if sym else None
    if kind == "run_screen":
        q = args.get("query")
        return {"query": q.strip()} if isinstance(q, str) and len(q.strip()) >= 3 else None
    if kind == "create_strategy_draft":
        p = args.get("prompt")
        return {"prompt": p.strip()} if isinstance(p, str) and len(p.strip()) >= 3 else None
    if kind == "place_order":
        sym = _clean_symbol(args.get("symbol")) or ctx_symbol
        side = str(args.get("side", "")).upper()
        otype = str(args.get("order_type", "MARKET")).upper()
        product = str(args.get("product", "CNC")).upper()
        try:
            qty = int(args.get("quantity") or 0)
        except (TypeError, ValueError):
            qty = 0
        if not sym or side not in ("BUY", "SELL") or qty <= 0:
            return None
        if otype not in ("MARKET", "LIMIT"):
            otype = "MARKET"
        if product not in ("CNC", "MIS", "NRML"):
            product = "CNC"
        out: Dict[str, Any] = {
            "symbol": sym, "side": side, "quantity": qty,
            "order_type": otype, "product": product,
        }
        if otype == "LIMIT":
            try:
                out["price"] = float(args.get("price"))
            except (TypeError, ValueError):
                return None  # a LIMIT order with no/invalid price is unsafe — drop it
        return out
    return None


async def propose_actions(
    *,
    message: str,
    route: Optional[str],
    symbol: Optional[str],
    user_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Map an NL request + page context to ≤3 validated, allowlisted proposals.
    Returns [] on a non-action message or any failure (never raises)."""
    from .llm import llm_for

    llm = llm_for("tool_planner")
    if not getattr(llm, "enabled", True):
        return []

    prompt = (
        f"User message: {message}\n"
        f"Current route: {route or '-'}\n"
        f"Symbol in context: {symbol or '-'}\n\n"
        "Propose the actions."
    )
    try:
        parsed = await llm.generate_json(
            prompt, _SCHEMA, system=_SYSTEM, user_id=user_id, feature="copilot_actions",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("propose_actions failed: %s", exc)
        return []

    raw = parsed.get("actions") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        return []

    out: List[Dict[str, Any]] = []
    for i, a in enumerate(raw[:3]):
        if not isinstance(a, dict):
            continue
        kind = a.get("kind")
        if kind not in _ALLOWED:
            continue
        cleaned = _validate(kind, a.get("args") or {}, symbol)
        if cleaned is None:
            continue
        out.append({
            "id": f"act-{i}",
            "kind": kind,
            "title": str(a.get("title") or kind.replace("_", " ").title())[:80],
            "summary": str(a.get("summary") or "")[:160],
            "args": cleaned,
            "danger": kind == "place_order",
        })
    return out
