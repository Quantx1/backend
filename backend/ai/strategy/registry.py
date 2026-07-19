"""
Strategy registry — PR-F per v2 design spec §7.3 + §7.5.

State machine over ``user_strategies.status``:

    draft ──► backtest ──► paper ──► live
                  │           │       │
                  └─► archived├──────►│
                              ▼       ▼
                            paused ◄──┘
                              │
                              ▼
                           paper or live (resume)

Transitions enforced HERE (not in DB triggers) so we can audit + tune
without an SQL deploy. Every state change writes ``deployed_at`` /
``paused_at`` / ``archived_at`` timestamps for forensics.

Tier gating: ``live`` requires user.tier in {pro, elite}. ``paper``
is free-tier-allowed. (Gate check is in the API layer — this module
doesn't know about user tiers.)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .dsl import Strategy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Allowed transitions — anything not here raises in transition()
# ─────────────────────────────────────────────────────────────────────


_TRANSITIONS: Dict[str, Set[str]] = {
    "draft": {"backtest", "paper", "archived"},
    "backtest": {"draft", "paper", "live", "archived"},
    "paper": {"paused", "live", "archived"},
    "live": {"paused", "archived"},
    "paused": {"paper", "live", "archived"},
    "archived": set(),  # terminal
}


class StrategyStateError(ValueError):
    """Raised when a transition isn't allowed or a precondition fails."""


def is_terminal(status: str) -> bool:
    return status == "archived"


def allowed_transitions(status: str) -> Set[str]:
    return _TRANSITIONS.get(status, set())


def validate_transition(from_status: str, to_status: str) -> None:
    """Raise StrategyStateError if ``from_status`` cannot move to ``to_status``."""
    if from_status == to_status:
        # No-op transitions are allowed (idempotency)
        return
    allowed = _TRANSITIONS.get(from_status, set())
    if to_status not in allowed:
        raise StrategyStateError(
            f"transition not allowed: {from_status} → {to_status}. "
            f"Allowed from {from_status}: {sorted(allowed) or 'none (terminal)'}",
        )


# ─────────────────────────────────────────────────────────────────────
# CRUD operations — thin wrappers around Supabase, NOT business logic.
# All return raw rows or raise. Business logic (tier checks, eligibility)
# lives in the API layer.
# ─────────────────────────────────────────────────────────────────────


def create_strategy(
    supabase,
    *,
    user_id: str,
    dsl: Dict[str, Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
    template_slug: Optional[str] = None,
    source: str = "user",
) -> Dict[str, Any]:
    """Create a new strategy in ``draft`` status.

    DSL is validated via the Strategy Pydantic model BEFORE the insert,
    so the DB never holds garbage.
    """
    # Validate DSL — raises ValidationError on bad input
    strategy = Strategy.model_validate(dsl)
    normalized_dsl = strategy.model_dump(mode="json")

    payload = {
        "user_id": user_id,
        "name": name or strategy.name,
        "description": description,
        "template_slug": template_slug,
        "dsl": normalized_dsl,
        "status": "draft",
        "source": source,
    }
    result = supabase.table("user_strategies").insert(payload).execute()
    if not result.data:
        raise RuntimeError("create_strategy: insert returned no row")
    return result.data[0]


def get_strategy(supabase, *, strategy_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single strategy. Returns None if not found or not owned."""
    # A non-UUID id would make the Postgres uuid cast raise (surfacing as a 500);
    # treat a malformed id as simply not-found so callers return a clean 404.
    import uuid as _uuid
    try:
        _uuid.UUID(str(strategy_id))
    except (ValueError, TypeError, AttributeError):
        return None
    result = (
        supabase.table("user_strategies")
        .select("*")
        .eq("id", strategy_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return (result.data or [None])[0]


def list_strategies(
    supabase,
    *,
    user_id: str,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """List the user's strategies, optionally filtered by status."""
    q = supabase.table("user_strategies").select("*").eq("user_id", user_id)
    if status:
        q = q.eq("status", status)
    result = q.order("updated_at", desc=True).limit(limit).execute()
    return result.data or []


def update_dsl(
    supabase,
    *,
    strategy_id: str,
    user_id: str,
    dsl: Dict[str, Any],
) -> Dict[str, Any]:
    """Edit the DSL. Allowed only while status in {draft, paused}.

    Live or paper strategies must be paused first — we don't allow
    in-flight DSL swaps because the scheduler might be mid-evaluation.
    """
    current = get_strategy(supabase, strategy_id=strategy_id, user_id=user_id)
    if current is None:
        raise StrategyStateError("strategy not found or not owned")
    if current["status"] not in ("draft", "paused"):
        raise StrategyStateError(
            f"cannot edit DSL while status={current['status']}. "
            f"Pause the strategy first.",
        )
    # Validate the new DSL
    strategy = Strategy.model_validate(dsl)
    normalized_dsl = strategy.model_dump(mode="json")

    result = (
        supabase.table("user_strategies")
        .update({"dsl": normalized_dsl, "name": strategy.name})
        .eq("id", strategy_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not result.data:
        raise RuntimeError("update_dsl: no row updated")
    return result.data[0]


def transition_status(
    supabase,
    *,
    strategy_id: str,
    user_id: str,
    new_status: str,
    capital_allocated: Optional[float] = None,
) -> Dict[str, Any]:
    """Move the strategy through the state machine.

    Side effects:
      - paper/live: sets deployed_at if first time
      - paused: sets paused_at
      - archived: sets archived_at + clears capital_allocated
      - capital_allocated update is only honored on paper/live transitions
    """
    current = get_strategy(supabase, strategy_id=strategy_id, user_id=user_id)
    if current is None:
        raise StrategyStateError("strategy not found or not owned")

    validate_transition(current["status"], new_status)

    now = datetime.now(timezone.utc).isoformat()
    update_payload: Dict[str, Any] = {"status": new_status}

    if new_status in ("paper", "live"):
        if current["status"] == "draft" or not current.get("deployed_at"):
            update_payload["deployed_at"] = now
        if capital_allocated is not None:
            if capital_allocated <= 0:
                raise StrategyStateError(
                    "capital_allocated must be > 0 when moving to paper/live",
                )
            update_payload["capital_allocated"] = capital_allocated
    elif new_status == "paused":
        update_payload["paused_at"] = now
    elif new_status == "archived":
        update_payload["archived_at"] = now
        update_payload["capital_allocated"] = 0

    result = (
        supabase.table("user_strategies")
        .update(update_payload)
        .eq("id", strategy_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not result.data:
        raise RuntimeError("transition_status: no row updated")
    logger.info(
        "strategy %s/%s: %s → %s",
        user_id, strategy_id, current["status"], new_status,
    )
    return result.data[0]


def record_backtest(
    supabase,
    *,
    strategy_id: str,
    user_id: str,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Stash the latest backtest result on the strategy row. Called after
    PR-G's backtest endpoint runs. Summary shape (suggested):
        {sharpe, win_rate, max_dd_pct, trades, equity_curve_id, ran_at}
    """
    now = datetime.now(timezone.utc).isoformat()
    enriched = {**summary, "ran_at": now}
    result = (
        supabase.table("user_strategies")
        .update({"last_backtest": enriched, "last_run_at": now})
        .eq("id", strategy_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not result.data:
        raise RuntimeError("record_backtest: no row updated")
    return result.data[0]


def log_execution(
    supabase,
    *,
    strategy_id: str,
    user_id: str,
    decision: str,
    mode: str,
    symbol: Optional[str] = None,
    trace: Optional[Dict[str, Any]] = None,
    trade_id: Optional[str] = None,
) -> None:
    """Append a tick row to strategy_executions. Best-effort — failures
    are logged but never raise (we don't want telemetry to break trading)."""
    try:
        supabase.table("strategy_executions").insert({
            "strategy_id": strategy_id,
            "user_id": user_id,
            "symbol": symbol,
            "decision": decision,
            "trace": trace or {},
            "trade_id": trade_id,
            "mode": mode,
        }).execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("log_execution failed for %s: %s", strategy_id, exc)


__all__ = [
    "StrategyStateError",
    "allowed_transitions",
    "validate_transition",
    "is_terminal",
    "create_strategy",
    "get_strategy",
    "list_strategies",
    "update_dsl",
    "transition_status",
    "record_backtest",
    "log_execution",
]
