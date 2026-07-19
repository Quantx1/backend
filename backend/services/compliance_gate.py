"""
Algorithmic-order compliance gate — SEBI algo-trading framework enforcement.

Every **automated** live order (fired by a strategy / cron, without a human
click) must pass ``check_algo_order`` before it reaches a broker. It fails
**closed** in production:

  * kill-switch active / auto-trader disabled → refuse (durable pause).
  * operator not exchange-empanelled          → refuse live automated orders.
      SEBI's algo-trading framework requires broker-approved, exchange-tagged
      algos carrying a registered strategy / algo ID (and a static IP on record).
      Until the operator is empanelled and sets ``ALGO_EMPANELMENT_ID`` +
      ``ALGO_TRADING_ENABLED``, the platform must NOT route live automated orders.
  * live options while only a SYNTHETIC (Black-Scholes) backtest exists → refuse.
      ``options_backtest`` always stamps ``synthetic_backtest=True``; a synthetic
      P&L curve must not authorise real-money option orders. Gated by
      ``ALLOW_LIVE_OPTIONS`` (default False).

Human-initiated orders (a user tapping Buy, copilot act-with-confirm) are NOT
automated — they go through the normal broker flow and are not gated here.

In development / paper (``APP_ENV != production`` or paper execution) the gate is
permissive **but still honours the kill-switch**, so tests and paper trading work
while the durable pause is always respected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class AlgoOrderDecision:
    allowed: bool
    reason: Optional[str] = None
    # Exchange-required audit tags to attach to the order / trade row.
    tags: Dict[str, str] = field(default_factory=dict)


def _is_production() -> bool:
    return (settings.APP_ENV or "").lower() in ("production", "prod")


def _profile_flags(supabase: Any, user_id: str) -> Dict[str, Any]:
    """Read the durable trading-control flags.

    In production, fail CLOSED on any error (treat the kill-switch as engaged) so
    we never route a real-money order we couldn't verify. In dev/test an infra
    error is non-fatal (returns empty → permissive), so local runs and the test
    suite aren't blocked by a mock/DB hiccup."""
    try:
        res = (
            supabase.table("user_profiles")
            .select("kill_switch_active, auto_trader_enabled")
            .eq("id", user_id)
            .single()
            .execute()
        )
        return res.data or {}
    except Exception as e:
        logger.warning("compliance_gate: profile-flag read failed: %s", e)
        if _is_production():
            return {"kill_switch_active": True, "auto_trader_enabled": False}
        return {}


def check_algo_order(
    *,
    supabase: Any,
    user_id: str,
    strategy_id: str = "",
    segment: str = "equity",   # "equity" | "options"
    automated: bool = True,
    live: bool = True,
) -> AlgoOrderDecision:
    """Gate a single order. Returns an :class:`AlgoOrderDecision`.

    ``tags`` carries the exchange-audit fields (algo id, static IP, order source,
    strategy id) that the caller should attach to the broker order + trades row.
    """
    tags: Dict[str, str] = {
        "order_source": "algo" if automated else "manual",
        "strategy_id": str(strategy_id or ""),
        "algo_id": settings.ALGO_EMPANELMENT_ID or "",
        "static_ip": settings.ALGO_STATIC_IP or "",
        "segment": segment,
    }

    # ── Durable pause: honoured in EVERY environment, paper included ──
    flags = _profile_flags(supabase, user_id)
    if flags.get("kill_switch_active"):
        return AlgoOrderDecision(False, "kill_switch_active", tags)
    # An automated order requires the auto-trader to be explicitly ON. The
    # durable kill-switch sets auto_trader_enabled=False, so this also enforces
    # the pause even if kill_switch_active were cleared out of band.
    if automated and flags.get("auto_trader_enabled") is False:
        return AlgoOrderDecision(False, "auto_trader_disabled", tags)

    # ── Paper / non-live: allowed (no real money, no exchange framework) ──
    if not live:
        return AlgoOrderDecision(True, None, tags)

    # ── Live options need a real (non-synthetic) options backtest ──
    if segment == "options" and not settings.ALLOW_LIVE_OPTIONS:
        return AlgoOrderDecision(False, "live_options_disabled_synthetic_backtest", tags)

    # ── Production live AUTOMATED orders require FULL exchange empanelment ──
    # Dev/paper skip this so tests and local runs work; production must have,
    # per the SEBI algo framework, ALL of: the master switch, a real
    # exchange-issued algo/empanelment ID, AND the registered static IP the
    # algo trades from. Any one missing → refuse to route. None of these can be
    # self-issued (see algo_readiness()).
    if automated and _is_production():
        if not _algo_live_ready():
            return AlgoOrderDecision(False, "algo_not_empanelled", tags)

    return AlgoOrderDecision(True, None, tags)


def _algo_live_ready() -> bool:
    return (
        bool(settings.ALGO_TRADING_ENABLED)
        and bool((settings.ALGO_EMPANELMENT_ID or "").strip())
        and bool((settings.ALGO_STATIC_IP or "").strip())
    )


def algo_readiness() -> Dict[str, Any]:
    """What is configured vs. still required to route LIVE automated orders in
    production (SEBI algo framework). The empanelment ID and static IP must come
    from real NSE/broker algo registration — they cannot be self-issued, so this
    is a readiness report, not a switch."""
    empanelment = bool((settings.ALGO_EMPANELMENT_ID or "").strip())
    static_ip = bool((settings.ALGO_STATIC_IP or "").strip())
    enabled = bool(settings.ALGO_TRADING_ENABLED)
    return {
        "live_automated_ready": empanelment and static_ip and enabled,
        "checklist": {
            "algo_trading_enabled": enabled,          # ALGO_TRADING_ENABLED (your switch)
            "exchange_empanelment_id": empanelment,   # ALGO_EMPANELMENT_ID (NSE-issued)
            "registered_static_ip": static_ip,        # ALGO_STATIC_IP (broker-whitelisted)
        },
        "note": (
            "Empanelment ID + static IP must come from real NSE/broker algo "
            "registration. Paper trading runs fully automated regardless."
        ),
    }
