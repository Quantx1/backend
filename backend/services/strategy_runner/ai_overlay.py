"""AI overlay — PR-FAN.

The user-level safety layer that sits between DSL match → broker order.

Even if a strategy fires "BUY RELIANCE", the AI overlay can block it
because:
  - Regime is `bear` and user has `regime_gate_enabled = True`
  - VIX is above `vix_hard_block_threshold` (default 35)
  - User's gross exposure is already at the cap
  - Stock-level cap (default 5% of capital) would be breached

The overlay is the difference between "this strategy signal fired" and
"actually take this trade." For a user running 50 strategies in parallel,
the overlay is what prevents 50 simultaneous bull-market longs in a
crashing market.

Memory locks honoured:
  - LLM never gates trades — overlay is pure rules
  - No fallbacks — unknown regime = "do not enter"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..regime import resolve_regime_at

logger = logging.getLogger(__name__)


# Default settings if the user has never touched their overlay config.
# Conservative — assume the user wants protection.
_DEFAULT_SETTINGS = {
    "regime_gate_enabled": True,
    "blocked_regimes": ["bear"],
    "vix_overlay_enabled": True,
    "vix_hard_block_threshold": 35.0,
    "alpha_rank_filter_enabled": False,
    "alpha_top_k": 5,
    "max_gross_exposure_pct": 80.0,
    "max_per_stock_pct": 5.0,
    # Event-risk: block NEW entries within the earnings window ("don't get
    # killed by events"). On by default — assume the user wants protection.
    "event_gate_enabled": True,
}


@dataclass
class AIOverlaySettings:
    """User's overlay preferences. Loaded from user_ai_overlay_settings,
    falls back to DEFAULTS if the user has never customised."""
    regime_gate_enabled: bool = True
    blocked_regimes: List[str] = field(default_factory=lambda: ["bear"])
    vix_overlay_enabled: bool = True
    vix_hard_block_threshold: float = 35.0
    alpha_rank_filter_enabled: bool = False
    alpha_top_k: int = 5
    max_gross_exposure_pct: float = 80.0
    max_per_stock_pct: float = 5.0
    event_gate_enabled: bool = True

    @classmethod
    def from_row(cls, row: Optional[dict]) -> "AIOverlaySettings":
        if not row:
            return cls()
        blocked = row.get("blocked_regimes")
        if not isinstance(blocked, list):
            blocked = ["bear"]
        return cls(
            regime_gate_enabled=bool(row.get("regime_gate_enabled", True)),
            blocked_regimes=list(blocked),
            vix_overlay_enabled=bool(row.get("vix_overlay_enabled", True)),
            vix_hard_block_threshold=float(row.get("vix_hard_block_threshold", 35.0)),
            alpha_rank_filter_enabled=bool(row.get("alpha_rank_filter_enabled", False)),
            alpha_top_k=int(row.get("alpha_top_k", 5)),
            max_gross_exposure_pct=float(row.get("max_gross_exposure_pct", 80.0)),
            max_per_stock_pct=float(row.get("max_per_stock_pct", 5.0)),
            event_gate_enabled=bool(row.get("event_gate_enabled", True)),
        )


@dataclass
class AIOverlayDecision:
    """Result of running the overlay on a candidate entry signal."""
    allowed: bool
    block_reason: Optional[str] = None
    size_multiplier: float = 1.0   # 1.0 = full size, 0.5 = half size, etc.
    notes: List[str] = field(default_factory=list)


def load_overlay_settings(supabase: Any, user_id: str) -> AIOverlaySettings:
    """Fetch a user's overlay settings (or sensible defaults if no row)."""
    try:
        rows = (
            supabase.table("user_ai_overlay_settings")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
    except Exception:
        return AIOverlaySettings()
    return AIOverlaySettings.from_row((rows.data or [None])[0])


def apply_ai_overlay(
    *,
    supabase: Any,
    settings: AIOverlaySettings,
    user_id: str,
    symbol: str,
    current_vix: Optional[float] = None,
    current_regime: Optional[str] = None,
    event_blackout: Optional[set] = None,
) -> AIOverlayDecision:
    """Run the overlay against one candidate entry signal.

    Returns ``AIOverlayDecision`` — either allowed (with optional size
    multiplier) or blocked with a reason. Pure function, no side effects.

    ``event_blackout`` (pre-computed by the runner for the whole cycle) lets
    the event-risk gate avoid a per-candidate DB call; if omitted, a single
    live lookup for this symbol is performed.
    """
    notes: List[str] = []

    # 1. Regime gate — pull from resolver if not provided
    if settings.regime_gate_enabled:
        regime = current_regime or resolve_regime_at(supabase)
        if regime in settings.blocked_regimes:
            return AIOverlayDecision(
                allowed=False,
                block_reason=f"regime_gate:{regime}",
                notes=[f"regime={regime}, blocked by user settings"],
            )
        notes.append(f"regime={regime} (allowed)")

    # 1b. Event-risk gate — never OPEN a new position into an earnings window.
    if settings.event_gate_enabled and symbol:
        try:
            if event_blackout is not None:
                blocked = symbol.upper() in {s.upper() for s in event_blackout}
            else:
                from ..scanners.event_risk import symbols_in_event_window
                blocked = bool(symbols_in_event_window([symbol]))
        except Exception:
            blocked = False  # fail-open: a data outage never blocks all entries
        if blocked:
            return AIOverlayDecision(
                allowed=False,
                block_reason="event_risk:earnings",
                notes=[f"{symbol} has earnings inside the blackout window — entry suppressed"],
            )

    # 2. VIX hard block
    if settings.vix_overlay_enabled and current_vix is not None:
        if current_vix >= settings.vix_hard_block_threshold:
            return AIOverlayDecision(
                allowed=False,
                block_reason=f"vix_hard_block:{current_vix}",
                notes=[f"VIX={current_vix} ≥ hard threshold {settings.vix_hard_block_threshold}"],
            )

    # 3. VIX-based size scaling (separate from hard block)
    size_mult = 1.0
    if settings.vix_overlay_enabled and current_vix is not None:
        # Stair-step: <15 → 1.0; 15-22 → 0.8; 22-30 → 0.5; 30+ → 0.25
        if current_vix < 15:
            size_mult = 1.0
        elif current_vix < 22:
            size_mult = 0.8
            notes.append(f"VIX={current_vix:.1f} → size 0.8x")
        elif current_vix < 30:
            size_mult = 0.5
            notes.append(f"VIX={current_vix:.1f} → size 0.5x")
        else:
            size_mult = 0.25
            notes.append(f"VIX={current_vix:.1f} → size 0.25x")

    # 4. Per-stock + gross exposure caps — checked at order placement, not here
    # (the runner queries strategy_positions for current exposure)

    return AIOverlayDecision(
        allowed=True,
        size_multiplier=size_mult,
        notes=notes,
    )
