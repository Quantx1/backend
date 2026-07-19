"""
F4 AutoPilot service — supervised-stack daily rebalancer.

2026-05-24 rewrite (post-v1 ship): RL was removed from v1 scope (see
project_rl_removed_2026_05_23.md). This service now orchestrates the
15:45 IST daily rebalance using the 4 PROD-promoted supervised models:

    Qlib LightGBM (Alpha158)    → cross-sectional stock ranker
    HMM regime classifier       → position-size multiplier (bull/sideways/bear)
    India VIX                   → exposure overlay
    Kelly criterion + caps      → final per-symbol weights

Pipeline per user per day:
    1. Resolve current regime (regime_hmm v20+)
    2. Qlib ranks all NSE liquid instruments → take top-N
    3. Convert ranks → target weights (Kelly-tilted, capped)
    4. Apply regime multiplier (bull=1.0, sideways=0.7, bear=0.3)
    5. Apply VIX overlay (RiskManagementEngine.apply_autopilot_overlays)
    6. Apply hard caps: 5% per stock, 20% per sector, 80% gross
    7. Diff against current live positions → emit trades

No RL inference. No FinRLXEnsemble. The previous code's no-op stub
(_build_observation always returned None) meant AutoPilot was silently
disabled for every Elite user. This rewrite makes it functional.

Safety:
    - dry_run flag still writes the decision row but skips broker emission
    - live-trade eligibility gate is checked twice (entry + per-trade)
    - per-position 5% cap means no single AI decision can blow up >5% of capital
    - daily -10% drawdown circuit breaker via RiskManagementEngine
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# AutoPilot constants — tunable but stable. Per project memory
# (project_v1_launch_4_models_2026_05_24.md) the supervised stack
# ships at v1 with these defaults.
TOP_N_HOLDINGS = 10            # max stocks AutoPilot holds at once
PER_STOCK_CAP = 0.05           # 5% max weight on any single stock
GROSS_EXPOSURE_CAP = 0.80      # 80% max gross exposure (always keep 20% cash)
KELLY_DECAY = 0.85             # geometric decay from top-1 to top-N

# Regime multipliers — applied AFTER Kelly weighting, BEFORE VIX overlay.
# Bear regime halves all positions; sideways scales down 30%.
REGIME_SIZING = {
    "bull": 1.00,
    "sideways": 0.70,
    "bear": 0.30,
}


@dataclass
class RebalanceDecision:
    user_id: str
    target_weights: Dict[str, float]
    regime: str
    blocked_reason: Optional[str] = None


def apply_tier_limits(
    weights: Dict[str, float],
    capital: float,
    limits: Dict[str, Any],
) -> tuple[Dict[str, float], float]:
    """Pricing v2 (2026-06-12) — clamp a rebalance to the user's tier.

    Two knobs, both from ``core.tiers.AUTO_TRADER_TIER_LIMITS``:
      * ``max_concurrent_positions`` — keep only the top-K weights (the
        Kelly-decayed ranking already orders conviction).
      * ``max_deployed_capital`` — AutoPilot Lite's ₹2L cap. Clamping the
        sizing capital bounds deployed value (weights sum ≤ gross cap < 1),
        so a Pro user with ₹10L profile capital still deploys ≤ ₹2L.
    """
    k = limits.get("max_concurrent_positions")
    if k and len(weights) > int(k):
        weights = dict(
            sorted(weights.items(), key=lambda kv: -kv[1])[: int(k)]
        )
    cap = limits.get("max_deployed_capital")
    if cap is not None and capital > float(cap):
        capital = float(cap)
    return weights, capital


class AutoPilotService:
    """Daily rebalancer invoked by the 15:45 IST scheduler job.

    Uses the 4 PROD supervised models. No RL inference. No FinRL-X
    ensemble (removed 2026-05-23, see project_rl_removed_2026_05_23.md).
    """

    def __init__(self, supabase_admin):
        self.supabase = supabase_admin
        self._qlib_engine = None
        self._qlib_load_attempted = False

    # ── Public API ─────────────────────────────────────────────────────

    async def daily_rebalance(self) -> Dict[str, Any]:
        """Loop over every Elite + AutoPilot-enabled user and rebalance.

        Returns a summary dict for the scheduler logger.
        """
        engine = self._get_qlib_engine()
        if engine is None:
            logger.warning(
                "AutoPilot: qlib_alpha158 engine unavailable — skipping rebalance",
            )
            return {"status": "skipped", "reason": "qlib_unavailable", "users": 0}

        # Compute the universe rank ONCE for the whole batch — same
        # Qlib output goes to every Elite user (sizing differs per user).
        ranks = engine.rank_universe(instruments="nse_all")
        if not ranks:
            logger.warning(
                "AutoPilot: qlib_alpha158 produced empty rank — skipping rebalance",
            )
            return {"status": "skipped", "reason": "empty_qlib_rank", "users": 0}

        # Resolve current regime once.
        regime = await self._current_regime()
        vix = await self._current_vix()

        # Compute target weights once — they're the same for every user.
        # User-level differs in: capital, VIX overlay sizing, current positions.
        base_weights = self._compute_base_weights(ranks, regime=regime)

        # Event-risk blackout (one batched query over the ~10 target names):
        # symbols with earnings inside the window are not ELIGIBLE to be opened
        # or added today — "don't get killed by events". Computed once and
        # shared across users; existing holdings can still be trimmed/exited.
        try:
            from ..services.scanners.event_risk import symbols_in_event_window
            event_blackout = symbols_in_event_window(base_weights.keys())
        except Exception as exc:  # noqa: BLE001
            logger.debug("autopilot event-risk lookup failed: %s", exc)
            event_blackout = set()
        if event_blackout:
            logger.info("AutoPilot event-risk blackout (no new entries): %s", sorted(event_blackout))

        users = self._enrolled_users()
        decisions: List[RebalanceDecision] = []
        for user in users:
            try:
                d = await self._rebalance_one(user, regime, vix, base_weights, event_blackout)
                decisions.append(d)
            except Exception as exc:  # noqa: BLE001 — keep going across users
                logger.exception(
                    "AutoPilot rebalance failed for user=%s", user.get("id"),
                )
                decisions.append(RebalanceDecision(
                    user_id=str(user.get("id")),
                    target_weights={},
                    regime=regime,
                    blocked_reason=f"exception: {type(exc).__name__}",
                ))

        ok = sum(1 for d in decisions if d.blocked_reason is None)
        return {
            "status": "ok",
            "regime": regime,
            "vix": vix,
            "qlib_universe_size": len(ranks),
            "target_holdings": len(base_weights),
            "users": len(decisions),
            "rebalanced": ok,
            "blocked": len(decisions) - ok,
            "decisions": [d.__dict__ for d in decisions],
        }

    # ── Core: weight computation ───────────────────────────────────────

    def _compute_base_weights(
        self, ranks: List[dict], *, regime: str,
    ) -> Dict[str, float]:
        """Top-N stocks → Kelly-decayed weights → regime multiplier → caps.

        Cross-sectional Qlib rank is a relative-strength score. We take
        the top N by score and assign Kelly-decayed weights so the #1
        rank gets the most capital. This avoids the all-or-nothing
        problem of equal-weight (which would treat rank-1 and rank-N the
        same) and the over-concentration of pure rank-1-only (which has
        no diversification).

        Output is a dict {symbol: weight} where weights sum to ≤
        GROSS_EXPOSURE_CAP × REGIME_SIZING[regime].
        """
        top = [r for r in ranks if r.get("qlib_score_raw", 0) > 0][:TOP_N_HOLDINGS]
        if not top:
            return {}

        # Geometric Kelly-decayed weights, normalized to sum=1, then
        # scaled to gross exposure cap.
        raw_weights = [KELLY_DECAY ** i for i in range(len(top))]
        total = sum(raw_weights)
        if total <= 0:
            return {}

        regime_mult = REGIME_SIZING.get(regime, REGIME_SIZING["sideways"])
        gross = GROSS_EXPOSURE_CAP * regime_mult
        weights: Dict[str, float] = {}
        for rank_row, w in zip(top, raw_weights):
            sym = str(rank_row.get("symbol") or "").upper()
            if not sym:
                continue
            # Normalize + scale to gross exposure
            target = (w / total) * gross
            # Hard cap per stock
            target = min(target, PER_STOCK_CAP)
            weights[sym] = round(target, 4)
        return weights

    # ── Per-user rebalance ─────────────────────────────────────────────

    async def _rebalance_one(
        self,
        user: Dict[str, Any],
        regime: str,
        vix: float,
        base_weights: Dict[str, float],
        event_blackout: Optional[set] = None,
    ) -> RebalanceDecision:
        user_id = str(user.get("id"))
        event_blackout = event_blackout or set()

        # Pricing v2 2026-06-12 — per-user mode (paper/live) + tier limits.
        from ..core.tiers import (  # noqa: PLC0415
            auto_trader_limits, resolve_autopilot_mode,
        )
        tier = str(user.get("tier") or "free")
        limits = auto_trader_limits(tier)
        mode = resolve_autopilot_mode(tier, user.get("auto_trader_config"))

        # PR 130 — eligibility gate (live money only; paper needs no broker).
        if mode == "live":
            from .eligibility import check_live_trade_eligibility  # noqa: PLC0415
            elig = check_live_trade_eligibility(user_id=user_id, supabase=self.supabase)
            if not elig.eligible:
                self._emit_blocked(user_id, elig.code or "ineligible", regime)
                return RebalanceDecision(
                    user_id=user_id, target_weights={}, regime=regime,
                    blocked_reason=elig.code,
                )

        # PR-AS — per-stream toggle gate. The daily-rebalance flow is the
        # SWING stream (Qlib top-N → HMM → VIX); users who haven't opted in
        # to the swing stream get skipped here even if autopilot_enabled=True
        # at the top level. Lets users disable swing while keeping
        # momentum/options/user-strategy streams active.
        from ..services.autopilot.streams import is_stream_enabled  # noqa: PLC0415
        if not is_stream_enabled(self.supabase, user_id=user_id, stream="swing"):
            logger.info(
                "autopilot._rebalance_one user=%s skipped: swing stream disabled",
                user_id,
            )
            return RebalanceDecision(
                user_id=user_id, target_weights={}, regime=regime,
                blocked_reason="swing_stream_disabled",
            )

        # VIX overlay + bear-regime halving + per-trade VaR cap.
        weights = dict(base_weights)  # copy — per-user overlays must not mutate base
        diag: Dict[str, Any] = {
            "qlib_top_n": len(base_weights),
            "regime_multiplier": REGIME_SIZING.get(regime, 0.70),
            "vix": vix,
        }
        try:
            from .risk import RiskManagementEngine  # noqa: PLC0415
            rme = RiskManagementEngine(self.supabase)
            capital = float(user.get("capital") or 0)
            weights, overlay_diag = rme.apply_autopilot_overlays(
                weights,
                vix_level=vix,
                regime=regime,
                capital=capital,
            )
            diag.update(overlay_diag or {})
        except Exception as exc:
            logger.debug("autopilot overlay skipped: %s", exc)

        # Pricing v2 — tier limits: trim to max positions, clamp deployed
        # capital (the AutoPilot Lite ₹2L cap). Pure + unit-tested.
        capital = float(user.get("capital") or 0)
        weights, capital = apply_tier_limits(weights, capital, limits)
        diag["mode"] = mode
        diag["tier"] = tier
        if limits.get("max_deployed_capital"):
            diag["max_deployed_capital"] = limits["max_deployed_capital"]
        if event_blackout:
            diag["event_blackout"] = sorted(event_blackout)

        # SAFETY — daily-loss circuit breaker. If the user has already breached
        # their daily loss limit, HALT new entries today (de-risking sells still
        # flow) before opening any fresh risk. Live + paper both honored.
        entries_halted = False
        try:
            from .risk import RiskManagementEngine, RISK_PROFILES  # noqa: PLC0415
            cfg = user.get("auto_trader_config") or {}
            rprofile = RISK_PROFILES.get(
                cfg.get("risk_profile") or user.get("risk_profile") or "moderate",
                RISK_PROFILES["moderate"],
            )
            ok, halt_reason = await RiskManagementEngine(self.supabase).check_loss_limits(user_id, rprofile)
            if not ok:
                entries_halted = True
                diag["entries_halted"] = halt_reason
                logger.info("AutoPilot: new entries halted for %s — %s", user_id, halt_reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AutoPilot loss-limit check failed for %s: %s — proceeding", user_id, exc)

        # Dry-run users: record decision but don't emit broker orders.
        dry_run = bool(user.get("autopilot_dry_run", False))
        if dry_run:
            diag["dry_run"] = True

        run_id = self._record_decision(user_id, regime, weights, diag)

        if dry_run:
            emitted = 0
        else:
            emitted = await self._emit_trades(
                user_id, weights, capital, regime, mode=mode,
                event_blackout=event_blackout, entries_halted=entries_halted,
            )

        if run_id:
            try:
                self.supabase.table("auto_trader_runs").update({
                    "trades_executed": emitted,
                    "actions_count": emitted,
                    "status": "executed" if emitted > 0 else "decided",
                }).eq("id", run_id).execute()
            except Exception:
                pass
        return RebalanceDecision(
            user_id=user_id, target_weights=weights, regime=regime,
        )

    # ── Engine + state loaders ─────────────────────────────────────────

    def _get_qlib_engine(self):
        if self._qlib_engine is not None:
            return self._qlib_engine
        if self._qlib_load_attempted:
            return None
        self._qlib_load_attempted = True
        try:
            from ..ai.qlib.engine import get_qlib_engine  # noqa: PLC0415
            eng = get_qlib_engine()
            if not eng.loaded and not eng.load():
                logger.warning("AutoPilot: qlib engine load failed")
                return None
            self._qlib_engine = eng
            logger.info("AutoPilot: Qlib engine loaded (PROD voter, ranker)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("AutoPilot: qlib engine import failed: %s", exc)
            self._qlib_engine = None
        return self._qlib_engine

    async def _current_regime(self) -> str:
        try:
            r = (
                self.supabase.table("regime_history")
                .select("regime")
                .order("detected_at", desc=True)
                .limit(1)
                .execute()
            )
            name = str((r.data or [{}])[0].get("regime") or "sideways").lower()
            if name not in ("bull", "sideways", "bear"):
                name = "sideways"
            return name
        except Exception as exc:
            logger.debug("regime read failed, defaulting to sideways: %s", exc)
            return "sideways"

    async def _current_vix(self) -> float:
        try:
            r = (
                self.supabase.table("regime_history")
                .select("vix")
                .order("detected_at", desc=True)
                .limit(1)
                .execute()
            )
            v = (r.data or [{}])[0].get("vix")
            return float(v) if v is not None else 15.0
        except Exception:
            return 15.0

    def _enrolled_users(self) -> List[Dict[str, Any]]:
        """auto_trader_enabled = TRUE (the column the toggle writes), ALL tiers.

        Pricing v2 2026-06-12: Pro runs AutoPilot Lite live (tier-capped),
        Free runs paper-only — mode + limits resolved per user in
        ``_rebalance_one`` via core.tiers.
        """
        try:
            rows = (
                self.supabase.table("user_profiles")
                .select(
                    "id, tier, auto_trader_enabled, autopilot_dry_run, "
                    "capital, live_trading_paused, auto_trader_config"
                )
                .eq("auto_trader_enabled", True)
                .execute()
            )
            return rows.data or []
        except Exception as exc:
            logger.warning("autopilot user enumeration failed: %s", exc)
            return []

    def _record_decision(
        self,
        user_id: str,
        regime: str,
        weights: Dict[str, float],
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        try:
            r = self.supabase.table("auto_trader_runs").insert({
                "user_id": user_id,
                "regime": regime,
                "target_weights": weights,
                "diagnostics": diagnostics or {},
                "status": "decided",
            }).execute()
            if r.data:
                return str(r.data[0].get("id"))
        except Exception as exc:
            logger.debug("auto_trader_runs insert skipped: %s", exc)
        return None

    # ── Trade emission: diff weights → minimal order set ───────────────

    _TRADE_EPSILON = 0.005  # 0.5% min weight delta to trigger an order

    async def _emit_trades(
        self,
        user_id: str,
        weights: Dict[str, float],
        capital: float,
        regime: str,
        mode: str = "live",
        event_blackout: Optional[set] = None,
        entries_halted: bool = False,
    ) -> int:
        """Compare desired weights vs current positions (same mode) and
        place the minimum set of orders that move us from current → target.

        ``mode='paper'`` (Free tier / paper opt-in) opens virtual positions
        via TradeExecutionService's DB-level path — no broker involved.
        Live orders re-check eligibility on entry per PR 130.

        ``event_blackout`` symbols are ENTRY-suppressed: a BUY/add (delta > 0)
        into a name with imminent earnings is skipped, but a trim/exit
        (delta < 0) of an existing holding is always allowed.
        """
        if capital <= 0:
            return 0
        event_blackout = {s.upper() for s in (event_blackout or set())}
        current = self._current_live_weights(user_id, capital, mode=mode)
        prices = self._latest_prices(set(weights.keys()) | set(current.keys()))

        try:
            from .execution import TradeExecutionService  # noqa: PLC0415
            tes = TradeExecutionService(self.supabase)
        except Exception as exc:
            logger.warning("AutoPilot trade-execution service unavailable: %s", exc)
            return 0

        emitted = 0
        all_symbols = set(weights.keys()) | set(current.keys())
        for sym in sorted(all_symbols):
            target = float(weights.get(sym, 0.0))
            cur = float(current.get(sym, 0.0))
            delta = target - cur
            if abs(delta) < self._TRADE_EPSILON:
                continue
            # Event-risk: never OPEN or ADD into an earnings-window name.
            # Trims/exits (delta < 0) of existing holdings still proceed.
            if delta > 0 and sym.upper() in event_blackout:
                logger.info("AutoPilot skip entry %s — event-risk blackout", sym)
                continue
            # Daily-loss circuit breaker: halt new entries/adds (delta > 0) while
            # the user is over their loss limit; de-risking sells still flow.
            if entries_halted and delta > 0:
                continue
            price = prices.get(sym)
            if not price or price <= 0:
                continue

            qty = int(abs(delta) * capital / price)
            if qty <= 0:
                continue

            direction = "LONG" if delta > 0 else "SHORT"
            # HIGH #4 (2026-05-31) — generate a brand-safe one-paragraph
            # thesis per trade. Hides per-model names per memory
            # `project_greek_branding_2026_04_19`; surfaces OUTCOMES +
            # general reasoning the user can trust without exposing IP.
            thesis = _build_brand_safe_thesis(
                symbol=sym, direction=direction, regime=regime,
                weight=weights.get(sym, 0.0),
            )
            trade_payload = {
                "user_id": user_id,
                "symbol": sym,
                "exchange": "NSE",
                "segment": "EQUITY",
                "direction": direction,
                "quantity": qty,
                "entry_price": price,
                "average_price": price,
                "stop_loss": None,
                "target": None,
                "execution_mode": mode,
                "status": "pending",
                "source": "autopilot",
                "regime_at_open": regime,
                "explanation_text": thesis,    # HIGH #4
            }
            try:
                ins = self.supabase.table("trades").insert(trade_payload).execute()
                trade_row = (ins.data or [{}])[0]
                await tes.execute(trade_row)
                emitted += 1
            except Exception as exc:
                logger.warning(
                    "AutoPilot trade emission failed for %s/%s: %s",
                    user_id, sym, exc,
                )
                continue
        return emitted

    def _current_live_weights(
        self, user_id: str, capital: float, mode: str = "live",
    ) -> Dict[str, float]:
        """Per-symbol current weight = position_value / capital, scoped to
        the user's execution mode so paper runs diff against paper positions."""
        if capital <= 0:
            return {}
        try:
            rows = (
                self.supabase.table("positions")
                .select("symbol, current_value, execution_mode, is_active")
                .eq("user_id", user_id)
                .eq("is_active", True)
                .eq("execution_mode", mode)
                .execute()
            )
        except Exception:
            return {}
        out: Dict[str, float] = {}
        for r in rows.data or []:
            sym = str(r.get("symbol") or "")
            val = float(r.get("current_value") or 0)
            if sym:
                out[sym] = out.get(sym, 0.0) + (val / capital)
        return out

    def _latest_prices(self, symbols) -> Dict[str, float]:
        symbols = [s for s in symbols if s]
        if not symbols:
            return {}
        try:
            from ..data.market import get_market_data_provider  # noqa: PLC0415
            provider = get_market_data_provider()
            return {
                s: float(provider.get_quote(s))
                for s in symbols if provider.get_quote(s)
            }
        except Exception:
            return {}

    def _emit_blocked(self, user_id: str, code: str, regime: str) -> None:
        try:
            from ..observability import EventName, track  # noqa: PLC0415
            track(EventName.AUTO_TRADE_BLOCKED, user_id, {
                "code": code,
                "regime": regime,
            })
        except Exception:
            pass


# HIGH #4 (2026-05-31) — brand-safe per-trade thesis builder.
# Per memory `project_greek_branding_2026_04_19`: NEVER expose real
# model names (Qlib, HMM, TFT, FinBERT) to users. Use general reasoning
# the user can trust without giving away IP.
def _build_brand_safe_thesis(*, symbol: str, direction: str, regime: str, weight: float) -> str:
    """One-paragraph English explanation suitable for the user's Inbox.

    Generic enough to protect proprietary alpha; specific enough that
    the user understands WHY the bot is taking this trade. Per memory
    locks, this rule-based text generator runs deterministically; LLM
    enrichment is added LATER once we have an OpenRouter gateway budget.
    """
    side_word = "buy" if direction == "LONG" else "sell"
    regime_phrase = {
        "bull": "supportive bull regime",
        "sideways": "range-bound regime with mean-reverting tendency",
        "bear": "defensive bear regime with capital protection bias",
    }.get((regime or "").lower(), "current regime")

    weight_phrase = (
        "high-conviction allocation" if weight >= 0.05
        else "moderate-size allocation" if weight >= 0.02
        else "scaled-in allocation"
    )

    return (
        f"AutoPilot decided to {side_word} {symbol} as a {weight_phrase} "
        f"in a {regime_phrase}. The decision combines our trend, momentum, "
        f"sentiment, and signal-quality engines. Risk is capped per the "
        f"5%-per-stock + Kelly-decayed sizing rules. Stop and target are "
        f"placed at the broker once the order fills."
    )


__all__ = ["AutoPilotService", "RebalanceDecision"]
