"""StrategyRunner — the fan-out engine. PR-FAN.

One tick:
  1. Load every user with an active strategy
  2. For each (user, strategy) → expand universe
  3. For each (user, strategy, symbol):
        - load OHLCV bars
        - apply AI overlay (regime + VIX gate, sizing scale)
        - if user already has an open position → evaluate EXIT
          else                              → evaluate ENTRY
  4. Emit signal events to the signals table; write open/close events
     to strategy_positions; record one summary row in strategy_runner_runs.

Output: every match becomes a row in ``signals`` with
``source="user_strategy"`` + ``strategy_id`` + ``user_id``. The user's
``autopilot_streams.user_strategy.enabled`` flag determines whether the
signal also fires a real broker order — the runner just writes signals;
the AutoPilot supervisor's intraday/post-market jobs route to brokers.

Timeframe-aware: daily strategies run at the daily tick (15:30 IST);
intraday strategies run at intraday ticks (every 5/15 min depending on
their tf). For v1 we ship the daily path only — intraday LSTM isn't
PROD yet so intraday DSL strategies have no realistic data to fire on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

_IST = timezone(timedelta(hours=5, minutes=30))


def _ist_hhmm() -> str:
    """Current wall-clock time in IST as 'HH:MM' — for auto square-off checks."""
    return datetime.now(_IST).strftime("%H:%M")

from ...ai.strategy.backtest import (
    DEFAULT_INITIAL_CAPITAL,
)
from ...ai.strategy.dsl import Strategy as DSLStrategy
from ...ai.strategy.indicators import MIN_LOOKBACK
from ...ai.strategy.interpreter import (
    EngineSignals,
    InterpreterContext,
    evaluate_condition,
)
from ..execution.paper_executor import execute_paper_order
from ..regime import resolve_regime_at
from .ai_overlay import apply_ai_overlay, load_overlay_settings
from .day_loss_breaker import evaluate_strategy_breaker, trip_strategy
from .universe_expander import expand_universe

logger = logging.getLogger(__name__)


# Hard cap to keep one bad strategy from blowing the runner. If a single
# (user, strategy) hits more than this many symbol-level matches in one
# tick, we log a warning and cap further matches for that strategy.
_MAX_MATCHES_PER_STRATEGY_PER_TICK = 20


@dataclass
class StrategySignalEvent:
    """One signal emitted by the runner. Mirrors the existing signals
    table shape so we can write directly without a translator."""
    user_id: str
    strategy_id: str
    symbol: str
    action: str            # 'buy' | 'sell' | 'close_long'
    entry_price: float
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    confidence: float = 0.5
    size_multiplier: float = 1.0
    overlay_notes: List[str] = field(default_factory=list)


@dataclass
class StrategyRunnerReport:
    tick_kind: str                              # 'daily' | 'intraday_5m' | 'manual'
    started_at: str
    finished_at: Optional[str] = None
    users_processed: int = 0
    strategies_evaluated: int = 0
    symbol_evaluations: int = 0
    signals_emitted: int = 0
    entries_emitted: int = 0
    exits_emitted: int = 0
    skipped_by_overlay: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick_kind": self.tick_kind,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "users_processed": self.users_processed,
            "strategies_evaluated": self.strategies_evaluated,
            "symbol_evaluations": self.symbol_evaluations,
            "signals_emitted": self.signals_emitted,
            "entries_emitted": self.entries_emitted,
            "exits_emitted": self.exits_emitted,
            "skipped_by_overlay": self.skipped_by_overlay,
            "error": self.error,
        }


class StrategyRunner:
    """One instance per process. Tick methods are async + safe to call
    multiple times — idempotent open/close position tracking via
    strategy_positions table."""

    def __init__(self, supabase_admin: Any):
        self.supabase = supabase_admin

    # ── Public tick entry points ─────────────────────────────────

    async def run_daily_tick(self) -> StrategyRunnerReport:
        """The 15:30 IST tick — evaluate every daily-timeframe strategy."""
        return await self._run_tick(tick_kind="daily", timeframes=("1d",))

    async def run_intraday_tick(self) -> StrategyRunnerReport:
        """Every 5/15-min IST during market hours — evaluate intraday
        strategies. NOTE: intraday data path requires LSTM PROD; v1
        ships this method but the actual intraday DSL strategies are
        rare today."""
        return await self._run_tick(
            tick_kind="intraday_5m",
            timeframes=("5m", "15m", "30m", "1h"),
        )

    async def run_for_user(
        self,
        user_id: str,
        *,
        timeframes: tuple = ("1d",),
    ) -> StrategyRunnerReport:
        """Manual single-user trigger — useful for admin testing
        and the "Run my strategies now" button in settings."""
        return await self._run_tick(
            tick_kind="manual",
            timeframes=timeframes,
            user_filter=user_id,
        )

    async def run_position_sweep(self) -> StrategyRunnerReport:
        """Intraday sweep — scan every OPEN strategy_position against
        live quotes. Fires exits on stop-loss / target hits using the
        same _emit_exit path as the daily tick, so paper portfolios
        auto-close at the moment price crosses the level instead of
        waiting until 15:30 IST.

        Production-quality requirement: a trader's stop-loss must
        trigger when price hits it, not at end-of-day.
        """
        from ..execution.paper_executor import execute_paper_order  # noqa: F401  # eager import

        report = StrategyRunnerReport(
            tick_kind="position_sweep",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            # All open positions across all users + their parent strategy
            # status (paper vs live) in one read.
            open_rows = (
                self.supabase.table("strategy_positions")
                .select(
                    "id, user_id, strategy_id, symbol, side, quantity, "
                    "entry_price, stop_loss, target_1"
                )
                .eq("status", "open")
                .limit(1000)
                .execute()
                .data
                or []
            )
            if not open_rows:
                report.finished_at = datetime.now(timezone.utc).isoformat()
                return report

            # Pull strategy_status for every position's strategy_id in
            # one batched query — otherwise we'd issue N queries inside
            # a hot loop.
            strategy_ids = list({p["strategy_id"] for p in open_rows if p.get("strategy_id")})
            statuses: Dict[str, str] = {}
            # strategy_id → square_off_time ("HH:MM") from the DSL, for the
            # auto square-off (uTrade-style EOD flatten of intraday positions).
            square_off: Dict[str, Optional[str]] = {}
            if strategy_ids:
                try:
                    srows = (
                        self.supabase.table("user_strategies")
                        .select("id, status, dsl")
                        .in_("id", strategy_ids)
                        .execute()
                    )
                    for r in (srows.data or []):
                        statuses[r["id"]] = r.get("status", "paper")
                        square_off[r["id"]] = (r.get("dsl") or {}).get("square_off_time")
                except Exception:
                    statuses = {}

            now_ist_hhmm = _ist_hhmm()

            # Fetch live quotes for every unique symbol in one batch.
            symbols = list({p["symbol"] for p in open_rows if p.get("symbol")})
            from ..data.market import get_market_data_provider
            provider = get_market_data_provider()
            try:
                quote_map = provider.get_quotes_batch(symbols[:200])
            except Exception as exc:
                logger.warning("position_sweep: batch quote fetch failed: %s", exc)
                quote_map = {}

            for pos in open_rows:
                symbol = pos["symbol"]
                report.symbol_evaluations += 1
                quote = quote_map.get(symbol)
                if not quote:
                    continue
                # Quote objects vs dicts — handle both
                try:
                    last_price = float(getattr(quote, "ltp", None) or quote.get("ltp", 0))
                except Exception:
                    continue
                if last_price <= 0:
                    continue

                # Auto square-off: once the strategy's IST square_off_time has
                # passed, flatten the position regardless of price. Checked
                # before the price stops so an EOD flatten always wins.
                reason: Optional[str] = None
                so_time = square_off.get(pos.get("strategy_id"))
                if so_time and now_ist_hhmm and now_ist_hhmm >= so_time:
                    reason = "square_off"
                else:
                    reason = self._hard_exit_for(pos, last_price)
                if reason is None:
                    continue

                # Use the strategy_id → status map. Default 'paper' if a
                # strategy row was archived under us — strategy_positions
                # row still gets closed, but no paper fill posted.
                strategy_status = statuses.get(pos["strategy_id"], "paper")
                try:
                    self._emit_exit(
                        pos["user_id"], pos["strategy_id"], pos,
                        last_price,
                        reason=reason,
                        strategy_status=strategy_status,
                    )
                    report.exits_emitted += 1
                    report.signals_emitted += 1
                except Exception as exc:
                    logger.warning(
                        "position_sweep: emit_exit raised for %s: %s",
                        pos.get("id"), exc,
                    )
        except Exception as exc:  # noqa: BLE001
            report.error = str(exc)[:240]
            logger.exception("position_sweep tick failed")

        report.finished_at = datetime.now(timezone.utc).isoformat()
        return report

    # ── Core driver ─────────────────────────────────────────────

    async def _run_tick(
        self,
        *,
        tick_kind: str,
        timeframes: tuple,
        user_filter: Optional[str] = None,
    ) -> StrategyRunnerReport:
        report = StrategyRunnerReport(
            tick_kind=tick_kind,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            current_regime = resolve_regime_at(self.supabase)
            current_vix = await self._current_vix()

            users = self._load_active_users(user_filter=user_filter)
            for user in users:
                user_id = user["id"]
                user_capital = float(user.get("capital") or DEFAULT_INITIAL_CAPITAL)
                strategies = self._load_live_strategies_for(user_id, timeframes)
                if not strategies:
                    continue
                report.users_processed += 1
                overlay_settings = load_overlay_settings(self.supabase, user_id)

                for strategy_row in strategies:
                    try:
                        await self._evaluate_strategy(
                            user_id=user_id,
                            user_capital=user_capital,
                            strategy_row=strategy_row,
                            overlay_settings=overlay_settings,
                            current_regime=current_regime,
                            current_vix=current_vix,
                            report=report,
                        )
                        report.strategies_evaluated += 1
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "strategy_runner: strategy %s for user %s failed",
                            strategy_row.get("id"), user_id,
                        )
        except Exception as exc:  # noqa: BLE001
            report.error = str(exc)[:240]
            logger.exception("strategy_runner tick failed")

        report.finished_at = datetime.now(timezone.utc).isoformat()
        await self._persist_run(report)
        return report

    # ── Per-strategy evaluation ─────────────────────────────────

    def _outcome_predictor_for(self, strategy_row: Dict[str, Any]):
        """Get the OutcomePredictor for this strategy's template_slug, if
        a trained model exists on disk. Returns None when no model is
        loaded (AI overlay Gate 4 is then a no-op — fail-open)."""
        try:
            from ...ai.outcome_models.registry import OutcomeModelRegistry
        except Exception:
            return None
        slug = strategy_row.get("template_slug")
        if not slug:
            # Try to read it from the DSL document if present
            dsl = strategy_row.get("dsl") or {}
            slug = dsl.get("template_slug") or dsl.get("name", "").lower().replace(" ", "-")
        if not slug:
            return None
        return OutcomeModelRegistry.get(slug)

    async def _evaluate_strategy(
        self,
        *,
        user_id: str,
        user_capital: float,
        strategy_row: Dict[str, Any],
        overlay_settings,
        current_regime: str,
        current_vix: Optional[float],
        report: StrategyRunnerReport,
    ) -> int:
        """Iterate over (universe symbols) and evaluate entry/exit for
        each. Returns the number of signals emitted for this strategy."""
        strategy_id = strategy_row["id"]
        dsl_json = strategy_row.get("dsl") or {}
        try:
            strategy = DSLStrategy.model_validate(dsl_json)
        except Exception as exc:
            logger.warning(
                "strategy_runner: skipping strategy %s — invalid DSL: %s",
                strategy_id, exc,
            )
            return 0

        # ── OPTIONS dispatch (PR-AU) ──────────────────────────────────
        # Multi-leg option strategies don't fit the per-symbol equity
        # loop: one strategy → one combined position on a single index
        # underlying, opened/closed atomically across legs. Route those
        # to the options-aware evaluator before doing any equity work.
        if getattr(strategy, "segment", None) and \
           str(strategy.segment.value).upper() == "OPTIONS":
            try:
                return await self._evaluate_options_strategy(
                    user_id=user_id,
                    strategy=strategy,
                    strategy_id=strategy_id,
                    strategy_status=strategy_row.get("status", "paper"),
                    current_vix=current_vix,
                    report=report,
                )
            except Exception as exc:
                logger.exception(
                    "strategy_runner: options evaluator raised for %s: %s",
                    strategy_id, exc,
                )
                return 0

        # Expand universe → concrete symbols
        try:
            symbols = expand_universe(
                strategy.universe.value,
                single_symbol=strategy.symbol,
            )
        except Exception:
            symbols = []
        if not symbols:
            return 0

        # Open positions for this strategy (so we know whether to evaluate
        # entry or exit per symbol)
        open_positions = self._load_open_positions(user_id, strategy_id)
        open_by_symbol = {p["symbol"]: p for p in open_positions}

        emitted = 0
        engine_signals = EngineSignals(regime=current_regime)

        for symbol in symbols:
            report.symbol_evaluations += 1
            bars = await self._load_bars(symbol)
            if bars is None or len(bars) < MIN_LOOKBACK + 5:
                continue
            ctx = InterpreterContext(bars=bars, engines=engine_signals)
            last_close = float(bars["close"].iloc[-1])

            if symbol in open_by_symbol:
                pos = open_by_symbol[symbol]

                # ── Hard SL/target gate (PR-AL fix) ────────────────────
                # Evaluate price-based stop loss / take profit BEFORE the
                # DSL exit condition. A stop loss must fire when price
                # crosses it regardless of whether the DSL exit condition
                # has been triggered — otherwise the user sees their loss
                # widen past their stated risk tolerance because the
                # strategy's exit rule hasn't matched yet.
                strategy_status = strategy_row.get("status", "paper")
                hard_exit_reason = self._hard_exit_for(pos, last_close)
                if hard_exit_reason is not None:
                    await self._emit_exit(
                        user_id, strategy_id, pos, last_close,
                        reason=hard_exit_reason,
                        strategy_status=strategy_status,
                    )
                    report.exits_emitted += 1
                    report.signals_emitted += 1
                    emitted += 1
                    continue

                # Otherwise → evaluate DSL exit
                if evaluate_condition(strategy.exit, ctx):
                    await self._emit_exit(
                        user_id, strategy_id, pos, last_close,
                        reason="dsl_exit",
                        strategy_status=strategy_status,
                    )
                    report.exits_emitted += 1
                    report.signals_emitted += 1
                    emitted += 1
            else:
                # No position → evaluate ENTRY
                if evaluate_condition(strategy.entry, ctx):
                    # ── Gate 1: continuation/reversal intent gate ──────
                    # Continuation strategies REQUIRE the previous bar to
                    # confirm direction. Reversal strategies skip this
                    # gate by design (they fire AGAINST momentum).
                    intent = self._strategy_intent_for(strategy_row)
                    if intent == "continuation":
                        if not self._prev_bar_confirms_long(bars):
                            report.skipped_by_overlay += 1
                            continue

                    # ── Gate 2: premium-confirmation ("falling knife") ──
                    # Reject if symbol's recent tick slope is sharply
                    # negative. Fail-open when no tick data exists yet.
                    from .premium_gate import check_premium_slope
                    pg = check_premium_slope(self.supabase, symbol=symbol)
                    if not pg.allowed:
                        report.skipped_by_overlay += 1
                        logger.info(
                            "strategy_runner: premium_gate blocked entry %s/%s — %s",
                            strategy_id, symbol, pg.note,
                        )
                        continue

                    # ── Gate 3: AI overlay (regime + VIX) ──────────────
                    decision = apply_ai_overlay(
                        supabase=self.supabase,
                        settings=overlay_settings,
                        user_id=user_id,
                        symbol=symbol,
                        current_vix=current_vix,
                        current_regime=current_regime,
                    )
                    if not decision.allowed:
                        report.skipped_by_overlay += 1
                        continue

                    # ── Gate 4: outcome-model P(win) gate ──────────────
                    # If we have a trained XGBoost outcome model for this
                    # template_slug, ask it: "given the current market
                    # state, would this strategy historically have won?"
                    # Reject if P(win) < 0.45 (slightly below random — leave
                    # room for the model's own noise).
                    # Fail-open when no model loaded for this slug.
                    predictor = self._outcome_predictor_for(strategy_row)
                    outcome_note = None
                    if predictor is not None:
                        try:
                            from ...ai.outcome_models.features import build_outcome_features
                            features = build_outcome_features(
                                bars, regime=current_regime, vix=current_vix,
                            )
                            p_win = predictor.predict_win_proba(features)
                            outcome_note = f"outcome_model_p_win={p_win:.3f}"
                            if p_win < 0.45:
                                report.skipped_by_overlay += 1
                                logger.info(
                                    "strategy_runner: outcome_model blocked %s/%s — P(win)=%.3f",
                                    strategy_id, symbol, p_win,
                                )
                                continue
                        except Exception as exc:
                            logger.debug("outcome model gate skipped: %s", exc)

                    # ── Gate 5: per-strategy day-loss breaker ──────────
                    # Stops a strategy from layering on more positions
                    # once the day's realized + unrealized P&L on this
                    # strategy has fallen below its max_daily_loss_pct.
                    # On trip: strategy is paused, no entry fires, audit
                    # signal is written so the user sees why.
                    breaker_check = evaluate_strategy_breaker(
                        self.supabase, user_id, strategy_id, strategy_row,
                    )
                    if breaker_check.breached:
                        trip_strategy(self.supabase, user_id, strategy_id, breaker_check)
                        report.skipped_by_overlay += 1
                        # No more entries for this strategy this tick —
                        # break out of the symbol loop entirely.
                        break

                    extra_notes = decision.notes + ([pg.note] if pg.note else [])
                    if outcome_note:
                        extra_notes.append(outcome_note)
                    await self._emit_entry(
                        user_id=user_id,
                        user_capital=user_capital,
                        strategy=strategy,
                        strategy_id=strategy_id,
                        strategy_status=strategy_row.get("status", "paper"),
                        symbol=symbol,
                        entry_price=last_close,
                        size_multiplier=decision.size_multiplier,
                        overlay_notes=extra_notes,
                    )
                    report.entries_emitted += 1
                    report.signals_emitted += 1
                    emitted += 1
                    if emitted >= _MAX_MATCHES_PER_STRATEGY_PER_TICK:
                        logger.warning(
                            "strategy %s hit per-tick match cap (%d) for user %s",
                            strategy_id, _MAX_MATCHES_PER_STRATEGY_PER_TICK, user_id,
                        )
                        break
        return emitted

    # ── OPTIONS evaluator (PR-AU) ───────────────────────────────────

    async def _evaluate_options_strategy(
        self,
        *,
        user_id: str,
        strategy,
        strategy_id: str,
        strategy_status: str,
        current_vix: Optional[float],
        report: StrategyRunnerReport,
    ) -> int:
        """Multi-leg option strategy evaluator.

        Contract is intentionally narrower than the equity loop:
          - Single underlying (strategy.symbol)
          - At most ONE open paper_option_position per (user, strategy)
          - Entry: DSL entry condition evaluated against the underlying
            bars, just like an equity strategy. If matched + no current
            open position → resolve legs + open a multi-leg paper position.
          - Exit: hard SL/target on combined unrealized P&L is checked
            FIRST; if neither triggers we evaluate the DSL exit condition.

        Live broker leg-by-leg routing comes in Slice 3 (PR-AV). For
        now status=='live' falls back to paper so the user sees the
        position move even before live wiring lands.
        """
        from ..execution.paper_options_executor import (
            open_paper_option_position,
            close_paper_option_position,
            mark_to_market,
        )

        underlying = strategy.symbol
        if not underlying:
            return 0

        legs = getattr(strategy, "legs", None) or []
        if not legs:
            logger.info(
                "options strategy %s has no legs — nothing to evaluate",
                strategy_id,
            )
            return 0

        # Load underlying bars for entry/exit evaluation. Index spot
        # comes from yfinance/Kite via the same data provider the equity
        # path uses.
        bars = await self._load_bars(underlying)
        if bars is None or len(bars) < MIN_LOOKBACK + 5:
            return 0

        # Spot at evaluation time
        last_close = float(bars["close"].iloc[-1])
        # VIX → sigma for BS pricing. If unavailable use a 20% default
        # (same as options_backtest fallback).
        sigma = float(current_vix or 20.0) / 100.0

        # Is there already an open multi-leg paper position for this
        # (user, strategy)?
        try:
            existing = (
                self.supabase.table("paper_option_positions")
                .select("*")
                .eq("user_id", user_id)
                .eq("strategy_id", strategy_id)
                .eq("status", "open")
                .limit(1)
                .execute()
                .data
                or []
            )
        except Exception:
            existing = []
        open_position = existing[0] if existing else None

        ctx = InterpreterContext(
            bars=bars,
            engines=EngineSignals(regime="sideways"),
        )

        # ── Exit path: position exists → check SL / target / DSL exit
        if open_position:
            mtm = mark_to_market(
                self.supabase, open_position,
                spot=last_close, sigma=sigma,
            )
            float(open_position.get("net_premium") or 0)
            max_loss = open_position.get("max_loss")
            max_profit = open_position.get("max_profit")

            # Defensive numeric comparisons
            try:
                max_loss_v = float(max_loss) if max_loss is not None else None
            except (TypeError, ValueError):
                max_loss_v = None
            try:
                max_profit_v = float(max_profit) if max_profit is not None else None
            except (TypeError, ValueError):
                max_profit_v = None

            reason: Optional[str] = None
            # Hard stop: unrealized loss has reached the max_loss limit.
            # max_loss is stored as positive rupees; unrealized < 0 on loss.
            if max_loss_v is not None and mtm.unrealized_pnl <= -abs(max_loss_v):
                reason = "stop_loss"
            elif max_profit_v is not None and mtm.unrealized_pnl >= abs(max_profit_v) * 0.9:
                # Take 90% of max profit — locking in is better than
                # waiting the last 10% which evaporates with theta.
                reason = "target"
            elif evaluate_condition(strategy.exit, ctx):
                reason = "dsl_exit"

            if reason is None:
                return 0  # hold

            # Live position → close via broker leg-by-leg; paper → paper.
            is_live = bool((open_position.get("metadata") or {}).get("live"))
            if is_live or strategy_status == "live":
                try:
                    from ..execution.live_options_executor import close_live_option_position
                    live_close = await close_live_option_position(
                        supabase=self.supabase,
                        user_id=user_id,
                        position_id=open_position["id"],
                        spot=last_close,
                        sigma=sigma,
                        reason=reason,
                    )
                except Exception as exc:
                    logger.warning(
                        "options LIVE auto-close raised %s/%s: %s",
                        strategy_id, underlying, exc,
                    )
                    return 0
                if live_close.ok:
                    report.exits_emitted += 1
                    report.signals_emitted += 1
                    logger.info(
                        "options LIVE auto-close %s/%s reason=%s",
                        strategy_id, underlying, reason,
                    )
                    return 1
                logger.warning(
                    "options LIVE auto-close failed %s/%s: %s",
                    strategy_id, underlying, live_close.reason,
                )
                return 0

            close_res = close_paper_option_position(
                supabase=self.supabase,
                position_id=open_position["id"],
                user_id=user_id,
                spot=last_close,
                sigma=sigma,
                reason=reason,
                source="user_strategy",
            )
            if close_res.ok:
                report.exits_emitted += 1
                report.signals_emitted += 1
                logger.info(
                    "options auto-close %s/%s: reason=%s realized=₹%.0f",
                    strategy_id, underlying, reason,
                    close_res.realized_pnl or 0,
                )
                return 1
            logger.warning(
                "options auto-close failed %s/%s: %s",
                strategy_id, underlying, close_res.reason,
            )
            return 0

        # ── Entry path: no position → check DSL entry
        if not evaluate_condition(strategy.entry, ctx):
            return 0

        if strategy_status == "live":
            # Live broker leg-by-leg path (PR-AV).
            try:
                from ..execution.live_options_executor import open_live_option_position
                live_res = await open_live_option_position(
                    supabase=self.supabase,
                    user_id=user_id,
                    underlying=underlying,
                    spot=last_close,
                    sigma=sigma,
                    legs=legs,
                    lots=1,
                    strategy_id=strategy_id,
                    template_slug=getattr(strategy, "template_slug", None),
                )
            except Exception as exc:
                logger.warning(
                    "options live auto-open raised %s/%s: %s",
                    strategy_id, underlying, exc,
                )
                return 0

            if not live_res.ok:
                logger.warning(
                    "options live auto-open rejected %s/%s: %s "
                    "(placed=%d failed=%d)",
                    strategy_id, underlying, live_res.reason,
                    len(live_res.placed_legs), len(live_res.failed_legs),
                )
                # If broker rejection but some legs already placed → the
                # paper_option_positions row was written 'partial';
                # surface that as one signal so the user knows.
                if live_res.placed_legs:
                    report.signals_emitted += 1
                return 0

            report.entries_emitted += 1
            report.signals_emitted += 1
            logger.info(
                "options LIVE auto-open %s/%s pos=%s legs=%d broker_orders=%d",
                strategy_id, underlying,
                (live_res.position_id or "")[:8],
                len(live_res.placed_legs),
                len(live_res.placed_legs),
            )
            return 1

        # Default: paper path
        open_res = open_paper_option_position(
            supabase=self.supabase,
            user_id=user_id,
            underlying=underlying,
            spot=last_close,
            sigma=sigma,
            legs=legs,
            lots=1,  # v1: always 1 lot per fire; UI controls bulk deploys
            strategy_id=strategy_id,
            template_slug=getattr(strategy, "template_slug", None),
            source="user_strategy",
        )
        if not open_res.ok:
            logger.info(
                "options auto-open skipped %s/%s: %s",
                strategy_id, underlying, open_res.reason,
            )
            return 0

        report.entries_emitted += 1
        report.signals_emitted += 1
        logger.info(
            "options auto-open %s/%s: pos=%s net=₹%.0f max_p=₹%s max_l=₹%s",
            strategy_id, underlying,
            (open_res.position_id or "")[:8],
            open_res.net_premium or 0,
            open_res.max_profit, open_res.max_loss,
        )
        return 1

    # ── DB writes ───────────────────────────────────────────────

    async def _emit_entry(
        self,
        *,
        user_id: str,
        user_capital: float,
        strategy,
        strategy_id: str,
        strategy_status: str,
        symbol: str,
        entry_price: float,
        size_multiplier: float,
        overlay_notes: List[str],
    ) -> None:
        """Insert into signals + strategy_positions atomically (best effort).

        Position sizing: strategy's position_size_pct × AI overlay size
        multiplier × user capital, floored to integer shares. We size
        explicitly here so the signal + strategy_positions rows reflect
        the real quantity instead of writing quantity=1 and hoping a
        downstream broker-routing job re-sizes it.
        """
        stop_loss = None
        target_1 = None
        if strategy.stop_loss_pct:
            stop_loss = entry_price * (1 - strategy.stop_loss_pct / 100)
        if strategy.take_profit_pct:
            target_1 = entry_price * (1 + strategy.take_profit_pct / 100)

        # Strategy says % of capital → rupees → integer share count.
        # Defensive floors: never deploy negative capital, never write
        # negative qty. 0-share entries (price > deployable capital) are
        # logged and skipped — emitting an entry the user can't afford
        # is worse than missing one.
        pct = max(0.0, float(strategy.position_size.value) * size_multiplier)
        capital_deployed = max(0.0, user_capital * (pct / 100.0))
        quantity = int(capital_deployed // max(entry_price, 1e-6))
        if quantity <= 0:
            logger.info(
                "strategy_runner: skipping %s/%s — capital %.0f at pct %.2f%% "
                "buys 0 shares at ₹%.2f",
                strategy_id, symbol, user_capital, pct, entry_price,
            )
            return

        try:
            self.supabase.table("signals").insert({
                "user_id": user_id,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "source": "user_strategy",
                "signal_type": "swing" if strategy.timeframe.value == "1d" else "intraday",
                "action": "buy",
                "entry_price": round(entry_price, 2),
                "stop_loss": round(stop_loss, 2) if stop_loss else None,
                "target_1": round(target_1, 2) if target_1 else None,
                "confidence": 0.65,           # baseline; AI overlay raised this implicitly
                "status": "active",
                "strategy_names": [strategy.name],
                "market_context": {
                    "overlay_size_multiplier": size_multiplier,
                    "overlay_notes": overlay_notes,
                    "requested_capital_pct": round(pct, 2),
                    "user_capital": round(user_capital, 2),
                    "capital_deployed": round(capital_deployed, 2),
                    "quantity": quantity,
                },
            }).execute()
        except Exception as exc:
            logger.warning("signals insert failed for %s/%s: %s", user_id, symbol, exc)
            return

        # Track open position so next tick evaluates exit, not entry.
        # Stop/target persisted here so the hard-SL exit gate can read
        # them on subsequent ticks (see _evaluate_strategy).
        try:
            self.supabase.table("strategy_positions").insert({
                "user_id": user_id,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "side": "long",
                "entry_price": round(entry_price, 2),
                "quantity": quantity,
                "capital_deployed": round(capital_deployed, 2),
                "stop_loss": round(stop_loss, 2) if stop_loss else None,
                "target_1": round(target_1, 2) if target_1 else None,
                "status": "open",
                "last_evaluated_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
        except Exception as exc:
            logger.warning("strategy_positions insert failed: %s", exc)

        # ── Auto-execute against the right execution surface ──────────
        # paper → paper portfolio (cash + paper_positions + paper_trades)
        # live  → real broker via live_executor.execute_live_order which
        #         handles broker auth + place_order + Zerodha GTT for
        #         SL/target + an idempotency check that refuses to layer
        #         a second live position on the same symbol.
        if strategy_status == "paper":
            try:
                result = execute_paper_order(
                    supabase=self.supabase,
                    user_id=user_id,
                    symbol=symbol,
                    action="buy",
                    quantity=quantity,
                    price=entry_price,
                    source="user_strategy",
                )
                if not result.ok:
                    logger.info(
                        "paper auto-buy skipped %s/%s: %s",
                        strategy_id, symbol, result.reason,
                    )
            except Exception as exc:
                logger.warning(
                    "paper auto-buy raised for %s/%s: %s",
                    strategy_id, symbol, exc,
                )
        elif strategy_status == "live":
            try:
                from ..execution.live_executor import execute_live_order
                result = await execute_live_order(
                    supabase=self.supabase,
                    user_id=user_id,
                    strategy_id=strategy_id,
                    symbol=symbol,
                    action="buy",
                    quantity=quantity,
                    price=entry_price,
                    stop_loss=stop_loss,
                    target=target_1,
                    reason="user_strategy_entry",
                )
                if not result.ok:
                    logger.warning(
                        "live auto-buy rejected %s/%s: %s",
                        strategy_id, symbol, result.reason,
                    )
                else:
                    logger.info(
                        "live auto-buy fired %s/%s broker_order=%s qty=%d",
                        strategy_id, symbol, result.broker_order_id, quantity,
                    )
            except Exception as exc:
                logger.warning(
                    "live auto-buy raised for %s/%s: %s",
                    strategy_id, symbol, exc,
                )

    # ── Entry gates (PR-DEPTH) ──────────────────────────────────

    def _strategy_intent_for(self, strategy_row: Dict[str, Any]) -> str:
        """Read the user_strategies.strategy_intent column. Defaults to
        'continuation' so existing strategies (which were authored before
        the intent column existed) get the continuation-gate by default."""
        return str(
            strategy_row.get("strategy_intent")
            or strategy_row.get("dsl", {}).get("strategy_intent")
            or "continuation"
        ).lower()

    def _prev_bar_confirms_long(self, bars) -> bool:
        """Continuation gate: previous bar must show non-bearish direction.

        Adapted from aaryansinha16/AI-trader's backend/app.py "prev-bar
        gate". If yesterday's close < yesterday's open by >0.1%, we
        treat that as a bearish prev-bar and skip continuation entries.
        Fail-open if not enough bars.
        """
        if bars is None or len(bars) < 2:
            return True
        try:
            prev = bars.iloc[-2]
            prev_close = float(prev["close"])
            prev_open = float(prev["open"])
            if prev_open <= 0:
                return True
            move_pct = (prev_close - prev_open) / prev_open
            # Allow if non-bearish (≥-0.1%); block only on clear prev-bar weakness
            return move_pct >= -0.001
        except Exception:
            return True

    # ── Exit emission ───────────────────────────────────────────

    def _hard_exit_for(
        self, position: Dict[str, Any], last_close: float,
    ) -> Optional[str]:
        """Return the exit reason if price has crossed the position's
        stop_loss or target_1, else None.

        Long positions only (v1). Side is checked defensively so this
        does the right thing if/when short positions are added.
        """
        if (position.get("side") or "long") != "long":
            return None
        sl = position.get("stop_loss")
        target = position.get("target_1")
        try:
            sl_val = float(sl) if sl is not None else None
        except (TypeError, ValueError):
            sl_val = None
        try:
            target_val = float(target) if target is not None else None
        except (TypeError, ValueError):
            target_val = None

        if sl_val is not None and last_close <= sl_val:
            return "stop_loss"
        if target_val is not None and last_close >= target_val:
            return "target"
        return None

    async def _emit_exit(
        self,
        user_id: str,
        strategy_id: str,
        position: Dict[str, Any],
        exit_price: float,
        *,
        reason: str = "dsl_exit",
        strategy_status: str = "paper",
    ) -> None:
        try:
            self.supabase.table("signals").insert({
                "user_id": user_id,
                "strategy_id": strategy_id,
                "symbol": position["symbol"],
                "source": "user_strategy",
                "signal_type": "swing",
                "action": "close_long",
                "entry_price": round(exit_price, 2),
                "confidence": 0.7,
                "status": "active",
                "market_context": {
                    "exit_reason": reason,
                    "entry_price": position.get("entry_price"),
                    "quantity": position.get("quantity"),
                },
            }).execute()
        except Exception as exc:
            logger.warning("exit signal insert failed: %s", exc)
        try:
            self.supabase.table("strategy_positions").update({
                "status": "closed",
                "exit_reason": reason,
                "exit_price": round(exit_price, 2),
                "last_evaluated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", position["id"]).execute()
        except Exception as exc:
            logger.warning("position close update failed: %s", exc)

        # ── Auto-close on the matching execution surface ──────────────
        # paper → paper portfolio sell
        # live  → broker sell via live_executor.execute_live_order which
        #         delegates to TradeExecutionService.close_position (the
        #         existing path that updates trades + positions rows and
        #         emits the broker exit order at market).
        qty = int(position.get("quantity") or 0)
        if qty <= 0:
            return

        if strategy_status == "paper":
            try:
                result = execute_paper_order(
                    supabase=self.supabase,
                    user_id=user_id,
                    symbol=position["symbol"],
                    action="sell",
                    quantity=qty,
                    price=exit_price,
                    source="user_strategy",
                )
                if not result.ok:
                    logger.info(
                        "paper auto-sell skipped %s/%s: %s",
                        strategy_id, position["symbol"], result.reason,
                    )
            except Exception as exc:
                logger.warning(
                    "paper auto-sell raised for %s/%s: %s",
                    strategy_id, position["symbol"], exc,
                )
        elif strategy_status == "live":
            try:
                from ..execution.live_executor import execute_live_order
                result = await execute_live_order(
                    supabase=self.supabase,
                    user_id=user_id,
                    strategy_id=strategy_id,
                    symbol=position["symbol"],
                    action="sell",
                    quantity=qty,
                    price=exit_price,
                    reason=reason,
                )
                if not result.ok:
                    logger.warning(
                        "live auto-sell rejected %s/%s: %s",
                        strategy_id, position["symbol"], result.reason,
                    )
                else:
                    logger.info(
                        "live auto-sell fired %s/%s reason=%s pnl=%s",
                        strategy_id, position["symbol"], reason,
                        result.realized_pnl,
                    )
            except Exception as exc:
                logger.warning(
                    "live auto-sell raised for %s/%s: %s",
                    strategy_id, position["symbol"], exc,
                )

    # ── DB reads ────────────────────────────────────────────────

    def _load_active_users(self, *, user_filter: Optional[str]) -> List[Dict[str, Any]]:
        """Users who have at least one live/paper strategy. Filter by
        ``user_filter`` if provided (manual single-user run). Each user
        dict also carries the user's ``capital`` so the runner can size
        positions correctly instead of writing ``quantity=1``.
        """
        try:
            q = (
                self.supabase.table("user_strategies")
                .select("user_id")
                .in_("status", ["paper", "live"])
                .limit(2000)
            )
            if user_filter:
                q = q.eq("user_id", user_filter)
            rows = q.execute()
        except Exception:
            return []
        ids = {r["user_id"] for r in (rows.data or []) if r.get("user_id")}
        if not ids:
            return []
        # One bulk capital lookup — much cheaper than per-user queries.
        try:
            profile_rows = (
                self.supabase.table("user_profiles")
                .select("id, capital")
                .in_("id", list(ids))
                .execute()
            )
            capitals = {
                p["id"]: float(p.get("capital") or DEFAULT_INITIAL_CAPITAL)
                for p in (profile_rows.data or [])
            }
        except Exception:
            capitals = {}
        return [
            {"id": uid, "capital": capitals.get(uid, float(DEFAULT_INITIAL_CAPITAL))}
            for uid in ids
        ]

    def _load_live_strategies_for(
        self, user_id: str, timeframes: tuple,
    ) -> List[Dict[str, Any]]:
        try:
            rows = (
                self.supabase.table("user_strategies")
                .select("id, dsl, status, name, strategy_intent, max_daily_loss_pct")
                .eq("user_id", user_id)
                .in_("status", ["paper", "live"])
                .limit(50)
                .execute()
            )
        except Exception:
            return []
        out = []
        for r in rows.data or []:
            dsl = r.get("dsl") or {}
            tf = dsl.get("timeframe", "1d")
            if tf in timeframes:
                out.append(r)
        return out

    def _load_open_positions(self, user_id: str, strategy_id: str) -> List[Dict[str, Any]]:
        try:
            rows = (
                self.supabase.table("strategy_positions")
                .select("id, symbol, entry_price, quantity, side, stop_loss, target_1")
                .eq("user_id", user_id)
                .eq("strategy_id", strategy_id)
                .eq("status", "open")
                .limit(200)
                .execute()
            )
            return rows.data or []
        except Exception:
            return []

    async def _load_bars(self, symbol: str):
        """Fetch the trailing ~1y of daily bars for the symbol via the
        existing market_data provider. Best-effort — returns None on failure."""
        try:
            from ...data.market import get_market_data_provider
            provider = get_market_data_provider()
            df = provider.get_historical(symbol.upper(), period="1y", interval="1d")
        except Exception:
            return None
        if df is None or len(df) < MIN_LOOKBACK:
            return None
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        return df

    async def _current_vix(self) -> Optional[float]:
        try:
            from ...data.market import get_market_data_provider
            provider = get_market_data_provider()
            df = provider.get_historical("INDIAVIX", period="5d", interval="1d")
        except Exception:
            return None
        if df is None or df.empty:
            return None
        return float(df["close"].iloc[-1])

    async def _persist_run(self, report: StrategyRunnerReport) -> None:
        try:
            payload = report.to_dict()
            if payload.get("finished_at") and payload.get("started_at"):
                started = datetime.fromisoformat(payload["started_at"].replace("Z", "+00:00"))
                finished = datetime.fromisoformat(payload["finished_at"].replace("Z", "+00:00"))
                payload["duration_ms"] = int((finished - started).total_seconds() * 1000)
            self.supabase.table("strategy_runner_runs").insert(payload).execute()
        except Exception:
            logger.debug("runner persist failed (non-fatal)")
