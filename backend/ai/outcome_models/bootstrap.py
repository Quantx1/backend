"""Bootstrap synthetic training data — PR-MODELS.

Problem: outcome models need ≥30 closed trades per template to train.
Real outcomes accumulate only after deployment + weeks of paper trading.

Solution: bootstrap by replaying each template through 1y of historical
NIFTY100 bars. Every entry → exit cycle becomes a synthetic outcome row.
Synthetic ≠ identical-to-live, but it gives the model enough signal to
discriminate "this strategy + this market state → typically wins."

Generated outcomes are tagged ``source='bootstrap'`` so they can be
weighted lower (or excluded) once real outcomes accumulate.

Memory locks honoured: synthetic outcomes use the same DSL backtest
that user-facing backtests use. No new heuristics; the labels come
from REAL price movement against REAL strategy rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# Universe of symbols to backtest each template against. Bigger universe →
# more synthetic outcomes, but slower. NIFTY50 ≈ 50 symbols is a good
# starting point.
DEFAULT_BOOTSTRAP_UNIVERSE = (
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "HCLTECH",
    "BAJFINANCE", "WIPRO", "SUNPHARMA", "M&M", "TITAN",
    "NESTLEIND", "ULTRACEMCO", "ADANIENT", "TATAMOTORS", "ONGC",
    "POWERGRID", "NTPC", "JSWSTEEL", "TATASTEEL", "INDUSINDBK",
)


@dataclass
class BootstrapResult:
    template_slug: str
    symbols_processed: int = 0
    backtests_run: int = 0
    outcomes_generated: int = 0
    skipped_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def bootstrap_outcomes_for_template(
    supabase_admin: Any,
    template_row: Dict[str, Any],
    *,
    universe: tuple = DEFAULT_BOOTSTRAP_UNIVERSE,
    lookback_period: str = "1y",
) -> BootstrapResult:
    """Replay one DSL template across `universe` symbols. Insert each
    trade as a synthetic row in strategy_outcomes.

    Args:
        template_row: row from strategy_catalog (must have slug, dsl, segment)
        universe: tuple of NSE tickers to replay against
        lookback_period: yfinance-style period string ("6mo", "1y", "2y")

    Skips OPTIONS-segment templates (those need tick data; v1.2).
    """
    slug = template_row.get("slug", "")
    result = BootstrapResult(template_slug=slug)

    if template_row.get("segment") != "EQUITY":
        result.skipped_reason = "options_template_skipped_v1"
        return result

    dsl_json = template_row.get("dsl")
    if not dsl_json:
        result.skipped_reason = "dsl_null"
        return result

    try:
        from ..strategy.dsl import Strategy as DSLStrategy
        from ..strategy.backtest import run_dsl_backtest
        from ..strategy.indicators import MIN_LOOKBACK
        from .features import build_outcome_features
    except Exception as exc:
        result.skipped_reason = f"import_failed: {exc}"
        return result

    try:
        strategy = DSLStrategy.model_validate(dsl_json)
    except Exception as exc:
        result.skipped_reason = f"dsl_invalid: {exc}"
        return result

    try:
        from ...data.market import get_market_data_provider
        provider = get_market_data_provider()
    except Exception as exc:
        result.skipped_reason = f"provider_failed: {exc}"
        return result

    # System-user UUID for bootstrap rows (Supabase requires user_id NOT NULL).
    # Use a deterministic UUID for bootstrap so we can find + filter them.
    BOOTSTRAP_USER_ID = "00000000-0000-0000-0000-000000000001"

    rows_to_insert: List[Dict[str, Any]] = []

    for symbol in universe:
        result.symbols_processed += 1
        try:
            ohlcv = provider.get_historical(symbol, period=lookback_period, interval="1d")
        except Exception:
            continue
        if ohlcv is None or len(ohlcv) < MIN_LOOKBACK + 10:
            continue

        # Normalize columns
        ohlcv = ohlcv.copy()
        ohlcv.columns = [c.lower() for c in ohlcv.columns]
        if "close" not in ohlcv.columns:
            continue

        try:
            backtest_result = run_dsl_backtest(
                strategy,
                ohlcv,
                symbol=symbol,
                initial_capital=500_000,
            )
        except Exception as exc:
            logger.debug("bootstrap %s/%s backtest failed: %s", slug, symbol, exc)
            continue
        result.backtests_run += 1

        # Each trade becomes one outcome row
        for trade in backtest_result.trades:
            # Match entry_date string against ohlcv index
            try:
                entry_date_str = trade.entry_date if isinstance(trade.entry_date, str) else str(trade.entry_date)
                # ohlcv.index entries are pd.Timestamp; compare ISO date prefix
                idx_strs = [str(ts.date()) for ts in ohlcv.index]
                match_positions = [i for i, s in enumerate(idx_strs) if s == entry_date_str[:10]]
                if not match_positions:
                    continue
                entry_idx = match_positions[0]
                bars_until_entry = ohlcv.iloc[: entry_idx + 1]
            except Exception:
                continue

            features = build_outcome_features(bars_until_entry, regime=None, vix=None)
            if not features:
                continue

            won = trade.net_pnl_pct > 0
            # DSLTrade doesn't have net_pnl_inr — derive a notional ₹ value
            # using a ₹100K notional just so the column has data. The model
            # only uses won/won-rate; the inr value is for display.
            notional_pnl_inr = round(trade.net_pnl_pct / 100 * 100_000, 2)
            rows_to_insert.append({
                "user_id": BOOTSTRAP_USER_ID,
                "strategy_id": None,
                "template_slug": slug,
                "symbol": symbol,
                "side": "long",
                "entry_at": str(trade.entry_date),
                "exit_at": str(trade.exit_date),
                "entry_price": float(trade.entry_price),
                "exit_price": float(trade.exit_price),
                "quantity": 1,
                "pnl_inr": notional_pnl_inr,
                "pnl_pct": float(trade.net_pnl_pct),
                "result": trade.exit_reason,
                "won": won,
                "bars_held": int(getattr(trade, "bars_held", 0) or 0),
                "features_at_entry": features,
                "regime_at_entry": None,
                "vix_at_entry": None,
                "source": "bootstrap",
            })
            result.outcomes_generated += 1

    # Bulk insert
    if rows_to_insert:
        try:
            # Insert in chunks of 100 to avoid Supabase limits
            CHUNK = 100
            for i in range(0, len(rows_to_insert), CHUNK):
                # Need to handle strategy_id NOT NULL constraint — convert to FK-allowed
                # Since strategy_id has FK constraint, we need either valid strategy
                # rows OR drop the FK on strategy_id. For bootstrap rows we'll let
                # the FK fail and catch it gracefully.
                chunk = rows_to_insert[i:i + CHUNK]
                try:
                    supabase_admin.table("strategy_outcomes").insert(chunk).execute()
                except Exception as exc:
                    msg = str(exc)[:200]
                    if "strategy_id" in msg or "foreign key" in msg.lower():
                        result.notes.append(
                            "strategy_id FK blocked bulk insert — schema needs nullable strategy_id for bootstrap rows")
                        result.outcomes_generated = 0
                        return result
                    logger.warning("bootstrap chunk insert failed: %s", msg)
        except Exception as exc:
            logger.warning("bootstrap insert failed: %s", exc)

    return result


def bootstrap_all_active_templates(
    supabase_admin: Any,
    *,
    universe: tuple = DEFAULT_BOOTSTRAP_UNIVERSE,
    lookback_period: str = "1y",
    only_slugs: Optional[List[str]] = None,
) -> List[BootstrapResult]:
    """Bootstrap synthetic outcomes for every active equity template.

    Returns one result per template. Filter by ``only_slugs`` if testing.
    """
    try:
        q = (
            supabase_admin.table("strategy_catalog")
            .select("slug, dsl, segment")
            .eq("is_active", True)
            .eq("segment", "EQUITY")
            .limit(200)
        )
        rows = q.execute()
    except Exception as exc:
        logger.warning("bootstrap fetch templates failed: %s", exc)
        return []

    templates = [r for r in (rows.data or []) if r.get("dsl")]
    if only_slugs:
        templates = [r for r in templates if r.get("slug") in only_slugs]

    results: List[BootstrapResult] = []
    for tpl in templates:
        logger.info("bootstrapping outcomes for %s", tpl.get("slug"))
        result = bootstrap_outcomes_for_template(
            supabase_admin, tpl,
            universe=universe, lookback_period=lookback_period,
        )
        results.append(result)
        logger.info(
            "  %s: backtests=%d outcomes=%d skipped=%s",
            tpl.get("slug"), result.backtests_run,
            result.outcomes_generated, result.skipped_reason,
        )
    return results
