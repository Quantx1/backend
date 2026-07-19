"""Full training pipeline — PR-MODELS.

Runs all three training stages end-to-end:
  1. Bootstrap synthetic outcomes for every active EQUITY template
     (replays DSL through 1-2y of NIFTY100 historical bars)
  2. Train one XGBoost outcome model per eligible template_slug
     (≥30 outcomes + 30/30 class balance)
  3. Train the RL exit Q-learning agent on all bootstrap trade journeys
     (HOLD + EXIT actions per bar; rewards = realized PnL)

Usage::

    python scripts/train/train_models_full.py
    python scripts/train/train_models_full.py --universe-size 30 --lookback 2y
    python scripts/train/train_models_full.py --only-slugs rsi-mean-reversion ema-crossover-swing

Idempotent — saves to ``artifacts/outcome/<slug>/`` and ``artifacts/rl/q_table.json``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()


_LARGE_UNIVERSE = (
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "HCLTECH",
    "BAJFINANCE", "WIPRO", "SUNPHARMA", "M&M", "TITAN",
    "NESTLEIND", "ULTRACEMCO", "ADANIENT", "TATAMOTORS", "ONGC",
    "POWERGRID", "NTPC", "JSWSTEEL", "TATASTEEL", "INDUSINDBK",
)


def stage_1_bootstrap(supabase, universe, lookback, only_slugs):
    from backend.ai.outcome_models.bootstrap import bootstrap_all_active_templates
    return bootstrap_all_active_templates(
        supabase,
        universe=tuple(universe),
        lookback_period=lookback,
        only_slugs=only_slugs,
    )


def stage_2_train_outcome_models(supabase):
    from backend.ai.outcome_models.trainer import (
        OutcomeModelTrainer, OutcomeModelConfig,
    )
    from backend.ai.outcome_models.registry import save_outcome_model

    trainer = OutcomeModelTrainer(supabase)
    rows = supabase.table("strategy_outcomes").select("template_slug").limit(20000).execute()
    counts: dict = {}
    for r in rows.data or []:
        slug = r.get("template_slug")
        if slug:
            counts[slug] = counts.get(slug, 0) + 1
    eligible = [s for s, c in counts.items() if c >= 30]
    print(f"  Eligible slugs (≥30 outcomes): {len(eligible)}")

    results = []
    for slug in eligible:
        result = trainer.train(OutcomeModelConfig(template_slug=slug))
        if result.trained:
            path = save_outcome_model(slug, result)
            auc = f"{result.auc:.3f}" if result.auc is not None else "N/A"
            print(f"    TRAINED {slug}: n={result.n_samples} wr={result.win_rate:.2f} auc={auc} → {path}")
        else:
            print(f"    SKIPPED {slug}: {result.skipped_reason}")
        results.append(result)
    return results


def stage_3_train_rl_exit(supabase):
    """Walk bootstrap outcomes, regenerate price trajectories from
    historical bars, train Q-learning on each trajectory."""
    from backend.ai.exit_engine.rl_exit_scaffold import RLExitAgent
    from backend.data.market import get_market_data_provider

    # Fetch outcomes with the journey context (entry_at, exit_at, symbol)
    rows = (
        supabase.table("strategy_outcomes")
        .select("template_slug, symbol, entry_at, exit_at, entry_price, exit_price, won, pnl_pct")
        .eq("source", "bootstrap")
        .order("exit_at", desc=True)
        .limit(5000)
        .execute()
    )
    outcomes = rows.data or []
    print(f"  Found {len(outcomes)} bootstrap outcomes to replay as RL journeys")

    if not outcomes:
        print("  No bootstrap outcomes — RL training skipped")
        return None

    provider = get_market_data_provider()
    agent = RLExitAgent()
    journeys_trained = 0
    skipped = 0

    # Cache OHLCV per symbol — fetch once, reuse for all trades on that symbol
    ohlcv_cache: dict = {}

    for outcome in outcomes:
        symbol = outcome.get("symbol")
        entry_at = outcome.get("entry_at", "")[:10]
        exit_at = outcome.get("exit_at", "")[:10]
        entry_price = float(outcome.get("entry_price") or 0)
        if not symbol or not entry_at or not exit_at or entry_price <= 0:
            skipped += 1
            continue

        # Lazy-fetch OHLCV per symbol
        if symbol not in ohlcv_cache:
            try:
                df = provider.get_historical(symbol, period="2y", interval="1d")
                if df is not None and not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    ohlcv_cache[symbol] = df
                else:
                    ohlcv_cache[symbol] = None
            except Exception:
                ohlcv_cache[symbol] = None
        df = ohlcv_cache.get(symbol)
        if df is None:
            skipped += 1
            continue

        # Slice the journey window [entry_at, exit_at] from this symbol's bars
        try:
            idx_strs = [str(ts.date()) for ts in df.index]
            entry_pos = next((i for i, s in enumerate(idx_strs) if s == entry_at), None)
            exit_pos = next((i for i, s in enumerate(idx_strs) if s == exit_at), None)
            if entry_pos is None or exit_pos is None or exit_pos <= entry_pos:
                skipped += 1
                continue
            journey_bars = df.iloc[entry_pos: exit_pos + 1]
        except Exception:
            skipped += 1
            continue

        # Build trajectory: one point per bar with close price + bars_held
        trajectory = [
            {"price": float(row["close"]), "bars_held": i + 1}
            for i, (_, row) in enumerate(journey_bars.iterrows())
        ]
        if len(trajectory) < 2:
            skipped += 1
            continue

        try:
            agent.fit_trajectory(
                trajectory=trajectory,
                entry_price=entry_price,
                sl_pct=0.08,
                target_pct=0.15,
                max_hold_bars=40,
            )
            journeys_trained += 1
        except Exception as exc:
            logging.debug("rl fit failed for %s: %s", symbol, exc)
            skipped += 1

    agent.is_loaded = len(agent.q_table) > 0
    saved = agent.save() if agent.is_loaded else None
    print(f"  RL agent: journeys_trained={journeys_trained} skipped={skipped}")
    print(f"           q_table_size={len(agent.q_table)} episodes={agent.training_episodes}")
    if saved:
        print(f"           saved → {saved}")
    return agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-size", type=int, default=30,
                        help="Number of NIFTY symbols to use (1-30)")
    parser.add_argument("--lookback", default="2y",
                        help="yfinance period string (1y, 2y, 5y)")
    parser.add_argument("--only-slugs", nargs="*", default=None,
                        help="Limit to specific template slugs (default: all active)")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip stage 1 (use existing strategy_outcomes rows)")
    parser.add_argument("--skip-rl", action="store_true",
                        help="Skip stage 3 (RL training)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

    from backend.core.database import get_supabase_admin
    sb = get_supabase_admin()

    universe = _LARGE_UNIVERSE[: max(1, min(args.universe_size, 30))]

    print(f"\n═══ STAGE 1: Bootstrap outcomes ═══")
    print(f"   universe: {len(universe)} symbols · lookback: {args.lookback}")
    if not args.skip_bootstrap:
        results = stage_1_bootstrap(sb, universe, args.lookback, args.only_slugs)
        total_outcomes = sum(r.outcomes_generated for r in results)
        print(f"   → {total_outcomes} synthetic outcomes generated across {len(results)} templates")
    else:
        print(f"   SKIPPED (--skip-bootstrap)")

    print(f"\n═══ STAGE 2: Train per-template XGBoost outcome models ═══")
    stage_2_train_outcome_models(sb)

    if not args.skip_rl:
        print(f"\n═══ STAGE 3: Train RL exit Q-learning agent ═══")
        stage_3_train_rl_exit(sb)
    else:
        print(f"\n═══ STAGE 3 SKIPPED (--skip-rl) ═══")

    print(f"\n✓ Full training complete")


if __name__ == "__main__":
    main()
