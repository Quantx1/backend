#!/usr/bin/env python3
"""
Strategy-by-Strategy Backtest Runner
=====================================
Backtests all 6 strategies on NSE stocks and prints per-strategy results.

Usage:
    python scripts/backtest/backtest_strategies.py [--stocks 30] [--period 2y]
    python scripts/backtest/backtest_strategies.py --stocks 500 --period 5y --full
"""

import sys
import os
import warnings
import argparse
import logging
import time
from pathlib import Path
from collections import defaultdict



# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd

from ml.backtest.engine import BacktestEngine, BacktestConfig
from ml.scanner import get_all_strategies
from ml.features.indicators import compute_all_indicators
from ml.features.patterns import walkforward_train, BreakoutMetaLabeler
from ml.strategies.consolidation_breakout import ConsolidationBreakout


def fetch_data_yfinance(symbols: list, period: str = "2y") -> dict:
    """Fetch OHLCV via yfinance with batch downloading for speed."""
    import yfinance as yf

    data = {}
    total = len(symbols)
    print(f"  Fetching {total} stocks via yfinance (period={period})...")

    # Batch download for speed (batches of 50)
    batch_size = 50
    for batch_start in range(0, total, batch_size):
        batch_syms = symbols[batch_start:batch_start + batch_size]
        try:
            batch_df = yf.download(batch_syms, period=period, progress=False,
                                   auto_adjust=True, group_by='ticker', threads=True)
            if batch_df is None or len(batch_df) == 0:
                continue

            for sym in batch_syms:
                try:
                    if len(batch_syms) == 1:
                        df = batch_df.copy()
                    else:
                        df = batch_df[sym].copy()

                    if df is None or len(df) < 200:
                        continue

                    df = df.dropna(subset=['Close'] if 'Close' in df.columns else ['close'])
                    if len(df) < 200:
                        continue

                    # Normalize columns to lowercase
                    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                    # Ensure required columns
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col not in df.columns:
                            break
                    else:
                        data[sym] = df
                except Exception:
                    pass
        except Exception:
            # Fallback to individual downloads
            for sym in batch_syms:
                try:
                    df = yf.download(sym, period=period, progress=False, auto_adjust=True)
                    if df is None or len(df) < 200:
                        continue
                    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col not in df.columns:
                            break
                    else:
                        data[sym] = df
                except Exception:
                    pass

        fetched = min(batch_start + batch_size, total)
        print(f"    {fetched}/{total} fetched ({len(data)} usable)")

    print(f"  Got {len(data)}/{total} stocks with 200+ bars\n")
    return data


def compute_indicators_sequential(stock_data: dict) -> dict:
    """Compute indicators for all stocks sequentially with progress."""
    total = len(stock_data)
    print(f"Computing indicators ({total} stocks)...")

    result_data = {}
    failed = 0
    done = 0
    t0 = time.time()

    for sym in list(stock_data.keys()):
        try:
            result_data[sym] = compute_all_indicators(stock_data[sym])
        except Exception as e:
            failed += 1

        done += 1
        if done % 50 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"    {done}/{total} done ({failed} failed) "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    print(f"  {len(result_data)} stocks ready ({failed} failed, {time.time()-t0:.0f}s total)\n")
    return result_data


def load_universe(max_stocks: int = 30, full: bool = False) -> list:
    """Load symbols from backtest universe file."""
    if full:
        path = ROOT / "data" / "full_backtest_universe.txt"
    else:
        path = ROOT / "data" / "backtest_universe.txt"

    symbols = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.split('#')[0].strip()
                if not line:
                    continue
                if not line.endswith(".NS"):
                    line = f"{line}.NS"
                symbols.append(line)
    else:
        symbols = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS",
            "ICICIBANK.NS", "BHARTIARTL.NS", "LT.NS", "ITC.NS", "KOTAKBANK.NS",
            "AXISBANK.NS", "MARUTI.NS", "TITAN.NS", "WIPRO.NS", "SUNPHARMA.NS",
            "HCLTECH.NS", "BAJFINANCE.NS", "NTPC.NS", "POWERGRID.NS", "TATASTEEL.NS",
        ]
    # Filter known delisted
    symbols = [s for s in symbols if s != "TATAMOTORS.NS"]
    return symbols[:max_stocks]


def aggregate_results(all_results: dict, strategy_name: str):
    """Aggregate multi-stock results for a single strategy."""
    all_trades = []
    for sym, result in all_results.items():
        all_trades.extend(result.trades)

    if not all_trades:
        print(f"\n  {strategy_name}: NO TRADES generated across all stocks")
        return None

    pnls = [t.net_pnl_pct for t in all_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Exit reason breakdown
    reasons = defaultdict(int)
    for t in all_trades:
        reasons[t.exit_reason] += 1

    # Profit factor
    win_amt = sum(t.quantity * t.entry_price * t.net_pnl_pct / 100 for t in all_trades if t.net_pnl_pct > 0)
    loss_amt = abs(sum(t.quantity * t.entry_price * t.net_pnl_pct / 100 for t in all_trades if t.net_pnl_pct <= 0))
    pf = round(win_amt / loss_amt, 2) if loss_amt > 0 else float('inf')

    # Sharpe
    sharpe = 0.0
    if len(pnls) > 1:
        std = np.std(pnls, ddof=1)
        if std > 0:
            sharpe = round((np.mean(pnls) / std) * np.sqrt(min(250, len(pnls))), 2)

    stats = {
        'strategy': strategy_name,
        'total_trades': len(all_trades),
        'win_rate': round(len(wins) / len(pnls) * 100, 1),
        'avg_win': round(np.mean(wins), 2) if wins else 0,
        'avg_loss': round(np.mean(losses), 2) if losses else 0,
        'avg_pnl': round(np.mean(pnls), 2),
        'profit_factor': pf,
        'sharpe': sharpe,
        'avg_hold': round(np.mean([t.hold_days for t in all_trades]), 1),
        'stocks_traded': len([s for s, r in all_results.items() if r.total_trades > 0]),
        'exit_reasons': dict(reasons),
    }

    return stats


def print_strategy_report(stats: dict):
    """Print formatted report for one strategy."""
    if stats is None:
        return

    s = stats
    print(f"\n{'~'*60}")
    print(f"  {s['strategy']}")
    print(f"{'~'*60}")
    print(f"  Total Trades:     {s['total_trades']:>6}   (across {s['stocks_traded']} stocks)")
    print(f"  Win Rate:         {s['win_rate']:>6.1f}%")
    print(f"  Avg Win:          {s['avg_win']:>+6.2f}%")
    print(f"  Avg Loss:         {s['avg_loss']:>+6.2f}%")
    print(f"  Avg P&L/trade:    {s['avg_pnl']:>+6.2f}%")
    print(f"  Profit Factor:    {s['profit_factor']:>6.2f}")
    print(f"  Sharpe Ratio:     {s['sharpe']:>6.2f}")
    print(f"  Avg Hold Days:    {s['avg_hold']:>6.1f}")
    print(f"  Exit Reasons:")
    for reason, count in sorted(s['exit_reasons'].items(), key=lambda x: -x[1]):
        pct = count / s['total_trades'] * 100
        print(f"    {reason:20s} {count:4d} ({pct:5.1f}%)")


def print_summary_table(all_stats: list):
    """Print comparison table of all strategies."""
    valid = [s for s in all_stats if s is not None]
    if not valid:
        print("\nNo strategies generated any trades!")
        return

    print(f"\n{'='*90}")
    print(f"  STRATEGY COMPARISON SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Strategy':<25} {'Trades':>7} {'WinRate':>8} {'AvgPnL':>8} {'PF':>7} {'Sharpe':>7} {'Hold':>6}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*6}")

    for s in sorted(valid, key=lambda x: x['avg_pnl'], reverse=True):
        verdict = "KEEP" if s['win_rate'] > 45 and s['profit_factor'] > 1.5 else "TUNE" if s['win_rate'] > 35 else "REVIEW"
        print(f"  {s['strategy']:<25} {s['total_trades']:>7} {s['win_rate']:>7.1f}% {s['avg_pnl']:>+7.2f}% {s['profit_factor']:>7.2f} {s['sharpe']:>7.2f} {s['avg_hold']:>5.1f}d {verdict}")

    print(f"\n  Legend: KEEP = WR>45%+PF>1.5  TUNE = WR 35-45%  REVIEW = WR<35%")
    print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description="Backtest all 6 strategies")
    parser.add_argument("--stocks", type=int, default=30, help="Number of stocks to test (default: 30)")
    parser.add_argument("--period", type=str, default="2y", help="Data period (default: 2y)")
    parser.add_argument("--strategy", type=str, default=None, help="Run specific strategy only")
    parser.add_argument("--full", action="store_true", help="Use full 500-stock universe")
    args = parser.parse_args()

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  SWING AI - STRATEGY BACKTEST")
    print(f"{'='*60}")
    print(f"  Stocks: {args.stocks} | Period: {args.period} | Full: {args.full}")

    # Load data
    symbols = load_universe(max_stocks=args.stocks, full=args.full)
    stock_data = fetch_data_yfinance(symbols, period=args.period)

    if not stock_data:
        print("ERROR: No stock data fetched. Check internet connection.")
        return

    # Compute indicators (once — engine will skip recomputation)
    stock_data = compute_indicators_sequential(stock_data)

    if not stock_data:
        print("ERROR: All indicator computations failed.")
        return

    # Train ML meta-labeler for consolidation breakout
    ml_labeler = None
    if len(stock_data) >= 50:
        print("Training ML meta-labeler on historical breakouts...")
        train_syms = list(stock_data.keys())[:len(stock_data) // 2]
        train_data = {s: stock_data[s] for s in train_syms}
        ml_labeler = walkforward_train(train_data, lookback=250, hold_period=15)
        if ml_labeler.is_trained:
            print(f"  ML meta-labeler trained on {len(train_syms)} stocks\n")
        else:
            print(f"  ML meta-labeler: not enough data, running without ML filter\n")
            ml_labeler = None
    else:
        print(f"  Skipping ML meta-labeler (need 50+ stocks, have {len(stock_data)})\n")

    # Configure
    config = BacktestConfig(
        initial_capital=500000,
        risk_per_trade_pct=3.0,
        min_confidence=55.0,
    )
    engine = BacktestEngine(config)
    strategies = get_all_strategies(ml_labeler=ml_labeler)

    if args.strategy:
        strategies = [s for s in strategies if s.name == args.strategy]
        if not strategies:
            print(f"ERROR: Strategy '{args.strategy}' not found")
            return

    # Run backtests per strategy
    all_stats = []
    for si, strategy in enumerate(strategies):
        t_strat = time.time()
        print(f"\n{'='*60}")
        print(f"  [{si+1}/{len(strategies)}] Running: {strategy.name} (max_hold={strategy.max_hold_bars})")
        print(f"{'='*60}")

        results = engine.run_multi_stock(stock_data, strategy)
        stats = aggregate_results(results, strategy.name)
        print_strategy_report(stats)
        all_stats.append(stats)

        print(f"  (completed in {time.time()-t_strat:.0f}s)")

    # Summary
    print_summary_table(all_stats)

    total_time = time.time() - t_start
    print(f"  Total backtest time: {total_time:.0f}s ({total_time/60:.1f}m)\n")


if __name__ == "__main__":
    main()
