"""Walk-forward scoring — split the bars into N contiguous folds and
score each independently so we can see whether the strategy decays.

A flat Sharpe across 3 folds means the edge is real. A 2.0 → 1.5 → 0.4
→ -0.2 trail means the strategy worked in 2024 and broke in 2026; you
shouldn't deploy it just because the in-sample blended Sharpe says 0.9.

Output is a list of per-fold dicts persisted into
`discovered_strategies.walk_forward` JSONB so the UI can render a
sparkline showing strategy decay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.ai.strategy.backtest import (
    DSLBacktestResult,
    run_dsl_backtest,
)
from backend.ai.strategy.dsl import InstrumentSegment, Strategy
from backend.ai.strategy.interpreter import EngineSignals
from backend.ai.strategy.options_backtest import (
    OptionsBacktestResult,
    run_options_backtest,
)

logger = logging.getLogger(__name__)


MIN_BARS_PER_FOLD = 80      # below this, fold Sharpe is meaningless


@dataclass
class FoldStat:
    """Per-fold metrics persisted into the walk_forward JSONB."""
    fold_index: int             # 0..N-1
    start_date: str
    end_date: str
    bars: int
    trades: int
    sharpe: float
    return_pct: float
    max_dd_pct: float
    win_rate: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "fold": self.fold_index,
            "start": self.start_date,
            "end": self.end_date,
            "bars": self.bars,
            "trades": self.trades,
            "sharpe": round(self.sharpe, 4),
            "return_pct": round(self.return_pct, 4),
            "max_dd_pct": round(self.max_dd_pct, 4),
            "win_rate": round(self.win_rate, 4),
        }
        if self.error:
            out["error"] = self.error[:200]
        return out


def split_bars(
    bars: pd.DataFrame, n_folds: int = 3,
) -> List[pd.DataFrame]:
    """Split a DataFrame into N contiguous folds by row count.

    Last fold absorbs the remainder so total bars are preserved.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if bars is None or len(bars) < n_folds * MIN_BARS_PER_FOLD:
        # Not enough data for meaningful walk-forward; return the whole
        # range as a single fold so the caller still gets a stat.
        return [bars] if bars is not None and len(bars) > 0 else []
    step = len(bars) // n_folds
    folds: List[pd.DataFrame] = []
    for i in range(n_folds):
        start = i * step
        end = (i + 1) * step if i < n_folds - 1 else len(bars)
        fold = bars.iloc[start:end]
        if len(fold) >= MIN_BARS_PER_FOLD:
            folds.append(fold)
    return folds


def score_walk_forward(
    strategy: Strategy,
    bars: pd.DataFrame,
    *,
    symbol: str,
    n_folds: int = 3,
    initial_capital: float = 100_000.0,
    engine_signals_by_date: Optional[Dict[pd.Timestamp, EngineSignals]] = None,
) -> List[FoldStat]:
    """Run the backtester independently on each fold, return per-fold stats.

    Each fold is a fresh capital-state backtest — we do NOT carry the
    equity curve from fold N into fold N+1. That makes the per-fold
    Sharpes directly comparable. Fold-level failures are isolated so a
    bad fold doesn't kill the whole walk-forward.
    """
    folds = split_bars(bars, n_folds)
    if not folds:
        return []

    stats: List[FoldStat] = []
    for i, fold_bars in enumerate(folds):
        start_d = str(fold_bars.index[0].date()) if hasattr(fold_bars.index[0], "date") else str(fold_bars.index[0])
        end_d = str(fold_bars.index[-1].date()) if hasattr(fold_bars.index[-1], "date") else str(fold_bars.index[-1])
        try:
            if strategy.instrument_segment == InstrumentSegment.OPTIONS:
                r: OptionsBacktestResult = run_options_backtest(
                    strategy, fold_bars, symbol=symbol,
                    initial_capital=initial_capital,
                    engine_signals_by_date=engine_signals_by_date,
                )
                stats.append(FoldStat(
                    fold_index=i, start_date=start_d, end_date=end_d,
                    bars=len(fold_bars), trades=r.total_trades,
                    sharpe=r.sharpe_ratio,
                    return_pct=r.total_return_pct,
                    max_dd_pct=r.max_drawdown_pct,
                    win_rate=r.win_rate,
                ))
            else:
                r: DSLBacktestResult = run_dsl_backtest(
                    strategy, fold_bars, symbol=symbol,
                    initial_capital=initial_capital,
                    engine_signals_by_date=engine_signals_by_date,
                )
                stats.append(FoldStat(
                    fold_index=i, start_date=start_d, end_date=end_d,
                    bars=len(fold_bars), trades=r.total_trades,
                    sharpe=r.sharpe_ratio,
                    return_pct=r.total_return_pct,
                    max_dd_pct=r.max_drawdown_pct,
                    win_rate=r.win_rate,
                ))
        except Exception as e:
            logger.debug("walk-forward fold %d failed for %s: %s", i, symbol, e)
            stats.append(FoldStat(
                fold_index=i, start_date=start_d, end_date=end_d,
                bars=len(fold_bars), trades=0, sharpe=0.0, return_pct=0.0,
                max_dd_pct=0.0, win_rate=0.0, error=str(e),
            ))
    return stats


def stability_score(folds: List[FoldStat]) -> float:
    """Cheap decay-detection metric: mean(Sharpe) − std(Sharpe).

    A strategy with [1.5, 1.4, 1.6] scores 1.5 − 0.08 = 1.42.
    A strategy with [2.0, 0.5, -0.5] scores 0.67 − 1.0 = -0.33.

    Returns 0.0 for empty folds.
    """
    if not folds:
        return 0.0
    sharpes = [f.sharpe for f in folds if f.error is None]
    if not sharpes:
        return 0.0
    mean = sum(sharpes) / len(sharpes)
    if len(sharpes) < 2:
        return mean
    var = sum((s - mean) ** 2 for s in sharpes) / len(sharpes)
    std = var ** 0.5
    return mean - std
