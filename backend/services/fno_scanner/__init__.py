"""F&O scanner — index option-chain snapshots + strategy suggestions.

PR-S19 (2026-05-31). Distinct from the equity SCANNER_FILTERS pipeline
because F&O signals are option-chain-centric (per-strike OI, IV per
strike, max-pain math) rather than per-symbol indicator filters.

Public entry points:
    snapshot.fetch_index_snapshot(symbol)  -> IndexSnapshot
    strategies.suggest_strategies(snap, vix_value) -> List[StrategySuggestion]
    lot_sizes.LOT_SIZES                    -> {symbol: lot_size} (Jan 2026 NSE revision)

Sources for all formulas: see the docstring of each module.
"""

from .lot_sizes import LOT_SIZES, FUTURE_TICK_SIZES
from .snapshot import IndexSnapshot, fetch_index_snapshot, teach_snapshot
from .strategies import StrategySuggestion, suggest_strategies, classify_vix_regime
from .adjustments import AdjustmentSuggestion, suggest_adjustments

__all__ = [
    "LOT_SIZES",
    "FUTURE_TICK_SIZES",
    "IndexSnapshot",
    "fetch_index_snapshot",
    "teach_snapshot",
    "StrategySuggestion",
    "suggest_strategies",
    "classify_vix_regime",
    "AdjustmentSuggestion",
    "suggest_adjustments",
]
