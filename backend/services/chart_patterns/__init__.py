"""Chart-pattern scanner service (PR-S5).

Wires our existing 2,166-line pattern algorithm (`ml/features/patterns.py`)
into a trader-grade scanner with the right gates:

  scan_universe() returns only patterns that:
    * Quality score >= MIN_QUALITY                            (rule-engine)
    * BreakoutMetaLabeler RF probability >= MIN_ML_THRESHOLD  (ML filter)
    * Aligned with the current market regime                  (don't bet
        long breakouts in bear, etc.)
    * Volume confirmed on the detection bar                   (no thin moves)

Each result carries the raw pattern + the ML score + the realised
forward-return statistics (computed offline + cached) so the UI can
show honest performance instead of a hard-coded "63.6% win rate" claim.
"""

from .scanner import (
    scan_universe, scan_universe_streaming, scan_symbol, PatternMatch,
    full_nse_universe, filter_by_sector,
)
from .explain import explain_symbol, PatternExplanation

__all__ = [
    "scan_universe", "scan_universe_streaming", "scan_symbol", "PatternMatch",
    "full_nse_universe", "filter_by_sector",
    "explain_symbol", "PatternExplanation",
]
