"""Signal generation package.

Public API (filled in as Tasks 2–7 of PR-A3 land)::

    from backend.ai.signals import (
        SignalGenerator,
        GeneratedSignal,
        EnsembleVoter,
        compute_ensemble_score,
        regime_bonus,
        save_signals, save_universe, cache_candles, resolve_catalog_id,
        OptionsSignalEngine,
        make_tft_voter, make_lgbm_voter, make_qlib_voter,
        make_regime_voter, WEIGHTS,
    )

See ``docs/superpowers/plans/2026-05-25-backend-structural-restructure.md``
for the design.
"""

from .types import EnsembleVoter, GeneratedSignal
from .ensemble import compute_ensemble_score, regime_bonus
from .persistence import (
    save_signals, save_universe, cache_candles, resolve_catalog_id,
)
from .options import OptionsSignalEngine
from .voters import (
    make_lgbm_voter, make_tft_voter, make_qlib_voter,
    make_regime_voter, WEIGHTS,
)
from .generator import SignalGenerator

__all__ = [
    "SignalGenerator",
    "EnsembleVoter",
    "GeneratedSignal",
    "compute_ensemble_score",
    "regime_bonus",
    "save_signals",
    "save_universe",
    "cache_candles",
    "resolve_catalog_id",
    "OptionsSignalEngine",
    "make_lgbm_voter",
    "make_tft_voter",
    "make_qlib_voter",
    "make_regime_voter",
    "WEIGHTS",
]
