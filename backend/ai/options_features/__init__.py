"""Options chain features — PR-FEATURES.

Borrowed from aaryansinha16/AI-trader's options-specific feature set
(features/option_chain_features.py). These features compute from option
chain snapshots (strike × CE/PE × OI/IV/volume) — useful for:

  - Outcome models (PCR + OI change as features per entry)
  - Options-strategy gates (e.g. skip Iron Condor entry when PCR is extreme)
  - F&O autopilot regime sizing

Pure-Python; expects an option_chain DataFrame from the broker (Kite
``instruments_oc`` or similar). Works against Supabase ``option_chain``
table once we ingest it.

7 features:
  pcr                       Put-Call Ratio (total PE OI / total CE OI)
  pcr_volume                Put-Call Volume Ratio (PE vol / CE vol)
  total_oi_change           Net OI change across the chain
  max_pain                  Strike with highest OI concentration
  iv_atm                    Average IV at ATM strikes
  iv_skew                   IV(PE_-2σ) - IV(CE_+2σ)  (positive = put-skew)
  theta_pressure            Sum of theta across short-DTE contracts
                            (proxy for time-decay risk in the chain)
"""

from .compute import (
    OptionsChainFeatures,
    compute_options_features,
    iv_percentile_from_history,
)

__all__ = [
    "OptionsChainFeatures",
    "compute_options_features",
    "iv_percentile_from_history",
]
