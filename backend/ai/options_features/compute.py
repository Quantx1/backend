"""Options chain feature computation — PR-FEATURES.

Input: pandas DataFrame with columns
  strike, option_type ('CE'|'PE'), oi, oi_change (optional),
  volume (optional), ltp (optional), iv (optional), delta (optional)

Output: ``OptionsChainFeatures`` dataclass + dict (JSON-safe) suitable
for outcome model features_at_entry JSONB column.

Pure-Python. No DB / network access here — caller fetches the chain
and passes it in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptionsChainFeatures:
    """One snapshot of options-chain features for ATM ± N strikes."""
    spot: float
    pcr: float                         # PE_OI / CE_OI
    pcr_volume: Optional[float] = None
    total_oi_change: float = 0.0
    max_pain: Optional[float] = None
    iv_atm: Optional[float] = None
    iv_skew: Optional[float] = None     # PE_OTM_IV - CE_OTM_IV
    theta_pressure: float = 0.0
    days_to_expiry: Optional[int] = None
    ce_volume_share: float = 0.5        # CE_vol / (CE_vol + PE_vol)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spot": round(self.spot, 2),
            "pcr": round(self.pcr, 3),
            "pcr_volume": round(self.pcr_volume, 3) if self.pcr_volume is not None else None,
            "total_oi_change": float(self.total_oi_change),
            "max_pain": round(self.max_pain, 2) if self.max_pain is not None else None,
            "iv_atm": round(self.iv_atm, 4) if self.iv_atm is not None else None,
            "iv_skew": round(self.iv_skew, 4) if self.iv_skew is not None else None,
            "theta_pressure": round(self.theta_pressure, 2),
            "days_to_expiry": self.days_to_expiry,
            "ce_volume_share": round(self.ce_volume_share, 3),
            "notes": self.notes,
        }

    @property
    def is_extreme_pcr(self) -> bool:
        """True if PCR is in the extreme zone (< 0.5 = put-thin / very
        bullish, or > 1.5 = put-heavy / very bearish). Useful as a
        contrarian gate — extreme readings often mark turning points."""
        return self.pcr < 0.5 or self.pcr > 1.5


def compute_options_features(
    chain_df: pd.DataFrame,
    *,
    spot_price: float,
    expiry_date: Optional[Any] = None,
    today: Optional[Any] = None,
) -> Optional[OptionsChainFeatures]:
    """Compute the 7-feature snapshot from an option chain DataFrame.

    Returns ``None`` if the chain is empty or doesn't have CE/PE rows.
    Missing columns degrade gracefully — feature fields stay None.
    """
    if chain_df is None or chain_df.empty:
        return None

    df = chain_df.copy()
    df["option_type"] = df["option_type"].str.upper()

    ce = df[df["option_type"] == "CE"]
    pe = df[df["option_type"] == "PE"]

    if ce.empty or pe.empty:
        return None

    feat = OptionsChainFeatures(spot=spot_price, pcr=1.0)

    # ─── PCR (OI-weighted) ──────────────────────────────────────
    ce_oi = float(ce["oi"].sum()) if "oi" in ce.columns else 0.0
    pe_oi = float(pe["oi"].sum()) if "oi" in pe.columns else 0.0
    feat.pcr = pe_oi / ce_oi if ce_oi > 0 else 1.0

    # ─── PCR (Volume-weighted) ──────────────────────────────────
    if "volume" in df.columns:
        ce_vol = float(ce["volume"].sum() or 0)
        pe_vol = float(pe["volume"].sum() or 0)
        if ce_vol + pe_vol > 0:
            feat.pcr_volume = pe_vol / ce_vol if ce_vol > 0 else 99.0
            feat.ce_volume_share = ce_vol / (ce_vol + pe_vol)

    # ─── OI change ──────────────────────────────────────────────
    if "oi_change" in df.columns:
        feat.total_oi_change = float(df["oi_change"].sum() or 0)

    # ─── Max Pain (strike with most OI concentration) ───────────
    if "oi" in df.columns:
        try:
            oi_by_strike = df.groupby("strike")["oi"].sum()
            feat.max_pain = float(oi_by_strike.idxmax())
        except Exception:
            pass

    # ─── ATM IV ──────────────────────────────────────────────
    if "iv" in df.columns:
        try:
            # Take 3 strikes nearest to spot, average their IV across CE+PE
            df["_dist"] = (df["strike"] - spot_price).abs()
            atm_rows = df.nsmallest(6, "_dist")        # 3 ATM strikes × 2 types
            iv_values = atm_rows["iv"].dropna()
            if len(iv_values) > 0:
                feat.iv_atm = float(iv_values.mean())
        except Exception:
            pass

    # ─── IV Skew (PE OTM IV - CE OTM IV) ─────────────────────────
    if "iv" in df.columns:
        try:
            # OTM PE: strike < spot. OTM CE: strike > spot.
            # Use 2-strike-wide OTM
            pe_otm = pe[(pe["strike"] < spot_price * 0.99)].sort_values("strike", ascending=False).head(3)
            ce_otm = ce[(ce["strike"] > spot_price * 1.01)].sort_values("strike").head(3)
            pe_iv = float(pe_otm["iv"].dropna().mean()) if len(pe_otm) else None
            ce_iv = float(ce_otm["iv"].dropna().mean()) if len(ce_otm) else None
            if pe_iv is not None and ce_iv is not None:
                feat.iv_skew = pe_iv - ce_iv
        except Exception:
            pass

    # ─── Days to expiry ──────────────────────────────────────────
    if expiry_date is not None:
        try:
            from datetime import date
            ed = expiry_date.date() if hasattr(expiry_date, "date") else expiry_date
            td = today or date.today()
            feat.days_to_expiry = max(0, (ed - td).days)
        except Exception:
            pass

    # ─── Theta Pressure ─────────────────────────────────────────
    # Sum of |theta| across the chain — high values = chain is bleeding
    # premium fast (close to expiry + high IV)
    if "theta" in df.columns:
        try:
            feat.theta_pressure = float(df["theta"].abs().sum())
        except Exception:
            pass
    elif feat.days_to_expiry is not None and feat.iv_atm is not None:
        # Approximate theta pressure when raw theta unavailable
        # Higher IV + fewer days = higher decay pressure
        approx = feat.iv_atm * 100 / max(feat.days_to_expiry, 1)
        feat.theta_pressure = round(approx, 2)
        feat.notes.append("theta_pressure approximated from iv_atm/dte")

    return feat


def iv_percentile_from_history(
    iv_history: pd.Series,
    *,
    current_iv: float,
    lookback_days: int = 252,
) -> Optional[float]:
    """Where does current IV sit in its 1-year distribution?

    Returns a value in [0, 100]: 0 = lowest IV in trailing year,
    100 = highest. None if insufficient history (< 30 bars).

    Aaryansinha uses this as an entry gate — "only sell premium when
    IV percentile > 50 (above-median vol)".
    """
    if iv_history is None or len(iv_history) < 30:
        return None
    recent = iv_history.tail(lookback_days).dropna()
    if len(recent) < 30:
        return None
    rank = (recent < current_iv).sum() / len(recent) * 100
    return float(rank)
