"""Historical / Realized Volatility (HV/RV) for an underlying.

Options traders read IV (implied, forward-looking) against HV (realized,
backward-looking) to judge whether options are rich or cheap:
    IV > HV  -> options pricing a premium over what the stock has actually
                delivered; favours option WRITERS (sell premium).
    IV < HV  -> options cheap relative to realized movement; favours BUYERS.

The HV math here mirrors EXACTLY ``ai/strategy/indicators._realized_volatility``:
annualised std of close-to-close log returns × sqrt(252) × 100. The compute
already exists in the ML indicator registry (``volatility_20`` / ``volatility_60``)
but was never surfaced to the F&O panel — this service exposes it for the
option-chain snapshot.

Cost: pure deterministic math over real daily closes (0 LLM tokens). Honest-empty:
returns ``None`` when there aren't enough closes to fill the smallest window.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# Trading days per year — same annualisation constant as the ML indicators.
_TRADING_DAYS = 252


def _annualized_hv(closes: pd.Series, window: int) -> Optional[float]:
    """Annualised realized volatility (%) over ``window`` close-to-close log
    returns. Mirrors ``indicators._realized_volatility``: std of log returns ×
    sqrt(252) × 100, read at the latest bar. Returns None if not enough data."""
    if closes is None or len(closes) < window + 1:
        return None
    log_ret = np.log(closes / closes.shift(1))
    vol = log_ret.rolling(window=window, min_periods=window).std() * np.sqrt(_TRADING_DAYS) * 100
    latest = vol.iloc[-1]
    if latest is None or (isinstance(latest, float) and np.isnan(latest)):
        return None
    return round(float(latest), 2)


def compute_hv(closes: Sequence[float], windows: Sequence[int] = (10, 20, 30)) -> Optional[Dict[str, Any]]:
    """Pure-function core: annualised HV for each window over a close series.

    Separated from ``realized_vol`` so the math is unit-testable without any
    market-provider / network access. Returns the same shape, or None when the
    series is too short for the smallest window. ``closes`` oldest-to-newest.
    """
    s = pd.Series([float(c) for c in closes], dtype=float)
    if (s <= 0).any():
        # log-returns need strictly positive prices; bail honestly rather than
        # producing NaN-laced volatility.
        return None
    valid = sorted(w for w in windows if len(s) >= w + 1)
    if not valid:
        return None

    hv: Dict[str, float] = {}
    for w in windows:
        v = _annualized_hv(s, w)
        if v is not None:
            hv[str(w)] = v
    if not hv:
        return None

    # "latest_hv" = the 20-window if present, else the largest computed window
    # (the most stable estimate available). This is what the IV-vs-HV teach
    # line compares against.
    if "20" in hv:
        latest_hv = hv["20"]
    else:
        latest_hv = hv[str(max(int(k) for k in hv))]

    return {
        "hv": hv,
        "latest_hv": latest_hv,
        "note": "Annualized realized (historical) volatility from daily close-to-close log returns.",
    }


def realized_vol(symbol: str, windows: Sequence[int] = (10, 20, 30)) -> Optional[Dict[str, Any]]:
    """Annualised HV for ``symbol`` across ``windows`` (default 10/20/30 day).

    Reads daily closes via the same market-data provider ``why_moving`` uses,
    then defers to ``compute_hv`` for the deterministic math. Honest-empty:
    returns None when the provider is offline or there aren't enough closes.
    """
    sym = symbol.strip().upper()

    # Need at least max(window)+1 closes; pull a 6-month daily window to be safe.
    try:
        from ...data.market import get_market_data_provider
        mp = get_market_data_provider()
        df = mp.get_historical(sym, period="6mo", interval="1d")
    except Exception as e:
        logger.debug("realized_vol: provider/historical failed for %s: %s", sym, e)
        return None

    if df is None or len(df) == 0:
        return None

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        return None

    closes = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(closes) < min(windows) + 1:
        return None

    return compute_hv(closes.tolist(), windows)


__all__ = ["realized_vol", "compute_hv"]
