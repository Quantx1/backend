"""In-memory tick -> rolling N-minute OHLCV bar aggregator (per symbol).

Fills the gap that ``PriceService.price_cache`` (latest-tick-only) leaves:
the intraday scanner needs a SERIES of completed bars. ``feed`` returns the
just-closed bar when a tick rolls into a new window, so a consumer can react
on bar close. Volume is per-bar (delta of the broker's cumulative day volume)."""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque, Dict, Optional

import pandas as pd


class IntradayBarAggregator:
    def __init__(self, interval_min: int = 5, max_bars: int = 80):
        self.interval = interval_min
        self.max_bars = max_bars
        self._cur: Dict[str, dict] = {}
        self._bars: Dict[str, Deque[dict]] = {}
        self._last_cum: Dict[str, float] = {}

    def _floor(self, ts: datetime) -> datetime:
        return ts.replace(second=0, microsecond=0,
                          minute=(ts.minute // self.interval) * self.interval)

    def feed(self, symbol: str, price: float, cum_volume: float, ts: datetime) -> Optional[dict]:
        """Add a tick. Returns the just-closed bar dict if this tick rolled the
        window, else None. ``cum_volume`` is the broker's cumulative day volume."""
        bucket = self._floor(ts)
        last_cum = self._last_cum.get(symbol)
        vol_delta = 0.0 if last_cum is None else max(0.0, float(cum_volume) - last_cum)
        self._last_cum[symbol] = float(cum_volume)

        cur = self._cur.get(symbol)
        if cur is None:
            self._cur[symbol] = {"ts": bucket, "open": price, "high": price,
                                 "low": price, "close": price, "volume": vol_delta}
            return None

        if bucket > cur["ts"]:
            self._bars.setdefault(symbol, deque(maxlen=self.max_bars)).append(cur)
            self._cur[symbol] = {"ts": bucket, "open": price, "high": price,
                                 "low": price, "close": price, "volume": vol_delta}
            return cur

        cur["high"] = max(cur["high"], price)
        cur["low"] = min(cur["low"], price)
        cur["close"] = price
        cur["volume"] += vol_delta
        return None

    def frame(self, symbol: str) -> Optional[pd.DataFrame]:
        """Completed bars as an OHLCV DataFrame with an IST DatetimeIndex
        (the shape the intraday scanner expects). None if no completed bars."""
        bars = list(self._bars.get(symbol, ()))
        if not bars:
            return None
        df = pd.DataFrame(bars).set_index("ts")
        df.index.name = None
        return df[["open", "high", "low", "close", "volume"]]
