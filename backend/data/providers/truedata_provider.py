"""TrueData market-data provider (spec 2026-06-15 §3.1) — the drop-in vendor.

Implements the DataProvider Protocol over the `truedata` package's historical
REST service (TD_hist). Enabled via DATA_PROVIDER=truedata once TRUEDATA_LOGIN
/ TRUEDATA_PASSWORD are set. Returns the same tidy long frame as FreeDataProvider
(['date','symbol','open','high','low','close','volume']) so train==serve data
shape is identical regardless of backend.

Trial caveats (sandbox, expires 2026-06-26): 50-symbol cap, bar history = 15
days, tick = 2 days, EOD = 3 years. A paid subscription is required for the
deep bar history the Intraday engine needs to TRAIN.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, time
from typing import List, Optional

import pandas as pd

from .base import OHLCVRequest

logger = logging.getLogger(__name__)

# OHLCVRequest.freq -> TrueData bar_size string.
_FREQ_MAP = {
    "eod": "eod", "week": "week", "month": "month",
    "1min": "1 min", "3min": "3 mins", "5min": "5 mins",
    "15min": "15 mins", "30min": "30 mins", "60min": "60 mins",
}
_OHLCV = ["open", "high", "low", "close", "volume"]


class TrueDataProvider:
    """DataProvider over TrueData's historical service. Satisfies base.DataProvider."""

    name = "truedata"

    def __init__(self, login: Optional[str] = None, password: Optional[str] = None,
                 _hist=None):
        self._login = login or os.environ.get("TRUEDATA_LOGIN")
        self._password = password or os.environ.get("TRUEDATA_PASSWORD")
        self._hist = _hist  # injectable for tests
        if not self._hist and not (self._login and self._password):
            raise RuntimeError(
                "TrueDataProvider needs TRUEDATA_LOGIN + TRUEDATA_PASSWORD "
                "(set in .env or pass explicitly)"
            )

    def _client(self):
        if self._hist is None:
            from truedata import TD_hist  # noqa: PLC0415 — lazy, optional dep
            self._hist = TD_hist(self._login, self._password)
        return self._hist

    def get_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        if req.freq == "tick":
            raise NotImplementedError(
                "tick is not an OHLCV bar; use a dedicated tick accessor"
            )
        if req.freq not in _FREQ_MAP:
            raise ValueError(f"unsupported freq {req.freq!r}")
        bar = _FREQ_MAP[req.freq]
        td = self._client()
        start_dt = datetime.combine(req.start, time.min)
        end_dt = datetime.combine(req.end, time.max)

        frames: List[pd.DataFrame] = []
        for sym in req.symbols:
            try:
                raw = td.get_historic_data(
                    sym, start_time=start_dt, end_time=end_dt, bar_size=bar,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("TrueData fetch failed for %s: %s", sym, e)
                continue
            if raw is None or len(raw) == 0:
                logger.warning("TrueData: no data for %s", sym)
                continue
            df = raw.copy()
            # EOD uses 'volume', intraday uses 'Volume' — normalize. 'timestamp' -> 'date'.
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"timestamp": "date"})
            if not {"date", *_OHLCV}.issubset(df.columns):
                logger.warning("TrueData %s missing cols %s", sym,
                               {"date", *_OHLCV} - set(df.columns))
                continue
            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = sym
            frames.append(df[["date", "symbol", *_OHLCV]])

        if not frames:
            raise RuntimeError(
                f"TrueData returned no OHLCV for any of {len(req.symbols)} symbols "
                f"({req.start}..{req.end}, {req.freq}) — check creds / trial limits "
                f"(50-symbol cap, bar=15d, eod=3y), not masking empty"
            )
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["symbol", "date"]).reset_index(drop=True)
