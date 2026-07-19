# M0 — Free-Data Plane + Feature/Label/CV Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared, model-agnostic foundation the four signal engines consume — a pluggable data provider (free-data first), a date-aware purged walk-forward CV, leakage-fixed labeling, a forward-return ranking labeler, and the Momentum feature builder — all on the existing `ml/` packages.

**Architecture:** Extend the existing `ml.data` / `ml.features` / `ml.training` packages. Add a `DataProvider` Protocol with a `FreeDataProvider` (wraps the existing pg-candle cache + yfinance/nselib) behind a single `load_ohlcv` facade, so a `TrueDataProvider` drops in later via config. Fix the two audit leakage bugs (triple-barrier ignores intra-bar high/low; embargo measured in rows not days) and add the absolute forward-return quantile labeler the LambdaRank rankers need. One Momentum feature builder is imported by both trainer and serving (train/serve parity by construction).

**Tech Stack:** Python 3.12, pandas, numpy, pytest. Existing: `ml.data.liquid_universe`, `ml.training.wfcv`, `ml.labeling.triple_barrier`, `ml.features.indicators`.

**Scope note:** This plan is M0 only and produces working, unit-testable libraries (no GPU, no model). The Momentum engine itself (trainer with TimesFM+Kronos+LGBM LambdaRank, serving engine, serving contract + smoke-load, risk engine, scheduler, frontend) is the **next plan (M1)**, which depends on this one. Spec: `docs/superpowers/specs/2026-06-15-quantx-4engine-mldl-design.md`.

---

## File Structure

- `src/backend/data/providers/base.py` (new) — `DataProvider` Protocol + `OHLCVRequest`.
- `src/backend/data/providers/free_provider.py` (new) — `FreeDataProvider` (pg cache → yfinance → nselib).
- `ml/data/data_loader.py` (new) — `load_ohlcv()` facade that selects the provider from `settings.DATA_PROVIDER`.
- `ml/data/liquid_universe.py` (modify) — add fail-loud `strict` mode (audit fix).
- `ml/labeling/triple_barrier.py` (modify) — add intra-bar `high`/`low` to the in-house path (audit fix).
- `ml/labeling/ranking_labels.py` (new) — absolute forward-return quantile relevance grades.
- `ml/training/purged_cv.py` (new) — date-aware purged walk-forward (day-sized embargo).
- `ml/features/momentum_features.py` (new) — the Momentum feature builder (train==serve).
- Tests mirror under `tests/ml/...`.

---

### Task 1: `DataProvider` Protocol + request type

**Files:**
- Create: `src/backend/data/providers/base.py`
- Test: `tests/ml/data/test_provider_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/data/test_provider_base.py
from datetime import date
from src.backend.data.providers.base import OHLCVRequest, DataProvider


def test_ohlcv_request_defaults_and_validation():
    req = OHLCVRequest(symbols=["RELIANCE"], start=date(2020, 1, 1), end=date(2021, 1, 1))
    assert req.freq == "eod"
    assert req.symbols == ["RELIANCE"]


def test_ohlcv_request_rejects_empty_symbols():
    import pytest
    with pytest.raises(ValueError):
        OHLCVRequest(symbols=[], start=date(2020, 1, 1), end=date(2021, 1, 1))


def test_dataprovider_is_protocol():
    # A class implementing the methods should satisfy the Protocol at runtime.
    class Dummy:
        name = "dummy"
        def get_ohlcv(self, req): ...
    assert isinstance(Dummy(), DataProvider)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/data/test_provider_base.py -v`
Expected: FAIL with `ModuleNotFoundError: src.backend.data.providers.base`

- [ ] **Step 3: Write minimal implementation**

```python
# src/backend/data/providers/base.py
"""Pluggable market-data provider interface (spec 2026-06-15 §3.0).

FreeDataProvider ships now; TrueDataProvider drops in later behind the
same Protocol. Engines depend on this interface, never a concrete vendor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Protocol, runtime_checkable

import pandas as pd

#: Supported bar frequencies. Free provider supports the daily set;
#: TrueData adds the intraday set later.
FREQS = ("eod", "week", "month", "1min", "3min", "5min", "15min", "30min", "60min", "tick")


@dataclass
class OHLCVRequest:
    """A request for OHLCV history.

    symbols: NSE trading symbols (no suffix), e.g. ["RELIANCE", "TCS"].
    start/end: inclusive date bounds.
    freq: one of FREQS. Free provider supports eod/week/month only.
    """

    symbols: List[str]
    start: date
    end: date
    freq: str = "eod"
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("OHLCVRequest.symbols must be non-empty")
        if self.freq not in FREQS:
            raise ValueError(f"freq {self.freq!r} not in {FREQS}")
        if self.end < self.start:
            raise ValueError("end must be >= start")


@runtime_checkable
class DataProvider(Protocol):
    """Every provider returns a tidy long OHLCV frame.

    Columns (exact): ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'].
    One row per (symbol, bar). Sorted by ['symbol', 'date'].
    MUST raise on total failure — never return an empty frame silently
    (audit: production_ohlcv empty-frame masking).
    """

    name: str

    def get_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ml/data/test_provider_base.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backend/data/providers/base.py tests/ml/data/test_provider_base.py
git commit -m "feat(data): add pluggable DataProvider Protocol + OHLCVRequest"
```

---

### Task 2: `FreeDataProvider` (pg cache → yfinance → nselib)

**Files:**
- Create: `src/backend/data/providers/free_provider.py`
- Test: `tests/ml/data/test_free_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/data/test_free_provider.py
from datetime import date
import pandas as pd
import pytest
from src.backend.data.providers.base import OHLCVRequest
from src.backend.data.providers.free_provider import FreeDataProvider


def _fake_cache(symbol, start, end):
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100},
        index=idx,
    )


def test_returns_tidy_long_frame(monkeypatch):
    p = FreeDataProvider(_loader=_fake_cache)
    df = p.get_ohlcv(OHLCVRequest(["RELIANCE", "TCS"], date(2020, 1, 1), date(2020, 1, 10)))
    assert list(df.columns) == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert set(df["symbol"]) == {"RELIANCE", "TCS"}
    assert df.sort_values(["symbol", "date"]).equals(df)  # already sorted


def test_raises_when_all_symbols_empty():
    p = FreeDataProvider(_loader=lambda s, a, b: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no OHLCV"):
        p.get_ohlcv(OHLCVRequest(["RELIANCE"], date(2020, 1, 1), date(2020, 1, 10)))


def test_intraday_freq_rejected():
    p = FreeDataProvider(_loader=_fake_cache)
    with pytest.raises(NotImplementedError, match="TrueData"):
        p.get_ohlcv(OHLCVRequest(["RELIANCE"], date(2020, 1, 1), date(2020, 1, 10), freq="5min"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/data/test_free_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: src.backend.data.providers.free_provider`

- [ ] **Step 3: Write minimal implementation**

```python
# src/backend/data/providers/free_provider.py
"""Free-data provider: pg candle cache → yfinance → nselib bhavcopy.

Daily/weekly/monthly only. Intraday + options require TrueData (later).
Fail-loud: raises if EVERY requested symbol comes back empty (audit fix).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Callable, Optional

import pandas as pd

from .base import OHLCVRequest

logger = logging.getLogger(__name__)

_DAILY_FREQS = {"eod": "1d", "week": "1wk", "month": "1mo"}


def _default_loader(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Load one symbol's daily OHLCV. Tries pg candles first, then yfinance.

    Returns a DatetimeIndex frame with columns open/high/low/close/volume,
    or an empty frame if nothing is available for this symbol.
    """
    # 1) pg candle cache (authoritative; corp-action adjusted)
    try:
        from ml.data.production_ohlcv import production_ohlcv  # noqa: PLC0415
        df = production_ohlcv(symbol, start=start, end=end)
        if df is not None and not df.empty:
            return df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    except Exception as e:  # noqa: BLE001
        logger.debug("pg candles miss for %s: %s", symbol, e)
    # 2) yfinance fallback
    try:
        import yfinance as yf  # noqa: PLC0415
        raw = yf.download(f"{symbol}.NS", start=str(start), end=str(end),
                          progress=False, auto_adjust=True)
        if raw is not None and not raw.empty:
            raw.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in raw.columns]
            return raw[["open", "high", "low", "close", "volume"]]
    except Exception as e:  # noqa: BLE001
        logger.debug("yfinance miss for %s: %s", symbol, e)
    return pd.DataFrame()


class FreeDataProvider:
    """DataProvider over free sources. Satisfies the base.DataProvider Protocol."""

    name = "free"

    def __init__(self, _loader: Optional[Callable[[str, date, date], pd.DataFrame]] = None):
        # _loader injectable for tests.
        self._loader = _loader or _default_loader

    def get_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        if req.freq not in _DAILY_FREQS:
            raise NotImplementedError(
                f"FreeDataProvider supports {sorted(_DAILY_FREQS)} only; "
                f"freq={req.freq!r} needs TrueData (enable DATA_PROVIDER=truedata)"
            )
        frames = []
        for sym in req.symbols:
            one = self._loader(sym, req.start, req.end)
            if one is None or one.empty:
                logger.warning("FreeDataProvider: no data for %s", sym)
                continue
            one = one.copy()
            one.index.name = "date"
            one = one.reset_index()
            one["symbol"] = sym
            frames.append(one)
        if not frames:
            raise RuntimeError(
                f"FreeDataProvider returned no OHLCV for any of {len(req.symbols)} "
                f"symbols ({req.start}..{req.end}) — check data source, not masking empty"
            )
        out = pd.concat(frames, ignore_index=True)
        out = out[["date", "symbol", "open", "high", "low", "close", "volume"]]
        return out.sort_values(["symbol", "date"]).reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ml/data/test_free_provider.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backend/data/providers/free_provider.py tests/ml/data/test_free_provider.py
git commit -m "feat(data): FreeDataProvider over pg cache + yfinance, fail-loud on empty"
```

---

### Task 3: `load_ohlcv` facade with provider selection

**Files:**
- Create: `ml/data/data_loader.py`
- Test: `tests/ml/data/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/data/test_data_loader.py
from datetime import date
import pandas as pd
from ml.data.data_loader import load_ohlcv, get_provider


def test_get_provider_defaults_to_free(monkeypatch):
    monkeypatch.delenv("DATA_PROVIDER", raising=False)
    assert get_provider().name == "free"


def test_load_ohlcv_uses_injected_provider():
    class Stub:
        name = "stub"
        def get_ohlcv(self, req):
            return pd.DataFrame({
                "date": pd.to_datetime(["2020-01-01"]), "symbol": ["RELIANCE"],
                "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [10],
            })
    df = load_ohlcv(["RELIANCE"], date(2020, 1, 1), date(2020, 1, 2), provider=Stub())
    assert len(df) == 1 and df.iloc[0]["symbol"] == "RELIANCE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/data/test_data_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: ml.data.data_loader`

- [ ] **Step 3: Write minimal implementation**

```python
# ml/data/data_loader.py
"""Single OHLCV entry point for trainers + serving engines (spec §3.5).

Provider chosen by settings.DATA_PROVIDER ("free" | "truedata"). Guarantees
the same tidy long frame regardless of backend, so train==serve data shape.
"""
from __future__ import annotations

import os
from datetime import date
from typing import List, Optional

import pandas as pd

from src.backend.data.providers.base import DataProvider, OHLCVRequest
from src.backend.data.providers.free_provider import FreeDataProvider


def get_provider() -> DataProvider:
    """Return the configured provider. Defaults to free."""
    name = os.environ.get("DATA_PROVIDER", "free").strip().lower()
    if name == "free":
        return FreeDataProvider()
    if name == "truedata":
        # Lazy import — only when explicitly enabled (creds required).
        from src.backend.data.providers.truedata_provider import TrueDataProvider  # noqa: PLC0415
        return TrueDataProvider()
    raise ValueError(f"unknown DATA_PROVIDER={name!r} (expected 'free' or 'truedata')")


def load_ohlcv(
    symbols: List[str],
    start: date,
    end: date,
    freq: str = "eod",
    provider: Optional[DataProvider] = None,
) -> pd.DataFrame:
    """Load OHLCV for symbols over [start, end]. See base.DataProvider for schema."""
    prov = provider or get_provider()
    return prov.get_ohlcv(OHLCVRequest(symbols=symbols, start=start, end=end, freq=freq))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ml/data/test_data_loader.py -v`
Expected: PASS (2 passed). Note: `test_get_provider_defaults_to_free` constructs a real `FreeDataProvider` (no network — construction only).

- [ ] **Step 5: Commit**

```bash
git add ml/data/data_loader.py tests/ml/data/test_data_loader.py
git commit -m "feat(data): load_ohlcv facade with DATA_PROVIDER selection"
```

---

### Task 4: Fix triple-barrier to use intra-bar high/low (audit leakage/correctness fix)

**Files:**
- Modify: `ml/labeling/triple_barrier.py:286-363` (the in-house `triple_barrier_events` loop)
- Test: `tests/ml/labeling/test_triple_barrier_intrabar.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/labeling/test_triple_barrier_intrabar.py
import numpy as np
from ml.labeling.triple_barrier import triple_barrier_events, TripleBarrierConfig


def test_intrabar_high_triggers_upper_even_if_close_does_not():
    # close never reaches the +2*ATR upper barrier, but bar-2 HIGH does.
    cfg = TripleBarrierConfig(profit_target_atr=2.0, stop_loss_atr=2.0,
                              vertical_barrier_days=3, min_atr_pct=0.0, asymmetric=False)
    close = np.array([100.0, 100.5, 101.0, 101.0, 101.0])
    high = np.array([100.0, 100.5, 103.0, 101.0, 101.0])   # bar 2 spikes to 103 (>= 100+2*1)
    low = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    atr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    labels, t1 = triple_barrier_events(close, atr, cfg, high=high, low=low)
    assert labels[0] == 1          # upper hit via intra-bar high
    assert t1[0] == 2              # at bar 2


def test_intrabar_low_triggers_stop():
    cfg = TripleBarrierConfig(profit_target_atr=2.0, stop_loss_atr=2.0,
                              vertical_barrier_days=3, min_atr_pct=0.0, asymmetric=False)
    close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    low = np.array([100.0, 100.0, 97.0, 100.0, 100.0])     # bar 2 drops to 97 (<= 100-2*1)
    atr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    labels, t1 = triple_barrier_events(close, atr, cfg, high=high, low=low)
    assert labels[0] == -1 and t1[0] == 2


def test_backward_compatible_without_high_low():
    # Old callers (close only) must still work — falls back to close as high/low.
    cfg = TripleBarrierConfig(min_atr_pct=0.0, vertical_barrier_days=3)
    close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    atr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    labels, t1 = triple_barrier_events(close, atr, cfg)
    assert labels.shape == close.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/labeling/test_triple_barrier_intrabar.py -v`
Expected: FAIL — `triple_barrier_events() got an unexpected keyword argument 'high'`

- [ ] **Step 3: Write the implementation**

Modify the in-house section of `triple_barrier_events` (the code after the mlfinlab fallback, currently lines ~328-363). Change the signature and the inner loop. Replace the function header and in-house loop:

```python
def triple_barrier_events(
    close: np.ndarray | pd.Series | Iterable[float],
    atr: np.ndarray | pd.Series | Iterable[float],
    cfg: Optional[TripleBarrierConfig] = None,
    *,
    high: np.ndarray | pd.Series | Iterable[float] | None = None,
    low: np.ndarray | pd.Series | Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
```

Keep the existing docstring; add one line: "If ``high``/``low`` are given, barrier touches use intra-bar extremes (correct); otherwise they fall back to ``close`` (legacy)." Keep the mlfinlab delegation as-is (it operates on close). In the in-house branch, after building `close_arr`/`atr_arr`, add:

```python
    high_arr = np.asarray(list(high), dtype=float) if high is not None else close_arr
    low_arr = np.asarray(list(low), dtype=float) if low is not None else close_arr
    if high_arr.shape != close_arr.shape or low_arr.shape != close_arr.shape:
        raise ValueError("high/low must match close shape")
```

Then replace the inner first-passage loop so it tests intra-bar extremes:

```python
        hit_at = i + vbd  # default: vertical barrier
        for j in range(1, vbd + 1):
            if high_arr[i + j] >= upper:
                labels[i] = 1
                hit_at = i + j
                break
            if low_arr[i + j] <= lower:
                labels[i] = -1
                hit_at = i + j
                break
        t1[i] = hit_at
```

(When high/low default to close, behavior is identical to today — backward compatible.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ml/labeling/test_triple_barrier_intrabar.py tests/ -k triple_barrier -v`
Expected: PASS (3 new + any existing triple-barrier tests still green)

- [ ] **Step 5: Commit**

```bash
git add ml/labeling/triple_barrier.py tests/ml/labeling/test_triple_barrier_intrabar.py
git commit -m "fix(labeling): triple-barrier uses intra-bar high/low (audit leakage fix)"
```

---

### Task 5: Absolute forward-return quantile labeler (for LambdaRank)

**Files:**
- Create: `ml/labeling/ranking_labels.py`
- Test: `tests/ml/labeling/test_ranking_labels.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/labeling/test_ranking_labels.py
import numpy as np
import pandas as pd
from ml.labeling.ranking_labels import forward_return_quantile_labels


def _panel():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    rows = []
    # 4 symbols; symbol A always best forward return, D always worst.
    prices = {"A": [10, 11, 13, 14], "B": [10, 10.5, 11, 11.2],
              "C": [10, 10.2, 10.3, 10.3], "D": [10, 9.8, 9.5, 9.4]}
    full_dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])
    for sym, px in prices.items():
        for d, p in zip(full_dates, px):
            rows.append({"date": d, "symbol": sym, "close": float(p)})
    return pd.DataFrame(rows)


def test_top_symbol_gets_highest_grade():
    out = forward_return_quantile_labels(_panel(), horizon=1, n_quantiles=4)
    first = out[out["date"] == pd.Timestamp("2020-01-01")].set_index("symbol")
    assert first.loc["A", "relevance"] == 3      # top quartile
    assert first.loc["D", "relevance"] == 0      # bottom quartile
    assert "fwd_return" in out.columns


def test_absolute_not_benchmark_relative():
    # fwd_return must equal raw close-to-close return, no index subtraction.
    out = forward_return_quantile_labels(_panel(), horizon=1, n_quantiles=4)
    a0 = out[(out.symbol == "A") & (out.date == pd.Timestamp("2020-01-01"))].iloc[0]
    assert abs(a0["fwd_return"] - (11 / 10 - 1)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/labeling/test_ranking_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: ml.labeling.ranking_labels`

- [ ] **Step 3: Write minimal implementation**

```python
# ml/labeling/ranking_labels.py
"""Absolute forward-return quantile labels for cross-sectional rankers.

Per spec §4.7: labels are ABSOLUTE (raw forward return), never relative to
a benchmark. Per rebalance date, symbols are bucketed into `n_quantiles`
by forward return; the bucket index is the LambdaRank relevance grade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return_quantile_labels(
    panel: pd.DataFrame,
    horizon: int,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """Return panel rows that have a full forward window, with labels.

    Args:
        panel: long frame with at least ['date', 'symbol', 'close'].
        horizon: forward window in bars (e.g. 20 trading days).
        n_quantiles: number of relevance buckets (10 = deciles).

    Returns:
        DataFrame ['date', 'symbol', 'fwd_return', 'relevance'] where
        relevance ∈ [0, n_quantiles-1] (higher = better forward return),
        computed per-date cross-sectionally. Rows without a full forward
        window are dropped.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    df = panel.sort_values(["symbol", "date"]).copy()
    df["fwd_close"] = df.groupby("symbol")["close"].shift(-horizon)
    df["fwd_return"] = df["fwd_close"] / df["close"] - 1.0
    df = df.dropna(subset=["fwd_return"])

    def _bucket(group: pd.DataFrame) -> pd.Series:
        # qcut with rank to handle ties; fall back to a single bucket when
        # a date has fewer symbols than quantiles.
        n = len(group)
        q = min(n_quantiles, n)
        if q < 2:
            return pd.Series(0, index=group.index)
        ranks = group["fwd_return"].rank(method="first")
        return pd.Series(
            pd.qcut(ranks, q, labels=False, duplicates="drop"), index=group.index
        )

    df["relevance"] = (
        df.groupby("date", group_keys=False).apply(_bucket).astype(int)
    )
    return df[["date", "symbol", "fwd_return", "relevance"]].reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ml/labeling/test_ranking_labels.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add ml/labeling/ranking_labels.py tests/ml/labeling/test_ranking_labels.py
git commit -m "feat(labeling): absolute forward-return quantile labels for rankers"
```

---

### Task 6: Date-aware purged walk-forward CV (day-sized embargo — audit fix)

**Files:**
- Create: `ml/training/purged_cv.py`
- Test: `tests/ml/training/test_purged_cv.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/training/test_purged_cv.py
import numpy as np
import pandas as pd
from ml.training.purged_cv import purged_walk_forward_by_date, PurgedCVConfig


def _dates():
    # 100 trading days, 3 symbols per day → 300 rows, date-pooled.
    days = pd.bdate_range("2020-01-01", periods=100)
    rows = [d for d in days for _ in range(3)]
    return pd.Series(pd.to_datetime(rows), name="date")


def test_no_symbol_date_splits_across_boundary():
    dates = _dates()
    cfg = PurgedCVConfig(n_folds=3, test_days=10, embargo_days=5, train_days=40)
    for train_idx, test_idx in purged_walk_forward_by_date(dates, cfg):
        train_dates = set(dates.iloc[train_idx])
        test_dates = set(dates.iloc[test_idx])
        assert train_dates.isdisjoint(test_dates)            # no shared date
        # embargo: gap of >= embargo_days trading days between train end & test start
        gap = (min(test_dates) - max(train_dates)).days
        assert gap >= 5


def test_embargo_measured_in_days_not_rows():
    # With 3 rows/day, a 5-day embargo must purge ~15 rows, not 5.
    dates = _dates()
    cfg = PurgedCVConfig(n_folds=2, test_days=10, embargo_days=5, train_days=40)
    folds = list(purged_walk_forward_by_date(dates, cfg))
    train_idx, test_idx = folds[0]
    last_train_day = dates.iloc[train_idx].max()
    first_test_day = dates.iloc[test_idx].min()
    n_business_between = len(pd.bdate_range(last_train_day, first_test_day)) - 2
    assert n_business_between >= 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/training/test_purged_cv.py -v`
Expected: FAIL with `ModuleNotFoundError: ml.training.purged_cv`

- [ ] **Step 3: Write minimal implementation**

```python
# ml/training/purged_cv.py
"""Date-aware purged walk-forward CV for cross-sectional panels.

Audit fix: ml.training.wfcv measures embargo in ROWS, which on a
date-pooled panel (N symbols per date) purges a fraction of one day.
This module groups by DATE and sizes train/test/embargo windows in
TRADING DAYS, so a single date's symbols never straddle the boundary and
the embargo actually covers the label horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import pandas as pd


@dataclass
class PurgedCVConfig:
    n_folds: int = 5
    test_days: int = 21        # ~1 month test window
    embargo_days: int = 20     # >= label horizon (audit: day-sized, not rows)
    train_days: int = 252 * 2  # expanding floor for fold 0

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if self.test_days < 1 or self.embargo_days < 0:
            raise ValueError("test_days >= 1 and embargo_days >= 0 required")


def purged_walk_forward_by_date(
    dates: pd.Series,
    cfg: PurgedCVConfig,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) positional arrays per fold.

    `dates` is the per-row date column of the panel (length = n_rows, with
    repeats for multiple symbols per date). Folds expand: train grows, test
    slides forward by `test_days` of unique trading days.
    """
    dser = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    uniq = np.array(sorted(dser.unique()))
    n_days = len(uniq)
    need = cfg.train_days + cfg.embargo_days + cfg.n_folds * cfg.test_days
    if need > n_days:
        raise ValueError(
            f"purged CV needs {need} trading days "
            f"(train {cfg.train_days} + embargo {cfg.embargo_days} + "
            f"{cfg.n_folds}x test {cfg.test_days}); panel has {n_days}"
        )
    # Map each unique day to row positions for fast membership.
    pos_by_day = {d: np.where(dser.values == d)[0] for d in uniq}

    def rows_for(days_slice) -> np.ndarray:
        return np.concatenate([pos_by_day[d] for d in days_slice]) if len(days_slice) else np.array([], int)

    for i in range(cfg.n_folds):
        test_start = cfg.train_days + cfg.embargo_days + i * cfg.test_days
        test_end = min(test_start + cfg.test_days, n_days)
        if test_end - test_start < 1:
            break
        train_end = test_start - cfg.embargo_days          # purge embargo days
        train_days = uniq[0:train_end]
        test_days = uniq[test_start:test_end]
        yield rows_for(train_days), rows_for(test_days)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ml/training/test_purged_cv.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add ml/training/purged_cv.py tests/ml/training/test_purged_cv.py
git commit -m "feat(training): date-aware purged walk-forward CV (day-sized embargo)"
```

---

### Task 7: Momentum feature builder (one builder, train==serve)

**Files:**
- Create: `ml/features/momentum_features.py`
- Test: `tests/ml/features/test_momentum_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/features/test_momentum_features.py
import numpy as np
import pandas as pd
from ml.features.momentum_features import build_momentum_features, MOMENTUM_FEATURE_ORDER


def _panel(n_days=300, symbols=("A", "B", "C")):
    days = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    rng = np.random.default_rng(0)
    for s_i, sym in enumerate(symbols):
        price = 100 + np.cumsum(rng.normal(0.1 * (s_i + 1), 1.0, n_days))
        for d, p in zip(days, price):
            rows.append({"date": d, "symbol": sym, "open": p, "high": p + 1,
                         "low": p - 1, "close": p, "volume": 1000 + s_i})
    return pd.DataFrame(rows)


def test_columns_match_feature_order():
    feats = build_momentum_features(_panel())
    for col in MOMENTUM_FEATURE_ORDER:
        assert col in feats.columns, f"missing {col}"
    assert {"date", "symbol"}.issubset(feats.columns)


def test_no_lookahead_in_returns():
    # ret_20d at row t must equal close[t]/close[t-20]-1 (uses only past).
    panel = _panel()
    feats = build_momentum_features(panel)
    a = feats[feats.symbol == "A"].sort_values("date").reset_index(drop=True)
    pa = panel[panel.symbol == "A"].sort_values("date").reset_index(drop=True)
    t = 100
    expected = pa["close"].iloc[t] / pa["close"].iloc[t - 20] - 1
    assert abs(a["ret_20d"].iloc[t] - expected) < 1e-9


def test_train_serve_parity_same_input_same_output():
    panel = _panel()
    f1 = build_momentum_features(panel)
    f2 = build_momentum_features(panel.copy())
    pd.testing.assert_frame_equal(f1, f2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/features/test_momentum_features.py -v`
Expected: FAIL with `ModuleNotFoundError: ml.features.momentum_features`

- [ ] **Step 3: Write minimal implementation**

```python
# ml/features/momentum_features.py
"""Momentum engine feature builder (spec §5.1).

THE single builder used by BOTH the trainer and the serving MomentumEngine
— importing this in both paths guarantees train/serve parity (audit: the
skew class of bugs). Absolute + intra-universe features only; relative-
strength vs index/sector is added by the caller when an index series is
supplied (kept as a feature; never in labels/outputs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Multi-horizon return windows (trading days).
_RET_WINDOWS = [5, 10, 21, 63, 126, 252]

MOMENTUM_FEATURE_ORDER = [
    *[f"ret_{w}d" for w in _RET_WINDOWS],
    "mom_consistency_63", "mom_accel", "vol_adj_mom_63",
    "dist_sma_50", "dist_sma_200", "above_high_63",
    "realized_vol_21", "drawdown_252",
    "rel_volume_21", "obv_slope_21",
    "xs_rank_ret_21", "xs_rank_ret_63",  # cross-sectional percentile ranks
]


def build_momentum_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum features for a long OHLCV panel.

    Args:
        panel: ['date','symbol','open','high','low','close','volume'].
    Returns:
        ['date','symbol', *MOMENTUM_FEATURE_ORDER]. Rows where the longest
        window is undefined contain NaN (caller drops for training).
    """
    df = panel.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", group_keys=False)

    for w in _RET_WINDOWS:
        df[f"ret_{w}d"] = g["close"].apply(lambda s, w=w: s / s.shift(w) - 1.0)

    # momentum quality
    daily_ret = g["close"].apply(lambda s: s.pct_change())
    df["__daily_ret"] = daily_ret
    df["mom_consistency_63"] = g["__daily_ret"].apply(
        lambda s: s.rolling(63).apply(lambda x: np.mean(x > 0), raw=True)
    )
    df["mom_accel"] = df["ret_21d"] - df["ret_63d"]
    df["realized_vol_21"] = g["__daily_ret"].apply(lambda s: s.rolling(21).std())
    df["vol_adj_mom_63"] = df["ret_63d"] / (df["realized_vol_21"] * np.sqrt(63) + 1e-9)

    # trend
    df["dist_sma_50"] = g["close"].apply(lambda s: s / s.rolling(50).mean() - 1.0)
    df["dist_sma_200"] = g["close"].apply(lambda s: s / s.rolling(200).mean() - 1.0)
    df["above_high_63"] = g["close"].apply(
        lambda s: (s >= s.rolling(63).max()).astype(float)
    )
    df["drawdown_252"] = g["close"].apply(lambda s: s / s.rolling(252).max() - 1.0)

    # volume confirmation
    df["rel_volume_21"] = g["volume"].apply(lambda s: s / (s.rolling(21).mean() + 1e-9))
    obv = g.apply(lambda x: (np.sign(x["close"].diff()).fillna(0) * x["volume"]).cumsum())
    df["__obv"] = obv.reset_index(level=0, drop=True) if isinstance(obv, pd.Series) else obv
    df["obv_slope_21"] = g["__obv"].apply(lambda s: s.diff(21) / (s.abs().rolling(21).mean() + 1e-9)) \
        if "__obv" in df else 0.0

    # cross-sectional percentile ranks (intra-universe; NOT vs benchmark)
    df["xs_rank_ret_21"] = df.groupby("date")["ret_21d"].rank(pct=True)
    df["xs_rank_ret_63"] = df.groupby("date")["ret_63d"].rank(pct=True)

    cols = ["date", "symbol"] + MOMENTUM_FEATURE_ORDER
    return df[cols].reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ml/features/test_momentum_features.py -v`
Expected: PASS (3 passed). If `obv_slope_21` groupby-apply shape errors, simplify to per-symbol transform; the test only requires the column to exist and parity to hold.

- [ ] **Step 5: Commit**

```bash
git add ml/features/momentum_features.py tests/ml/features/test_momentum_features.py
git commit -m "feat(features): momentum feature builder (train==serve parity)"
```

---

### Task 8: Fix `liquid_universe` silent fallback (audit fail-loud)

**Files:**
- Modify: `ml/data/liquid_universe.py` (add `strict` to `LiquidUniverseConfig`, raise instead of NIFTY-200 fallback when `strict=True`)
- Test: `tests/ml/data/test_liquid_universe_strict.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/data/test_liquid_universe_strict.py
import pytest
from ml.data.liquid_universe import liquid_universe, LiquidUniverseConfig


def test_strict_raises_when_source_unavailable(monkeypatch):
    # Force the underlying fetch to yield nothing; strict must raise, not
    # silently return NIFTY_200_FALLBACK (audit fix).
    import ml.data.liquid_universe as lu
    monkeypatch.setattr(lu, "_fetch_liquidity_table", lambda cfg: __import__("pandas").DataFrame())
    with pytest.raises(RuntimeError, match="liquid universe"):
        liquid_universe(LiquidUniverseConfig(top_n=50, strict=True))
```

> Note: read `ml/data/liquid_universe.py` first to find the exact internal
> fetch function name; if it differs from `_fetch_liquidity_table`, patch the
> real one. The test asserts the *behavior* (strict → raise on empty).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/data/test_liquid_universe_strict.py -v`
Expected: FAIL — `LiquidUniverseConfig` has no `strict` field (TypeError) or no raise.

- [ ] **Step 3: Write the implementation**

In `ml/data/liquid_universe.py`: add `strict: bool = False` to `LiquidUniverseConfig`. In `liquid_universe()`, locate the branch that currently falls back to `NIFTY_200_FALLBACK` on empty/failed fetch and wrap it:

```python
    if table is None or table.empty:
        if cfg.strict:
            raise RuntimeError(
                "liquid universe build failed: data source returned nothing "
                "and strict=True (refusing silent NIFTY-200 fallback — audit fix)"
            )
        logger.warning("liquid universe: source empty — using NIFTY_200_FALLBACK")
        return NIFTY_200_FALLBACK[: cfg.top_n]
```

Trainers will pass `strict=True`; ad-hoc/dev callers keep the lenient default.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ml/data/test_liquid_universe_strict.py tests/ -k liquid_universe -v`
Expected: PASS (new test + existing liquid_universe tests still green)

- [ ] **Step 5: Commit**

```bash
git add ml/data/liquid_universe.py tests/ml/data/test_liquid_universe_strict.py
git commit -m "fix(data): liquid_universe strict mode fails loud (audit fix)"
```

---

## Self-Review

**1. Spec coverage (vs spec §3–§4):**
- §3.0 pluggable provider → Tasks 1–3 ✓ · §3.4 universe fail-loud → Task 8 ✓ · §3.5 loader → Task 3 ✓
- §4.1 feature factory (one builder, parity) → Task 7 ✓ · §4.2 labeling (intra-bar fix + ranking labels) → Tasks 4, 5 ✓
- §4.3 purged+embargoed walk-forward, day-sized → Task 6 ✓
- **Deferred to M1 (noted):** §4.4 serving contract + smoke-load, §4.5 registry/gate plumbing fix, §4.6 risk engine, §4.7 output `style` enum, §4.8 unified-runner trainer wiring — all first *exercised* by a real model, so they live in the Momentum (M1) plan.

**2. Placeholder scan:** No TBD/TODO; every code step has complete code. The two *modify* tasks (4, 8) reference exact existing functions; Task 8 instructs reading the file to confirm the internal fetch name before patching (behavioral test pins the contract).

**3. Type consistency:** `OHLCVRequest`/`DataProvider` (Task 1) used identically in Tasks 2–3. Long-frame schema `['date','symbol','open','high','low','close','volume']` consistent across Tasks 1, 2, 3, 7. `MOMENTUM_FEATURE_ORDER` defined once (Task 7) and is the single source the M1 trainer + serving will import. `PurgedCVConfig`/`TripleBarrierConfig` names match their modules.

**Gaps:** none blocking M0. M1 plan will open with the serving contract + smoke-load so no model promotes before round-tripping through its production predictor.
