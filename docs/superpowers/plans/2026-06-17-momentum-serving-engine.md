# Momentum Serving Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the already-trained `momentum_lambdarank` model a live, frontend-visible engine with its own distinct output, establishing the reusable contract the other 3 style engines reuse.

**Architecture:** A per-style output schema (`StyleSignal`/`MomentumSignal`) + a separate ATR risk engine (`derive_levels`) + a `MomentumEngine` that resolves the model, builds the 18-feature contract per symbol, predicts, ranks cross-sectionally, and attaches risk levels + an on-demand 60s-cached endpoint + a dedicated frontend page.

**Tech Stack:** Python 3.12, FastAPI, LightGBM, pandas; Next.js/TypeScript frontend. CPU-only (no GPU/forecasters at serve).

**Spec:** `docs/superpowers/specs/2026-06-17-quantx-momentum-serving-engine-design.md`

**Run tests with:** `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest <path> -v`
**Full gate:** `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest -q` → expect `872 passed` (873+ after new tests) · `lint-imports` → `1 kept, 0 broken`.

---

## File Structure

- Create `backend/ai/signals/style_types.py` — `Style` enum, `StyleSignal`, `MomentumSignal`. (New file, not editing the v1 `types.py`, to keep the v1 ensemble signal untouched.)
- Create `backend/trading/risk_engine.py` — `derive_levels` + `RISK_PARAMS`.
- Create `backend/ai/signals/engines/__init__.py` + `backend/ai/signals/engines/momentum.py` — `MomentumEngine`.
- Modify `backend/api/signals_routes.py` — add `GET /api/signals/momentum` + 60s cache.
- Modify `frontend/lib/api.ts` — add `signals.getMomentum()`.
- Modify `frontend/components/signals/categories.ts` — add `'momentum'` category.
- Modify `frontend/components/signals/CategorySignalsPage.tsx` — per-style data source for momentum.
- Create `frontend/app/signals/momentum/page.tsx`.
- Tests: `tests/trading/test_risk_engine.py`, `tests/ml/serving/test_momentum_engine.py`, `tests/api/test_momentum_signals_route.py`.

---

### Task 1: Per-style output schema

**Files:**
- Create: `backend/ai/signals/style_types.py`
- Test: `tests/ml/serving/test_style_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/serving/test_style_types.py
from backend.ai.signals.style_types import Style, StyleSignal, MomentumSignal


def test_momentum_signal_is_style_signal_and_serializes():
    sig = MomentumSignal(
        symbol="RELIANCE", style=Style.MOMENTUM, rank=1, percentile=1.0,
        confidence=100.0, direction="BUY", entry_price=100.0, stop_loss=85.0,
        target=130.0, risk_reward=2.0, reasons=["top of book"],
        expected_return=0.0369, top_decile_prob=1.0,
    )
    assert isinstance(sig, StyleSignal)
    assert sig.style == Style.MOMENTUM
    d = sig.to_dict()
    assert d["style"] == "momentum"
    assert d["symbol"] == "RELIANCE"
    assert d["expected_return"] == 0.0369
    assert d["top_decile_prob"] == 1.0
    assert d["risk_reward"] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/ml/serving/test_style_types.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.ai.signals.style_types`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/ai/signals/style_types.py
"""Per-style signal output schema for the 4-engine serving layer.

Each style engine (momentum/swing/positional/intraday) emits its own
StyleSignal subclass with style-specific fields (spec §4.7). Kept separate
from the v1 ensemble `GeneratedSignal` in types.py — this is additive.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List


class Style(str, Enum):
    MOMENTUM = "momentum"
    # SWING / POSITIONAL / INTRADAY added when their engines land.


@dataclass
class StyleSignal:
    """Base output every style engine produces. Levels come from the risk
    engine; the engine fills rank/percentile/confidence."""
    symbol: str
    style: Style
    rank: int
    percentile: float
    confidence: float
    direction: str
    entry_price: float
    stop_loss: float
    target: float
    risk_reward: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["style"] = self.style.value
        return d


@dataclass
class MomentumSignal(StyleSignal):
    """Momentum ranker output (spec §4.7)."""
    expected_return: float = 0.0
    top_decile_prob: float = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/ml/serving/test_style_types.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/ai/signals/style_types.py tests/ml/serving/test_style_types.py
git commit -m "feat(signals): per-style output schema (StyleSignal + MomentumSignal)"
```

---

### Task 2: ATR risk engine

**Files:**
- Create: `backend/trading/risk_engine.py`
- Test: `tests/trading/test_risk_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/trading/test_risk_engine.py
import pytest
from backend.ai.signals.style_types import Style
from backend.trading.risk_engine import derive_levels, RISK_PARAMS


def test_momentum_buy_levels_from_atr():
    # ref=100, atr=10, momentum mults (sl=1.5, tp=3.0)
    entry, sl, target, rr = derive_levels("BUY", ref_price=100.0, atr=10.0, style=Style.MOMENTUM)
    assert entry == 100.0
    assert sl == 85.0      # 100 - 1.5*10
    assert target == 130.0 # 100 + 3.0*10
    assert rr == pytest.approx(2.0)  # (130-100)/(100-85)


def test_momentum_params_present():
    assert Style.MOMENTUM in RISK_PARAMS
    sl_mult, tp_mult = RISK_PARAMS[Style.MOMENTUM]
    assert (sl_mult, tp_mult) == (1.5, 3.0)


def test_rejects_nonpositive_atr():
    with pytest.raises(ValueError):
        derive_levels("BUY", ref_price=100.0, atr=0.0, style=Style.MOMENTUM)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/trading/test_risk_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.trading.risk_engine`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/trading/risk_engine.py
"""Per-style ATR risk engine (spec §4.6). Models emit expected_return +
confidence; this turns a reference price + ATR into entry/SL/target levels.

Separate from backend/trading/risk.py (position sizing / day-loss / exposure
limits) — different responsibility, no overlap. Pure functions, no I/O.
"""
from __future__ import annotations

from typing import Dict, Tuple

from backend.ai.signals.style_types import Style

#: style -> (stop_loss_atr_mult, take_profit_atr_mult)
RISK_PARAMS: Dict[Style, Tuple[float, float]] = {
    Style.MOMENTUM: (1.5, 3.0),
}


def derive_levels(
    direction: str, ref_price: float, atr: float, style: Style
) -> Tuple[float, float, float, float]:
    """Return (entry, stop_loss, target, risk_reward). BUY-only for now
    (the style engines are long-only rankers). Raises ValueError on bad input."""
    if atr <= 0:
        raise ValueError(f"atr must be > 0, got {atr}")
    if ref_price <= 0:
        raise ValueError(f"ref_price must be > 0, got {ref_price}")
    if style not in RISK_PARAMS:
        raise ValueError(f"no risk params for style {style}")
    if direction != "BUY":
        raise ValueError(f"only BUY supported, got {direction}")

    sl_mult, tp_mult = RISK_PARAMS[style]
    entry = float(ref_price)
    stop_loss = round(entry - sl_mult * atr, 2)
    target = round(entry + tp_mult * atr, 2)
    if entry <= stop_loss:
        raise ValueError("degenerate levels: entry <= stop_loss")
    risk_reward = round((target - entry) / (entry - stop_loss), 2)
    return entry, stop_loss, target, risk_reward
```

- [ ] **Step 4: Run test to verify it passes**

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/trading/test_risk_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/trading/risk_engine.py tests/trading/test_risk_engine.py
git commit -m "feat(trading): per-style ATR risk engine (derive_levels)"
```

---

### Task 3: MomentumEngine serving class

**Files:**
- Create: `backend/ai/signals/engines/__init__.py` (empty package marker with docstring)
- Create: `backend/ai/signals/engines/momentum.py`
- Test: `tests/ml/serving/test_momentum_engine.py`

The engine takes injectable deps for testability (default = real registry/disk + cached_universe + load_ohlcv), mirroring the `FreeDataProvider(_loader=...)` pattern.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/serving/test_momentum_engine.py
import numpy as np
import pandas as pd
from backend.ai.signals.engines.momentum import MomentumEngine
from backend.ai.signals.style_types import MomentumSignal


class _FakeBooster:
    """Returns a deterministic score per row = mean of the feature row, so
    symbols with higher features rank higher."""
    def predict(self, X):
        arr = np.asarray(X, dtype=float)
        return arr.mean(axis=1)


def _ramp_ohlcv(sym_offset: float) -> pd.DataFrame:
    # 400 trading days, gently rising price so momentum features are finite.
    idx = pd.date_range("2022-01-01", periods=400, freq="B")
    base = 100.0 + sym_offset
    close = base + np.linspace(0, 40 + sym_offset, 400)
    return pd.DataFrame({
        "date": idx, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": 1_000_000,
    })


def test_ranks_and_levels_and_outputs():
    syms = ["AAA", "BBB", "CCC"]
    offsets = {"AAA": 0.0, "BBB": 5.0, "CCC": 10.0}

    def fake_loader(symbols, start, end, freq="eod", provider=None):
        frames = []
        for s in symbols:
            df = _ramp_ohlcv(offsets[s]); df["symbol"] = s
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    eng = MomentumEngine(_booster=_FakeBooster(),
                         _feature_order=None,  # use the real MOMENTUM_FEATURE_ORDER
                         _universe=lambda limit=None: syms,
                         _load_ohlcv=fake_loader)
    sigs = eng.run(top_n=3)
    assert len(sigs) == 3
    assert all(isinstance(s, MomentumSignal) for s in sigs)
    # rank is a 1..N permutation, percentile in [0,1], descending by rank
    ranks = [s.rank for s in sigs]
    assert sorted(ranks) == [1, 2, 3]
    assert sigs[0].rank == 1 and sigs[0].percentile == 1.0
    # every signal has trade levels + momentum outputs
    for s in sigs:
        assert s.direction == "BUY"
        assert s.stop_loss < s.entry_price < s.target
        assert s.risk_reward > 0
        assert -1.0 <= s.expected_return <= 1.0
        assert 0.0 <= s.top_decile_prob <= 1.0


def test_honest_empty_when_model_missing():
    def boom(*a, **k):
        raise LookupError("no prod version")
    eng = MomentumEngine(_model_loader=boom, _universe=lambda limit=None: ["AAA"],
                         _load_ohlcv=lambda *a, **k: pd.DataFrame())
    sigs = eng.run()
    assert sigs == []
    assert eng.status == "model_not_loaded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/ml/serving/test_momentum_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.ai.signals.engines`

- [ ] **Step 3: Write minimal implementation**

Create `backend/ai/signals/engines/__init__.py`:

```python
"""Per-style serving engines (momentum first; swing/positional/intraday reuse)."""
```

Create `backend/ai/signals/engines/momentum.py`:

```python
"""MomentumEngine — serves the trained momentum_lambdarank ranker.

Resolves the model (registry-first, disk fallback), builds the 18-feature
serve contract per symbol, predicts, ranks cross-sectionally, and attaches
ATR-derived levels. CPU-only — no forecaster columns at serve time
(MOMENTUM_FEATURE_ORDER == feature_order.json). Honest-empty if the model
is missing (no heuristic fallback).
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from backend.ai.signals.style_types import MomentumSignal, Style
from backend.trading.risk_engine import derive_levels

logger = logging.getLogger(__name__)

_MODEL_NAME = "momentum_lambdarank"
_ROOT = Path(__file__).resolve().parents[4]  # engines -> signals -> ai -> backend -> repo
_DISK_DIR = _ROOT / "artifacts" / "models" / "momentum_lambdarank"


def _default_model_loader():
    """Registry-first; disk fallback. Returns (booster, feature_order, decile_spread)."""
    import lightgbm as lgb  # noqa: PLC0415
    txt: Optional[Path] = None
    fo_path: Optional[Path] = None
    metrics_path: Optional[Path] = None
    try:
        from backend.ai.registry import get_registry  # noqa: PLC0415
        d = get_registry().resolve(_MODEL_NAME)
        txt = d / "momentum_lambdarank.txt"
        fo_path = d / "feature_order.json"
        metrics_path = d / "metrics.json"
    except Exception as exc:  # registry miss → disk fallback
        logger.info("momentum registry resolve failed (%s); trying disk", exc)
    if txt is None or not txt.exists():
        txt = _DISK_DIR / "momentum_lambdarank.txt"
        fo_path = _DISK_DIR / "feature_order.json"
        metrics_path = _DISK_DIR / "metrics.json"
    if not txt.exists():
        raise LookupError(f"momentum model artifact not found at {txt}")
    booster = lgb.Booster(model_file=str(txt))
    feature_order = json.loads(fo_path.read_text()) if fo_path and fo_path.exists() \
        else list(booster.feature_name())
    decile = 0.0369
    if metrics_path and metrics_path.exists():
        decile = float(json.loads(metrics_path.read_text()).get("decile_spread_mean", decile))
    return booster, feature_order, decile


def _atr14(df: pd.DataFrame) -> float:
    """Simple ATR(14) on the latest bar from a single-symbol OHLCV frame."""
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else 0.0


class MomentumEngine:
    def __init__(
        self,
        *,
        _booster=None,
        _feature_order: Optional[List[str]] = None,
        _decile_spread: float = 0.0369,
        _model_loader: Optional[Callable] = None,
        _universe: Optional[Callable] = None,
        _load_ohlcv: Optional[Callable] = None,
    ):
        self.status = "ok"
        self._booster = _booster
        self._feature_order = _feature_order
        self._decile = _decile_spread
        self._model_loader = _model_loader or _default_model_loader
        if _universe is None:
            from ml.training.trainers.momentum_lambdarank import cached_universe  # noqa: PLC0415
            _universe = cached_universe
        self._universe = _universe
        if _load_ohlcv is None:
            from ml.data.data_loader import load_ohlcv  # noqa: PLC0415
            _load_ohlcv = load_ohlcv
        self._load_ohlcv = _load_ohlcv

    def _ensure_model(self) -> bool:
        if self._booster is not None:
            if self._feature_order is None:
                from ml.features.momentum_features import MOMENTUM_FEATURE_ORDER  # noqa: PLC0415
                self._feature_order = list(MOMENTUM_FEATURE_ORDER)
            return True
        try:
            self._booster, self._feature_order, self._decile = self._model_loader()
            return True
        except Exception as exc:
            logger.warning("MomentumEngine model load failed: %s", exc)
            self.status = "model_not_loaded"
            return False

    def run(self, top_n: int = 20, universe_limit: Optional[int] = None) -> List[MomentumSignal]:
        from ml.features.momentum_features import build_momentum_features, MOMENTUM_WARMUP_BARS  # noqa: PLC0415
        if not self._ensure_model():
            return []
        syms = self._universe(limit=universe_limit)
        if not syms:
            self.status = "no_data"
            return []
        end = date.today()
        start = end - timedelta(days=int(MOMENTUM_WARMUP_BARS * 2.2) + 30)
        panel = self._load_ohlcv(syms, start, end, freq="eod")
        if panel is None or panel.empty:
            self.status = "no_data"
            return []

        rows = []  # (symbol, score, close, atr)
        for sym, g in panel.groupby("symbol"):
            g = g.sort_values("date")
            if len(g) < MOMENTUM_WARMUP_BARS:
                continue
            try:
                feats = build_momentum_features(g[["date", "symbol", "open", "high", "low", "close", "volume"]])
                feats = feats.dropna(subset=self._feature_order)
                if feats.empty:
                    continue
                last = feats.iloc[[-1]][self._feature_order]
                score = float(self._booster.predict(last)[0])
                atr = _atr14(g)
                close = float(g["close"].iloc[-1])
                if atr > 0 and close > 0:
                    rows.append((sym, score, close, atr))
            except Exception as exc:
                logger.debug("momentum feature/predict failed for %s: %s", sym, exc)

        if not rows:
            self.status = "no_data"
            return []

        rows.sort(key=lambda r: r[1], reverse=True)
        n = len(rows)
        out: List[MomentumSignal] = []
        for i, (sym, score, close, atr) in enumerate(rows[:top_n]):
            rank = i + 1
            percentile = 1.0 if n == 1 else round(1.0 - (rank - 1) / (n - 1), 4)
            expected_return = round(self._decile * (2 * percentile - 1), 4)
            top_decile_prob = round(percentile, 4)
            confidence = round(percentile * 100, 1)
            entry, sl, target, rr = derive_levels("BUY", close, atr, Style.MOMENTUM)
            out.append(MomentumSignal(
                symbol=sym, style=Style.MOMENTUM, rank=rank, percentile=percentile,
                confidence=confidence, direction="BUY", entry_price=entry, stop_loss=sl,
                target=target, risk_reward=rr,
                reasons=[f"Momentum rank {rank}/{n}", f"percentile {percentile:.0%}"],
                expected_return=expected_return, top_decile_prob=top_decile_prob,
            ))
        self.status = "ok"
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/ml/serving/test_momentum_engine.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Live smoke (real model + cached data) — verify end-to-end**

Run:
```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -c "
from backend.ai.signals.engines.momentum import MomentumEngine
e = MomentumEngine()
sigs = e.run(top_n=10, universe_limit=60)
print('status:', e.status, 'signals:', len(sigs))
for s in sigs[:5]:
    print(f'  {s.symbol:<10} rank={s.rank} pct={s.percentile:.2f} exp_ret={s.expected_return:+.3f} entry={s.entry_price:.1f} SL={s.stop_loss:.1f} tgt={s.target:.1f}')
"
```
Expected: `status: ok signals: 10` with real symbols, descending ranks, finite levels.

- [ ] **Step 6: Commit**

```bash
git add backend/ai/signals/engines/ tests/ml/serving/test_momentum_engine.py
git commit -m "feat(signals): MomentumEngine serving class (CPU, registry+disk model)"
```

---

### Task 4: HTTP endpoint with 60s cache

**Files:**
- Modify: `backend/api/signals_routes.py` (add the route + a small TTL cache; place near the other GETs)
- Test: `tests/api/test_momentum_signals_route.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_momentum_signals_route.py
from backend.ai.signals.style_types import MomentumSignal, Style


def test_momentum_route_shape(monkeypatch):
    from backend.api import signals_routes as sr

    fake = [MomentumSignal(symbol="AAA", style=Style.MOMENTUM, rank=1, percentile=1.0,
                           confidence=100.0, direction="BUY", entry_price=100.0,
                           stop_loss=85.0, target=130.0, risk_reward=2.0, reasons=["r"],
                           expected_return=0.04, top_decile_prob=1.0)]

    class _FakeEngine:
        status = "ok"
        def run(self, top_n=20, universe_limit=None):
            return fake

    monkeypatch.setattr(sr, "_momentum_engine", lambda: _FakeEngine())
    sr._momentum_cache.clear()
    payload = sr._compute_momentum(top_n=20)
    assert payload["status"] == "ok"
    assert payload["count"] == 1
    assert payload["signals"][0]["symbol"] == "AAA"
    assert payload["signals"][0]["style"] == "momentum"
    assert payload["signals"][0]["expected_return"] == 0.04
```

- [ ] **Step 2: Run test to verify it fails**

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/api/test_momentum_signals_route.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_compute_momentum'`

- [ ] **Step 3: Write minimal implementation**

Add to `backend/api/signals_routes.py` (near the top-level helpers; reuse the existing `router`, `current_user_tier`, `_get_user_profile_dep` already imported in the file):

```python
import time as _time  # add if not already imported

# ── Momentum engine (per-style serving) — 60s TTL cache (scanner pattern) ──
_momentum_cache: dict = {}  # key -> (ts, payload)
_MOMENTUM_TTL_S = 60


def _momentum_engine():
    from ..ai.signals.engines.momentum import MomentumEngine  # noqa: PLC0415
    return MomentumEngine()


def _compute_momentum(top_n: int = 20) -> dict:
    eng = _momentum_engine()
    sigs = eng.run(top_n=top_n)
    return {
        "signals": [s.to_dict() for s in sigs],
        "count": len(sigs),
        "status": eng.status,
        "style": "momentum",
    }


@router.get("/api/signals/momentum")
async def get_momentum_signals(
    top_n: int = Query(20, ge=1, le=100),
    profile=Depends(_get_user_profile_dep()),
    tier: UserTier = Depends(current_user_tier),
):
    """Momentum ranker — top-of-book by expected forward return (spec §5.1).
    On-demand with a 60s in-process cache (no persistence)."""
    key = f"momentum:{top_n}"
    now = _time.time()
    hit = _momentum_cache.get(key)
    if hit and now - hit[0] < _MOMENTUM_TTL_S:
        return hit[1]
    payload = _compute_momentum(top_n=top_n)
    _momentum_cache[key] = (now, payload)
    return payload
```

(If `time`/`Query`/`UserTier`/`current_user_tier`/`_get_user_profile_dep` are already imported in the file — they are, per the existing `/today` route — do not re-import; only add `import time as _time` if absent.)

- [ ] **Step 4: Run test to verify it passes**

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/api/test_momentum_signals_route.py -v`
Expected: PASS

- [ ] **Step 5: App-import + route smoke**

Run:
```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -c "
import backend.api.app
from backend.api.app import app
paths = [r.path for r in app.routes]
assert '/api/signals/momentum' in paths, 'route not registered'
print('route registered OK')
"
```
Expected: `route registered OK`

- [ ] **Step 6: Commit**

```bash
git add backend/api/signals_routes.py tests/api/test_momentum_signals_route.py
git commit -m "feat(api): GET /api/signals/momentum (on-demand, 60s cache)"
```

---

### Task 5: Frontend — momentum category + page + data source

**Files:**
- Modify: `frontend/lib/api.ts`
- Modify: `frontend/components/signals/categories.ts`
- Modify: `frontend/components/signals/CategorySignalsPage.tsx`
- Create: `frontend/app/signals/momentum/page.tsx`

**Gate (frontend "test"):** `cd frontend && npx tsc --noEmit` must be clean (no new errors), then a manual/Playwright render of `/signals/momentum`.

- [ ] **Step 1: Add the API client method**

In `frontend/lib/api.ts`, inside the `signals: { ... }` object (next to `getToday`/`getIntraday`), add:

```typescript
    getMomentum: (topN = 20) =>
      http<{
        signals: Array<Record<string, unknown>>
        count: number
        status: string
        style: string
      }>(`/api/signals/momentum?top_n=${topN}`),
```

(Use the same `http`/request helper the sibling methods use — match `getIntraday`'s exact call style.)

- [ ] **Step 2: Add the momentum category**

In `frontend/components/signals/categories.ts`:
- Change `export type CategoryId = 'intraday' | 'swing' | 'positional'` → `... | 'momentum'`.
- Add a `CATEGORIES.momentum` entry mirroring the shape of `CATEGORIES.swing` with momentum copy:

```typescript
  momentum: {
    id: 'momentum',
    name: 'Momentum',
    blurb: 'Stocks ranked by expected forward return',
    what:
      'Momentum trades ride stocks already moving with strength — the engine ranks the entire NSE cross-section by expected forward return (multi-factor: trend persistence, acceleration, volatility-adjusted strength) and surfaces the top of the book. Weekly rebalance, long-only.',
    how:
      'A LightGBM LambdaRank model scores every name; the top decile becomes signals with entry, stop and target pre-computed by the risk engine.',
    // copy the remaining required fields from CATEGORIES.swing's shape (icon, accent, etc.)
  },
```

- Add `CATEGORIES.momentum` to the ordered list array that currently lists intraday/swing/positional.
- In `categoryOf`, keep momentum mapping to its own page only where relevant; the momentum page fetches its own endpoint (Step 3), so `categoryOf` does not need to route momentum into swing anymore for the dedicated page.

- [ ] **Step 3: Make CategorySignalsPage fetch the per-style endpoint for momentum**

In `frontend/components/signals/CategorySignalsPage.tsx`, where it currently does:

```typescript
  const today = useSWR(demo ? null : 'signals:today', () => api.signals.getToday(), {...})
```

add a momentum branch so that when `category === 'momentum'` the page sources from the momentum endpoint and maps rows to the display shape:

```typescript
  const isMomentum = category === 'momentum'
  const momentum = useSWR(
    demo || !isMomentum ? null : 'signals:momentum',
    () => api.signals.getMomentum(50),
    { revalidateOnFocus: false },
  )
  // when isMomentum, build todayRows from momentum.data.signals (already ranked);
  // map each to the existing DisplaySignal shape: symbol, direction='BUY',
  // entry_price, stop_loss, target, risk_reward, confidence, plus show
  // rank / percentile / expected_return. Otherwise keep the existing
  // getToday()+categoryOf path unchanged.
```

Render `rank`, `percentile` (as %), and `expected_return` (as %) in the momentum rows (extend the existing card/row with these momentum-only fields, shown only when `isMomentum`).

- [ ] **Step 4: Create the page**

```tsx
// frontend/app/signals/momentum/page.tsx
import { CategorySignalsPage } from '@/components/signals/CategorySignalsPage'

export default function MomentumSignalsPage() {
  return <CategorySignalsPage category="momentum" />
}
```

- [ ] **Step 5: Typecheck + render**

Run: `cd frontend && npx tsc --noEmit`
Expected: no new errors.
Then (if dev server available): load `/signals/momentum` and confirm it renders the ranked momentum list (or the honest empty state if the backend returns `status: "model_not_loaded"`).

- [ ] **Step 6: Commit**

```bash
git add frontend/lib/api.ts frontend/components/signals/categories.ts frontend/components/signals/CategorySignalsPage.tsx frontend/app/signals/momentum/page.tsx
git commit -m "feat(frontend): /signals/momentum page wired to the momentum engine"
```

---

### Task 6: Final integration gate

- [ ] **Step 1: Full backend suite + contract**

Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest -q`
Expected: all pass (872 prior + the new tests).

Run: `lint-imports`
Expected: `Contracts: 1 kept, 0 broken.`

- [ ] **Step 2: End-to-end serve smoke**

Run:
```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -c "
from backend.api.signals_routes import _compute_momentum
p = _compute_momentum(top_n=10)
print('status', p['status'], 'count', p['count'])
assert p['style'] == 'momentum'
print('OK')
"
```
Expected: `status ok count 10` (or honest-empty `model_not_loaded`/`no_data` if data/model absent) + `OK`.

- [ ] **Step 3: Commit any final touch-ups, then push**

```bash
git push origin feat/mldl-4engine
```

---

## Notes for the implementer
- **No forecaster at serve.** The model uses `MOMENTUM_FEATURE_ORDER` (18 OHLCV features). Do not import or call TimesFM/Kronos in the engine.
- **Honest-empty, never fallback.** If the model can't load, return `[]` + a status — never synthesize signals.
- **Import boundary.** `backend.ai.signals.engines.momentum` may import `ml.*` (allowed). Never make `ml.*` import `backend.*` (import-linter blocks it; `lint-imports` must stay green).
- **Long-only.** Momentum signals are all `direction="BUY"`; the rank is the signal.
- **This is the template.** Swing/positional/intraday later add a `Style` value, a `<Style>Signal` subclass, a `RISK_PARAMS` entry, a `<Style>Engine`, an endpoint, and a category — reusing everything here.
