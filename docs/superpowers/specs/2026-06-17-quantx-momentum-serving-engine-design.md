# Quant X — Momentum Serving Engine (4-Engine Template) — Design Spec

**Date:** 2026-06-17
**Status:** Approved (brainstorming) → ready for implementation plan
**Supersedes/relates:** Implements the serving layer of
`docs/superpowers/specs/2026-06-15-quantx-4engine-mldl-design.md` (§4.6 risk engine,
§4.7 output schema, §5.1 Momentum). Foundation (M0) + the Momentum trainer are already built;
this spec adds the **serving** path that turns the trained `momentum_lambdarank` model into a
live, frontend-visible engine — and establishes the reusable contract that Swing, Positional,
and Intraday slot into later.

---

## 0. One-paragraph summary

Build the Momentum **serving slice** end-to-end: a per-style output schema, a separate ATR-based
risk engine, a `MomentumEngine` serving class, an on-demand cached HTTP endpoint, and a dedicated
frontend page. Each of the 4 style engines is a **separate use case with a distinct output**
(spec §4.7); this slice makes that real for Momentum and is the **template** the other three reuse.

## 1. Goals & non-goals

**Goals**
- Make Momentum a true standalone engine: distinct output (`expected_return, rank, percentile,
  top_decile_prob, confidence` + risk-derived levels), its own endpoint, its own frontend page.
- Establish reusable scaffolding: a generic `StyleSignal` base, a per-style-configurable
  `RiskEngine`, and the endpoint/page pattern — so Swing/Positional/Intraday later only add a
  subclass + engine class + endpoint.
- CPU-only, no GPU, no paid data — fully unblocked today.

**Non-goals (YAGNI)**
- No persistence / cron / history. Serving is **on-demand + 60s cache** (the 77-scanner pattern).
  History can be added later without changing the output contract.
- No forecaster (TimesFM/Kronos) at serve time — see §3.
- No Swing/Positional/Intraday engines in this slice (they reuse this template later).
- No change to the existing v1 ensemble signals (`/api/signals/today`) — additive only.

## 2. Load-bearing constraint (why this is unblocked)

The trained `momentum_lambdarank` model's serve contract is
`artifacts/models/momentum_lambdarank/feature_order.json` = **18 pure-OHLCV features**
(`ret_5d…ret_252d`, `mom_consistency_63`, `mom_accel`, `vol_adj_mom_63`, `dist_sma_50/200`,
`above_high_63`, `realized_vol_21`, `drawdown_252`, `rel_volume_21`, `obv_slope_21`,
`xs_rank_ret_21/63`). **No TimesFM/Kronos columns.** The trainer's `with_forecasts` GPU features
are additive *behind the same `MOMENTUM_FEATURE_ORDER` contract* (default `False`). Therefore the
engine serves entirely on CPU. `atr_14` already exists in `ml/features/indicators.py` for the risk
engine.

## 3. Architecture (data flow)

```
GET /api/signals/momentum            (tier-gated; 60s in-process TTL cache)
 └─ MomentumEngine.run(universe)
      1. resolve model: registry.resolve("momentum_lambdarank") → lgb.Booster
         (registry-first; disk fallback artifacts/models/momentum_lambdarank/momentum_lambdarank.txt)
         → if absent: honest-empty {signals: [], status: "model_not_loaded"}
      2. universe: cached_universe()  (cache → nse_tiers fallback)
      3. per symbol: load OHLCV (FreeDataProvider) → build_momentum_features (18-col contract)
         → skip symbol if < MOMENTUM_WARMUP_BARS (252) bars or feature build fails
      4. model.predict(features[feature_order]) → raw score per symbol (latest bar)
      5. cross-section across symbols for the as-of date:
           rank (1..N desc by score), percentile (0..1),
           top_decile_prob (1 if percentile ≥ 0.9 else scaled), confidence (calibrated 0..100),
           expected_return (score mapped to a return estimate via the trainer's decile spread)
      6. RiskEngine.derive_levels(BUY, ref_price=close, atr=atr_14, style=MOMENTUM)
           → entry, stop_loss, target, risk_reward
   → List[MomentumSignal]
 → cache 60s → JSON
Frontend /signals/momentum
 → CategorySignalsPage(category="momentum")
 → api.signals.getMomentum() → renders rank · percentile · expected_return + entry/SL/target
```

Momentum is **long-only** (top of the cross-sectional book), so `direction = BUY` for all signals;
the ranking *is* the signal. `expected_return` and the percentile/decile fields are the
distinct momentum output that differentiates it from the v1 ensemble signal.

## 4. Components (each isolated, one responsibility, independently testable)

### 4.1 Output schema — `backend/ai/signals/types.py` (extend)
- `class Style(str, Enum)`: `MOMENTUM = "momentum"` (swing/positional/intraday added later).
- `@dataclass class StyleSignal`: `symbol: str`, `style: Style`, `rank: int`, `percentile: float`,
  `confidence: float`, `direction: str`, `entry_price: float`, `stop_loss: float`, `target: float`,
  `risk_reward: float`, `reasons: list[str]`, plus `to_dict()` for JSON.
- `@dataclass class MomentumSignal(StyleSignal)`: adds `expected_return: float`,
  `top_decile_prob: float`.
- Left untouched: the existing `GeneratedSignal` (v1 ensemble) — this is additive.

### 4.2 Risk engine — `backend/trading/risk_engine.py` (new)
- Pure, no I/O. `derive_levels(direction: str, ref_price: float, atr: float, style: Style)
  -> tuple[entry, stop_loss, target, risk_reward]`.
- `RISK_PARAMS: dict[Style, tuple[float, float]]` = `{Style.MOMENTUM: (1.5, 3.0)}`
  (sl_atr_mult, tp_atr_mult); other styles added later.
- BUY: `entry = ref_price`, `stop_loss = entry - sl_mult*atr`, `target = entry + tp_mult*atr`,
  `risk_reward = (target-entry)/(entry-stop_loss)`. Guards: `atr > 0`, `entry > stop_loss`.
- Separate from `backend/trading/risk.py` (which governs position sizing / day-loss / exposure
  limits) — different responsibility, no overlap.

### 4.3 Serving engine — `backend/ai/signals/engines/momentum.py` (new package `engines/`)
- `class MomentumEngine`: holds the resolved booster + feature_order; `run(universe=None,
  as_of=None) -> list[MomentumSignal]`.
- Model load: registry-first via `backend.ai.registry`, disk fallback to the artifacts path;
  validates `booster.feature_name() == feature_order` (reuses the serve-smoke contract idea).
- Honest-empty (no heuristic fallback) if the model is missing — returns `[]` and a status flag.
- Reuses `ml.features.momentum_features.build_momentum_features` (already serve-safe for single &
  multi symbol) and `ml.data` provider for OHLCV.
- Import boundary: lives under `backend.ai` and may import `ml.*` (allowed by import-linter;
  `ml` must not import `backend`, which is unaffected here).

### 4.4 Endpoint — `backend/api/signals_routes.py` (extend)
- `GET /api/signals/momentum` → `{ signals: [...], count, status, as_of }`.
- 60s in-process TTL cache keyed on `(universe, as_of_day)` (same pattern as
  `_scanner_cached` in `screener_routes.py`).
- Tier gating consistent with `/api/signals/today` (reuse its dependency).

### 4.5 Frontend
- `frontend/lib/api.ts`: add `signals.getMomentum()` → `GET /api/signals/momentum`, typed to the
  momentum payload.
- `frontend/components/signals/categories.ts`: add `'momentum'` to `CategoryId`, a `CATEGORIES.momentum`
  entry (copy: what momentum is + how the ranker finds it), and add it to the ordered list.
  Update `categoryOf` so momentum no longer silently folds into swing for the momentum page.
- `frontend/app/signals/momentum/page.tsx`: `<CategorySignalsPage category="momentum" />`.
- `CategorySignalsPage`: make the data source per-category — when `category === 'momentum'`, fetch
  `api.signals.getMomentum()` (distinct output) instead of `getToday()` + client filter; map
  `MomentumSignal` into the display rows and surface `rank`, `percentile`, `expected_return`.
- Nav: add the momentum page link alongside the existing signal category links.

## 5. Error handling

| Condition | Behavior |
|---|---|
| Model artifact absent | `{signals: [], status: "model_not_loaded"}`, HTTP 200 (honest-empty, no fallback) |
| Per-symbol feature build fails | log + skip that symbol, continue (scanner pattern) |
| `< 252` warmup bars | skip symbol |
| `atr <= 0` or degenerate levels | skip the risk-level derivation for that symbol (drop signal) |
| Universe empty / data provider down | `{signals: [], status: "no_data"}` |

## 6. Testing

- **risk_engine** (`tests/.../test_risk_engine.py`): BUY levels for known atr; RR math; per-style
  param lookup; guards (atr≤0, entry≤stop) raise/skip.
- **MomentumEngine** (`tests/ml/.../test_momentum_engine.py`): tiny fixture universe from cached
  CSVs → asserts rank is a permutation 1..N, percentile monotonic with score, every signal has
  entry/SL/target/RR and `expected_return`; honest-empty when the model path is monkeypatched away.
- **types**: `MomentumSignal.to_dict()` round-trip + `Style` enum value.
- **endpoint**: route test with a mocked `MomentumEngine` → asserts JSON shape + cache hit on 2nd call.
- **frontend**: `tsc` clean; `categories.ts`/`CategorySignalsPage` compile with the new category.
- **Gate**: full `python -m pytest -q` stays green (currently 872); `lint-imports` 1 kept/0 broken.

## 7. Scope boundaries

- In: momentum schema + risk engine + engine + endpoint + frontend page, all CPU, on-demand+cache.
- Out (future, reuse this template): persistence/cron/history; Swing/Positional/Intraday engines;
  forecaster-enriched momentum (GPU); calibration of `expected_return` against realized returns
  (start with the trainer's decile-spread mapping, refine later).

## 8. Reuse path for the other 3 engines (why this is a "template")

Each later engine adds only: (a) a `Style` enum value + a `<Style>Signal(StyleSignal)` subclass with
its distinct fields (§4.7); (b) a `RISK_PARAMS[Style.X]` entry; (c) a `<Style>Engine` class with its
model + feature builder; (d) a `GET /api/signals/<style>` endpoint; (e) a category + page. The
schema base, risk engine, cache pattern, and `CategorySignalsPage` per-style data source are shared.
