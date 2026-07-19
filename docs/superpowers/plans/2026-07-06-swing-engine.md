# Swing Engine (build #2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** The swing style engine end-to-end: `swing_features.py` (~55 price-first features, 5–21d emphasis) + a Chronos-2 forecast adapter + `SwingTrainer` on the canonical 9-stage spine + the serving slice (Style.SWING → `/api/signals/swing` → SignalsHub swing tab), gated exactly like momentum.

**Architecture:** Mirrors momentum verbatim — same spine, same cache mechanics, same serving template. Swing differs only in: horizon (10d vs 20d), feature families (mean-reversion/short-trend vs long-momentum), and one new forecaster (Chronos-2; it also REUSES momentum's cached TimesFM/Kronos parquets read-only — zero recompute).

**Canonical templates (implementers: READ these, mirror their structure exactly):**
`ml/features/momentum_features.py` (builder rules) · `ml/features/forecast_features.py` (adapter shape + cache) · `ml/training/trainers/momentum_lambdarank.py` (PipelineTrainer + cache top-up + CLI) · `backend/ai/signals/{style_types.py, engines/momentum.py}` + `backend/trading/risk_engine.py` + `backend/api/signals_routes.py` `/api/signals/momentum` block (serving slice) · `scripts/runpod/{smoke_momentum_local.sh, train_momentum_gpu.sh}`.

**Iron rules (identical to momentum; violations were real bugs):** per-symbol ops via `groupby(sym).transform` ONLY (never `.apply` returning frames — single-symbol serving crash); cross-sectional ranks computed LAST on the full panel; benchmark fail-soft (None → NaN RS cols); every feature uses only past data; `pct_change(fill_method=None)`; Mac CPU smoke green BEFORE any pod; brand firewall in UI copy.

**Run tests:** `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest <path> -v`

---

## Task 1: `ml/features/swing_features.py` (+ tests)

Create `SWING_FEATURE_ORDER` (49 base + 6 RS = 55) and `build_swing_features(panel, benchmark=None)`. Mirror `momentum_features.py`'s structure (docstring rules, `_EPS`, helper reuse, output = `['date','symbol',*ORDER]`). `SWING_WARMUP_BARS = 63`.

**Exact features (formulas; `r` = per-symbol daily `close.pct_change(fill_method=None)`; all rolling = trailing):**

A. Returns (8): `ret_1d,ret_2d,ret_3d,ret_5d,ret_10d,ret_21d,ret_42d,ret_63d` = `close/close.shift(w)-1`.

B. Mean-reversion (12): `rsi_2`, `rsi_14` (ta.momentum.RSIIndicator on close, per symbol via the index-safe per-group helper pattern used for ADX in momentum); `zscore_10` = `(close-sma10)/(std10+eps)`; `zscore_20` likewise; `dist_sma_5`, `dist_sma_10`, `dist_sma_20` = `close/sma_w - 1`; `dist_ema_9` = `close/ema9-1`; `boll_pos_20` = `(close-sma20)/(2*std20+eps)`; `pullback_from_high_21` = `close/rolling_max21 - 1`; `bounce_from_low_21` = `close/rolling_min21 - 1`; `close_pos_21` = `(close-low21min)/(high21max-low21min+eps)`.

C. Gaps/range (4): `gap_open` = `open/close.shift(1)-1` (per symbol!); `gap_abs_mean_5` = rolling(5).mean of `|gap_open|`; `range_pct_1d` = `(high-low)/(close+eps)`; `close_pos_1d` = `(close-low)/(high-low+eps)`.

D. Short trend (5): `sma_5_20_align` = `(sma5>sma20).astype(float)`; `sma20_slope_10` = `sma20.pct_change(10, fill_method=None)`; `ema_9_21_spread` = `(ema9-ema21)/(close+eps)`; `adx_14` (ta, per-group helper); `macd_hist_norm` = `(ema12-ema26 - ema9_of_that)/ (close+eps)`.

E. Volatility (6): `realized_vol_5`, `realized_vol_10`, `realized_vol_21` = `r.rolling(w).std()`; `vol_ratio_5_21` = `rv5/(rv21+eps)`; `parkinson_vol_10` (as momentum's, window 10); `atr_pct_14` (vectorized TR as momentum).

F. Volume (6): `rel_volume_5`, `rel_volume_21` = `vol/rolling_mean+eps`; `vol_zscore_10`; `up_vol_ratio_10` (as momentum's, window 10); `obv_slope_10` (index-preserving cumsum as momentum); `volume_breakout` = `(rel_volume_5>2).astype(float)`.

G. Base for ranks (2): `vol_adj_mom_21` = `ret_21d/(realized_vol_21*sqrt(21)+eps)`; `mom_consistency_10` = `_rolling_mean_positive(r,10)`.

H. RS-vs-NIFTY (6, fail-soft NaN without benchmark): `rs_index_5`, `rs_index_10`, `rs_index_21` = `ret_wd - bench_ret_wd` (bench shifts PER SYMBOL after merge — copy momentum's Section G exactly incl the sort-order guard); `rs_index_slope_5` = `rs_index_10.diff(5)` per symbol; `beta_index_63`, `corr_index_63` (copy momentum's).

I. Cross-sectional ranks — LAST (6): `xs_rank_ret_5`, `xs_rank_ret_10`, `xs_rank_ret_21`, `xs_rank_zscore_20`, `xs_rank_vol_adj_mom_21`, `xs_rank_rs_index_10` = `groupby("date")[col].rank(pct=True)`.

**Tests** (`tests/ml/features/test_swing_features.py`, mirror momentum's file): columns==ORDER; no-lookahead on ret_5d; train/serve parity (same input twice → identical); single-symbol no-crash; expanded-set finite on a 400-bar uptrend WITH benchmark; RS fail-soft without benchmark; 2-symbol no-cross-leak on rs_index_10 (copy momentum's RS leak test shape).

Commit: `feat(swing): swing feature factory (55 features, 5-21d families)`.

## Task 2: Chronos-2 adapter in `forecast_features.py` (+ test)

Add `CHRONOS_FEATURES = ["chronos_fwd_ret", "chronos_uncert"]` and:

```python
def chronos_forecast_features(
    panel: pd.DataFrame,
    horizon: int = 10,
    context: int = 512,
    stride: int = 5,
    batch_size: int = 64,
    min_history: int = 252,
    min_date=None,
    model_id: str = "amazon/chronos-2",
) -> pd.DataFrame:
    """Rolling Chronos-2 zero-shot forecast features (verified API, chronos-forecasting 2.2.2):
    BaseChronosPipeline.from_pretrained(model_id, device_map=device) then
    quantiles, mean = pipe.predict_quantiles(inputs=[torch.tensor(series), ...],
        prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])
    quantiles[k] has shape (horizon, 3); take step horizon-1:
      chronos_fwd_ret = q50/last_close - 1 ; chronos_uncert = (q90-q10)/last_close.
    Mirror timesfm_forecast_features EXACTLY otherwise: rdates probe + empty
    early-exit (min_date), per-date batched loop over symbols with >=min_history,
    context tail slice, log every 20 dates, return ['date','symbol',*CHRONOS_FEATURES].
    """
```

Also extend `merge_forecast_features`'s ensemble: `ens_fwd_ret` = mean of whichever of `tsfm_fwd_ret/kronos_fwd_ret/chronos_fwd_ret` are present (≥2 present → mean; else NaN) — update `FORECAST_FEATURES` to include the chronos cols. Update the Task-3 ensemble test accordingly (mean of 3 when all present; NaN when only 1).

**Test** (`tests/ml/features/test_chronos_adapter.py`): monkeypatch a fake pipeline object (`predict_quantiles` returns known tensors) injected via monkeypatching `BaseChronosPipeline.from_pretrained`; assert output rows/columns/values and the min_date early-exit returns an empty typed frame WITHOUT calling from_pretrained.

Commit: `feat(swing): Chronos-2 forecast adapter + 3-way ensemble`.

## Task 3: `SwingTrainer` on the spine (+ smoke script)

`ml/training/trainers/swing_lambdarank.py` — copy `momentum_lambdarank.py`'s shape wholesale, with:
- `SwingConfig`: `horizon=10`, `n_quantiles=10`, `start=2020-01-01`, `end=default_factory=date.today`, `with_forecasts=False`, `forecast_stride=5`, `hpo_trials=0`, `cv=PurgedCVConfig(n_folds=5, test_days=63, embargo_days=10, train_days=378)`, same `lgbm_params`.
- `SwingTrainer(PipelineTrainer)`: `name="swing_lambdarank"`, `skip_promote_gate=True`; `engine_spec` → `EngineSpec(horizon=10, hpo_trials=cfg.hpo_trials, cv=..., eval=EvalSpec(min_ic=0.02, min_icir=0.5), eda=EDASpec(min_abs_ic=0.0, max_leakage_corr=0.999, max_constant_features=8))`.
- `build_features`: `build_swing_features(panel, benchmark=load_nifty_benchmark(...))`; forecasts branch = the SAME cache/_topup pattern as momentum but: chronos under `swing_chronos.parquet` (own top-up at horizon=10), tsfm/kronos READ-ONLY from `momentum_{tsfm,kronos}.parquet` (no recompute, no save; if absent → skip those frames, cols come out NaN — fail-soft).
- `build_labels`: `forward_return_quantile_labels(horizon=10)`.
- `fit_args`/`make_model`/`search_space`/`serve_smoke("swing_lambdarank.txt")`/`__main__` — same as momentum (search space identical).
- `scripts/runpod/smoke_swing_local.sh`: copy momentum's smoke; 6 symbols, `FORECAST_DEVICE=cpu`, `with_forecasts=True`, stride 120, assert `n_features >= 55` + serve-contract via `smoke_artifact(out, "swing_lambdarank.txt")`.

**Tests** (`tests/ml/training/test_swing_pipeline.py`, mirror momentum's): price-only 8-symbol run via spine on the real cache → metrics carry `eda/feature_audit/hpo`, `feature_order.json == SWING_FEATURE_ORDER`, report.md + drift_baseline exist, serve_smoke green. Plus `runner --list` shows `swing_lambdarank`.

Commit: `feat(swing): SwingTrainer on the canonical spine + local smoke`.

## Task 4: Serving slice (backend) — momentum template, verbatim pattern

- `style_types.py`: `Style.SWING = "swing"`; `SwingSignal(StyleSignal)` with `expected_return: float = 0.0`, `top_decile_prob: float = 0.0` (same fields as momentum — same meta bar).
- `risk_engine.py`: `RISK_PARAMS[Style.SWING] = (1.2, 2.4)` (tighter than momentum's (1.5, 3.0) — 10d horizon).
- `backend/ai/signals/engines/swing.py`: copy `momentum.py`; `_MODEL_NAME="swing_lambdarank"`, disk fallback `artifacts/models/swing_lambdarank/`, features via `build_swing_features` on the FULL panel with `load_nifty_benchmark` (import from `ml.data.benchmark` — NOT `_load_ohlcv(["NSEI"])`), warmup `SWING_WARMUP_BARS`, emits `SwingSignal` with `Style.SWING` levels.
- `signals_routes.py`: `GET /api/signals/swing` — copy the `/momentum` block (60s TTL cache, tier gate, honest-empty statuses).
- **Tests**: `tests/ml/serving/test_swing_engine.py` (mirror momentum's 3 tests: ranks/levels/outputs with NSEI in fake loader; xs-rank non-degenerate; honest-empty on model missing).

Commit: `feat(swing): serving slice — Style.SWING, SwingEngine, /api/signals/swing`.

## Task 5: Frontend wiring + gates

- `frontend/lib/api.ts` (or the api client home of `getMomentum`): `api.signals.getSwing()` typed like momentum's raw type.
- SignalsHub swing tab: data path → `/api/signals/swing` when the ML feed returns `status: ok` (same pattern as the momentum tab; ×100 display scaling for percentile/expected_return — the fractions rule). Brand firewall: copy says "Alpha ranks the swing horizon…" style language, NO model names.
- Gates: full pytest suite green · `lint-imports` 1 kept/0 broken · `bash scripts/runpod/smoke_swing_local.sh` FULL STACK GREEN on Mac CPU.

Commit: `feat(swing): hub swing tab wired to the ML engine`.

## Task 6 (Phase 3.4 of master plan — after Tasks 1–5 green): GPU train + promote

Pod runbook (proven): secure pod w/ `PUBLIC_KEY` env → clone → `COPYFILE_DISABLE=1` cache tarball incl `artifacts/forecast_cache/*.parquet` → `HPO_TRIALS=30 bash scripts/runpod/train_swing_gpu.sh` (copy momentum's GPU script; same dep line incl scikit-learn) → watchdog + monitor → evaluate gates → scp artifacts + `swing_chronos.parquet` → delete pod. Only Chronos computes (~1–2 h, ~$1–2). Then `runner --only swing_lambdarank --promote` AFTER the serving slice is live. tft_swing voter retirement = separate follow-up commit once swing is validated in prod.

---

**Out of scope here:** fundamentals/FII-DII/sentiment features (data still thin — layered in a later swing iteration); momentum Phase 2 deploy tasks (tracked separately in the master plan).
