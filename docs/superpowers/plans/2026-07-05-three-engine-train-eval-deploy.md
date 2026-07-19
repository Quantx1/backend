# Three-Engine ML Program — Train · Test · Evaluate · Deploy (Master Plan)

> **Scope (founder-locked 2026-07-05):** THREE engines — Momentum, Swing, Positional. Intraday ML is CUT (no PatchTST, no paid TrueData bar-history). "Real-time" = the locked two-speed serving: daily EOD CPU scoring (~2 min, signals served in ms) + weekly GPU forecast batch → cache. GPU never serves a request.
>
> **Execution discipline:** each code-heavy task group (3.1–3.3, 4.1–4.2) gets its own detailed TDD plan via superpowers:writing-plans at execution time; everything else executes directly from this doc. Gates are hard — a phase does not start until the previous phase's gate is green.

**Goal:** All 3 style engines trained on fresh data through the canonical 9-stage spine (EDA → quality → purged-CV → HPO → DSR/PBO eval → report), promoted through the registry gates, and serving daily ranked signals in production (backend cron + SignalsHub tabs).

**Already built (do NOT rebuild):** 9-stage spine (`ml/training/pipeline.py` + specs/report/baseline_drift), MomentumTrainer on the spine (75 feats incl TimesFM+Kronos, HPO-ready), serve-smoke train/serve contract gate, momentum serving slice (`MomentumEngine`, `/api/signals/momentum`, hub tab), turnkey RunPod recipe (`scripts/runpod/train_momentum_gpu.sh` — HPO plumbed, deps complete, PEP-668 handled, forecast backfill persists to `artifacts/forecast_cache/*.parquet`), registry + promote gates + unified runner.

**Cost/time budget (total ≈ $10–15 GPU, ~2–3 weeks of build):**

| Phase | GPU | Wall-clock | Build effort |
|---|---|---|---|
| 0 Data refresh | $0 | ~1 h (local, unattended) | small |
| 1 Momentum train+eval | ~$4–7 (ONE-time backfill) | 5–10 h pod (unattended) | none (turnkey) |
| 2 Momentum deploy | $0 | — | 1–2 sessions |
| 3 Swing build+train+deploy | ~$1–2 (Chronos only; reuses cache) | 1–2 h pod | 3–4 sessions |
| 4 Positional build+train+deploy | ~$0.5 (reuses cache) | ~1 h pod | 2–3 sessions |
| 5 Production hardening | ~$0.35/week ongoing | — | 1–2 sessions |

---

## Phase 0 — Data freshness (BLOCKER: cache ends 2026-02-06, 5 months stale)

A model deployed in July must train on data through July. Licensing: internal backfill via yfinance/jugaad is the sanctioned path (Path A — display licensing is separate and unaffected).

- [ ] **0.1** Refresh all 322 equity CSVs in `data/cache/*_NS_10y.csv` through the latest close (script: extend each CSV via `yf.download(f"{sym}.NS", start=<last_date+1>)`, same tz-aware format; skip + log symbols that fail).
- [ ] **0.2** Refresh `data/cache/NSEI_10y.csv` the same way via `^NSEI`.
- [ ] **0.3** Quality pass: re-run `run_quality_checks` over the refreshed frames; drop/flag symbols with gaps or stale runs; report count.
- [ ] **0.4** Re-run the local momentum pipeline test (`tests/ml/training/test_momentum_pipeline.py`) on refreshed data — green.

**Gate 0:** ≥300 clean symbols whose max(date) is within 3 trading days of today; NSEI current; test green. (Cache stays gitignored — it ships to pods via the scp tarball flow, `COPYFILE_DISABLE=1 tar` to avoid AppleDouble files.)

---

## Phase 1 — Momentum: train → evaluate (the one long GPU run)

- [ ] **1.1** Launch pod (MCP, secure RTX 4090, `PUBLIC_KEY` env set) → clone branch → upload cache tarball → `HPO_TRIALS=30 FORECAST_STRIDE=5 bash scripts/runpod/train_momentum_gpu.sh` in nohup + watchdog + session monitor. (Exact runbook = what was executed 2026-07-05; all fixes committed.)
- [ ] **1.2** On completion: scp home `artifacts/models/momentum_lambdarank/` (model, feature_order, metrics, report.md, PNGs, drift_baseline) **and `artifacts/forecast_cache/*.parquet`** (the reusable 6-year backfill — this is the run's second product). Stop + delete pod.
- [ ] **1.3** Evaluate against gates: `momentum_lambdarank_quality_pass` (rank-IC ≥ 0.02, ICIR ≥ 0.5) — required; DSR ≥ 0.5 and PBO ≤ 0.4 — desirable (report if missed, decide); verify `tsfm_uncert` is non-constant on CUDA (was dead on CPU); read feature_importance.
- [ ] **1.4** Feature prune: drop ~zero-gain features from `MOMENTUM_FEATURE_ORDER`, retrain **incrementally** (reuses forecast parquet — ~30 min CPU-capable) if the prune is material; else skip.
- [ ] **1.5** Register + promote: `python -m ml.training.runner --only momentum_lambdarank --promote` (serve-smoke + quality gate enforced by the runner). Commit `feature_order.json` + `metrics.json` + report.

**Gate 1:** quality_pass = true · serve-smoke green · artifact + forecast parquet safely local + in B2/registry.

---

## Phase 2 — Momentum: deploy (backend + frontend)

- [ ] **2.1** Forecast-cache READ path: `ml/features/forecast_features.py::load_cached_forecasts(engine, cache_dir)` returns the persisted rebalance-date frames; trainer's `build_features` uses it when fresh (skip recompute unless `FORCE_FORECAST_BACKFILL=1`); serving merges the latest row per symbol. TDD.
- [ ] **2.2** Serving update (the old momentum plan's Task 6): `MomentumEngine.run()` switches benchmark to `ml.data.benchmark.load_nifty_benchmark` (offline+^NSEI fallback — already built), left-merges cached forecast cols, scores the artifact's new ≥70-col `feature_order`. Extend `tests/ml/serving/test_momentum_engine.py`. NaN-tolerant when cache absent (already verified).
- [ ] **2.3** Daily scoring cron: extend `backend/platform/scheduler.py` with `generate_style_signals` at 15:55 IST (after AutoPilot's 15:50) — runs each live engine's `run()`, persists ranked signals (existing signals tables/endpoints). Idempotent, honest-empty on holidays.
- [ ] **2.4** Weekly forecast refresh v1: `scripts/runpod/refresh_forecast_cache.sh` — spin pod, compute ONLY dates newer than the parquet's max(date) (~322 symbols × ≤5 dates ≈ minutes), merge-append parquet, upload to B2, kill pod. Manual-trigger weekly for now; Modal cron later (Phase 5).
- [ ] **2.5** Frontend verify: SignalsHub momentum tab renders the new model's signals (fields unchanged — same `MomentumSignalRaw` contract, ×100 display scaling rules); Playwright smoke on `/signals/momentum`.
- [ ] **2.6** Full gates: pytest suite green · `lint-imports` clean · live serve smoke returns ranked signals from the NEW artifact.

**Gate 2:** New momentum model serving in prod daily; weekly refresh runbook works once end-to-end.

---

## Phase 3 — Swing engine (build → train → deploy)

Price-first v1 (fundamentals/FII-DII/sentiment caches are still thin — layer them in a later iteration as real history accrues). Detailed TDD plan via writing-plans before 3.1.

- [ ] **3.1** `ml/features/swing_features.py` — `SWING_FEATURE_ORDER` (~60–80): short-horizon returns (1/3/5/10/21d), mean-reversion (RSI-family, distance-from-short-MAs, gap stats, high-low position), volatility, volume confirmation, RS-vs-NIFTY (reuse benchmark hook), cross-sectional ranks. Same builder rules as momentum: `groupby.transform` only, benchmark fail-soft, single-symbol-safe, computed-on-full-panel ranks.
- [ ] **3.2** Chronos-2 adapter in `forecast_features.py`: `chronos_forecast_features(panel, horizon=10, stride=5)` → `chronos_fwd_ret`, `chronos_uncert` (quantile spread); batched zero-shot (`chronos-forecasting>=2` already pinned). Swing's forecast set = Chronos cols + REUSE the cached TimesFM/Kronos parquet (same (date,symbol) keys — zero recompute). **TFT deferred:** it is not zero-shot (needs its own training); revisit only if Chronos+reused columns underperform.
- [ ] **3.3** `SwingTrainer(PipelineTrainer)` — `EngineSpec(name="swing_ranker", horizon=10, hpo_trials=30, eval gates IC≥0.02/ICIR≥0.5)`, hooks mirroring momentum, LGBM HPO space, `--limit/--hpo-trials` CLI. Local CPU smoke (`scripts/runpod/smoke_swing_local.sh`, 6 symbols) green BEFORE any pod.
- [ ] **3.4** GPU train: pod runbook; only the Chronos backfill is new compute (~1–2 h). Persist `swing_chronos.parquet` to the same cache dir. Evaluate vs gates; prune; promote via runner.
- [ ] **3.5** Serving slice (momentum template, verbatim pattern): `Style.SWING` + `SwingSignal` + `RISK_PARAMS[Style.SWING]=(1.2, 2.4)` (tighter than momentum — 10d horizon) + `SwingEngine` + `GET /api/signals/swing` (60s cache, tier-gated) + SignalsHub swing tab wired to it. Brand firewall: UI copy uses public engine names only.
- [ ] **3.6** Cutover: with swing live and gated, retire the legacy `tft_swing` voter from the v1 ensemble (the planned v1→v2 de-congestion) — separate commit, feature-flagged first week.

**Gate 3:** swing quality_pass + serve-smoke green · `/api/signals/swing` returns ML signals · hub tab live · suite+lint green.

---

## Phase 4 — Positional engine (build → train → deploy)

- [ ] **4.1** `ml/features/positional_features.py` — `POSITIONAL_FEATURE_ORDER` (~50–70): long-horizon returns (63/126/252d + 12-1), long MAs + slopes, vol regime, drawdown/recovery, RS-vs-NIFTY long windows, liquidity, cross-sectional ranks.
- [ ] **4.2** `PositionalTrainer(PipelineTrainer)` — horizon=60, `CVSpec(test_days=126, train_days=504)`, hpo_trials=30. Forecast features: REUSE the cached 20d TimesFM/Kronos/Chronos columns as inputs (they are features, not the label; a dedicated 60d backfill is an optional later improvement — decide from feature_importance).
- [ ] **4.3** Local CPU smoke → GPU train (nearly all cached — ~1 h) → evaluate → prune → promote.
- [ ] **4.4** Serving slice: `Style.POSITIONAL` + `RISK_PARAMS[Style.POSITIONAL]=(2.0, 4.0)` (wider — 60d horizon) + `PositionalEngine` + `GET /api/signals/positional` + hub tab. Monthly retrain cadence.

**Gate 4:** same as Gate 3, for positional.

---

## Phase 5 — Production hardening (all 3 live)

- [ ] **5.1** Unified weekly batch: ONE job refreshes TimesFM+Kronos+Chronos forecasts for the whole universe (shared calls serve all 3 engines), appends parquets, uploads B2. Move from manual script → Modal cron (scale-to-zero GPU) when stable.
- [ ] **5.2** Drift live: weekly job compares serving-window feature stats vs each artifact's `drift_baseline.json` (PSI/KS from `ml/eval/drift`); alert on breach → retrain runbook.
- [ ] **5.3** Retrain cadence: momentum+swing weekly (incremental, ~30 min), positional monthly. All through `runner --promote` (gates enforce quality; bad models never ship — no-fallback rule).
- [ ] **5.4** Registry/admin: 3 prod `model_versions` rows; admin ML panel shows each engine's report + rolling performance.
- [ ] **5.5** E2E: Playwright across the 3 hub tabs; API contract tests; docs + memory updated.

**Gate 5 (program DONE):** 3 engines live · daily cron ≈2 min CPU · weekly GPU ≈30–60 min ≈$0.35–1 · monthly bill ≈$5–15 · every promoted model passed EDA/quality/purged-CV/DSR/PBO/serve-smoke.

---

## Standing rules
- Mac CPU smoke green BEFORE any pod (never debug on the meter).
- Watchdog + session monitor on every pod; artifacts on `/workspace` volume; stop/delete promptly.
- Forecast backfills are write-once, reuse-forever (`artifacts/forecast_cache/`).
- Absolute outputs, separate risk engine for levels, brand firewall in all UI copy, no fallback models.
