# Quant X — Production Repository Reorganization Plan (2026-06-16)

## Quant X — Production Repository Reorganization (full-app)

### Why this matters now
The 4-engine ML/DL build (Momentum/Swing/Positional/Intraday) is about to add ~10 new shared-foundation modules (`pipeline.py`, `serve_smoke.py`, `specs.py`, `baseline_drift.py`, `factory.py`, `risk_engine.py`, per-engine feature builders + trainers) plus a TrueData provider. Those modules sit exactly on the fault line the 2026-06-15 audit blamed for **train/serve skew on every PROD model**. The current tree has no enforced boundary between *what is trained* and *what is served*, so the same code drifts into two copies. Fixing the structure first is the cheapest way to make "feature builder shared train==serve" (spec §4.1) and "serve-smoke before promote" (spec §4.4 / pipeline Stage 8b) structurally true instead of a convention people forget.

### The single root cause: there is no training↔serving boundary
The repo has two top-level Python trees, `ml/` (research/training) and `src/backend/` (serving/app), but they import each other in **both directions** and one of them does both jobs:

- `ml/` imports the serving app: `ml/data/data_loader.py` → `src.backend.data.providers.{base,free_provider,truedata_provider}`; `ml/data/kite_source.py` → `src.backend.services.kite_data_provider`; `ml/training/base.py` + `ml/training/trainers/lgbm_signal_gate.py` → `src.backend.ai.registry`; `ml/data/sentiment_history.py` → `src.backend.ai.sentiment.finbert_india`.
- `src/backend/` imports the research tree **at request/serve time**: `ai/signals/generator.py` → `ml.regime_detector` + `ml.scanner`; `platform/scheduler.py` → `ml.regime_detector` + `ml.training.runner` + `ml.eval.drift_monitor` + `ml.data.fii_dii_history`; `ai/feature_engineering.py` → `ml.features.lgbm_v2`; `ai/signals/options.py` → `ml.strategies`.

So `ml/` is **not** a clean research package — `ml/scanner.py`, `ml/risk_manager.py`, `ml/regime_detector.py`, `ml/strategies/`, `ml/backtest/` are serving/business modules misfiled under the training tree, and `ml/data/` duplicates the serving data layer in `src/backend/data/providers/`. The canonical `data_loader` even lives in `ml/` but wraps providers in `src/backend/`.

### Design principle for the target
Make **three concentric rings** with a one-directional dependency rule, enforced by an import-linter contract in CI:

1. **`core/` (pure domain + contracts)** — dataclasses (`MarketDepth`, signal `types`, `EngineSpec`/`LabelSpec`), provider Protocols, registry interface, output schema. Depends on nothing internal. Both rings 2 & 3 may import it.
2. **`ml/` (training & research)** — data factory, features, labeling, CV, eval, trainers, the 9-stage pipeline, RunPod entry. May import `core/`. **May NOT import the serving app.**
3. **`backend/` (serving app)** — FastAPI api/services/trading/data-serving/registry-impl/scheduler. May import `core/` and the **serialized model contract** in `core/`, plus call `ml/` only through the registry artifact path. **May NOT import `ml/` trainers at request time.**

The two legitimate `backend → ml` couplings today (regime features at serve, lgbm inference features at serve) are resolved by moving the *shared* feature/label code into `core.features` (pure, no-deps, train/serve-identical — which is exactly what spec §4.1 mandates anyway) and leaving only fit/HPO/eval in `ml/`. The scheduler's "kick off a training run" call moves behind a thin `ml.cli` subprocess/job boundary, not an in-process import.

### Naming conventions (locked)
- One import root. Drop the `src/` wrapper: app becomes `backend.api.app:app` (was `src.backend.api.app:app`). Add a `pyproject.toml` with `[tool.setuptools] packages = ["core","ml","backend"]` and editable install so `import backend` / `import ml` / `import core` work without sys.path hacks (kills the `sys.path.insert` lines in conftest + every backfill script).
- Layer-suffix modules consistently: trainers `*_trainer.py`, serving engines `*_engine.py`, feature builders `*_features.py`, label builders `*_labels.py`, providers `*_provider.py`, API routers `*_routes.py` (already the norm).
- Per-engine symmetry: for each style `X` there is exactly one `core/features/X_features.py` (shared), one `ml/training/trainers/X_trainer.py`, one `backend/ai/engines/X_engine.py`. A test asserts the trinity exists for every `EngineStyle` enum member.
- Artifacts: ONE local cache root `artifacts/models/<model>/<version>/` (merge `ml/models/`, top-level `models/`, `.model_cache/`); committed tiny configs allowed, large weights gitignored + sourced from B2. `data/` stays for static reference inputs only (universes, holidays, tiers), `var/` (gitignored) for runtime caches (bhavcopy, parquet caches).
- Scripts collapse into a thin `python -m ml.cli` / `python -m backend.cli` Typer app; only orchestration shell (`runpod_*.sh`, `dev.sh`, `qa/`, `release/`) stays under `scripts/`.

### Migration is phased, shim-guarded, test-gated — never big-bang
Every move uses a re-export shim at the old path (`from new.path import *  # DEPRECATED, remove after <date>`) so existing imports keep working; the full test suite (130 tests) + `uvicorn ... --check` import + a new import-linter contract run green after each phase before the next starts. Shims carry a removal date and a CI grep that fails if a NEW import uses an old path. This mirrors the codebase's existing shim policy (structural target 2026-05-25) and route-alias deprecation pattern.

### Files relevant to executing this
- `pyproject.toml` (new, repo root) — declares the 3 packages + editable install.
- `core/` (new top-level) — receives shared contracts/features/labels.
- `nixpacks.toml`, `Dockerfile`, `railway.toml`, `scripts/dev.sh` — the 4 places that hardcode `src.backend.api.app:app` and must flip to `backend.api.app:app` in Phase 1.
- `tests/conftest.py` + `scripts/*.py` — drop sys.path hacks once editable install lands.
- `ml/__init__.py` — currently documents a stale rule-based structure; rewrite to the new layout.
- Import-linter config (new) in `pyproject.toml` — encodes the 3-ring contract; wired into `.github/workflows/backend-ci.yml`.

## Target tree

```
Swing_AI_Final/
├── pyproject.toml                  # NEW: declares core/ ml/ backend/ as packages; editable install; import-linter 3-ring contract; ruff/mypy/pytest config
├── README.md  Dockerfile  nixpacks.toml  railway.toml  vercel.json  runtime.txt
├── requirements/                   # split from 2 loose files at root
│   ├── base.txt                    # serving runtime (was requirements.txt)
│   ├── train.txt                   # GPU/forecaster stack (was requirements-train.txt)
│   └── dev.txt                     # pytest, ruff, mypy, import-linter
│
├── core/                           # RING 1 — pure domain + contracts (NO internal deps; both ml/ & backend/ import this)
│   ├── contracts/                  # provider Protocols, registry interface, model serving contract (feature_order/dataset_params sidecar schema)
│   ├── types/                      # signal output schema (style/horizon enum, expected_return/rank/confidence), MarketDepth, EngineSpec/LabelSpec/FeatureSpec/CVSpec
│   ├── features/                   # SHARED feature builders — identical train & serve (spec §4.1)
│   │   ├── factory.py              #   Qlib Alpha158 + TA-Lib orchestration (spec wants ml/features/factory.py; lives in core so serve can use it w/o importing ml/)
│   │   ├── momentum_features.py  swing_features.py  positional_features.py  intraday_features.py
│   │   ├── indicators.py  frac_diff.py  patterns.py  volume_analysis.py  forecast_features.py
│   │   └── regime_features.py      # ← from ml/regime_detector compute_regime_features (serve uses this without importing ml/)
│   └── labeling/                   # SHARED label builders (used by trainers; pure)
│       ├── triple_barrier.py  ranking_labels.py  sample_weights.py
│
├── ml/                             # RING 2 — TRAINING & RESEARCH ONLY (imports core/; NEVER imports backend serving app)
│   ├── cli.py                      # Typer entry: `python -m ml.cli train --engine momentum --promote` (folds scripts/train_*.py, retrain_pipeline.py, smoke_all.py)
│   ├── data/                       # research data factory (the load/cache/quality side)
│   │   ├── data_loader.py          # uniform loader; depends on core.contracts provider Protocol (impl injected, not imported from backend)
│   │   ├── liquid_universe.py  production_ohlcv.py  bhavcopy_source.py  corporate_actions.py
│   │   ├── delisted_registry.py  fundamentals_pit.py  fii_dii_history.py  sentiment_history.py
│   │   ├── news_ingester.py  quality_check.py
│   │   └── providers/              # research-side concrete providers (free/kite/yfinance/nselib/truedata) implementing core.contracts
│   ├── preprocessing/eda.py
│   ├── training/
│   │   ├── pipeline.py             # NEW: 9-stage template-method spine (Stage enum + PipelineContext + run_pipeline)
│   │   ├── base.py  runner.py  discovery.py  purged_cv.py  wfcv.py  cpcv.py  optuna_search.py
│   │   ├── specs.py                # NEW: EngineSpec/LabelSpec/FeatureSpec/CVSpec (re-exports core/types)
│   │   ├── serve_smoke.py          # NEW Stage 8b: round-trip artifact THROUGH backend predictor (the #1 audit fix)
│   │   ├── baseline_drift.py       # NEW: writes PSI/KS train-window baseline
│   │   ├── verbose.py  smoke.py
│   │   └── trainers/               # auto-discovered; one *_trainer.py per engine
│   │       ├── momentum_trainer.py        # ← momentum_lambdarank.py (TimesFM+Kronos→LGBM LambdaRank)
│   │       ├── swing_trainer.py           # TFT+Chronos→LGBM Ranker  (← tft_swing.py)
│   │       ├── positional_trainer.py      # Kronos+TimesFM→LGBM Ranker
│   │       ├── intraday_trainer.py        # PatchTST→LGBM  (M4, TrueData-gated)
│   │       ├── regime_hmm_trainer.py  qlib_alpha158_trainer.py  lgbm_signal_gate_trainer.py
│   │   └── adapters/               # forecaster wrappers: timesfm.py kronos.py chronos.py tft.py patchtst.py
│   ├── eval/                       # backtest_eval, lambdarank_ic, overfitting, spa, impact_cost, kelly, drift, drift_monitor
│   ├── backtest/engine.py          # research backtester (← ml/backtest)
│   └── research/                   # legacy rule-based strategies kept for backtest harness only (← ml/strategies + scanner.py + risk_manager.py)
│
├── backend/                        # RING 3 — SERVING APP (was src/backend; imports core/ + reads registry artifacts; NO request-time ml/ trainer imports)
│   ├── cli.py                      # Typer entry: backfills, seed, schema (folds scripts/backfill_*.py, seed_*.py, apply_migrations.py callers)
│   ├── api/
│   │   ├── app.py  main.py         # app:app  → run as `backend.api.app:app`
│   │   ├── routes/                 # *_routes.py moved out of api/ root into routes/ (50 routers)
│   │   └── admin/                  # admin/*.py routers
│   ├── core/                       # config, database, security, tiers, public_models  (app bootstrap, not domain core/)
│   ├── ai/                         # serving AI (peer pkg)
│   │   ├── engines/                # NEW: serving signal engines, one *_engine.py per style
│   │   │   └── momentum_engine.py swing_engine.py positional_engine.py intraday_engine.py
│   │   ├── signals/                # ensemble, generator, voters, persistence, options, types→core
│   │   ├── registry/               # b2_client, model_registry (impl of core.contracts registry), versions, compat
│   │   ├── qlib/                   # serving qlib (engine, ranking, data_handler) — feature math delegated to core.features.factory
│   │   ├── agents/  sentiment/  strategy/  strategy_discovery/  vision/  earnings/  digest/  weekly_review/
│   │   ├── exit_engine/  microstructure/  options_features/  outcome_models/
│   ├── trading/                    # autopilot_service, execution, eligibility, pnl, risk, fo/
│   │   └── risk_engine.py          # NEW (spec §4.6): expected_return+ATR → entry/SL/target. Models NEVER emit levels.
│   ├── data/                       # SERVING data layer
│   │   ├── providers/              # live providers (free/kite/yfinance/nselib/truedata) implementing core.contracts
│   │   ├── brokers/  fundamentals/  reference/  screener/  tick_collector/
│   │   ├── market.py  ohlc_store.py  orderflow_store.py  universe.py  market_calendar.py
│   ├── services/                   # grouped by domain (was 70-file flat)
│   │   ├── screener/  fno/  intraday/  autopilot/  portfolio/  news/  options/  market/  strategy_runner/  assistant/
│   ├── platform/                   # scheduler, depth_bus, realtime, push, events, alerts, referrals, cron_lock, system_flags, whatsapp
│   ├── middleware/  observability/  schemas/  utils/
│
├── frontend/                       # Next.js app (unchanged layout; already clean)
│   ├── app/  components/  lib/  hooks/  contexts/  types/  public/  tests/
│
├── artifacts/                      # NEW: single local artifact root (merges ml/models + models/ + .model_cache)
│   ├── models/<model>/<version>/   # registry-managed; large weights gitignored, sourced from B2
│   └── rl/q_table.json
├── data/                           # STATIC reference inputs only (committed): universes, nse_holidays, nse_tiers, fno_instruments
├── var/                            # NEW (gitignored): runtime caches (bhavcopy, parquet, *_NS_10y.csv) — was data/cache + ml/data/cache
│
├── infra/                          # ← infrastructure/ (renamed)
│   └── database/                   # complete_schema.sql, migrations/, archive/
├── scripts/                        # ONLY orchestration shell + CI gates (business logic moved to ml.cli/backend.cli)
│   ├── runpod/                     # runpod_*.sh, pod_bootstrap.sh, train_momentum_gpu.sh
│   ├── qa/  release/               # CI gate scripts (kept)
│   └── dev.sh  preflight.sh  check_legacy_branding.sh  check_frontend_hex_literals.sh
├── docs/                           # specs/, plans/, audits, feature docs
└── tests/                          # mirrors core/ ml/ backend/ frontend/
    ├── core/  ml/  backend/  integration/  contracts/   # contracts/ = import-linter + feature-parity + serve-smoke guards
```

## Key moves

- `ml/regime_detector.py (compute_regime_features) + ml/features/lgbm_v2.py (compute_inference_features)` → `core/features/regime_features.py + core/features/lgbm_features.py (shared, pure)` — These are imported BY THE SERVING APP at request time (generator.py, scheduler.py, feature_engineering.py) yet live in the research tree, forcing backend->ml coupling and risking train/serve skew. Moving the shared compute into ring-1 core/ lets both the trainer and the live engine import identical code, which is exactly what spec §4.1 mandates and what serve_smoke (Stage 8b) verifies.
- `ml/data/data_loader.py + provider Protocols (currently in src/backend/data/providers/base.py)` → `core/contracts/providers.py (Protocol) + ml/data/data_loader.py (consumes the Protocol)` — Today the canonical loader lives in ml/ but imports concrete providers from src/backend/, creating the cycle. Putting the provider Protocol in core/ breaks the cycle: ml/ depends on the abstract contract, and both the research providers (ml/data/providers/) and serving providers (backend/data/providers/) implement it.
- `ml/scanner.py, ml/risk_manager.py, ml/strategies/, ml/backtest/` → `ml/research/ (and serving usages re-pointed: generator.py scanner import → backend serving engine)` — These rule-based modules are misfiled: some are pure research (backtest harness) and some are pulled into the live app (ml.scanner in generator.py, ml.strategies in signals/options.py). Quarantining them under ml/research/ and replacing the serve-time imports with backend engines makes the training tree free of serving responsibilities.
- `ml/training/trainers/*.py (momentum_lambdarank, tft_swing, regime_hmm, qlib_alpha158, lgbm_signal_gate)` → `ml/training/trainers/*_trainer.py refactored onto the new pipeline.run_pipeline spine via 3 hooks (build_features/build_labels/fit_model) + predict_for_serve` — Each trainer re-implements the 9 stages inline and they have drifted (audit root cause). One template-method spine + per-engine closures collapses the copy-paste drift and gives every engine the same EDA/quality/CV/eval/serve-smoke/promote gate.
- `src/backend/ (run as src.backend.api.app:app)` → `backend/ (run as backend.api.app:app) + pyproject.toml editable install` — Removes the redundant src/ wrapper, gives a stable import root, and lets `import backend`/`import ml`/`import core` resolve without the sys.path.insert hacks now scattered across tests/conftest.py and ~6 backfill scripts. Only 4 deploy files hardcode the old path.
- `ml/models/, top-level models/, .model_cache/` → `artifacts/models/<model>/<version>/ (single root; large weights gitignored + B2-sourced) + artifacts/rl/` — Three overlapping local artifact roots make 'where does a trained model live' ambiguous and contributed to the audit's version-drift finding (served on-disk artifact matched no current trainer). One registry-managed root with versioned subdirs makes provenance unambiguous.
- `scripts/train_*.py, backfill_*.py, retrain_pipeline.py, smoke_all.py, seed_*.py, apply_migrations.py-callers (~30 of 53 scripts)` → `ml/cli.py (training/backfill subcommands) + backend/cli.py (db/seed subcommands); scripts/ keeps only shell orchestration (runpod_*, dev.sh, qa/, release/, check_*.sh)` — Business logic in argv-parsing scripts is untestable and duplicates module code. Folding it behind two Typer CLIs makes it importable, testable, and discoverable while leaving genuine shell orchestration where shell belongs.
- `src/backend/services/ (70-file flat catch-all)` → `backend/services/{screener,fno,intraday,autopilot,portfolio,news,options,market,strategy_runner,assistant}/` — A flat 70-file directory hides domain ownership and only half-realizes the ai/+trading/+data/+platform/ peer target (memory 2026-05-25). Domain subpackages make boundaries and code ownership legible.
- `src/backend/data/providers/* AND ml/data/* duplicated data access` → `core/contracts/providers.py (one Protocol) implemented by backend/data/providers/ (live) and ml/data/providers/ (research/backfill)` — Two parallel data layers with no shared contract drift independently. A single Protocol in core/ with two explicit implementations (live-serving vs research-backfill) makes the split intentional and testable rather than accidental duplication.
- `infrastructure/, data/cache/ + ml/data/cache/` → `infra/ (rename) + var/ (gitignored runtime caches) while data/ keeps ONLY static committed reference inputs` — Separates committed reference data (universes, holidays, tiers, fno_instruments) from regenerable runtime caches (bhavcopy, *_NS_10y.csv, parquet), so the repo stops carrying churny cache files and the static/runtime distinction is enforced by .gitignore.
- `src/backend/api/*_routes.py (50 routers at api/ root)` → `backend/api/routes/*_routes.py` — Pulling routers into a routes/ subpackage (admin/ already follows this) unclutters api/ to just app.py/main.py/bootstrap and groups the router surface for easier nav and ownership.
- `spec-required new serving module: risk engine (currently signals derive levels inside models, e.g. TFT _derive_levels)` → `backend/trading/risk_engine.py` — Spec §4.6 + audit: models must emit only expected_return/confidence; a separate risk engine derives entry/SL/target from ATR. Giving it a dedicated home under trading/ enforces 'models never emit levels' structurally.

## Migration phases

1. Phase 0 — Scaffolding & guards (no moves). Add pyproject.toml (declare future packages + editable install + import-linter config initially in 'report only' mode), requirements/ split, tests/contracts/ with a (currently lenient) import-linter contract and a per-engine-trinity test. Add a CI grep for deprecated-import detection. Goal: tooling exists before any file moves. Gate: full suite + app import green.
2. Phase 1 — Import root flip (src/backend → backend). Rename src/backend → backend (git mv), update the 4 deploy spots (nixpacks.toml, Dockerfile, railway.toml, scripts/dev.sh) to backend.api.app:app, leave a src/backend shim package re-exporting backend.*, remove sys.path hacks now covered by editable install. Codemod internal `src.backend` → `backend` imports. Gate: uvicorn import check + 130 tests green; deploy a preview to confirm boot.
3. Phase 2 — Carve out core/ (break the cycle). Create core/contracts (provider Protocol moved out of backend/data/providers/base.py), core/types (signal schema + EngineSpec family), core/features (move regime_features + lgbm_features + the shared *_features builders), core/labeling (triple_barrier/ranking_labels/sample_weights). Leave shims at every old path. Re-point ml/data/data_loader.py and the serve-time backend imports to core/. Run import-linter in ENFORCE mode for the core->{} no-dep rule. Gate: feature-parity test (train builder == serve builder) added and green; suite green.
4. Phase 3 — Quarantine serving code out of ml/. Move ml/scanner.py, ml/risk_manager.py, ml/strategies/, ml/backtest/ → ml/research/; replace the serve-time imports in backend (generator.py ml.scanner, signals/options.py ml.strategies) with backend engines/shims; move ml/data/providers concrete classes so research providers implement core.contracts. Enforce the ml↛backend import rule in import-linter. Gate: scheduler + signal generation smoke (scripts/staging/verify_signal_runtime.py) green; suite green.
5. Phase 4 — Pipeline spine + trainer refactor (lands with the 4-engine build). Add ml/training/{pipeline.py, specs.py, serve_smoke.py, baseline_drift.py} + ml/training/adapters/. Rename trainers to *_trainer.py and refactor each onto run_pipeline via the 3+1 hooks. Wire serve_smoke as a promote precondition in runner.py. Gate: serve-smoke guard green for every promoted model; per-engine trinity test green.
6. Phase 5 — Services domain-grouping + api/routes/ + risk engine. Group backend/services/ flat files into domain subpackages, move *_routes.py under api/routes/, add backend/trading/risk_engine.py and strip model-derived levels. Shims for moved service modules. Gate: route inventory test (no router dropped) + suite green.
7. Phase 6 — Scripts → CLIs + artifact/data consolidation. Fold train_*/backfill_*/seed_*/smoke_all into ml.cli + backend.cli (keep thin shim scripts during transition); consolidate ml/models + models/ + .model_cache → artifacts/, split data/ into data/ (static) + var/ (gitignored runtime), rename infrastructure/ → infra/. Update .gitignore + registry artifact paths + RunPod scripts. Gate: a training run writes to artifacts/ and the registry resolves it; CLI subcommands tested.
8. Phase 7 — Shim sweep & contract lock. After a soak period (set a removal date per the shim policy), delete all re-export shims, flip import-linter to strict on all three ring rules + the no-deprecated-import grep to hard-fail, rewrite ml/__init__.py docstring + README to the new tree. Gate: zero deprecated imports remain; full contract enforced in CI; clean deploy.