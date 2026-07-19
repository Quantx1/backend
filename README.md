# Quant X — Backend

FastAPI backend for the Quant X AI/ML trading-signal platform (4-engine ML/DL, signals, screener, copilot, broker + paper trading).

Part of the [Quantx1](https://github.com/Quantx1) org: [landing](https://github.com/Quantx1/landing) · [frontend](https://github.com/Quantx1/frontend) · **backend** · [ml](https://github.com/Quantx1/ml)

## Setup

```bash
# IMPORTANT: the ml package is a git submodule at ./ml
git clone --recurse-submodules https://github.com/Quantx1/backend.git
cd backend
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you cloned without `--recurse-submodules`, run `git submodule update --init`.

## Run

```bash
uvicorn backend.api.app:app --reload --port 8000
```

Health check: `GET /health`.

## Test

```bash
pytest tests/
```

## Layout

- `backend/` — the application package (api, ai, services, trading, data, platform)
- `ml/` — git submodule → [Quantx1/ml](https://github.com/Quantx1/ml) (feature engineering, trainers, regime detection)
- `artifacts/` — trained model artifacts loaded at serve time
- `data/` — NSE tier lists + paper-trading baselines
- `scripts/` — ops/train/eval/backtest scripts (imported by some backend modules)
- `infrastructure/database/` — SQL migrations
- `docs/` — project documentation and audits

## Deploy

Railway/Nixpacks config in `railway.toml` + `nixpacks.toml`; Docker via `Dockerfile`. The build must fetch submodules (`git submodule update --init` pre-build, or enable submodule checkout in your platform settings).

### Updating the ml submodule

```bash
cd ml && git pull origin main && cd ..
git commit -am "bump ml submodule"
```
