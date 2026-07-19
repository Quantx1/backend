# QuantX — Production Deployment Guide

> Indian equities swing/intraday trading SaaS (SEBI-RA regulated). This document covers **where to deploy, how, and what it costs**.

---

## 1. What you're deploying

QuantX has four independently-hosted parts, plus in-process ML and scheduling.

| Part | Tech | Hosted on (as configured in repo) |
|------|------|-----------------------------------|
| **Frontend** | Next.js 14 / React 18 / Tailwind | **Vercel** |
| **Backend API** | FastAPI + Uvicorn (Python 3.12) | **Railway** (Docker / Nixpacks) |
| **Database + Auth + Realtime** | Supabase (Postgres 17) | **Supabase cloud** |
| **Cache / pub-sub** | Redis | **Railway addon** or Upstash |
| **Model artifacts** | LightGBM / TFT / Chronos | **Backblaze B2** + baked into image |
| ML inference | LightGBM, XGBoost, Chronos, TFT, ONNX | **CPU-only, in-process** |
| Scheduler | APScheduler — 8 daily cron jobs | **in-process** |

**Key fact: ML is CPU-only inference (~1–2 GB RAM). No GPU required.** This is what keeps hosting cheap.

The repo already ships with deployment config — you are filling in secrets, not building from scratch:
- `Dockerfile` (Python 3.12-slim, TA-Lib compiled)
- `nixpacks.toml` (Railway build)
- `railway.toml` (start cmd + `/health` healthcheck)
- `vercel.json` (frontend build)
- `supabase/config.toml` (local dev)
- `.github/workflows/deploy.yml` (CI → auto-deploy backend to Railway, frontend to Vercel)

---

## 2. Architecture

```
                    ┌──────────────┐
   Users  ───────▶  │   Vercel     │   Next.js frontend
                    │ (frontend)   │
                    └──────┬───────┘
                           │  NEXT_PUBLIC_API_URL  (/api, /ws proxy)
                           ▼
                    ┌──────────────┐        ┌───────────────┐
                    │   Railway    │◀──────▶│    Redis      │  pub/sub + locks
                    │ FastAPI +    │        └───────────────┘
                    │ Uvicorn      │
                    │ + ML models  │        ┌───────────────┐
                    │ + scheduler  │◀──────▶│  Supabase     │  Postgres/Auth/Realtime
                    └──────┬───────┘        └───────────────┘
                           │
          ┌────────────────┼────────────────┬─────────────┐
          ▼                ▼                 ▼             ▼
     Zerodha/Angel/    Razorpay         Backblaze B2    Resend / Sentry
     Upstox brokers    (payments)       (model registry) (email / errors)
```

---

## 3. Deployment steps (recommended stack)

Deploy in this order — later steps need values from earlier ones.

### Step 1 — Supabase (database, auth, realtime)
1. Create a project at [supabase.com]. **Region: Mumbai (`ap-south-1`)** for latency to Indian users and broker APIs.
2. Run migrations in `infrastructure/database/migrations/`.
3. Collect these values (used by backend + frontend):
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `SUPABASE_SERVICE_KEY`
   - `SUPABASE_JWT_SECRET`  ← critical for JWT signature verification

### Step 2 — Model artifacts → Backblaze B2
1. Create a B2 bucket (e.g. `swingai-models`).
2. Upload the contents of `artifacts/models/` (LightGBM gate ~15 MB, TFT ckpt ~1.5 MB, momentum ranker).
3. Collect `B2_APPLICATION_KEY_ID`, `B2_APPLICATION_KEY`, `B2_BUCKET_MODELS`.
   - *Chronos 205M is downloaded at runtime from HF — no upload needed, but adds ~500 MB to first-boot memory.*

### Step 3 — Backend → Railway
1. New Railway project → deploy from GitHub repo. `railway.toml` handles build/start/healthcheck.
2. **Provision ≥ 2 GB RAM** (models load at startup; 1 GB will OOM).
3. Add the **Redis** plugin (one-click) → auto-sets `REDIS_URL`. Set `ENABLE_REDIS=True`.
4. Set every backend env var (see §5). Region: Mumbai if available.
5. Confirm `/health` returns 200.

### Step 4 — Frontend → Vercel
1. Import repo; `vercel.json` points builds at `frontend/`.
2. Set env vars:
   - `NEXT_PUBLIC_API_URL` = Railway backend URL
   - `NEXT_PUBLIC_WS_URL` = `wss://<railway-host>`
   - `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`
3. Deploy. Point your custom domain at Vercel.

### Step 5 — CI/CD (already built)
Add these **GitHub repo secrets** so `deploy.yml` auto-deploys on push to `main`:
`RAILWAY_TOKEN`, `VERCEL_TOKEN`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`.

---

## 4. Cost — three tiers (USD/month)

### Tier A — MVP / beta (<100 users)
| Service | Plan | Cost |
|---|---|---|
| Vercel | Hobby (free) | $0 |
| Railway | Backend ~2 GB, usage-based | ~$10–20 |
| Supabase | Free | $0 |
| Redis | Railway addon / Upstash free | $0–5 |
| Backblaze B2 | few GB | <$1 |
| Sentry / PostHog / Resend | Free tiers | $0 |
| Domain | ~$12/yr | ~$1 |
| **Total** | | **~$15–25/mo** |

### Tier B — Production launch (few hundred paying users) ← most likely
| Service | Plan | Cost |
|---|---|---|
| Vercel | Pro | $20 |
| Railway | Backend 2–4 GB always-on + Redis | ~$25–50 |
| Supabase | Pro (8 GB DB, daily backups, no auto-pause) | $25 |
| Backblaze B2 | model + data storage | ~$2–5 |
| Sentry | Team (or free) | $0–26 |
| Resend | free → $20 as volume grows | $0–20 |
| Domain | | ~$1 |
| **Total** | | **~$75–130/mo** |

### Tier C — Scale (thousands of users, multiple replicas)
| Service | Cost |
|---|---|
| Railway backend replicas (4 GB × 2–3) | $100–250 |
| Supabase Pro + compute add-on | $25–100+ |
| Redis (dedicated) | $10–30 |
| Vercel Pro + bandwidth | $20–60 |
| Observability (paid) | $30–100 |
| **Total** | **~$250–600/mo** |

### Pass-through / usage costs (not in tables)
- **Razorpay**: ~2% per transaction (revenue-linked).
- **Broker API subscriptions**: Zerodha Kite Connect ≈ **₹500/user/mo**; Angel SmartAPI free; Upstox free. Can dominate cost if many users connect live Zerodha — factor into pricing.
- **LLM (OpenRouter)**: pay-per-token, only if `ENABLE_ENHANCED_AI=True` (disabled by default → $0).

**Bottom line: launch a real product for ~$75–130/mo; beta for under $25/mo.**

---

## 5. Environment variables

Full canonical list is in `.env.example`. Grouped essentials:

**Application**
`APP_ENV=production`, `DEBUG=False`, `SECRET_KEY`, `APP_VERSION`, `ADMIN_EMAILS`, `LOG_LEVEL=INFO`

**Supabase**
`SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `SUPABASE_JWT_SECRET`

**Brokers**
`ZERODHA_API_KEY`, `ZERODHA_API_SECRET`, `ZERODHA_REDIRECT_URI`,
`KITE_ADMIN_API_KEY`, `KITE_ADMIN_API_SECRET`, `KITE_ADMIN_ACCESS_TOKEN`, `KITE_ADMIN_USER_ID`, `KITE_ADMIN_PASSWORD`, `KITE_ADMIN_TOTP_SECRET`,
`ANGEL_API_KEY`, `ANGEL_REDIRECT_URI`,
`UPSTOX_API_KEY`, `UPSTOX_API_SECRET`, `UPSTOX_REDIRECT_URI`,
`BROKER_ENCRYPTION_KEY`  ← encrypts stored broker tokens; keep strong, never in git

**Payments**
`RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`, `RAZORPAY_WEBHOOK_SECRET`

**Storage / cache**
`B2_APPLICATION_KEY_ID`, `B2_APPLICATION_KEY`, `B2_BUCKET_MODELS`,
`REDIS_URL`, `REDIS_PASSWORD`, `ENABLE_REDIS=True`, `MODEL_CACHE_DIR`, `MODEL_STORAGE_BUCKET`

**Market data / ML**
`DATA_PROVIDER=free`, `ML_MODEL_PATH`, `TFT_MODEL_PATH`, `TFT_CONFIG_PATH`, `XGBOOST_MODEL_PATH`

**Feature flags**
`ENABLE_SCHEDULER=True`, `ENABLE_BROKER_TICKER`, `ENABLE_ENHANCED_AI=False`

**Notifications / observability**
`RESEND_API_KEY`, `EMAIL_FROM`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
`VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY`, `VAPID_CLAIMS_EMAIL`, `SENTRY_DSN`

**Compliance**
`SEBI_RA_REG_NUMBER`, `SEBI_RA_VALID_UNTIL`, `REQUIRE_SUITABILITY_QUIZ_FOR_LIVE=True`

**Frontend (Vercel)**
`NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_WS_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`

---

## 6. Pre-launch checklist (fix before real money)

- [ ] **Scheduler single-runner.** APScheduler runs in-process. With >1 Railway replica, cron jobs (EOD scan, token refresh, trade execution) fire on **every** replica → duplicate trades. Fix: run the scheduler in a **single dedicated worker service** (1 replica) with stateless API replicas, OR guard each job with a Redis lock. Critical for live trading.
- [ ] **Supabase Pro** before real users — the free tier auto-pauses and has no daily backups. Unacceptable for a trading DB.
- [ ] **Region alignment** — Railway + Supabase + users all in Mumbai / `ap-south-1`. Broker APIs are India-hosted; cross-region latency hurts order execution.
- [ ] **Strong secrets** — `SUPABASE_JWT_SECRET`, `BROKER_ENCRYPTION_KEY`, `SECRET_KEY`. Plan encryption-key rotation for broker tokens.
- [ ] **SEBI compliance** — valid `SEBI_RA_REG_NUMBER`, suitability quiz gate (`REQUIRE_SUITABILITY_QUIZ_FOR_LIVE=True`) enforced before enabling live (not paper) trading.
- [ ] **Backend RAM ≥ 2 GB** — verify no OOM on model load at startup.
- [ ] **Razorpay + broker webhooks** point at production URLs; verify signature secrets.
- [ ] **Sentry DSN set**, healthcheck green, `/health` monitored (Railway restarts on failure, max 10 retries).

---

## 7. Known gaps (not blockers, but plan for them)

- No Kubernetes / Terraform — Railway + Vercel are managed; infra is click-configured, not IaC.
- No documented DB replica/backup strategy beyond Supabase managed backups.
- No emergency broker-token refresh endpoint (refresh is daily cron only).
- Model registry (B2) credentials are set up manually.

---

*Generated as a deployment reference. Stack, costs, and gaps reflect the repo state as explored; verify pricing against current Vercel/Railway/Supabase plans before committing.*
