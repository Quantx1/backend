# Quant X — BYOB Broker Connection + Honest Data Model

- **Date:** 2026-06-22
- **Status:** Approved design → implementation plan next
- **Branch:** feat/xai-redesign
- **Owner:** Rishi

## 1. Context & problem

Quant X needs users to connect their own broker ("bring your own broker" / BYOB)
so the app can show **live** market data sourced from each user's *own* exchange
entitlement, rather than redistributing exchange data centrally (which needs an
NSE/BSE data-vendor license). The broker plumbing is already ~90% built (OAuth
endpoints for Zerodha/Upstox/Angel, encrypted token storage, a broker-connect
onboarding page, the `/broker/callback` handler, settings tiles, a
`useBrokerStatus` hook, and a `BrokerLock` gate). What's missing is: an easy,
one-click, *early* connect experience; correct data-source labeling so nothing
is misrepresented as live; one schema bug; and production-grade OAuth-state +
token-refresh handling.

### Regulatory framing (decided)

Two independent axes — do not conflate:

1. **Exchange market-data licensing (NSE/BSE).** Displaying/redistributing
   quotes, depth, OHLC to users needs an exchange data license. **BYOB addresses
   this for per-user, in-session data** (the user views data sourced from their
   own broker entitlement — a "tool pointed at your own feed", like Streak /
   Sensibull). It does **not** legitimize centralized, market-wide display.
2. **SEBI RA/IA (Research Analyst / Investment Adviser).** Paid stock-specific
   buy/sell recommendations may require registration **regardless of data
   source**. Out of scope here (founder legal task), but flagged.

**Decision:** soft + prominent BYOB. Centralized market-wide surfaces are
sourced from EOD/permissible data and **clearly labeled "Delayed · EOD
research"**; live data lights up **only** per-symbol from the user's broker once
connected. This is the cheapest defensible path with no exchange license today
(legal review still recommended on the signals/RA axis).

> Not legal advice. A SEBI / exchange-data compliance opinion should confirm
> this model before public launch.

## 2. Goals / non-goals

**Goals**
- One-click, *early* (broker-first) broker connection after signup, with an
  explicit "explore with a virtual ₹10L portfolio" escape.
- Inline Angel One credential entry during onboarding (no bounce-to-settings).
- Persistent, app-wide "connect your broker" nudge until connected.
- Correct, honest data labeling: **Live · your broker** vs **EOD research**.
- Fix `/broker/connections` (missing `expires_at` column).
- Production-grade OAuth state (Redis) and proactive token refresh.

**Non-goals**
- Registering broker OAuth developer apps / supplying `*_API_KEY/SECRET/REDIRECT_URI`
  + `BROKER_ENCRYPTION_KEY` (founder action; without these OAuth cannot complete).
- Acquiring the NSE EOD/derived data license (deferred; EOD surfaces use
  permissible/free EOD sources + clear labeling until then).
- SEBI RA/IA registration / legal opinion (founder legal task).
- Hard-gating the app behind a broker (explicitly rejected — preserves the
  free/paper/EOD-research funnel).

## 3. Current state (verified)

- **Backend** `backend/api/broker_routes.py`: OAuth initiate/callback for
  Zerodha (Kite) + Upstox, credential connect for Angel One + Zerodha enctoken,
  disconnect, `GET /broker/connections`, positions/holdings/margin. OAuth
  `state` held in an **in-memory dict** `_oauth_states` (10-min TTL). **Bug:**
  `/broker/connections` selects `expires_at`, which is **absent** from the
  `broker_connections` table → endpoint errors.
- **Adapters** `backend/data/brokers/integration.py`: Zerodha/Angel/Upstox
  implemented; Angel + Upstox have `refresh_session()`; Zerodha enctoken has no
  refresh (needs re-login from stored creds). `credentials.py` = Fernet encrypt.
- **Frontend**: `app/onboarding/{risk-quiz,broker-connect,complete}` all wired;
  platform layout bounces incomplete users to **risk-quiz first**. `broker/callback`
  is complete but returns to `/settings`. `lib/hooks/useBrokerStatus.ts` is the
  client source of truth. `components/broker/BrokerLock.tsx` gates live surfaces.
  Angel One in onboarding bounces to settings with an error. Landing copy
  (`app/page.tsx`) has a broken half-sentence. No broker CTA in signup/login.
- **Gating**: soft already — live AutoPilot/execute enforce broker server-side
  (`auto_trader_routes.py`); paper/signals/scanner need no broker.

## 4. Design

### 4.1 Broker-first onboarding (re-sequence)
- After signup, the onboarding redirect target becomes
  `/onboarding/broker-connect` (was `/onboarding/risk-quiz`). New order:
  **connect → risk-quiz → complete**.
- `app/(platform)/layout.tsx` onboarding bounce updated to send incomplete users
  to broker-connect first (or whichever onboarding step is unfinished).
- The broker-connect page gains a prominent **"Skip — explore with a virtual
  ₹10L portfolio"** action that advances to the risk quiz without connecting.
- On successful OAuth callback, return **into onboarding** (→ risk-quiz) rather
  than `/settings`. The callback distinguishes onboarding vs settings origin via
  a `return_to` param stored alongside the OAuth state.

### 4.2 Connect UX
- One-click OAuth buttons (Zerodha/Upstox) with loading state (existing
  `initiateOAuth` → broker consent → `/broker/callback`).
- **Inline Angel One credential form** on the broker-connect page (reuse the
  fields from the settings Angel modal) → `api.broker.connect(...)`; no bounce.
- "Connected ✓" confirmation surfaced in `onboarding/complete` (read
  `useBrokerStatus`), and which broker.
- **Unconfigured-broker degraded state:** when a broker's OAuth keys are not
  configured server-side, the initiate endpoint returns a typed
  `broker_not_configured` response and the tile shows a clear "Not available yet"
  state instead of a raw error.

### 4.3 Persistent connect banner + soft gating
- A dismissible (per session) app-wide banner: "Connect your broker to unlock
  live data + trading", shown when `useBrokerStatus().isConnected === false`,
  with a one-click connect action. Rendered in the platform shell; hidden on
  auth/onboarding routes.
- Live-only surfaces keep their existing `BrokerLock` gate. No new hard gates.

### 4.4 Honest data model + DataBadge
- New `<DataBadge mode="live" | "eod" />` component:
  - **"Live · your broker"** — rendered on per-user surfaces fed by BYOB
    (symbol quote/depth, watchlist, positions, P&L, charts) **only when
    connected**; otherwise those surfaces show EOD or an empty/connect state.
  - **"EOD research"** — rendered on centralized market-wide surfaces (AI
    Scanner, top movers on `/stocks`, universe signals) which are sourced from
    EOD/permissible data and must never be presented as live.
- Audit the centralized surfaces and attach the EOD badge + reinforce the
  "research / educational, not investment advice" disclaimer.
- Per-user live numbers must resolve their source from the user's broker
  connection; when disconnected they fall back to EOD or a connect prompt — never
  to a centralized live feed.

### 4.5 Backend: `expires_at` + schema
- Add `expires_at TIMESTAMPTZ` to `broker_connections` (additive migration via
  `scripts/apply_migrations.py`; reflect in `complete_schema.sql` Part B).
- Persist token expiry on connect/callback where the broker returns it (Upstox
  expiry, Angel JWT expiry, Kite daily expiry ≈ next 06:00 IST). `/broker/connections`
  returns it (already expected by the client type).

### 4.6 Infra: OAuth state → Redis (in scope)
- Replace the in-memory `_oauth_states` dict with Redis-backed state:
  `SETEX oauth_state:{state} 600 {json:{user_id,broker,return_to,created_at}}`;
  on callback `GET`+`DEL`, verify presence + broker match. Reuse the existing
  Redis client (same one powering DepthBus). Makes OAuth correct across restarts
  and multiple instances.
- If Redis is unavailable at runtime: fail the initiate with a clear 503 rather
  than silently using process-local memory (correctness over degraded multi-instance).

### 4.7 Infra: proactive token refresh (in scope)
- `_ensure_fresh(connection)` helper invoked before any broker API use
  (positions/holdings/margin/quote/depth): if `expires_at` is within a threshold
  (e.g. 5 min) or past, attempt refresh:
  - **Upstox / Angel One:** call `refresh_session()`; on success update encrypted
    `access_token` + `expires_at`.
  - **Zerodha enctoken (stored password+TOTP):** re-run `_zerodha_auto_login`.
  - **Zerodha Kite OAuth (no stored creds):** cannot refresh silently → set
    `status='expired'` so the UI prompts reconnect.
- On refresh failure → mark `status='expired'`; the connect banner + settings
  tile surface "reconnect". Optional lightweight scheduled sweep to refresh
  connections nearing expiry (sub-task; on-demand refresh is the core).

## 5. Data flow

1. **Signup** → backend creates user (`onboarding_completed=false`) → auto-login
   → onboarding redirect → **broker-connect** (first).
2. **Connect** → `initiateOAuth(broker)` → backend stores state in Redis →
   returns `auth_url` → broker consent → `/broker/callback?code|request_token&state`
   → backend verifies Redis state, exchanges token, encrypts + upserts
   `broker_connections` (+ `expires_at`), sets `user_profiles.broker_connected`
   → callback returns to onboarding (`return_to`) → risk-quiz → complete → dashboard.
   (Angel One: inline form → `connect()` → same persistence, no redirect.)
3. **Data resolution per surface:** per-user live → BYOB connection (refresh if
   stale) when connected, else EOD/empty; centralized → EOD source + EOD badge.

## 6. Error handling
- **Broker keys unconfigured:** typed `broker_not_configured` → tile "Not
  available yet" (no raw 500).
- **OAuth failure / state missing/expired:** callback shows a clear error + a
  retry link back to the connect step.
- **Redis down on initiate:** 503 with a user-facing "try again shortly".
- **Token expired / refresh failed:** `status='expired'` → reconnect prompt;
  live surfaces fall back to EOD/empty, never to a centralized live feed.
- **Disconnected:** live surfaces show the connect banner / `BrokerLock`.

## 7. Testing
- Backend: `expires_at` migration applied + `/broker/connections` returns it;
  Redis-backed state set/verify/expire; `_ensure_fresh` refresh paths per broker
  (mock adapters), including the Kite-OAuth "can't refresh → expired" branch;
  `broker_not_configured` typed response when keys absent.
- Frontend: onboarding order (connect→quiz→complete) + skip path; inline Angel
  form submit; callback `return_to` routing (onboarding vs settings); banner
  visibility keyed on `useBrokerStatus`; DataBadge renders live vs EOD per surface.
- Route-seam / brand-firewall tests stay green; no model names in UI.

## 8. Phasing
- **Phase 1 — BYOB connect flow + infra:** onboarding re-sequence, inline Angel,
  skip-to-virtual, callback `return_to`, connect banner, `expires_at` migration,
  Redis OAuth state, `_ensure_fresh` token refresh, landing-copy fix.
- **Phase 2 — Honest data labeling:** `<DataBadge>` + audit/label centralized
  surfaces as EOD research; ensure per-user live resolves only from BYOB.

## 9. Out of scope (founder / deferred)
- Registering broker OAuth developer apps + setting env keys / `BROKER_ENCRYPTION_KEY`.
- NSE EOD/derived data license (EOD surfaces use permissible sources + labeling until then).
- SEBI RA/IA legal opinion / registration.

## 10. Open / legal items
- Confirm each broker's API terms permit per-user in-session display (they
  generally do; no caching/redistribution/cross-user use).
- SEBI RA/IA opinion on paid signals before charging.
- Exchange-data compliance opinion on the EOD-research labeling.
