# BYOB Broker Connection + Honest Data Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make connecting a broker a one-click, broker-first onboarding step, label data honestly (per-user LIVE via the user's broker vs centralized "EOD research"), fix the `expires_at` bug, and make OAuth state + token refresh production-grade.

**Architecture:** Reuse the existing ~90%-built broker plumbing. Backend: move OAuth `state` from an in-process dict to Redis, add an `expires_at` column, add a refresh-before-use helper, and return a typed `broker_not_configured` signal. Frontend: re-sequence onboarding (connect → quiz → complete), add inline Angel One, route the callback back to its origin, add an app-wide connect banner, and add a `DataBadge` to mark Live vs EOD surfaces.

**Tech Stack:** FastAPI + Supabase (Postgres) + `redis.asyncio` (backend, repo root `backend/`); Next.js 14 App Router + SWR + Tailwind (frontend `frontend/`); pytest at repo-root `tests/`; frontend verified via `npx tsc --noEmit` + Playwright render checks (no jest/vitest in repo). Spec: `docs/superpowers/specs/2026-06-22-quantx-byob-broker-and-data-model-design.md`.

**Conventions:**
- Branch: `feat/xai-redesign`. Commit after each task. Co-author trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- No raw hex in Tailwind classNames (use tokens / `bg-signature` etc.); no model names in UI (brand firewall).
- Backend migrations applied via `.venv/bin/python scripts/apply_migrations.py` (DATABASE_URL, port 5432).

---

## File structure

**Create**
- `backend/platform/oauth_state.py` — async Redis-backed OAuth state (store/consume, TTL, broker + return_to payload).
- `backend/data/brokers/freshness.py` — `ensure_fresh(connection)` refresh-before-use + `compute_expiry(broker, token_payload)`.
- `infrastructure/database/migrations/2026-06-22_broker_expires_at.sql` — additive `expires_at` column.
- `frontend/components/broker/ConnectBrokerBanner.tsx` — app-wide "connect your broker" nudge.
- `frontend/components/common/DataBadge.tsx` — "Live · your broker" vs "EOD research" badge.
- `tests/api/test_oauth_state.py`, `tests/data/brokers/test_freshness.py` — backend unit tests.

**Modify**
- `backend/api/broker_routes.py` — replace `_oauth_states`/`generate_state`/`verify_state` with the Redis module; thread `return_to`; call `ensure_fresh` before data calls; persist `expires_at`; return `broker_not_configured`.
- `infrastructure/database/complete_schema.sql` — reflect `expires_at` in Part B.
- `frontend/app/(platform)/layout.tsx:53` — onboarding bounce → `/onboarding/broker-connect`.
- `frontend/app/onboarding/broker-connect/page.tsx` — step label, skip → risk-quiz, set `broker_oauth_return`, inline Angel One form.
- `frontend/app/broker/callback/page.tsx` — read `broker_oauth_return`, route there on success.
- `frontend/components/<shell>` (platform layout) — mount `ConnectBrokerBanner`.
- `frontend/app/page.tsx` — fix the broken landing sentence.
- Centralized surfaces (`frontend/app/scanner/*`, `frontend/app/stocks/page.tsx`, `frontend/app/signals/*`) — attach `DataBadge mode="eod"`; per-user live surfaces get `mode="live"`.

---

# Phase 1 — BYOB connect flow + infra

## Task 1: Add `expires_at` column to `broker_connections`

**Files:**
- Create: `infrastructure/database/migrations/2026-06-22_broker_expires_at.sql`
- Modify: `infrastructure/database/complete_schema.sql` (Part B broker_connections block)

- [ ] **Step 1: Write the migration**

```sql
-- 2026-06-22 — broker_connections.expires_at: token expiry tracking.
-- /broker/connections + freshness checks read this; previously absent → endpoint errored.
ALTER TABLE public.broker_connections
    ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;
```

- [ ] **Step 2: Apply it**

Run: `.venv/bin/python scripts/apply_migrations.py`
Expected: applies cleanly; re-run is a no-op (`IF NOT EXISTS`).

- [ ] **Step 3: Verify the column exists**

Run: `.venv/bin/python - <<'PY'`
```python
import sys; sys.path.insert(0,'.')
from dotenv import load_dotenv; load_dotenv('.env')
from backend.data.ohlc_store import pg_connect
c=pg_connect(); cur=c.cursor()
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='broker_connections' AND column_name='expires_at'")
print('expires_at present:', bool(cur.fetchone())); c.close()
```
Expected: `expires_at present: True`

- [ ] **Step 4: Reflect in complete_schema.sql** — in the `broker_connections` CREATE TABLE block add `expires_at TIMESTAMPTZ,` after `last_synced_at`.

- [ ] **Step 5: Commit**

```bash
git add infrastructure/database/migrations/2026-06-22_broker_expires_at.sql infrastructure/database/complete_schema.sql
git commit -m "fix(broker): add broker_connections.expires_at column"
```

## Task 2: Redis-backed OAuth state module

**Files:**
- Create: `backend/platform/oauth_state.py`
- Test: `tests/api/test_oauth_state.py`

- [ ] **Step 1: Write the failing test** (uses `fakeredis.aioredis`; falls back to skip if unavailable)

```python
import pytest, asyncio
fakeredis = pytest.importorskip("fakeredis")
from backend.platform import oauth_state as os_mod

@pytest.fixture
def store(monkeypatch):
    from fakeredis import aioredis as fra
    client = fra.FakeRedis(decode_responses=True)
    monkeypatch.setattr(os_mod, "_get_redis", lambda: client)
    return os_mod

def test_store_then_consume_roundtrips(store):
    state = asyncio.run(store.store_state("u1", "zerodha", return_to="onboarding"))
    data = asyncio.run(store.consume_state(state))
    assert data == {"user_id": "u1", "broker": "zerodha", "return_to": "onboarding"}

def test_consume_is_single_use(store):
    state = asyncio.run(store.store_state("u1", "upstox", return_to="settings"))
    assert asyncio.run(store.consume_state(state)) is not None
    assert asyncio.run(store.consume_state(state)) is None  # already consumed

def test_unknown_state_returns_none(store):
    assert asyncio.run(store.consume_state("nope")) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/api/test_oauth_state.py -v`
Expected: FAIL (`module backend.platform.oauth_state not found`).

- [ ] **Step 3: Implement the module**

```python
"""Redis-backed OAuth state for broker connect flows.

Replaces the in-process dict (lost on restart / wrong across instances).
State is single-use with a 10-minute TTL. Payload carries the user, broker,
and where to return the user after a successful callback.
"""
from __future__ import annotations
import json
import secrets
from typing import Optional

import redis.asyncio as aioredis

from backend.core.config import settings

_TTL_SECONDS = 600
_PREFIX = "oauth_state:"
_client: Optional["aioredis.Redis"] = None


def _get_redis() -> "aioredis.Redis":
    global _client
    if _client is None:
        _client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _client


async def store_state(user_id: str, broker: str, return_to: str = "settings") -> str:
    state = secrets.token_urlsafe(32)
    payload = json.dumps({"user_id": user_id, "broker": broker, "return_to": return_to})
    await _get_redis().setex(_PREFIX + state, _TTL_SECONDS, payload)
    return state


async def consume_state(state: str) -> Optional[dict]:
    if not state:
        return None
    r = _get_redis()
    key = _PREFIX + state
    raw = await r.get(key)
    if raw is None:
        return None
    await r.delete(key)  # single-use
    return json.loads(raw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/api/test_oauth_state.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/platform/oauth_state.py tests/api/test_oauth_state.py
git commit -m "feat(broker): Redis-backed OAuth state (multi-instance safe)"
```

## Task 3: Wire initiate/callback to Redis state + `return_to`

**Files:**
- Modify: `backend/api/broker_routes.py` (remove `_oauth_states` line 54, `generate_state` 198-207, `verify_state` 210-222; update all initiate/callback endpoints)

- [ ] **Step 1: Delete the in-memory state.** Remove line 54 `_oauth_states: Dict[str, Dict] = {}` and the `generate_state`/`verify_state` functions (lines 198-222).

- [ ] **Step 2: Import the module** near the top imports:

```python
from backend.platform import oauth_state
```

- [ ] **Step 3: Update each initiate endpoint** — replace `state = generate_state(user.id, "<broker>")` with:

```python
return_to = (payload.return_to if payload and getattr(payload, "return_to", None) else "settings")
state = await oauth_state.store_state(user.id, "<broker>", return_to=return_to)
```

For the GET-style initiates (`zerodha_auth_initiate` etc. take no body), accept an optional `return_to: str = Query("settings")` param and pass it through.

- [ ] **Step 4: Update each callback endpoint** — replace `state_data = verify_state(state)` with:

```python
state_data = await oauth_state.consume_state(state)
if not state_data:
    raise HTTPException(status_code=400, detail="invalid_or_expired_state")
```

After a successful token exchange, include `return_to` in the JSON response:

```python
return {"success": True, "broker": "<broker>", "account_id": account_id,
        "return_to": state_data.get("return_to", "settings")}
```

- [ ] **Step 5: Verify import + app boot**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0,'.'); import backend.api.broker_routes; print('import ok')"`
Expected: `import ok` (no NameError from removed helpers).

- [ ] **Step 6: Commit**

```bash
git add backend/api/broker_routes.py
git commit -m "feat(broker): use Redis OAuth state + thread return_to through callbacks"
```

## Task 4: Proactive token refresh + expiry persistence

**Files:**
- Create: `backend/data/brokers/freshness.py`
- Test: `tests/data/brokers/test_freshness.py`
- Modify: `backend/api/broker_routes.py` (`_load_broker` / data endpoints 822-933 call `ensure_fresh`; connect/callback persist `expires_at`)

- [ ] **Step 1: Write the failing test**

```python
import asyncio
from datetime import datetime, timedelta, timezone
from backend.data.brokers import freshness

def _iso(dt): return dt.astimezone(timezone.utc).isoformat()

def test_fresh_token_not_refreshed():
    exp = _iso(datetime.now(timezone.utc) + timedelta(hours=2))
    called = {"n": 0}
    class A:  # adapter
        def refresh_session(self): called["n"] += 1; return True
    assert freshness.needs_refresh(exp, threshold_s=300) is False

def test_expired_token_needs_refresh():
    exp = _iso(datetime.now(timezone.utc) - timedelta(minutes=1))
    assert freshness.needs_refresh(exp, threshold_s=300) is True

def test_kite_oauth_cannot_refresh_marks_expired():
    # Kite OAuth (no stored creds) -> refreshable() False
    assert freshness.refreshable("zerodha", {"access_token": "x", "api_key": "k"}) is False
    # Zerodha enctoken (stored password+totp) -> True
    assert freshness.refreshable("zerodha", {"password": "p", "totp_secret": "s"}) is True
    assert freshness.refreshable("upstox", {"refresh_token": "r", "api_key": "k", "api_secret": "x"}) is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/brokers/test_freshness.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `freshness.py`**

```python
"""Refresh-before-use for stored broker credentials.

`needs_refresh` decides from expires_at; `refreshable` decides whether a silent
refresh is even possible for the broker + stored creds (Kite OAuth tokens are
daily and can't be refreshed silently → caller marks the connection expired).
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional


def needs_refresh(expires_at_iso: Optional[str], threshold_s: int = 300) -> bool:
    if not expires_at_iso:
        return False  # unknown expiry → don't churn; rely on 401 handling
    try:
        exp = datetime.fromisoformat(expires_at_iso)
    except ValueError:
        return False
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return (exp - datetime.now(timezone.utc)).total_seconds() <= threshold_s


def refreshable(broker: str, creds: dict) -> bool:
    if broker == "upstox":
        return all(creds.get(k) for k in ("refresh_token", "api_key", "api_secret"))
    if broker == "angelone":
        return bool(creds.get("refresh_token") or (creds.get("password") and creds.get("totp_secret")))
    if broker == "zerodha":
        # enctoken auto-login needs stored password+TOTP; Kite OAuth tokens cannot refresh silently
        return bool(creds.get("password") and creds.get("totp_secret"))
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/brokers/test_freshness.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Wire into broker_routes data path.** In `_load_broker` (or each of positions/holdings/margin at 822-933), after decrypting creds and before the broker API call, add:

```python
from backend.data.brokers import freshness
if freshness.needs_refresh(conn.get("expires_at")):
    creds = dict(decrypted)
    if freshness.refreshable(broker_name, creds) and broker.refresh_session():
        # re-encrypt + persist new token; recompute expires_at
        supabase_admin.table("broker_connections").update({
            "access_token": encrypt_credentials({**creds, "access_token": broker.access_token}),
            "expires_at": _compute_expiry(broker_name, broker),
        }).eq("user_id", user.id).eq("broker_name", broker_name).execute()
    else:
        supabase_admin.table("broker_connections").update({"status": "expired"}).eq(
            "user_id", user.id).eq("broker_name", broker_name).execute()
        raise HTTPException(status_code=409, detail="broker_token_expired")
```

Add a small `_compute_expiry(broker, adapter)` helper near the top: Upstox/Angel from the token payload if present; Kite → next 06:00 IST; default `None`.

- [ ] **Step 6: Persist `expires_at` on connect/callback.** In each successful connect/callback upsert, add `"expires_at": _compute_expiry(broker, broker_adapter)` to the row.

- [ ] **Step 7: Verify import**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0,'.'); import backend.api.broker_routes; print('ok')"`
Expected: `ok`.

- [ ] **Step 8: Commit**

```bash
git add backend/data/brokers/freshness.py tests/data/brokers/test_freshness.py backend/api/broker_routes.py
git commit -m "feat(broker): proactive token refresh before use + expiry persistence"
```

## Task 5: `broker_not_configured` typed response

**Files:**
- Modify: `backend/api/broker_routes.py` (each initiate endpoint)

- [ ] **Step 1: Guard each initiate** — before building the auth URL, if the broker's API key env is blank:

```python
if not settings.ZERODHA_API_KEY:   # ANGEL_API_KEY / UPSTOX_API_KEY respectively
    raise HTTPException(status_code=409, detail="broker_not_configured")
```

- [ ] **Step 2: Verify** the import still boots (`python -c "import backend.api.broker_routes"`). Expected: ok.

- [ ] **Step 3: Commit**

```bash
git add backend/api/broker_routes.py
git commit -m "feat(broker): typed broker_not_configured when OAuth keys absent"
```

## Task 6: Broker-first onboarding (re-sequence)

**Files:**
- Modify: `frontend/app/(platform)/layout.tsx:53`
- Modify: `frontend/app/onboarding/broker-connect/page.tsx`

- [ ] **Step 1: Repoint the onboarding bounce.** In `layout.tsx`, change line 53:

```tsx
if (!s.completed) router.replace('/onboarding/broker-connect')
```

- [ ] **Step 2: Re-sequence the connect page.** In `broker-connect/page.tsx`:
  - Change the eyebrow `Step 2 of 3` → `Step 1 of 3`.
  - Change the skip button `router.push('/onboarding/complete')` → `router.push('/onboarding/risk-quiz')`, label `Skip — explore with a virtual ₹10L portfolio`.
  - Replace the `← Risk profile` back link with nothing (this is now the first step), or a plain "You can connect later in Settings → Broker" note.
  - In `onConnect`, before redirect, set the return origin:

```tsx
try { sessionStorage.setItem('broker_oauth_return', 'onboarding') } catch {}
```

- [ ] **Step 3: Verify tsc**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 4: Verify render** (Playwright): navigate `http://localhost:3000/onboarding/broker-connect` → assert eyebrow reads "Step 1 of 3" and the skip button text contains "virtual ₹10L".

- [ ] **Step 5: Commit**

```bash
git add frontend/app/'(platform)'/layout.tsx frontend/app/onboarding/broker-connect/page.tsx
git commit -m "feat(onboarding): broker-first flow (connect -> quiz -> complete)"
```

## Task 7: Inline Angel One on the onboarding connect page

**Files:**
- Modify: `frontend/app/onboarding/broker-connect/page.tsx`

- [ ] **Step 1: Add inline Angel form state + UI.** Angel One is credential-based (no OAuth redirect). For the `angel` card, instead of `onConnect`, toggle an inline form (API key, client id, password, TOTP secret) that submits to `api.broker.connect({ broker_name: 'angelone', ... })`. On success, `mutate` the broker status and advance to `/onboarding/risk-quiz`. Reuse the field set from the settings Angel modal (`app/settings/page.tsx` Angel modal).

```tsx
const [angelOpen, setAngelOpen] = useState(false)
const [angel, setAngel] = useState({ api_key: '', client_id: '', password: '', totp_secret: '' })
const submitAngel = async () => {
  setPending('angel')
  try {
    await api.broker.connect({ broker_name: 'angelone', ...angel })
    toast.success('Angel One connected')
    router.push('/onboarding/risk-quiz')
  } catch (e) {
    toast.error('Angel One connect failed', { description: handleApiError(e) })
    setPending(null)
  }
}
```

- [ ] **Step 2: Render the inline form** under the Angel card when `angelOpen` (4 inputs + a Connect button), instead of bouncing.

- [ ] **Step 3: Verify tsc** — `cd frontend && npx tsc --noEmit`. Expected: clean.

- [ ] **Step 4: Verify render** (Playwright): click the Angel One card → inline form fields appear (no navigation away).

- [ ] **Step 5: Commit**

```bash
git add frontend/app/onboarding/broker-connect/page.tsx
git commit -m "feat(onboarding): inline Angel One credential connect (no settings bounce)"
```

## Task 8: Callback returns to its origin

**Files:**
- Modify: `frontend/app/broker/callback/page.tsx`

- [ ] **Step 1: Read the return origin** near the top of the effect (after reading state/broker):

```tsx
const returnTo = sessionStorage.getItem('broker_oauth_return') || ''
sessionStorage.removeItem('broker_oauth_return')
const successDest = returnTo === 'onboarding' ? '/onboarding/risk-quiz' : '/settings'
```

- [ ] **Step 2: Use it on success** — replace the two `router.push('/settings')` success redirects (line 85) and the success copy with `router.push(successDest)` and "Redirecting…". The backend response's `return_to` may also be used if present (prefer the server value when the fetch returns JSON with `return_to`).

- [ ] **Step 3: Verify tsc** — `cd frontend && npx tsc --noEmit`. Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add frontend/app/broker/callback/page.tsx
git commit -m "feat(broker): callback returns to onboarding or settings by origin"
```

## Task 9: Persistent "connect your broker" banner

**Files:**
- Create: `frontend/components/broker/ConnectBrokerBanner.tsx`
- Modify: the platform shell that wraps authed pages (mount the banner; e.g. `frontend/components/shell/AppShell.tsx` main pane or `app/(platform)/layout.tsx`)

- [ ] **Step 1: Implement the banner** (uses the existing `useBrokerStatus` hook; dismissible per session; hidden when connected or dismissed):

```tsx
'use client'
import { useState } from 'react'
import Link from 'next/link'
import { Zap, X } from '@/lib/icons'
import { useBrokerStatus } from '@/lib/hooks/useBrokerStatus'

export function ConnectBrokerBanner() {
  const { isConnected, isLoading } = useBrokerStatus()
  const [dismissed, setDismissed] = useState(false)
  if (isLoading || isConnected || dismissed) return null
  return (
    <div className="flex items-center gap-3 border-b border-line bg-wrap px-4 py-2 text-[12.5px] text-d-text-secondary">
      <Zap className="h-4 w-4 shrink-0 text-signature" />
      <span className="min-w-0 flex-1">
        Connect your broker to unlock <span className="text-d-text-primary">live data</span> and live trading. Until then you're on the virtual ₹10L portfolio.
      </span>
      <Link href="/settings#broker" className="shrink-0 rounded-md bg-primary px-3 py-1 font-medium text-primary-foreground hover:bg-primary-hover">
        Connect
      </Link>
      <button onClick={() => setDismissed(true)} aria-label="Dismiss" className="shrink-0 text-d-text-muted hover:text-d-text-primary">
        <X className="h-4 w-4" />
      </button>
    </div>
  )
}
```

- [ ] **Step 2: Mount it** at the top of the authed main pane (above page content), e.g. in `AppShell.tsx` just inside `<main>`. Do NOT render it on `/login`, `/signup`, `/onboarding/*`, `/broker/callback` (the shell already only wraps platform routes; if mounting in a broader spot, guard on `pathname`).

- [ ] **Step 3: Verify tsc** — `cd frontend && npx tsc --noEmit`. Expected: clean.

- [ ] **Step 4: Verify render** (Playwright): with no broker connected, load `/stocks` → banner visible with a Connect link; click ✕ → it disappears.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/broker/ConnectBrokerBanner.tsx frontend/components/shell/AppShell.tsx
git commit -m "feat(broker): app-wide connect-your-broker banner (soft BYOB nudge)"
```

## Task 10: Fix the broken landing copy

**Files:**
- Modify: `frontend/app/page.tsx` (the "No broker, no card… Connect" sentence)

- [ ] **Step 1: Complete the sentence.** Replace the truncated copy with: `No broker, no card. A virtual ₹10L portfolio is seeded at signup. Connect your broker any time to trade live with your own data.`

- [ ] **Step 2: Verify tsc** — `cd frontend && npx tsc --noEmit`. Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add frontend/app/page.tsx
git commit -m "fix(landing): complete the broker/paper copy"
```

---

# Phase 2 — Honest data model (EOD vs Live labeling)

## Task 11: `DataBadge` component

**Files:**
- Create: `frontend/components/common/DataBadge.tsx`

- [ ] **Step 1: Implement it**

```tsx
'use client'
import { Radio, Clock } from '@/lib/icons'
import { Tooltip } from '@/components/foundation'

export function DataBadge({ mode, className = '' }: { mode: 'live' | 'eod'; className?: string }) {
  if (mode === 'live') {
    return (
      <Tooltip content="Live data from your connected broker feed.">
        <span className={`inline-flex items-center gap-1 rounded-full border border-up/30 bg-up/10 px-2 py-0.5 text-[10px] font-medium text-up ${className}`}>
          <Radio className="h-3 w-3" /> Live · your broker
        </span>
      </Tooltip>
    )
  }
  return (
    <Tooltip content="End-of-day research data — delayed, not live. Connect a broker for live data.">
      <span className={`inline-flex items-center gap-1 rounded-full border border-line px-2 py-0.5 text-[10px] font-medium text-d-text-muted ${className}`}>
        <Clock className="h-3 w-3" /> EOD research
      </span>
    </Tooltip>
  )
}
```

- [ ] **Step 2: Verify the icons exist** in the shim — `cd frontend && grep -oE "(Radio|Clock)\b" lib/icons.tsx | sort -u`. Expected: both listed. (If `Radio` is absent, substitute `Activity`.)

- [ ] **Step 3: Verify tsc** — `cd frontend && npx tsc --noEmit`. Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/common/DataBadge.tsx
git commit -m "feat(data): DataBadge — Live (broker) vs EOD research"
```

## Task 12: Label centralized surfaces EOD; per-user surfaces Live

**Files:**
- Modify: `frontend/app/scanner/*` (AI Scanner header), `frontend/app/stocks/page.tsx` (top movers header), `frontend/app/signals/page.tsx` + `frontend/app/signals/[id]/page.tsx` (signal header)
- Modify per-user live surfaces (watchlist quote, positions, stock-detail live price) to render `mode="live"` only when `useBrokerStatus().isConnected`

- [ ] **Step 1: Add `<DataBadge mode="eod" />`** next to the page title on the AI Scanner, `/stocks` top-movers, and `/signals` headers, plus a one-line "Delayed end-of-day research — not investment advice." caption where signals/screeners are shown.

- [ ] **Step 2: Add `<DataBadge mode="live" />`** to per-user live surfaces, gated on broker connection:

```tsx
const { isConnected } = useBrokerStatus()
{isConnected ? <DataBadge mode="live" /> : <DataBadge mode="eod" />}
```

- [ ] **Step 3: Verify tsc** — `cd frontend && npx tsc --noEmit`. Expected: clean.

- [ ] **Step 4: Verify render** (Playwright): `/scanner` and `/stocks` show "EOD research"; with no broker, per-user live spots also show "EOD research" (never a fake "Live").

- [ ] **Step 5: Commit**

```bash
git add frontend/app/scanner frontend/app/stocks/page.tsx frontend/app/signals
git commit -m "feat(data): label centralized surfaces EOD, per-user live gated on broker"
```

---

## Self-review

**Spec coverage:** §4.1 broker-first onboarding → Task 6; §4.2 connect UX + inline Angel + unconfigured state → Tasks 5,7; §4.3 banner + soft gating → Task 9; §4.4 DataBadge + EOD labeling → Tasks 11,12; §4.5 expires_at → Task 1; §4.6 Redis OAuth state → Tasks 2,3; §4.7 token refresh → Task 4; §5 data flow (return_to) → Tasks 3,8; landing copy (§4.2) → Task 10. All covered.

**Placeholder scan:** No TBD/TODO; every code step shows code; commands have expected output. (Task 4 Step 5 references `_compute_expiry`, defined in the same step; Task 12 globs page dirs — the executor confirms exact files at run time.)

**Type/name consistency:** `store_state`/`consume_state` (Tasks 2,3); `needs_refresh`/`refreshable` (Task 4); `broker_oauth_return` sessionStorage key set in Task 6/7, read in Task 8; `DataBadge` props `{mode}` consistent (Tasks 11,12); banner uses `useBrokerStatus` (matches existing hook). Consistent.

**Notes for the executor:** broker OAuth cannot complete end-to-end in this env until the founder registers developer apps and sets `*_API_KEY/SECRET/REDIRECT_URI` + `BROKER_ENCRYPTION_KEY`; the `broker_not_configured` (Task 5) + EOD fallback (Task 12) keep the UI honest until then.
