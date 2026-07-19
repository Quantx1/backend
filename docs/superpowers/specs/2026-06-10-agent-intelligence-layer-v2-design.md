# Agent Intelligence Layer v2 — Design

**Goal:** Make every AI agent across Quant X reason with a strong open model and behave consistently, add genuine multi-step depth to the copilot, and put a grounded agent on every feature surface — all behind a hardened cost layer that keeps total LLM spend under a **$50/mo metered ceiling**.

**Architecture:** A single shared *Agent Intelligence Layer* (AIL) that all surfaces route through: a role→model **router**, a two-tier **persistent cache**, a **per-feature cap** dependency applied everywhere, and a standard *facts → deterministic drivers → optional grounded narrative* contract. Power comes from **routing + structure + caching**, not from spending more.

**Tech stack:** FastAPI (`backend/`), OpenRouter gateway (`ai/agents/llm.py`), Supabase Postgres (cache + memory tables), Next.js 14 frontend (embedded-agent cards). Models: **Qwen3-235B** (default strong brain), **DeepSeek-R1** (Elite-only deep mode, off by default), a tiny free model for classifier/planner fast-paths.

---

## 1. Context — current state (audited 2026-06-09)

A 3-front read of the live code established the baseline this design corrects:

- **Every agent role currently uses the same free model.** `AGENT_MODEL_MAP` (`core/config.py`) is honored by `llm_for(role)` / `complete_sync(role=)` (`ai/agents/llm.py:103-117`) but is **empty by default**, so all roles fall back to `LLM_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"`. A one-line classifier and the Portfolio Doctor synthesis run on the *same* model. **The biggest power lever, and nearly free.**
- **Per-feature caps are defined but only `chat` is enforced.** `LLM_FEATURE_CAPS` / `LLM_FEATURE_CAP_WINDOW` (`core/tiers.py:128-136`) list caps for `strategy_gen`, `scanner_thesis`, `chart_vision`, `debate`, `fno_advisor`, `portfolio_doctor`, but only `_enforce_copilot_cap` (`api/ai_routes.py:87-119`) actually consumes credits, and only on the two `/copilot/chat*` endpoints. Debate, Doctor, F&O advisor, vision, scanner-thesis, strategy-gen have **tier gates but no per-feature credit cap** → an Elite user can run the 8-call Debate or 5-call Doctor unbounded. **The cost hole.**
- **The grounded cache is in-process only.** `grounded.py:19-20` uses a module-level `_CACHE` dict (6h TTL). Wiped on restart, not shared across instances → identical `(symbol, day)` answers are **re-paid**. **A silent cost leak that gets worse with the strong model.**
- **The copilot is single-shot.** `COPILOT_GRAPH` (`ai/agents/copilot.py:423-431`) runs Classifier → ToolPlanner → ToolCaller → Responder once, with no re-plan loop, no self-check, and no server-side memory (history lives in browser `sessionStorage`). Solid and cheap, but it can't chase a multi-step question.
- **Two ad-hoc LLM callers bypass the grounded pattern:** `services/chart_patterns/explain.py` (`complete_sync(role="scanner_thesis")`) and `services/screener_v2/nl_screen.py` (`complete_sync(role="tool_planner")`). Both work but are uncached and inconsistent.
- **Bare surfaces with no reasoning agent:** earnings calendar, news/broker feed (raw headlines, no synthesis), watchlist (no "what changed" digest), rebalancer (suggestions exist, no grounded "why").
- **What is already good and stays:** OpenRouter-only gateway with free→paid fallback (`llm.py:65-96`), `$X/mo` kill-switch via `UsageMeter` (`observability/llm_budget.py`), best-effort usage accounting to `llm_usage_events` (`track_llm_usage`), tier gates (`middleware/tier_gate.py`), the `grounded_reason` contract, and the embedded-agent frontend pattern (`frontend/components/copilot/EmbeddedAgent.tsx`).

## 2. Locked decisions

| Decision | Value |
|---|---|
| Spend posture | Aggressive — a **strong model is the default brain**, not a last resort |
| Default strong model | **Qwen3-235B** for responder, all grounded cards/explainers, **Doctor, Debate, strategy-gen** |
| Deep-reasoning model | **DeepSeek-R1**, reserved as an **Elite-only "deep mode" toggle, shipped OFF** |
| Fast-path model | tiny free model (Qwen3-8B-class `:free`) for `classifier` / `tool_planner` (regex still catches ~95%) |
| Sentiment / vision | unchanged (finbert_india + existing vision model) |
| Monthly ceiling | **$50/mo**, metered, hard kill-switch + graceful degrade to free models when exceeded |
| Cost ordering | Cost hardening (cache + caps) ships **before** strong models go broad |

Rationale for 235B-default over R1-default at $50: R1 emits large hidden-reasoning token volumes, making Doctor ≈ $0.037 and Debate ≈ $0.05 per run; on 235B the same paths are ≈ $0.003 / $0.004 (~12× cheaper) while staying genuinely strong. This is what makes a realistic **~1,000–2,500 active users** fit in $50 (see Appendix A). R1 remains wired so "deep mode" can be enabled per-Elite later without code change.

## 3. Architecture — the shared Agent Intelligence Layer

All LLM-backed features route through four shared pieces. No surface calls a provider directly or caches on its own.

```
feature service ──► assemble_facts() ──► deterministic drivers (always, 0 tokens)
                                          │
                                          └─ if use_llm & cap-ok & cache-miss:
                                               router.model_for(role, ctx)  ─► LLM gateway ─► cache.set
                                          ▲                                                      │
                  enforce_llm_cap(feature) (route dep)                      cache.get (L1 mem + L2 Supabase)
```

### 3.1 Model Router
- Populate **in-code defaults** for `AGENT_MODEL_MAP` (not env-only) in `core/config.py`, so correct routing works out of the box; env JSON still overrides.
- New helper `model_for(role, *, deep=False, tier=None)` in `ai/agents/llm.py`:
  - `classifier`, `tool_planner` → free tiny model.
  - `responder`, `grounded_reason`, `doctor`, `debate`, `strategy_gen`, `scanner_thesis` → Qwen3-235B.
  - When `deep=True` **and** caller is Elite **and** the `DEEP_MODE` feature flag is on → DeepSeek-R1; otherwise 235B.
- The monthly budget setting (the cap read by `UsageMeter`, `core/config.py:~249`, currently $20) → **$50**. Kill-switch + `build_models(allow_paid=not over_budget)` unchanged: when over $50, paid (strong) entries are stripped and calls degrade to free models. (The plan pins the exact setting name from the live code.)

### 3.2 Persistent response cache (`ai/agents/response_cache.py`, new)
- Two-tier: **L1** in-process LRU (fast, per-instance) + **L2** Supabase table `llm_response_cache`.
- Interface: `cache_get(key) -> dict | None`, `cache_set(key, payload, *, ttl_seconds, surface, model)`.
- Table `llm_response_cache`: `cache_key text pk`, `surface text`, `payload jsonb`, `model text`, `created_at timestamptz`, `expires_at timestamptz`. Index on `expires_at` for sweeps. RLS: service-role only (server-written).
- `grounded.py` swaps its `_CACHE` dict for `cache_get/Set` (keeps L1 as the in-process layer). Keys keep the existing `surface:symbol:YYYY-MM-DD` shape.
- Multi-agent outputs (Doctor, Debate) cache by `doctor:{portfolio_hash}:{day}` / `debate:{signal_id}:{day}`.

### 3.3 Universal cap enforcement (`middleware/llm_caps.py`, new)
- Generalize `AssistantCreditLimiter` into `enforce_llm_cap(feature: str)` — a FastAPI dependency that reads the cap + window from `LLM_FEATURE_CAPS` / `LLM_FEATURE_CAP_WINDOW`, consumes one credit, and raises `HTTP 402 {error: "credit_cap", feature, limit, window, upgrade_url}` when exceeded. Admin bypass preserved.
- Apply to **every** LLM endpoint: `chat` (keep), `debate`, `portfolio_doctor`, `fno_advisor`, `scanner_thesis`, `chart_vision`, `strategy_gen`. `_enforce_copilot_cap` becomes a thin wrapper over the generic dependency.

### 3.4 Standard agent contract
One documented shape every feature agent follows (already used by `why_moving.py`): `assemble_facts()` (real data, never fabricated) → `_drivers(facts)` (deterministic bullets, always returned, 0 tokens) → optional `grounded_reason(facts, question, cache_key=…, role=…)` gated by `use_llm`. New agents and the two refactored ad-hoc callers conform to this.

## 4. Phases

Each phase is independently shippable, keeps the existing **342-test** services+data suite green, and adds its own tests.

### Phase 1 — Foundation (router + cost hardening)
Ships the spine and the brakes **before** strong models go broad.
1. In-code `AGENT_MODEL_MAP` defaults + `model_for()` + `deep`/`tier` resolution + `DEEP_MODE` flag (default off).
2. Monthly budget setting → 50 (resolve the exact name from `core/config.py`); verify degrade-to-free path.
3. `response_cache.py` + `llm_response_cache` migration; wire into `grounded.py` (and a daily-expiry sweep).
4. `enforce_llm_cap(feature)` dependency + apply to all 7 LLM endpoints; `LLM_FEATURE_CAPS` reconciled with the new defaults.
**Deliverable:** every existing agent now runs on the right-sized model, never re-pays a cached answer, and is cap-protected. Spend is bounded and observable before depth/breadth land.

### Phase 2 — Agentic depth
5. Bounded re-plan loop in `copilot.py`: after `ToolCaller`, deterministic `needs_more(state)` (requested data missing / tool error / planner emitted a follow-up); if true and `rounds < 2`, loop to `ToolPlanner` with the observation. Hard cap 2 rounds, budget-gated.
6. `validate_grounding(reply, facts)` self-check (0 tokens): every numeric token in the reply must appear in tool results/facts; otherwise strip/flag. Runs before the `done` event.
7. Server-side memory: table `copilot_memory(user_id pk, summary text, updated_at)` + retrieval into the responder context + a cheap throttled summary refresh (235B, capped). History stops being browser-only.
8. Doctor + Debate honor `deep` (R1) when the Elite toggle is on; both cached per (entity, day).
**Deliverable:** the copilot takes multiple steps, checks its own grounding, and remembers across sessions; deep mode is available to flip on.

### Phase 3 — Consistency + breadth
9. Refactor `chart_patterns/explain.py` and `screener_v2/nl_screen.py` onto the shared layer (cache + cap + role routing).
10. Four new grounded agents on bare surfaces, each following §3.4 (facts → drivers → cached narrative, capped):
    - **Earnings preview** — upcoming earnings, expected-move/IV/past-surprise facts → narrative.
    - **News/sentiment synthesis** — headlines + finbert scores → "what the news means" digest.
    - **Watchlist daily digest** — moves/signals/alerts across the user's watchlist → one "what changed today" summary.
    - **Rebalancer "why"** — grounded narrative over the existing deterministic rebalance suggestions.
11. Page audit: every feature page mounts its embedded agent consistently (`EmbeddedAgent`).
**Deliverable:** no ad-hoc LLM paths remain; an agent lives on every surface; "throughout all features" is met.

## 5. Data model changes
- `llm_response_cache` (Phase 1) — see §3.2.
- `copilot_memory` (Phase 2) — `user_id uuid pk`, `summary text`, `updated_at timestamptz`. RLS: owner-read, service-write.
- No change to `llm_usage_events` (accounting already correct).
- Migrations applied via `scripts/apply_migrations.py` (DATABASE_URL, port 5432) and reflected into `complete_schema.sql` Part B.

## 6. Cost guardrails (invariants)
Every LLM call, on every path, must: (a) resolve its model via `model_for(role)`; (b) pass the `_guard_budget` check (paid stripped when over $50); (c) record to `llm_usage_events`; (d) consult/populate the persistent cache when the surface is cacheable; (e) sit behind `enforce_llm_cap(feature)` at the route. Classifier/planner regex fast-paths stay, so trivial calls never reach a strong model. No fabrication: deterministic drivers are always returned independent of the LLM.

## 7. Testing strategy
- **Phase 1:** `model_for` returns the correct model per role/tier/deep-flag; over-budget strips paid; `cache_get/Set` hit/miss + TTL expiry; `enforce_llm_cap` returns 402 past the limit and respects day vs month windows; grounded answers are served from L2 cache on a cold process.
- **Phase 2:** re-plan loop is bounded to 2 rounds and triggers only on `needs_more`; `validate_grounding` strips a fabricated number not present in facts; memory summary round-trips and is injected into responder context.
- **Phase 3:** refactored ad-hoc callers produce identical output shape and now cache; each new agent returns deterministic drivers with `use_llm=false` (0 tokens) and a grounded narrative on click; honest-empty when data is unavailable.
- **Live smoke:** budget meter increments on a paid call; over-cap degrades to free; each surface returns grounded output; per-feature 402 fires.
- Pure-function tests, no network/DB/LLM in unit tests (monkeypatch the gateway), matching the repo's existing test style.

## 8. Non-goals / out of scope
- Raising the cap above $50 or enabling R1 deep-mode by default (both are post-traction toggles).
- Replacing the open-source-only model strategy (no closed/frontier models).
- "Live-feed" features that need a paid tick/depth license (footprint ticks, broker L2, intraday OI) — unrelated to the agent layer.
- A full agent framework rewrite — we extend the existing graph/registry, not replace it.

## 9. Risks
- **R1 token blow-up if deep-mode is enabled broadly** — mitigated by Elite-gating + per-feature cap + it shipping off.
- **Cache staleness** — daily TTL keys (`:YYYY-MM-DD`) bound this to one trading day; intraday-sensitive surfaces use short TTLs.
- **Re-plan loop runaway** — hard 2-round cap + budget guard.
- **Price drift** — Appendix A numbers are planning estimates; confirm live OpenRouter/DeepInfra prices before go-live (does not affect the architecture).

---

## Appendix A — Capacity economics ($50/mo, 235B default)

Pricing assumptions (planning estimates, confirm before launch): Qwen3-235B ≈ $0.15/M in, $0.70/M out; DeepSeek-R1 ≈ $0.45/M in, $2.2/M out; free models $0.

| Action | Model | ~Cost each (uncached) |
|---|---|---|
| Grounded card / explainer | 235B | $0.0003 |
| Copilot chat turn | 235B | $0.0006 |
| Strategy-gen / chart vision | 235B | $0.001–0.002 |
| Portfolio Doctor (5 agents) | 235B | ~$0.003 |
| Bull/Bear Debate (8 agents) | 235B | ~$0.004 |
| Portfolio Doctor / Debate (deep mode) | R1 | $0.037 / $0.05 |

$50/mo buys ≈ 80,000 chat/card answers **or** ~16,000 Doctor runs **or** ~12,000 Debates (any mix). Caching by `(symbol/entity, day)` means popular symbols are computed once per day, so **cost-per-user falls as users grow**. Realistic blended capacity: **~1,000–2,500 active users**. Worst case is bounded by the per-feature caps.
