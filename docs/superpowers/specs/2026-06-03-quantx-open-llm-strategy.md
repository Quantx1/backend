# Quant X — Open-Source LLM Strategy (per-agent model routing)

> **Status:** Research complete + **direction locked 2026-06-03** (see §9). Next artefact: TDD implementation plan for Steps 0–5.
> **Locked decisions:** (1) **Serverless hybrid** — no self-host at this scale. (2) **Pure open-source end-state — Gemini removed entirely** (kept only as a transition-only rollback during migration; final fallback is cross-provider *open* models). (3) **US serverless with no-retention/enterprise endpoints** — cross-border prompt egress accepted, zero data retention required.
> **Goal:** Move the LLM agents off a single `gemini-2.5-flash` wrapper onto open-weight models, a *specific model per agent role*, optimised for ultra-low cost without losing quality, for ~2,000 users.
> **Method:** 4 parallel deep-research passes (open-model landscape · finance-LLM reality check · hosting economics · routing architecture), all verified against current mid-2026 sources. Citations live in each research thread; key ones inline below.

---

## 1. Executive verdict (TL;DR)

1. **Don't self-host yet.** At ~2,000 users (~**400M tokens/month** estimated), serverless token APIs cost **~$150–300/mo**. A self-hosted GPU good enough for the heavy agents is **~$2,100/mo (1× H100) idle most of the day**, realistically **~$4,200/mo** with the 235B tier + redundancy. Self-host break-even is **~2–6 billion tokens/month** — **10–30× your current scale**. Keep RunPod for **training only**.
2. **Per-agent model routing is the right design** — and your codebase already supports it (`Agent.__init__(llm=…)` → `Agent.llm` returns `self._llm_override or get_llm()`). No graph rewrites needed.
3. **Your 3 instincts, graded by the evidence:**
   - ✅ **Qwen3-32B for orchestration** — correct. Best *verified* open function-calling (BFCL), ideal for the `tool_planner`.
   - ✅ **DeepSeek for deep analysis** — correct. (DeepSeek **V4 Pro/Flash**, MIT, is the mid-2026 successor to R1; near-frontier reasoning at trivial cost.)
   - ❌ **FinGPT for sentiment/earnings** — **drop it.** It's stale 2023 LoRA adapters (Llama-2 base), not a deployable product, with a documented **bullish bias** that would literally poison your bull/bear debate. On **Indian** equities, US finance models *underperform* general models.
4. **Sentiment is not an LLM swap.** Keep your in-prod `finbert_india`; the high-leverage upgrade is **RAG + multi-source context**, not a new model (evidence: a Dec-2025 NIFTY-50 study where off-the-shelf FinBERT was the *worst* model and a tiny general model + RAG won).
5. **The gateway is LiteLLM** (one OpenAI-compatible interface in front of every provider). **End-state is pure open-source — no Gemini.** Resilience comes from **cross-provider fallback on the same open model** (e.g. Qwen3-32B on Fireworks → DeepInfra → Together). Gemini stays wired *only* as a transition-time rollback (clear one env var) and is removed in the final hardening step.
6. **The #1 risk is tool-call JSON reliability**, not model quality. Fix it with **schema-constrained decoding** (`response_format: json_schema strict`), which also lets us delete the `_extract_json` regex hack for the planner.
7. **The biggest cost lever isn't the provider — it's chat input tokens** (~222M/mo from re-sending portfolio context + history every turn). Prompt-trimming + caching can roughly **halve** the bill.

---

## 2. The agent inventory (what we're routing)

Three graphs, from `backend/ai/agents/`:

- **Copilot** (Main Chat, interactive, streaming): `classifier` → `tool_planner` → `tool_caller` → `responder`
- **FinRobot Portfolio Doctor** (background, reasoning): `fundamental` → `management_tone` → `promoter_holding` → `peer_comparison` → `synthesizer`
- **TradingAgents Debate** (Elite, background, high-stakes): `fundamentals/technical/sentiment_analyst` → `manager` → `bull/bear_researcher` → `risk_manager` → `trader`

Plus **sentiment** (news/earnings) which today runs the fine-tuned `finbert_india` PROD model — *not* a chat LLM.

---

## 3. Per-agent model map (the centrepiece)

All prices **USD per 1M tokens (input / output)**, on-demand serverless, fetched 2026-06-03. Prices move ~monthly — treat absolutes as "good for a quarter", the *relative* picks as durable.

| Agent role | Workload | **Recommended model** | Provider (sweet spot) | ~$/Mtok (in/out) | Why this model |
|---|---|---|---|---|---|
| `copilot.classifier` | interactive · tiny · mostly regex-skipped already | **Llama 3.1 8B** or **Qwen3 8B** | **DeepInfra** | $0.02 / $0.05 | Binary in/out-of-scope; near-free; JSON mode is plenty |
| `copilot.tool_planner` | interactive · **JSON-critical · load-bearing** | **Qwen3-32B** + strict JSON schema | **Fireworks** (4× faster structured output) | $0.08 / $0.28 (DeepInfra) | Best *verified* open function-calling (BFCL ~0.70–0.76). Reliability > cost here |
| `copilot.responder` | interactive · **streaming · low TTFT** | **Llama 3.3 70B** or **Qwen3-32B** or **gpt-oss-20B** | **Groq / Cerebras** | $0.59 / $0.79 (70B Groq) | Sub-500ms first token, 600–1,000+ tok/s; no strict-JSON need |
| `finrobot.*` (Doctor, 5 steps) | background · reasoning · quality | **Qwen3-235B-A22B** or **DeepSeek V4 Flash** | **DeepInfra** | $0.071 / $0.10 (235B) · $0.10 / $0.20 (V4 Flash) | Frontier-class reasoning at trivial cost; latency irrelevant |
| `tradingagents.*` (Debate, 8 agents, Elite) | background · high-stakes deep reasoning | **DeepSeek V4 Pro** or **Qwen3-235B** | DeepInfra / Fireworks | $1.30 / $2.60 (V4 Pro) · $0.071 / $0.10 (235B) | V4 Pro for the ceiling; 235B for value. Reserve reasoning models for background only |
| **sentiment** (news/earnings) | classification | **KEEP `finbert_india` + add RAG** | self (already) | ~$0 | India-tuned; general/US finance LLMs underperform on Indian text |

**Provider roles:** **DeepInfra** = cost floor + default; **Groq/Cerebras** = the latency-critical streaming responder; **Fireworks** = best structured-output (planner JSON); **OpenRouter/Together** = automatic fallback. **Gemini** = terminal fallback during/after migration.

> Reasoning models (DeepSeek-R1/V4 "thinking") burn large hidden token budgets (median latency ~20s) — **never** in the interactive stream, **only** in the background Doctor/Debate graphs.

### 3.1 Provider decision (LOCKED research 2026-06-03) — OpenRouter vs Together vs the field

Verdict: **OpenRouter as the gateway → DeepInfra (primary host) + Groq (streaming responder), ZDR forced on. Skip Together.**

- **OpenRouter = the gateway.** Pass-through pricing (no per-token markup) + a **~5.5% credit-purchase fee**; one key → all 5 models, automatic cross-provider failover, **Auto Exacto** tool-call-quality routing (de-risks the planner), enforceable **Zero-Data-Retention** (`provider.zdr:true`). Cannot front a self-hosted vLLM box — moot while serverless. It *replaces LiteLLM's router role*; keep LiteLLM/OpenAI-SDK only as the thin in-process client (escape hatch).
- **DeepInfra = primary host.** Cheapest on 4 of 5 models, **ZDR by default**. Routed to *through* OpenRouter.
- **Groq = the streaming responder only.** ~321 tok/s, <1s TTFT for Llama-3.3-70B; DeepInfra is ~13× slower (25 tok/s) → bad for live chat. OpenRouter auto-fails-back Groq→DeepInfra.
- **Together = SKIP.** Most expensive of the three (3–10× DeepInfra on shared models) **and missing 3 of the 5 target models** (Qwen3-8B, Qwen3-32B, DeepSeek-V3.1). Only revisit for enterprise SLA / managed fine-tuning / SOC2-HIPAA paper — none needed now.

Per-agent → host (current $/Mtok in/out, via OpenRouter):

| Agent | Model (pin this slug) | Host | $/Mtok |
|---|---|---|---|
| classifier | Qwen3-8B | Novita / DeepInfra | ~$0.04 / $0.14 |
| tool_planner | Qwen3-32B (+ `structured-outputs-2025-11-13` header for strict tools) | DeepInfra FP8 | $0.08 / $0.28 |
| responder (stream) | Llama-3.3-70B | **Groq** | $0.59 / $0.79 |
| Portfolio Doctor | **Qwen3-235B-A22B-2507** (NOT `-04-28` → ~6–18× pricier) | DeepInfra | $0.071 / $0.10 |
| Debate | **DeepSeek-V3.1** (V4-Pro is ~5× cost for marginal gain) | DeepInfra | $0.21 / $0.79 |

**Estimated cost @ ~400M tok/mo ≈ $170 + ~5% OpenRouter ≈ ~$179/mo** (responder ≈ 75% of it; route responder to DeepInfra for non-interactive paths to cut toward ~$70/mo). Squarely inside the $150–300 target.

**Founder action:** sign up for **OpenRouter** (+ ensure **DeepInfra** and **Groq** are enabled as upstreams / add their keys via BYOK), force ZDR on, top up credits.

### 3.2 HARD $20/month budget tier (LOCKED 2026-06-03 — supersedes the $179 responder choice)

Founder cap: **≤ $20/month**, free + small models by default, big model only on escalation. Verified by a 6-agent workflow with two adversarial verifiers. **Key finding: the dollar cap is NOT the binding constraint — free-tier *rate limits* are.**

**The honest reframe (what the verifiers proved):**
- **$20 is trivially met.** At realistic scale (~2–4% DAU ≈ 60 DAU, ~6–24M tokens/mo — *not* 400M) expected spend is **~$1–5/mo**. The cap doesn't break until **~700–1,050 DAU**.
- **"Free models" cannot *serve* the market-open burst.** Free tiers choke on **TPM/RPM**, not tokens: Groq free = 30 RPM / **6,000 TPM (org-wide)** → only ~2 planner/responder calls/min before 429; OpenRouter `:free` = 20 RPM / 1,000 RPD (after one-time $10), and **failed requests still count**. Free serving collapses at **single-digit concurrent users** at 09:15 IST, *not* the hundreds the first draft assumed.
- **Therefore: cheap-paid-by-default during market hours; free tiers as a cost-shaver for the classifier + off-peak.** DeepInfra 8B ($0.02–0.08/M) is so cheap that running *everything* paid through the bell still costs a few $/mo at this DAU.

**Per-agent (budget tier):**

| Agent | Default model | Lane | Escalates to | ~$/mo |
|---|---|---|---|---|
| classifier | Qwen3-1.7B (regex fast-path first) / Groq free Llama-3.1-8B (14,400 RPD) | free + cheap-paid | — (it's the router) | ~$0 |
| tool_planner | **Qwen3-8B** + strict JSON schema (only small model with trustworthy tool JSON, ~0.93 F1; Llama-8B ~0.57 → avoid) | free → DeepInfra 8B paid overflow | stays 8B | ~$0.1–0.3 |
| responder (stream) | **Qwen3-8B** (Llama-3.1-8B cheap 2nd voice) | free → DeepInfra paid overflow | Qwen3-32B **only on `hard=true`** | ~$0.2–0.5 |
| Portfolio Doctor (Elite, bg) | **Qwen3-32B** (DeepInfra $0.08/$0.28; free Groq QwQ-32B for batch) | paid | — (is the escalation tier) | ~$0.6 |
| Debate (Elite, rare, bg) | **Qwen3-32B** | paid | — | ~$0.3 |

> This **supersedes §3.1's Llama-3.3-70B/Groq responder** (that was 75% of the $179): the $20 tier runs **Qwen3-8B for all interactive work** and reserves Qwen3-32B for classifier-flagged hard queries + Elite background. Quality tradeoff: 8B narration is thinner than Gemini on hard reasoning — the `hard=true` trigger must be tuned (false-negatives ship shallow answers).

**Non-negotiable build primitives (the "cents/month" is not real until these exist):**
1. **Per-token usage + cost meter with a hard $20 kill-switch** — today there is *zero* metering (single `GeminiWrapper`). Build FIRST; it makes the budget observable + enforceable.
2. **Trim the tool_planner's schema resend** — it currently `json.dumps(indent=2)`s the *entire* `tool_registry.schema()` every call (`copilot.py:169`, ~1–3K tok). Send a compact tool-name list / cached static schema (~5–10× planner-input cut). **Mandatory** — TPM counts input tokens even when cached, so this is what keeps Groq's 6K TPM from 429-ing.
3. **Prompt-cache prefixes** (system + schema) → 45–65% token savings on the re-sent prefix across the 3 hops.
4. **Concurrency-aware limiter (token-bucket per provider)** — spill to cheap paid the instant free TPM/RPM headroom < 1; **never retry a 429 against the same free bucket** (retries burn OpenRouter's 1,000 RPD).
5. **`hard=true` escalation gate** to Qwen3-32B with a **monthly 32B token budget + circuit-breaker**, so a load spike can't silently route interactive traffic to the expensive model.

**Realistic monthly cost:** ~$1–5/mo expected; stays < $20 up to ~600–700 DAU. **Don't architect around one-time signup credits** ($25 Together / $10 OpenRouter unlock / $5 DeepInfra) — they don't renew; budget on PAYG DeepInfra rates. **Decoys to exclude from the serving path:** Cerebras (8K context + 5–30 RPM), Mistral free (~2 RPM), Cloudflare (~15–25 gens/day) — huge tokens, unusable RPM.

### 3.3 The Quant-Researcher / generation tier (where the money is — spend the budget HERE)

The LLM doesn't only narrate — it **generates new strategies and scanners** from market conditions / assets / research, which are then backtested + Sharpe-gated, and only the profitable ones go live ([studio.py](../../../backend/ai/strategy/studio.py) → [backtest.py](../../../backend/ai/strategy/backtest.py) → [registry.py](../../../backend/ai/strategy/registry.py) `draft→backtest→paper→live`; ref the 2026-05-31 "AI trade gating reversed" decision). This is a **different, harder role** than chat — and it changes the budget priority.

**Two truths:**
1. **Money is protected by the GATE, not the model.** A generated strategy *cannot* go live unless it passes the backtest + Sharpe gate. So model choice for generation = **alpha YIELD** (how many generated candidates actually prove out), not safety. A weak model just wastes compute proposing junk that fails the gate.
2. **Generation is the one role that deserves a STRONG model** — "deep intelligence". Use **Qwen3-235B-A22B-2507** (DeepInfra, **$0.071/$0.10**) — frontier-class open reasoning at trivial cost. (DeepSeek-V4-Flash $0.10/$0.20 as an alt; *not* the cheap 8B — it mostly produces gate-failing overfit.)

**Why it still fits $20:** generation is **rare + background + high-value** (a research batch, not a per-message call), so even the big model is cheap in absolute terms. E.g. ~100 candidate strategies/day × ~12K tok ≈ 36M tok/mo on Qwen3-235B ≈ **~$3–5/mo**. Allocate the *bulk* of the $20 here (highest-value spend) with a hard **research-volume cap** (N candidates/day) + the kill-switch. Chat narration on cheap 8B is a rounding error; **spend where it makes money (generation), not where it just talks.**

| Role | Model | Provider | ~$/mo | Gate that protects money |
|---|---|---|---|---|
| **Strategy generator** (market-condition / research → DSL) | **Qwen3-235B-2507** | DeepInfra | ~$3–8 | `dsl.py` schema validate → `run_dsl_backtest` → Sharpe gate → `registry` state machine |
| **Scanner generator** (research → filter spec) | **Qwen3-235B-2507** | DeepInfra | ~$1–3 | filter-schema validate → run vs universe → signal backtest |

**⚠️ Non-negotiable: the gate must be OUT-OF-SAMPLE / walk-forward, not in-sample Sharpe.** A creative LLM + a naive in-sample backtest = overfit strategies that backtest beautifully and lose live (cf. the retired "63.6% WR" pattern-engine claim → 35–40% OOS). The model finds candidates; **the rigor of the walk-forward gate is what turns candidates into real money.** Invest in gate rigor (OOS window, transaction costs, regime robustness) as much as in the model.

---

## 4. Cost model (~2,000 users)

**Assumptions (tunable):** 30% DAU (600), 8 chat msgs/active/day, ~1,500 in + 400 out per msg (+ classifier/planner overhead), 250 Doctor runs/day, 120 Debate runs/day, 22 trading days/mo → **~319M input + ~90M output ≈ 409M tokens/month**, of which chat is ~277M (cheap-model-eligible) and heavy agents ~132M.

| Scenario | Est. $/month | Notes |
|---|---|---|
| All-serverless, one mid model (DeepInfra DeepSeek V3.1/V4-Flash) | **~$140** | simplest |
| All-serverless, premium provider (Together/Fireworks) | $245–370 | pay for brand/SLA |
| **Hybrid tiered serverless (recommended)** | **~$18–250** | cheap small-model chat + 235B heavy agents on DeepInfra |
| Self-host always-on 1× H100 | ~$2,110 | + your DevOps/on-call; idle ~90% of day; single point of failure |
| Self-host 2× H100 (235B + redundancy) | ~$4,220 | "real" production self-host |

**Break-even (always-on H100 ≈ $2,110/mo vs serverless):** ~2.3B tok/mo (vs Fireworks 70B) up to ~21B tok/mo (vs DeepInfra hybrid). You're at ~0.4B. **You'd need 6–50× current traffic before self-host math flips** — and that assumes a *saturated* GPU; at realistic ~20% utilisation, multiply by ~5.

**Conclusion: serverless, hybrid routing. Budget ~$150–300/mo with headroom.** ~7–14× cheaper than the cheapest viable self-host, with near-zero ops.

---

## 5. Why NOT the things you'd expect to do

- **Self-host (Ollama/vLLM/SGLang on RunPod):** economically wrong until ~10–30× scale (§4). vLLM/SGLang are the *right* serving stack **if/when** you cross break-even — note them for later, don't build now. Ollama is a dev/laptop tool, not a 2,000-user serving layer.
- **FinGPT:** stale LoRA adapters on Llama-2 (latest checkpoint Oct-2023), no hosted endpoint, QA exact-match 3.8–28% vs GPT-4's ~70%, "failed to generate coherent summaries", and a **consistent bullish bias** (arXiv 2507.08015). For an auto-trading product that bias is a liability.
- **Finance-specific LLMs generally (Palmyra-Fin / FinBERT-successors / BloombergGPT):** BloombergGPT is closed; Palmyra-Fin is 70B + non-commercial-ish + US/CFA-centric; the Dec-2025 "Is GPT-OSS All You Need?" study (arXiv 2512.14717) shows general open models match/beat finance-tuned ones across 10 financial NLP tasks (the "efficiency paradox"). On **Indian** news (NIFTY-50 study, arXiv 2512.20082) off-the-shelf **FinBERT was the worst model (0.50 F1)**; a 3B general model + RAG hit 0.61+.
- **Reasoning model in the chat stream:** ~20s median latency kills the streaming UX. Background graphs only.

---

## 6. Architecture: one gateway, per-agent models, Gemini fallback

```
                       ┌─────────────────────────────────────────────┐
  Agent (per role) ──▶ │  LLM wrapper (model-aware)                   │
   classifier          │   • complete / complete_stream / generate_json│
   tool_planner        └───────────────┬─────────────────────────────┘
   responder                           │ OpenAI-compatible base_url
   doctor.*                            ▼
   debate.*               ┌──────────────────────────┐
                          │  LiteLLM (Router / Proxy) │  ← one interface
                          │  fallbacks · retries ·    │
                          │  cost tracking · cache    │
                          └───┬───┬───┬───┬───┬───────┘
                              │   │   │   │   │
                     DeepInfra│Groq│Fireworks│vLLM(future)│ Gemini(fallback)
```

- **Gateway pick: LiteLLM** — the only option that puts hosted providers **and** a future self-hosted vLLM box behind one OpenAI-style interface, free + self-hostable, with fallbacks/retries/cost-tracking built in. Start with the **in-process Python Router** (no extra service); graduate to the **proxy container** only when a second service needs it. (OpenRouter can't front a private vLLM box; Portkey is the alternative if we later want semantic caching — its gateway went Apache-2.0 in Mar-2026.)
- **Per-agent injection already exists.** `CopilotToolPlanner(llm=LLM(model=...))` etc. A missing mapping → `get_llm()` → **Gemini**, so nothing breaks unflagged.

---

## 7. Tool-calling reliability (the #1 risk) — use constrained decoding

Today `generate_json` leans on Gemini "being good at JSON" + `_extract_json` regex recovery. **Open models via `tool_choice="auto"` do NOT guarantee valid JSON** (vLLM: "arguments may be malformed … prefer `tool_choice="required"` or named function"). Rules:

1. **Send a real JSON Schema**, not a prose `schema_hint`; pass `response_format={"type":"json_schema","strict":true,…}` (or `tool_choice="required"`). Routes through xgrammar/guided-json → **schema-valid output guaranteed**. Lets us **delete `_extract_json` for the planner**.
2. **Pin the planner to a strong model** (Qwen3-32B / GLM-4.5 / DeepSeek-V3-class), served with strict mode (**Fireworks** = ~4× faster structured output).
3. **Qwen3 gotcha:** with `enable_thinking=false` + `guided_json`, output can be invalid — keep thinking on or set `enable_in_reasoning=True`.
4. Keep `_extract_json` only as a belt-and-suspenders fallback rung.

---

## 8. Migration plan (incremental, reversible, planner-last)

Env-flag driven; **Gemini is always the terminal fallback**; clear the env var → instant full rollback to today's behaviour.

- **Step 0 — Gateway in front of Gemini only.** Stand up LiteLLM Router in-process with one entry (`gemini-2.5-flash`). Prove identical outputs. (Decouples "switch gateway" from "switch model".)
- **Step 1 — Make `LLM` model-aware.** Add `model: Optional[str]`; route to the gateway when `LLM_GATEWAY_BASE_URL` is set, else the existing `GeminiWrapper`. Keep `track_llm_usage` telemetry (populate real provider/model). Upgrade `generate_json` to accept a JSON Schema + strict `response_format`.
- **Step 2 — Env model map (the rollback switch):**
  ```bash
  AGENT_MODEL_MAP='{"classifier":"deepinfra/qwen3-8b","tool_planner":"fireworks/qwen3-32b",
    "responder":"groq/llama-3.3-70b","doctor":"deepinfra/qwen3-235b","debate":"deepinfra/deepseek-v4"}'
  LLM_GATEWAY_BASE_URL="http://localhost:4000/v1"   # unset = pure Gemini
  ```
  Inject per agent at graph construction. Missing key → `None` → Gemini.
- **Step 3 — Migrate one agent at a time, lowest blast-radius first:** `classifier` → `responder` (A/B, watch TTFT) → **`tool_planner` LAST**, only with strict JSON schema + shadow-run vs Gemini + a LiteLLM fallback list `provider → Gemini`.
- **Step 4 — Background graphs** (Doctor, Debate) in parallel: point at a reasoning model; lower risk (no streaming, no user waiting). (`LAB_MODEL`/`ANTHROPIC_API_KEY` shows per-surface model selection is already a familiar pattern here.)
- **Step 5 — Hardening + Gemini removal (pure-open end-state):** replace every alias's terminal fallback **Gemini → a second open provider** (same model, different host); retries/backoff; exact-match Redis cache; evaluate semantic cache (Portkey-OSS behind LiteLLM) on planner+responder for 30–50% savings; prompt-context trimming (biggest lever). Once all agents are validated on open models, **delete the Gemini wiring** and drop `GEMINI_API_KEY` from the runtime path.

**Risk controls:** every step behind an env flag; **during migration** Gemini is the terminal rollback (clear two env vars → back on pure Gemini, zero code revert); agents migrate independently; the load-bearing planner goes last + constrained + shadow-validated. **After Step 5** the rollback target becomes the cross-provider open fallback, not Gemini.

---

## 9. Decisions (LOCKED 2026-06-03)

1. **Deployment model → SERVERLESS HYBRID.** No self-host at this scale (RunPod stays training-only). Revisit only at ~10–30× traffic.
2. **Gemini → PURE OPEN-SOURCE END-STATE, Gemini removed.** Kept only as a transition-time rollback during migration; final fallback is **cross-provider open models** (same model, multiple hosts). `GEMINI_API_KEY` drops out of the runtime path at Step 5.
3. **Data residency → US SERVERLESS + NO-RETENTION.** Cross-border prompt egress accepted; require zero-data-retention / enterprise endpoints (DeepInfra/Together). No on-shore mandate → serverless stands.

**Provider accounts needed to go live (founder action):** **DeepInfra** (primary — cheapest, widest catalog, the cost floor), **Groq** (fast streaming responder), **Fireworks** (structured-output planner). Optionally **OpenRouter/Together** as fallback. The code lands behind an env flag and defaults to current behaviour until these keys exist.

Next artefact: a TDD implementation plan (writing-plans skill) for Steps 0–5.

---

## 10. Sources (load-bearing)

- Open-model landscape + serverless pricing — Together/DeepInfra/Groq/Fireworks pricing pages, Artificial Analysis (DeepSeek V4), BFCL (llm-stats), all fetched 2026-06-03.
- Finance-LLM reality check — FinGPT assessment arXiv 2507.08015; "Is GPT-OSS All You Need?" arXiv 2512.14717; NIFTY-50 sentiment arXiv 2512.20082; "Reasoning or Overthinking" arXiv 2506.04574; Open FinLLM Leaderboard (TheFinAI).
- Hosting economics — RunPod pricing + serverless docs; vLLM/SGLang throughput (databasemart, cerebrium); self-host-vs-API break-even (DevTk, Braincuber).
- Routing/tool-calling — LiteLLM routing docs; vLLM tool-calling + structured-outputs docs; SGLang tool parser; XGrammar-2 (MLC, May-2026); OpenRouter structured outputs; BFCL v4.
