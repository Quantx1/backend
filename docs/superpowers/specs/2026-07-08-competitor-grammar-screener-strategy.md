# Competitor-grammar match: screener + strategy (feat/xai-redesign) — 2026-07-08

Goal: make the screener + strategy-generation surfaces match the *interaction grammar* of
uTrade Algos, Tradomate, and FinStocks AI — **their mechanics/IA/flow, rendered in our locked
xAI dark language** (near-black, white pills, GeistMono numbers, duotone #3FB950 up / #F85149
down; NO raw hex in className — git hook blocks it; use existing tokens/foundation). English-only.

Source: workflow gap analysis wf_e7bb9c5c-cfe + `scratchpad/deliverables/INDIA-AI-COMPETITOR-MATRIX`.
The branch already HAS: NL→DSL + clarifying loop, vision-to-strategy, walk-forward OOS gate,
discovery engine, compare, template wall, NL Screener Agent (confluence), Power-tab editable
checkboxes, per-scanner WR (scanner_stats). Gaps are presentation/editability/inline-actions.

## SLICE 1 (this session) — the two flagship matches

### PKG-A · Strategy Builder: uTrade-grade inline action row + gate badge
uTrade's signature is a single row of validate-in-place actions beneath every generated
strategy: **Backtest · Pay-off · Margin · Deploy**, plus FinStocks' persistent **Paper/Live**
toggle (Paper default) and a prominent **gate** verdict.

Backend (worktree `backend/`):
- NEW `GET /api/strategies/{id}/gate` in `api/strategies_routes.py` — run `evaluate_gate(row.last_backtest)`
  (reuse the import already used by transition) → `{has_backtest, passed, failures:[...], metrics:{...}}`.
  Auth = owner. No transition, no side effects. This makes the gate verdict readable without
  attempting a deploy.

Frontend (worktree `frontend/`):
- NEW `components/strategies/StrategyActionRow.tsx` — props `{dsl, draft, phase, onSave, onBacktest, btResult, onReset}`.
  Renders (xAI dark, foundation components):
  - **Paper/Live segmented pill** (Paper default; white-pill active) — the persistent Mock/Live control.
  - Inline action buttons: **[Run backtest]** (existing handler) · **[Payoff]** · **[Margin]** · **[Deploy]**.
    - Payoff: enabled only when `dsl.instrument_segment === 'OPTIONS'` && legs present → opens a
      Dialog with the existing `components/strategy/PayoffDiagram.tsx` (legs + spotPrice from dsl).
      Else a disabled chip "Equity — no option payoff".
    - Margin: client-side **estimate** (labeled "est.") from position_size × capital
      (percent_of_capital → ₹ per trade; options → Σ premium×lot est.). Honest, no broker call.
    - Deploy: Paper → `POST /{id}/transition {to:'paper'}` (toast); Live → foundation Dialog confirm
      → transition `{to:'live'}`; on 422 gate_failed render the server `message` verbatim in a
      down-tone callout (never mask). Disabled until a draft exists.
  - **Gate badge**: after backtest, `GET /{id}/gate` → up-tone `GATE PASS` / down-tone `NEEDS WORK`
    (+ failures list in a popover); before backtest → muted "Run a backtest to check the gate".
- EDIT `app/(platform)/strategies/page.tsx` Builder post-compile view: keep the Symbol/Lookback/
  Capital inputs; replace the `[Save as draft][Run backtest][Start over]` row with
  `<StrategyActionRow …/>`. Keep BacktestViewer below.

### PKG-B · Screener: editable resolved-scanner chips + working Save (Tradomate editability)
Today the NL Screener Agent resolves to a bag of scanner IDs shown only as narration prose; the
`[Save as screen][Refine in Power tab]` labels are dead. Match Tradomate's editable screen + fix
the loop — additively, without touching the streaming choreography.

Frontend (worktree `frontend/`):
- NEW `components/scanner/ScreenRefinePanel.tsx` — props `{scannerIds, catalog, minHits, onRerun, onSave}`.
  Renders below the agent artifact once a scan resolves:
  - resolved scanners as **removable chips** (name + ✕), each with its per-scanner WR if present.
  - **"+ Add scanner"** popover from `/v2/scanner-catalog` (grouped by category).
  - **min-hits stepper** (2/3/4/5) with the "match ≥N" label (our confluence's boolean).
  - **[Re-run]** → `api.screener.powerConfluence({scanners, min_hits, limit})` → replaces the result table in place.
  - **[Save as screen]** → `createSavedScan({name, scanner_ids, min_hits})` (existing hourly saved-scan) → toast.
- EDIT `app/(platform)/scanner/page.tsx` `ScreenerAgentHero`: capture `scanners_used` from the
  nl-scan/confluence response; mount `ScreenRefinePanel` beneath the agent; wire re-run + save.
  Leave EmbeddedAgent internals untouched.

## Non-goals (later slices, tracked)
Per-screen WR gauge (needs `backfill_scanner_outcomes.py` run first — honest-empty otherwise),
returns/vol/DD columns on result rows, three-layer backtest-a-screen analytics, no-code nested
AND/OR block builder, Strategy Console (chart|orders|activity), DSL-on-every-card, Basic/Advanced
toggle, Progress+References dual-rail, AI Stock one-pager, Hindi prompts. These are real and
sequenced but out of this slice.

## Verify
pytest (new gate endpoint test) · `tsc --noEmit` · `next build` (or dev compile) ·
Playwright on :3000 authed: Builder compile → action row (backtest→gate badge→margin→deploy paper);
Scanner NL → refine chips → re-run → save. Screenshot each.
