# Quant X — xAI Redesign Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Quant X's visual language with the xAI design system (near-black canvas, white outline pills, Inter 400 + Geist Mono, minimal duotone), re-skin the `components/foundation/*` primitives, and rebuild `/copilot` as the approved north-star — all on the frozen backend.

**Architecture:** The whole theme is driven by CSS variables in `app/globals.css` consumed through `tailwind.config.ts`. We remap the variable *definitions* to xAI values (high leverage — most token-using surfaces re-skin automatically), fix the malformed/undefined shadcn channel vars, make the app dark-only, then re-skin each foundation primitive and verify every change with Playwright computed-style assertions against a new `/preview-design` kitchen-sink. The Copilot surface is rebuilt last using the re-skinned primitives plus a curated, mono-restyled delight layer.

**Tech Stack:** Next.js 14.1 App Router · React 18 · TypeScript · Tailwind 3.4 · Radix · `next/font/google` (Inter + Geist_Mono) · shadcn registry (for Aceternity/Magic UI delight) · Playwright (visual + computed-style verification). No new runtime deps beyond what shadcn components pull.

**Spec:** `docs/superpowers/specs/2026-06-18-quantx-xai-redesign-design.md`. **xAI tokens:** `DESIGN-x.ai.md`. **Audit:** `docs/QUANTX_DEEP_AUDIT_2026_06_18.md`.

**Hard guardrails (NEVER touch):** `lib/api.ts` paths/contract · money-path guards (kill-switch / OOS gate 422 / role checks) · brand firewall `public_label()` · no-fallback discipline · safety-friction UX. Every feature keeps its route. Backend untouched.

---

## ⚠️ RE-BASE ADDENDUM (2026-06-19) — this plan now targets `feat/mldl-4engine`, NOT `main`

The redesign branch was re-based onto `feat/mldl-4engine` (the real app, 301 commits ahead of `main`). The task text below was authored against `main`; apply it to mldl's ACTUAL files with these deltas:

- **globals.css line numbers in the tasks are indicative (main-relative).** mldl's `globals.css` differs by ~228 lines. The token VALUES and approach are unchanged — locate mldl's `:root` `--color-*` block, the `html.light` block(s), the shadcn channel vars (currently HSL), and the `--success`/`--danger`/`--text-*` references, and apply the same xAI remap (canvas `#0a0a0a` etc.; HSL→RGB channels; DEFINE the undefined `--success`/`--danger`/`--warning`/`--text-*`/`--background-*`; remove ALL `html.light`; `.numeric`→`var(--font-mono)`; add `--font-mono: var(--font-geist-mono)`).
- **Foundation set on mldl differs:** it ADDS `ConfirmDialog`, `Reveal`, `Spark`, `Verdict`, `DisclaimerFooter` — re-skin these too (same xAI rules; `Verdict` BUY/SELL uses duotone, `Spark` like `Sparkline`). It does NOT have `DropdownMenu` (skip it in Task 14).
- **These NEW files are ALREADY carried over (do NOT re-create):** `components/foundation/EyebrowMono.tsx`, `tests/e2e/design-system.spec.ts`, `app/preview-design/{page,OverlaysDemo}.tsx`, `components/ui/{typing-animation,terminal,blur-fade,dot-pattern}.tsx`, `components/copilot/{MarkdownMessage,SuggestedPrompts}.tsx`. You MUST still export `EyebrowMono` from `components/foundation/index.ts`.
- **Copilot (Tasks 18-21) targets the REAL surface on mldl:** `components/copilot/EmbeddedAgent.tsx` + `artifacts.tsx` + `types.ts` + `app/(platform)/copilot/page.tsx` (NOT `CopilotPanel.tsx`, which is main-only). Read the real files; PRESERVE whatever data flow exists (SSE stream if present, else the JSON `copilotChat` call) byte-identically. Re-skin EmbeddedAgent/artifacts to mono + duotone-numbers; wire in MarkdownMessage/SuggestedPrompts/Terminal/BlurFade.
- **deps:** do NOT add the `motion` package (delight components already use `framer-motion@10`); `providers.tsx` gains `<MotionConfig reducedMotion="user">`. Run `npx shadcn@latest init` only if mldl lacks `components.json`; the 4 delight components are already vendored.

---

## File Structure

**Create:**
- `frontend/app/preview-design/page.tsx` — dev-only kitchen-sink rendering every primitive (the visual + test target).
- `frontend/components/foundation/EyebrowMono.tsx` — new primitive: uppercase tracked Geist Mono section label.
- `frontend/tests/e2e/design-system.spec.ts` — Playwright computed-style assertions for the token system + primitives.
- `frontend/components/copilot/MarkdownMessage.tsx` — the single xAI markdown renderer (replaces ad-hoc renderers).
- `frontend/components/copilot/SuggestedPrompts.tsx` — empty-state prompt pills.

**Modify:**
- `frontend/app/globals.css` — remap `--color-*`, fix shadcn channel vars, define dead `--success`/`--danger`/`--text-*`, remove `html.light` + `!important` overrides.
- `frontend/tailwind.config.ts` — add `borderRadius.pill`, prune dead legacy palettes/keyframes (optional cleanup), keep token mappings.
- `frontend/app/layout.tsx` — swap DM_Sans/DM_Mono → Inter/Geist_Mono; `<html className="dark">`.
- `frontend/app/providers.tsx` — `forcedTheme="dark"`; simplify `ThemedToaster` to mono.
- `frontend/middleware.ts` — add `/preview-design` to public prefixes (dev).
- `frontend/components/foundation/{Button,Card,StatCard,Badge,ChangeBadge,Input,NumericInput,Select,Tabs,DataTable,Dialog,Sheet,Popover,Tooltip,DropdownMenu,Toast,Skeleton,EmptyState,PageHeader,UsageMeter,Sparkline}.tsx` — re-skin to xAI.
- `frontend/components/foundation/index.ts` — export `EyebrowMono`.
- `frontend/app/(platform)/copilot/page.tsx` + `frontend/components/copilot/CopilotPanel.tsx` — rebuild surface.
- `frontend/components/copilot/artifacts.tsx`, `EmbeddedAgent.tsx` — re-skin GenUI to mono + duotone numbers.

**xAI token reference (used throughout):**
| Purpose | Hex | RGB channels |
|---|---|---|
| canvas | `#0a0a0a` | `10 10 10` |
| canvas-soft | `#1a1c20` | `26 28 32` |
| canvas-card | `#191919` | `25 25 25` |
| canvas-mid | `#363a3f` | `54 58 63` |
| hairline | `#212327` | `33 35 39` |
| ink | `#ffffff` | `255 255 255` |
| body | `#dadbdf` | `218 219 223` |
| mute | `#7d8187` | `125 129 135` |
| up (duotone) | `#3FB950` | `63 185 80` |
| down (duotone) | `#F85149` | `248 81 73` |
| accent-sunset | `#ff7a17` | `255 122 23` |

---

## Task 0: Tooling — fonts dep check + shadcn init

**Files:**
- Modify: `frontend/components.json` (create if missing)

- [ ] **Step 1: Install the `geist` package (Geist Mono loader) + confirm Inter**

Geist Mono is NOT in `next/font/google` on Next 14.1.0 (it was added in 14.2). Use Vercel's official `geist` package, which self-hosts the real typeface and works on 14.1. Inter stays on `next/font/google`.

Run: `cd frontend && npm install geist`
Then verify Inter is available: `cd frontend && node -e "console.log(require('next/font/google').Inter ? 'ok' : 'missing')"`
Expected: `geist` installs; prints `ok`.

- [ ] **Step 2: Initialize shadcn so delight components install into our tree**

Run: `cd frontend && npx shadcn@latest init --defaults --yes`
Expected: creates/updates `components.json`. If it prompts, accept: style `default`, base color `neutral`, CSS vars `yes`, components alias `@/components`, utils `@/lib/utils` (already exists).

- [ ] **Step 3: Verify the alias resolves and build is unbroken**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS (no new errors vs baseline).

- [ ] **Step 4: Commit**

```bash
cd frontend && git add components.json && git commit -m "chore(redesign): shadcn init for xAI delight components"
```

---

## Task 1: Remap the token system to xAI (the high-leverage change)

**Files:**
- Modify: `frontend/app/globals.css` (`:root` block ~lines 10-72, `html.light` block ~lines 83-136, `!important` override block ~lines 1137-1200)

- [ ] **Step 1: Remap the `--color-*` family in `:root` to xAI hex**

In `:root`, set these exact values (replace the existing hex):

```css
:root {
  --color-main:        #0a0a0a;   /* canvas */
  --color-wrap:        #191919;   /* canvas-card */
  --color-wrap-hover:  #1a1c20;   /* canvas-soft */
  --color-hover:       #1a1c20;
  --color-card-hover:  #1a1c20;
  --color-surface-2:   #141414;
  --color-chart-bg:    #0a0a0a;
  --color-line:        #212327;   /* hairline */
  --color-wrap-line:   #363a3f;   /* canvas-mid (accent border) */
  --color-light:       #ffffff;   /* ink — text primary */
  --color-desc:        #dadbdf;   /* body — text secondary */
  --color-muted:       #7d8187;   /* mute — text tertiary */
  --color-up:          #3FB950;   /* duotone up */
  --color-down:        #F85149;   /* duotone down */
  --color-highlight:   #ff7a17;   /* sunset accent (rare) */
  --color-orange:      #ff7a17;
}
```

- [ ] **Step 2: Convert the shadcn channel vars from HSL→RGB channels and set to xAI mono**

These are consumed by `tailwind.config.ts` as `rgb(var(--x) / <alpha>)`, so they MUST be space-separated RGB channels (the current HSL triplets like `163 82% 62%` are malformed under `rgb()`). Replace the `:root` shadcn block:

```css
  --background: 10 10 10;
  --foreground: 255 255 255;
  --card: 25 25 25;
  --card-foreground: 255 255 255;
  --popover: 25 25 25;
  --popover-foreground: 255 255 255;
  --primary: 255 255 255;            /* xAI primary = white */
  --primary-foreground: 10 10 10;
  --secondary: 26 28 32;
  --secondary-foreground: 255 255 255;
  --muted: 26 28 32;
  --muted-foreground: 125 129 135;
  --accent: 255 122 23;              /* sunset, rare */
  --accent-foreground: 10 10 10;
  --destructive: 248 81 73;
  --destructive-foreground: 255 255 255;
  --border: 33 35 39;                /* hairline */
  --ring: 255 255 255;
```

- [ ] **Step 3: Define the previously-undefined channel vars (kills the colorless-token bug)**

`tailwind.config.ts` references `--success`, `--danger`, `--warning`, `--text-primary`, `--text-secondary`, `--text-muted` which were never defined. Add to `:root`:

```css
  --success: 63 185 80;
  --danger: 248 81 73;
  --warning: 255 122 23;
  --text-primary: 255 255 255;
  --text-secondary: 218 219 223;
  --text-muted: 125 129 135;
```

- [ ] **Step 4: Remove light mode**

Delete the entire `html.light { … }` block (~lines 83-136) and the `html.light .text-white { … !important }` + `html.light .bg-white\/… { … !important }` + `html.light .text-white\/… { … }` override blocks (~lines 1137-1200). Dark is now the only theme; `:root` is authoritative. If `html.light .editorial-shell` (~line 141) and other `html.light …` contextual rules remain, delete them too (grep `grep -n "html.light" app/globals.css` and remove every match).

Run after editing: `cd frontend && grep -c "html.light" app/globals.css`
Expected: `0`

- [ ] **Step 5: Verify the file still compiles (no dangling braces)**

Run: `cd frontend && npx prettier --check app/globals.css || true` then `cd frontend && npm run build 2>&1 | tail -5`
Expected: build reaches CSS processing without a PostCSS/parse error. (Full build verified in Task 16.)

- [ ] **Step 6: Commit**

```bash
cd frontend && git add app/globals.css && git commit -m "feat(redesign): remap design tokens to xAI palette (dark-only, duotone, fix dead tokens)"
```

---

## Task 2: Tailwind config — add pill radius, keep token mappings

**Files:**
- Modify: `frontend/tailwind.config.ts:346-349` (borderRadius), `:16` (darkMode)

- [ ] **Step 1: Add the pill + sm radius tokens**

In `theme.extend.borderRadius`, add:

```ts
      borderRadius: {
        'sm': '8px',      // xAI card radius
        'pill': '9999px', // xAI interactive shape
        '4xl': '2rem',
        '5xl': '2.5rem',
      },
```

- [ ] **Step 2: Remove dead `darkMode: 'class'` reliance (we force dark)**

Leave `darkMode: 'class'` in place (harmless — next-themes still sets `class="dark"`), but DO NOT add any `dark:` variants in new code. (No edit required; this step is a note.)

- [ ] **Step 3: Verify config typechecks**

Run: `cd frontend && npx tsc --noEmit -p tsconfig.json`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
cd frontend && git add tailwind.config.ts && git commit -m "feat(redesign): add xAI pill + 8px card radius tokens"
```

---

## Task 3: Fonts — Inter + Geist Mono, dark `<html>`

**Files:**
- Modify: `frontend/app/layout.tsx:3,18-29,92-96`

- [ ] **Step 1: Swap the font imports + loaders**

Inter from `next/font/google`; Geist Mono from the `geist` package (its variable is fixed to `--font-geist-mono`). Replace line 3 and the DM_Sans/DM_Mono loaders (lines 18-29):

```ts
import { Inter } from 'next/font/google'
import { GeistMono } from 'geist/font/mono'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
  weight: ['400', '500'],
})
// GeistMono is a ready font object — use GeistMono.variable (sets --font-geist-mono)
```

- [ ] **Step 2: Make the document dark and apply the font variables on `<html>`**

Put BOTH font variables on `<html>` (so `:root`-scope can alias them) and make it dark. Replace lines 92-96:

```tsx
    <html lang="en" className={`dark ${inter.variable} ${GeistMono.variable}`} suppressHydrationWarning>
      <body
        className="font-sans bg-main antialiased"
        style={{ color: 'var(--color-light)' }}
      >
```

Then in `app/globals.css` `:root`, alias the mono token to Geist so Tailwind's `font-mono` (→ `var(--font-mono)`) resolves to Geist Mono:

```css
  --font-mono: var(--font-geist-mono);
```

Also update `viewport.themeColor` (line 15) to `'#0a0a0a'`. (Drop the `noise-overlay` class only if it references removed light styles; otherwise keep — it is mono-safe.)

- [ ] **Step 3: Build to confirm fonts resolve**

Run: `cd frontend && npm run build 2>&1 | tail -8`
Expected: build downloads Inter + Geist Mono and completes the compile step without a font error.

- [ ] **Step 4: Commit**

```bash
cd frontend && git add app/layout.tsx && git commit -m "feat(redesign): Inter 400 + Geist Mono, dark-only document"
```

---

## Task 4: Providers — force dark, mono toaster

**Files:**
- Modify: `frontend/app/providers.tsx:55-88` (ThemedToaster), `:98-105` (ThemeProvider)

- [ ] **Step 1: Force the dark theme**

Replace the `<ThemeProvider …>` opening (lines 99-105):

```tsx
    <ThemeProvider
      attribute="class"
      forcedTheme="dark"
      storageKey="quantx.theme"
      disableTransitionOnChange
    >
```

- [ ] **Step 2: Collapse `ThemedToaster` to a single mono style**

Replace the `ThemedToaster` body (lines 55-88) with:

```tsx
function ThemedToaster() {
  return (
    <Toaster
      theme="dark"
      position="top-right"
      toastOptions={{
        style: {
          background: 'var(--color-wrap)',
          border: '1px solid var(--color-line)',
          borderRadius: '8px',
          color: 'var(--color-light)',
        },
      }}
    />
  )
}
```

(Remove the now-unused `useTheme`, `useState`, `useEffect`, `mounted` imports/usages in this function. `richColors` is removed — duotone is driven by our tokens.)

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS (no unused-import errors).

- [ ] **Step 4: Commit**

```bash
cd frontend && git add app/providers.tsx && git commit -m "feat(redesign): force dark theme + mono toaster"
```

---

## Task 5: `/preview-design` kitchen-sink route

**Files:**
- Create: `frontend/app/preview-design/page.tsx`
- Modify: `frontend/middleware.ts` (public prefixes)

- [ ] **Step 1: Allow the route publicly in dev**

In `frontend/middleware.ts`, find the public-prefix list (grep `PUBLIC_PREFIXES`) and add `'/preview-design'`.

- [ ] **Step 2: Add `data-testid`/`...rest` passthrough to the primitives the harness measures**

The harness asserts computed styles on the real primitive DOM nodes, so `Card`, `Badge`, and `Input` must forward `data-testid` to their root element (`Button` already spreads `...rest`). Add `...rest` to each root element's props now (1-line change each; full styling happens in their own tasks). Example for `Badge.tsx`:

```tsx
interface Props extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: Tone
}
export const Badge = ({ tone = 'muted', className, children, ...rest }: Props) => (
  <span className={cn(/* … */)} {...rest}>{children}</span>
)
```

Do the equivalent for `Card` (root `div`, extend `React.HTMLAttributes<HTMLDivElement>`) and confirm `Input` already forwards `...rest` to the `<input>` (most do — add if missing).

- [ ] **Step 3: Create the kitchen-sink page**

Uses only primitives that exist today; the eyebrow is inline markup here (Task 7 swaps it for the `EyebrowMono` component, and Task 15 adds an `EmptyState` example).

```tsx
// frontend/app/preview-design/page.tsx
import { notFound } from 'next/navigation'
import { Button, Card, Badge, Input } from '@/components/foundation'

export default function PreviewDesign() {
  if (process.env.NODE_ENV === 'production') notFound()
  return (
    <main data-testid="preview-root" className="min-h-screen bg-main text-d-text-primary p-8 space-y-10">
      <section data-testid="sec-type" className="space-y-3">
        <p className="font-mono uppercase tracking-[0.1em] text-xs text-d-text-muted">Typography</p>
        <h1 className="text-display-lg font-sans">Engineered restraint</h1>
        <p className="text-d-text-secondary">Body copy in Inter weight 400.</p>
      </section>

      <section data-testid="sec-buttons" className="flex gap-3">
        <Button data-testid="btn-primary" variant="primary">Primary</Button>
        <Button data-testid="btn-secondary" variant="secondary">Outline pill</Button>
        <Button variant="ghost">Ghost</Button>
      </section>

      <section data-testid="sec-cards" className="grid grid-cols-2 gap-4">
        <Card data-testid="card-default"><p className="text-d-text-primary">Card content</p></Card>
        <Card><p className="text-d-text-secondary">Another card</p></Card>
      </section>

      <section data-testid="sec-badges" className="flex gap-2">
        <Badge tone="up" data-testid="badge-up">+2.40%</Badge>
        <Badge tone="down" data-testid="badge-down">-1.10%</Badge>
        <Badge tone="muted">NEUTRAL</Badge>
      </section>

      <section data-testid="sec-input" className="max-w-sm">
        <Input data-testid="search-input" placeholder="Search symbol" />
      </section>
    </main>
  )
}
```

(Add `Tabs`/`DataTable`/overlay/`EmptyState` examples to this page when those primitives are touched in Tasks 12-15.)

- [ ] **Step 3: Run the dev server and open the route**

Run: `cd frontend && npm run dev` (separate shell), then visit `http://localhost:3000/preview-design`.
Expected: page renders on near-black; primitives still look "old" (re-skin happens next). No 404, no auth redirect.

- [ ] **Step 4: Commit**

```bash
cd frontend && git add app/preview-design/page.tsx middleware.ts && git commit -m "feat(redesign): /preview-design kitchen-sink (dev-only)"
```

---

## Task 6: Playwright design-system harness (failing assertions first — TDD)

**Files:**
- Create: `frontend/tests/e2e/design-system.spec.ts`

- [ ] **Step 1: Write the computed-style assertions**

```ts
// frontend/tests/e2e/design-system.spec.ts
import { test, expect } from '@playwright/test'

const rgb = (s: string) => s.replace(/\s+/g, '')

test.describe('xAI design system', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/preview-design')
  })

  test('canvas is near-black #0a0a0a', async ({ page }) => {
    const bg = await page.locator('[data-testid="preview-root"]').evaluate(
      (el) => getComputedStyle(el).backgroundColor)
    expect(rgb(bg)).toBe('rgb(10,10,10)')
  })

  test('primary button is a white-filled pill', async ({ page }) => {
    const btn = page.locator('[data-testid="btn-primary"]')
    const s = await btn.evaluate((el) => {
      const c = getComputedStyle(el)
      return { radius: c.borderRadius, bg: c.backgroundColor }
    })
    expect(parseInt(s.radius)).toBeGreaterThanOrEqual(999)   // pill
    expect(rgb(s.bg)).toBe('rgb(255,255,255)')               // white-filled
  })

  test('secondary button is an outline pill with 1px border', async ({ page }) => {
    const s = await page.locator('[data-testid="btn-secondary"]').evaluate((el) => {
      const c = getComputedStyle(el)
      return { radius: c.borderRadius, borderWidth: c.borderTopWidth }
    })
    expect(parseInt(s.radius)).toBeGreaterThanOrEqual(999)
    expect(parseInt(s.borderWidth)).toBe(1)
  })

  test('cards are 8px charcoal with hairline border, no shadow', async ({ page }) => {
    const s = await page.locator('[data-testid="card-default"]').evaluate((el) => {
      const c = getComputedStyle(el)
      return { radius: c.borderRadius, bg: c.backgroundColor, shadow: c.boxShadow, border: c.borderTopColor }
    })
    expect(s.radius).toBe('8px')
    expect(rgb(s.bg)).toBe('rgb(25,25,25)')
    expect(s.shadow).toBe('none')
    expect(rgb(s.border)).toBe('rgb(33,35,39)')
  })

  test('duotone: up is green, down is red', async ({ page }) => {
    const up = await page.locator('[data-testid="badge-up"]').evaluate((el) => getComputedStyle(el).color)
    const down = await page.locator('[data-testid="badge-down"]').evaluate((el) => getComputedStyle(el).color)
    expect(rgb(up)).toBe('rgb(63,185,80)')
    expect(rgb(down)).toBe('rgb(248,81,73)')
  })

  test('eyebrow is uppercase mono', async ({ page }) => {
    const s = await page.locator('[data-testid="sec-type"] p').first().evaluate((el) => {
      const c = getComputedStyle(el)
      return { transform: c.textTransform, family: c.fontFamily }
    })
    expect(s.transform).toBe('uppercase')
    expect(s.family.toLowerCase()).toContain('mono')
  })
})
```

- [ ] **Step 2: Run — expect FAIL on primitive assertions (canvas already passes)**

Run: `cd frontend && npx playwright test tests/e2e/design-system.spec.ts --reporter=line`
Expected: the canvas test PASSES (Task 1 done); button/card/eyebrow tests FAIL (primitives not yet re-skinned). This is the TDD baseline.

- [ ] **Step 3: Commit**

```bash
cd frontend && git add tests/e2e/design-system.spec.ts && git commit -m "test(redesign): design-system computed-style harness"
```

---

## Task 7: `EyebrowMono` primitive (new)

**Files:**
- Create: `frontend/components/foundation/EyebrowMono.tsx`
- Modify: `frontend/components/foundation/index.ts`

- [ ] **Step 1: Create the component**

```tsx
// frontend/components/foundation/EyebrowMono.tsx
import * as React from 'react'
import { cn } from '@/lib/utils'

interface Props extends React.HTMLAttributes<HTMLParagraphElement> {
  children: React.ReactNode
}

/** xAI section eyebrow: uppercase tracked Geist Mono, reads like a code comment. */
export const EyebrowMono = ({ className, children, ...rest }: Props) => (
  <p
    className={cn(
      'font-mono uppercase tracking-[0.1em] text-xs text-d-text-muted',
      className,
    )}
    {...rest}
  >
    {children}
  </p>
)
EyebrowMono.displayName = 'EyebrowMono'
```

- [ ] **Step 2: Export it**

In `frontend/components/foundation/index.ts` add: `export { EyebrowMono } from './EyebrowMono'`

- [ ] **Step 3: Run the eyebrow test**

Run: `cd frontend && npx playwright test tests/e2e/design-system.spec.ts -g "eyebrow" --reporter=line`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
cd frontend && git add components/foundation/EyebrowMono.tsx components/foundation/index.ts && git commit -m "feat(redesign): EyebrowMono primitive"
```

---

## Task 8: `Button` → outline pill + white-filled primary + active scale

**Files:**
- Modify: `frontend/components/foundation/Button.tsx`

- [ ] **Step 1: Re-skin variants, radius, and press feedback**

Replace the whole file:

```tsx
import * as React from 'react'
import { cn } from '@/lib/utils'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
export type ButtonSize = 'sm' | 'md' | 'lg'

interface Props extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
}

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  // rare white-filled pill — the primary CTA
  primary: 'bg-white text-main border border-white hover:bg-white/90',
  // canonical outline pill — translucent-white border
  secondary: 'bg-transparent text-d-text-primary border border-white/20 hover:bg-white/[0.06]',
  ghost: 'bg-transparent text-d-text-secondary hover:text-d-text-primary hover:bg-white/[0.04]',
  danger: 'bg-transparent text-down border border-down/40 hover:bg-down/[0.08]',
}

const SIZE_CLASSES: Record<ButtonSize, string> = {
  sm: 'h-8 px-4 text-xs',
  md: 'h-9 px-5 text-sm',
  lg: 'h-11 px-6 text-sm',
}

export const Button = React.forwardRef<HTMLButtonElement, Props>(
  ({ variant = 'primary', size = 'md', className, children, ...rest }, ref) => (
    <button
      ref={ref}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-pill font-normal',
        'transition-[transform,background-color,border-color] duration-150 ease-out',
        'active:scale-[0.97]',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/40',
        VARIANT_CLASSES[variant],
        SIZE_CLASSES[size],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  ),
)
Button.displayName = 'Button'
```

(`font-medium`→`font-normal`: xAI is weight 400. `rounded-md`→`rounded-pill`. Added `active:scale-[0.97]` per emil-design-eng.)

- [ ] **Step 2: Run the button tests**

Run: `cd frontend && npx playwright test tests/e2e/design-system.spec.ts -g "button" --reporter=line`
Expected: PASS (primary + secondary).

- [ ] **Step 3: Commit**

```bash
cd frontend && git add components/foundation/Button.tsx && git commit -m "feat(redesign): Button → xAI outline/white-filled pills + active scale"
```

---

## Task 9: `Card` + `StatCard` → 8px charcoal hairline, no shadow

**Files:**
- Modify: `frontend/components/foundation/Card.tsx`, `frontend/components/foundation/StatCard.tsx`

- [ ] **Step 1: Read the current files**

Run: `cd frontend && sed -n '1,80p' components/foundation/Card.tsx components/foundation/StatCard.tsx`

- [ ] **Step 2: Apply the xAI card mapping**

In both files, set the container classes so resolved styles are: background `bg-wrap` (→#191919), border `border border-line` (→#212327), radius `rounded-sm` (8px), and **remove** any `shadow-*`, `glass-card`, `backdrop-blur`, gradient, or `bg-[#hex]` literal. Ensure `data-testid`/`...rest` passes through to the root element. Canonical container class:

```
'bg-wrap border border-line rounded-sm p-6'
```

(Card interior padding = 24px = `p-6`. Keep existing header/footer subcomponents; just swap their surface/border/radius/shadow classes to the tokens above. Remove any `hover:shadow`/`hover:-translate` motion — xAI cards are flat.)

- [ ] **Step 3: Run the card test**

Run: `cd frontend && npx playwright test tests/e2e/design-system.spec.ts -g "cards" --reporter=line`
Expected: PASS (radius 8px, bg rgb(25,25,25), shadow none, border rgb(33,35,39)).

- [ ] **Step 4: Commit**

```bash
cd frontend && git add components/foundation/Card.tsx components/foundation/StatCard.tsx && git commit -m "feat(redesign): Card/StatCard → xAI flat hairline charcoal"
```

---

## Task 10: `Badge` + `ChangeBadge` + duotone discipline

**Files:**
- Modify: `frontend/components/foundation/Badge.tsx`, `frontend/components/foundation/ChangeBadge.tsx`

- [ ] **Step 1: Tighten Badge radius (tone colors already follow the remapped vars)**

`Badge`'s `up`/`down` tones already resolve to the new tokens (the duotone test passes once `--color-up/down` are remapped — done in Task 1). Set radius `rounded`→`rounded-sm`, weight `font-medium`→`font-normal`, and make the muted tone mono:

```tsx
const TONE: Record<Tone, string> = {
  primary: 'bg-white/[0.06] text-d-text-primary border-white/15',
  up: 'bg-up/10 text-up border-up/20',
  down: 'bg-down/10 text-down border-down/20',
  warning: 'bg-warning/10 text-warning border-warning/20',
  muted: 'bg-wrap-hover text-d-text-muted border-line',
}
```

Base: `'inline-flex items-center px-2 py-0.5 text-xs font-normal rounded-sm border'`.

- [ ] **Step 2: Read + re-skin ChangeBadge**

Run: `cd frontend && sed -n '1,80p' components/foundation/ChangeBadge.tsx`
Apply: positive→`text-up`, negative→`text-down`, neutral→`text-d-text-muted`; radius `rounded-sm`; no glow/shadow; keep the lakh/crore formatting logic untouched.

- [ ] **Step 3: Run the duotone test (still green)**

Run: `cd frontend && npx playwright test tests/e2e/design-system.spec.ts -g "duotone" --reporter=line`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
cd frontend && git add components/foundation/Badge.tsx components/foundation/ChangeBadge.tsx && git commit -m "feat(redesign): Badge/ChangeBadge → xAI mono chrome + duotone numbers"
```

---

## Task 11: `Input` + `NumericInput` + `Select`

**Files:**
- Modify: `frontend/components/foundation/{Input,NumericInput,Select}.tsx`

- [ ] **Step 1: Read current files**

Run: `cd frontend && sed -n '1,120p' components/foundation/Input.tsx components/foundation/NumericInput.tsx components/foundation/Select.tsx`

- [ ] **Step 2: Apply the xAI input mapping**

Field/input classes resolve to: `bg-wrap-hover` (→#1a1c20), `border border-line`, `rounded-sm` (8px), text `text-d-text-primary`, placeholder `placeholder:text-d-text-muted`, focus `focus-visible:ring-1 focus-visible:ring-white/40 focus-visible:border-white/30`. Remove any glow/gradient. Preserve `NumericInput`'s wheel-blur guard, formatters, and clamp logic (do NOT touch behavior). Canonical input class:

```
'h-9 w-full bg-wrap-hover border border-line rounded-sm px-3 text-sm text-d-text-primary placeholder:text-d-text-muted focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/40'
```

- [ ] **Step 3: Add Input assertion + run**

Append to `design-system.spec.ts`:

```ts
test('inputs use hairline border + 8px radius', async ({ page }) => {
  const s = await page.locator('[data-testid="sec-input"] input').evaluate((el) => {
    const c = getComputedStyle(el)
    return { radius: c.borderRadius, border: c.borderTopColor }
  })
  expect(s.radius).toBe('8px')
  expect(s.border.replace(/\s+/g,'')).toBe('rgb(33,35,39)')
})
```

Run: `cd frontend && npx playwright test tests/e2e/design-system.spec.ts -g "inputs" --reporter=line`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
cd frontend && git add components/foundation/Input.tsx components/foundation/NumericInput.tsx components/foundation/Select.tsx tests/e2e/design-system.spec.ts && git commit -m "feat(redesign): Input/NumericInput/Select → xAI fields"
```

---

## Task 12: `Tabs` → mono-caps labels + underline indicator

**Files:**
- Modify: `frontend/components/foundation/Tabs.tsx`, `frontend/app/preview-design/page.tsx`

- [ ] **Step 1: Read + re-skin**

Run: `cd frontend && sed -n '1,140p' components/foundation/Tabs.tsx`
Apply: tab labels `font-mono uppercase tracking-[0.08em] text-xs`; active = `text-d-text-primary` with a 1px `border-b border-white` underline indicator; inactive = `text-d-text-muted hover:text-d-text-secondary`; tablist bottom border `border-b border-line`; **no** animated sliding background, no glow. Keep keyboard/aria behavior.

- [ ] **Step 2: Add a Tabs example + visual check**

Add a `<Tabs>` example to `preview-design/page.tsx` under `data-testid="sec-tabs"`; run the dev server and confirm mono-caps + underline render.

- [ ] **Step 3: Commit**

```bash
cd frontend && git add components/foundation/Tabs.tsx app/preview-design/page.tsx && git commit -m "feat(redesign): Tabs → xAI mono-caps + underline"
```

---

## Task 13: `DataTable` → mono-caps header, hairline rows

**Files:**
- Modify: `frontend/components/foundation/DataTable.tsx`, `frontend/app/preview-design/page.tsx`, `frontend/tests/e2e/design-system.spec.ts`

- [ ] **Step 1: Read + re-skin**

Run: `cd frontend && sed -n '1,200p' components/foundation/DataTable.tsx`
Apply: `<th>` = `font-mono uppercase tracking-[0.08em] text-xs text-d-text-muted bg-wrap-hover`; `<td>` = `text-sm text-d-text-primary`; row border `border-b border-line`; cell padding `px-4 py-3`; sticky header keeps `bg-wrap-hover`; remove zebra/glow. Preserve sorting, keyboard nav, skeleton/empty/error slots, and the generic typing.

- [ ] **Step 2: Add example + assertion**

Add a small `<DataTable>` to the preview page (`data-testid="sec-table"`). Append:

```ts
test('datatable header is mono-caps', async ({ page }) => {
  const th = page.locator('[data-testid="sec-table"] thead th').first()
  const s = await th.evaluate((el) => {
    const c = getComputedStyle(el)
    return { transform: c.textTransform, family: c.fontFamily }
  })
  expect(s.transform).toBe('uppercase')
  expect(s.family.toLowerCase()).toContain('mono')
})
```

Run: `cd frontend && npx playwright test tests/e2e/design-system.spec.ts -g "datatable" --reporter=line`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd frontend && git add components/foundation/DataTable.tsx app/preview-design/page.tsx tests/e2e/design-system.spec.ts && git commit -m "feat(redesign): DataTable → xAI mono-caps header + hairline rows"
```

---

## Task 14: Overlays — `Dialog`/`Sheet`/`Popover`/`Tooltip`/`DropdownMenu`/`Toast`

**Files:**
- Modify: `frontend/components/foundation/{Dialog,Sheet,Popover,Tooltip,DropdownMenu,Toast}.tsx`, `frontend/app/preview-design/page.tsx`

- [ ] **Step 1: Read each file**

Run: `cd frontend && sed -n '1,120p' components/foundation/Dialog.tsx components/foundation/Sheet.tsx components/foundation/Popover.tsx components/foundation/Tooltip.tsx components/foundation/DropdownMenu.tsx components/foundation/Toast.tsx`

- [ ] **Step 2: Apply the overlay mapping (per file)**

Each overlay surface: `bg-wrap border border-line rounded-sm`, text `text-d-text-primary`, scrim `bg-black/60` (no blur, or `backdrop-blur-sm` max). Motion per emil-design-eng: enter `ease-out` ≤200ms, `scale(0.97)+opacity` (never `scale(0)`); **popovers/tooltips/dropdowns** set `transform-origin` to the Radix var (e.g. `origin-[var(--radix-popover-content-transform-origin)]`); **Dialog stays centered**. Tooltip duration 125ms. Remove glass/glow; one subtle `shadow-soft` allowed on Dialog only.

- [ ] **Step 3: Verify open/close + origins visually**

Add trigger examples to the preview page; run dev server; confirm each opens, scales from the correct origin, and is mono-skinned.

- [ ] **Step 4: review-animations gate**

Invoke the `review-animations` skill on the diff of these six files. Apply any Block-level fixes before committing.

- [ ] **Step 5: Commit**

```bash
cd frontend && git add components/foundation/Dialog.tsx components/foundation/Sheet.tsx components/foundation/Popover.tsx components/foundation/Tooltip.tsx components/foundation/DropdownMenu.tsx components/foundation/Toast.tsx app/preview-design/page.tsx && git commit -m "feat(redesign): overlays → xAI hairline surfaces, origin-aware motion"
```

---

## Task 15: `Skeleton`/`EmptyState`/`PageHeader`/`UsageMeter`/`Sparkline`

**Files:**
- Modify: `frontend/components/foundation/{Skeleton,EmptyState,PageHeader,UsageMeter,Sparkline}.tsx`

- [ ] **Step 1: Read each file**

Run: `cd frontend && sed -n '1,120p' components/foundation/Skeleton.tsx components/foundation/EmptyState.tsx components/foundation/PageHeader.tsx components/foundation/UsageMeter.tsx components/foundation/Sparkline.tsx`

- [ ] **Step 2: Apply mappings**

- `Skeleton`: `bg-wrap-hover` base, subtle `animate-pulse`, `rounded-sm`; no shimmer/gradient.
- `EmptyState`: centered, optional `EyebrowMono` label, `text-d-text-secondary` body, an outline-pill CTA; `bg-wrap-hover rounded-sm p-12` frame.
- `PageHeader`: title `text-display-sm` (32px) Inter-400, optional `EyebrowMono` above it, hairline `border-b border-line` under the header.
- `UsageMeter`: track `bg-wrap-hover`, fill `bg-white` (mono), near-limit→`bg-down`; `rounded-pill`.
- `Sparkline`: stroke uses `var(--color-up)`/`var(--color-down)` by sign, mono otherwise. Remove the non-token chart palette.

- [ ] **Step 3: Visual check + commit**

Run dev server, confirm on the preview page.
```bash
cd frontend && git add components/foundation/Skeleton.tsx components/foundation/EmptyState.tsx components/foundation/PageHeader.tsx components/foundation/UsageMeter.tsx components/foundation/Sparkline.tsx && git commit -m "feat(redesign): skeleton/empty/header/meter/sparkline → xAI"
```

---

## Task 16: Full design-system gate (all primitives green)

**Files:** none (verification)

- [ ] **Step 1: Run the full design-system spec**

Run: `cd frontend && npx playwright test tests/e2e/design-system.spec.ts --reporter=line`
Expected: ALL tests PASS.

- [ ] **Step 2: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build 2>&1 | tail -6`
Expected: tsc PASS; build completes.

- [ ] **Step 3: Milestone commit**

```bash
cd frontend && git commit --allow-empty -m "chore(redesign): foundation primitives re-skinned to xAI — design system green"
```

---

## Task 17: Install + mono-restyle the Copilot delight components

**Files:**
- Create (via shadcn registry, then restyle): Magic UI `typing-animation`, `terminal`, `blur-fade`, `dot-pattern`; Aceternity `placeholders-and-vanish-input`.

- [ ] **Step 1: Install the allowlisted components**

Run:
```bash
cd frontend && npx shadcn@latest add "https://magicui.design/r/typing-animation" "https://magicui.design/r/terminal" "https://magicui.design/r/blur-fade" "https://magicui.design/r/dot-pattern" --yes
cd frontend && npx shadcn@latest add "https://ui.aceternity.com/registry/placeholders-and-vanish-input.json" --yes
```
Expected: components land in `frontend/components/ui/` (or the configured alias). If a URL 404s, fetch the correct registry path from the MCP (`mcp__magic-ui__getRegistryItem` / `mcp__aceternityui__get_installation_info`).

- [ ] **Step 2: Restyle each to monochrome**

Open each new file and replace any gradient/neon/color classes with mono tokens: text `text-d-text-primary`, dim `text-d-text-muted`, surfaces `bg-wrap`/`bg-wrap-hover`, borders `border-line`, dot-pattern fill `fill-white/[0.06]`. Remove colored glow. Keep the motion timing but confirm it obeys emil-design-eng (ease-out, ≤300ms, reduced-motion).

- [ ] **Step 3: review-animations gate**

Invoke `review-animations` on the diff of the installed components. Apply Block fixes.

- [ ] **Step 4: Commit**

```bash
cd frontend && git add components/ && git commit -m "feat(redesign): vendor + mono-restyle Copilot delight components"
```

---

## Task 18: Copilot empty state

**Files:**
- Read: `frontend/app/(platform)/copilot/page.tsx`, `frontend/components/copilot/CopilotPanel.tsx`
- Create: `frontend/components/copilot/SuggestedPrompts.tsx`

- [ ] **Step 1: Read the current Copilot surface**

Run: `cd frontend && cat "app/(platform)/copilot/page.tsx" components/copilot/CopilotPanel.tsx`

- [ ] **Step 2: Build the suggested-prompt pills**

```tsx
// frontend/components/copilot/SuggestedPrompts.tsx
'use client'
import { Button } from '@/components/foundation'

const PROMPTS = [
  'What is the market regime today?',
  'Scan for high-conviction swing signals',
  'Explain my portfolio risk',
  'Any momentum setups in NIFTY 50?',
]

export function SuggestedPrompts({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex flex-wrap justify-center gap-2">
      {PROMPTS.map((p) => (
        <Button key={p} variant="secondary" size="sm" onClick={() => onPick(p)}>
          {p}
        </Button>
      ))}
    </div>
  )
}
```

- [ ] **Step 3: Render the xAI empty state in the Copilot panel**

When there are no messages, render: a `DotPattern` background (mono, `fill-white/[0.06]`), a centered `text-display-md` Inter-400 headline (e.g. "Ask Quant X"), an `EyebrowMono` above it ("MAIN CHAT"), and `<SuggestedPrompts onPick={…} />` wired to submit. Reuse the existing send handler in `CopilotPanel` — do NOT change the API call.

- [ ] **Step 4: Visual check**

Run dev server, open `/copilot` logged-in (or use the dev auth bypass). Confirm the empty state.

- [ ] **Step 5: Commit**

```bash
cd frontend && git add "app/(platform)/copilot/page.tsx" components/copilot/CopilotPanel.tsx components/copilot/SuggestedPrompts.tsx && git commit -m "feat(redesign): Copilot xAI empty state + suggested-prompt pills"
```

---

## Task 19: Copilot composer

**Files:**
- Modify: `frontend/components/copilot/CopilotPanel.tsx`

- [ ] **Step 1: Re-skin the composer**

The input row becomes a single hairline pill: container `bg-wrap-hover border border-line rounded-pill`, mono placeholder, a white-filled send `Button` (pill). Optionally swap the textarea for the `placeholders-and-vanish-input` component for the vanish-on-submit flourish — but keep the existing submit handler / streaming call exactly. Preserve Enter-to-send and disabled-while-streaming behavior.

- [ ] **Step 2: Visual check + commit**

```bash
cd frontend && git add components/copilot/CopilotPanel.tsx && git commit -m "feat(redesign): Copilot xAI pill composer"
```

---

## Task 20: Copilot message stream + single markdown renderer

**Files:**
- Create: `frontend/components/copilot/MarkdownMessage.tsx`
- Modify: `frontend/components/copilot/CopilotPanel.tsx`

- [ ] **Step 1: Find the existing markdown renderers (to consolidate)**

Run: `cd frontend && grep -rln "react-markdown\|marked(\|ReactMarkdown" components/ app/ | head`
Note the markdown library already in use (and any raw-HTML injection sites). Reuse the installed lib; do NOT add a new markdown dep, and prefer the lib's safe rendering over raw HTML injection.

- [ ] **Step 2: Create the single xAI markdown renderer**

`MarkdownMessage.tsx` wraps the existing markdown lib with xAI prose classes: headings Inter-400, body `text-d-text-secondary`, code blocks `bg-wrap-hover border border-line rounded-sm font-mono text-xs`, links `text-d-text-primary underline underline-offset-2`, lists tightened. Mirror the existing renderer's props so it is a drop-in.

- [ ] **Step 3: Re-skin the message stream**

User messages right-aligned in a `bg-wrap-hover rounded-sm` bubble; assistant left-aligned, no bubble, rendered via `MarkdownMessage`. Each message wrapped in `BlurFade` (mono, ≤300ms, staggered). Role label via `EyebrowMono` ("YOU" / "QUANT X"). Replace any other inline markdown renderer in this panel with `MarkdownMessage`.

- [ ] **Step 4: Confirm streaming still works**

Run dev server, send a message, confirm SSE tokens stream and render (the `/copilot/chat/stream` call is unchanged). Verify no console errors.

- [ ] **Step 5: Commit**

```bash
cd frontend && git add components/copilot/MarkdownMessage.tsx components/copilot/CopilotPanel.tsx && git commit -m "feat(redesign): Copilot xAI message stream + single markdown renderer"
```

---

## Task 21: Copilot tool-trace terminal + GenUI artifacts re-skin

**Files:**
- Modify: `frontend/components/copilot/artifacts.tsx`, `frontend/components/copilot/EmbeddedAgent.tsx` (if it renders the trace), `frontend/components/copilot/CopilotPanel.tsx`

- [ ] **Step 1: Read the artifact + trace code**

Run: `cd frontend && cat components/copilot/artifacts.tsx components/copilot/EmbeddedAgent.tsx`

- [ ] **Step 2: Render any tool-call / "thinking" trace as a mono Terminal block**

Where the panel shows tool steps, wrap them in the installed `Terminal` component (mono, `bg-wrap border border-line rounded-sm`). No color except duotone on numeric results.

- [ ] **Step 3: Re-skin artifacts (ChipRow/ArtifactCard/Bars/StatPills/Gauge/ActionRow)**

Surfaces → `bg-wrap border border-line rounded-sm`; labels → `EyebrowMono`/mono; **numbers only** → `text-up`/`text-down` by sign; charts (Bars/Sparkline/Gauge) stroke/fill from `var(--color-up)`/`var(--color-down)`/`white`, removing the hardcoded `#8B5CF6` and any non-token palette. Keep the data wiring intact.

- [ ] **Step 4: review-animations gate + visual check**

Invoke `review-animations` on the Copilot diff. Run dev server; trigger an artifact-producing prompt; confirm mono chrome + duotone numbers + working charts.

- [ ] **Step 5: Commit**

```bash
cd frontend && git add components/copilot/artifacts.tsx components/copilot/EmbeddedAgent.tsx components/copilot/CopilotPanel.tsx && git commit -m "feat(redesign): Copilot tool-trace terminal + mono/duotone GenUI artifacts"
```

---

## Task 22: Copilot north-star verification

**Files:**
- Create: `frontend/tests/e2e/copilot-xai.spec.ts`

- [ ] **Step 1: Write a smoke + visual spec**

```ts
// frontend/tests/e2e/copilot-xai.spec.ts
import { test, expect } from '@playwright/test'

// Uses the authed storage state already configured for tests/e2e/authed.
test.use({ storageState: 'tests/e2e/.auth/user.json' })

test('copilot empty state renders xAI shell', async ({ page }) => {
  await page.goto('/copilot')
  await expect(page.getByText(/MAIN CHAT/i)).toBeVisible()
  const bg = await page.locator('body').evaluate((el) => getComputedStyle(el).backgroundColor)
  expect(bg.replace(/\s+/g,'')).toBe('rgb(10,10,10)')
  await page.screenshot({ path: 'test-results/copilot-xai-empty.png', fullPage: true })
})
```

(If the repo's authed Playwright setup uses a different storageState path, match it — grep `storageState` in `playwright.config.ts`.)

- [ ] **Step 2: Run it**

Run: `cd frontend && npx playwright test tests/e2e/copilot-xai.spec.ts --reporter=line`
Expected: PASS; screenshot saved.

- [ ] **Step 3: Commit**

```bash
cd frontend && git add tests/e2e/copilot-xai.spec.ts && git commit -m "test(redesign): Copilot xAI north-star smoke"
```

---

## Task 23: Phase 1 final verification

**Files:** none

- [ ] **Step 1: Typecheck + production build**

Run: `cd frontend && npx tsc --noEmit && npm run build 2>&1 | tail -10`
Expected: tsc clean; build succeeds.

- [ ] **Step 2: Full e2e suite (existing tests stay green)**

Run: `cd frontend && npx playwright test --reporter=line`
Expected: design-system + copilot specs PASS; pre-existing specs unchanged (investigate any new failure — it likely means a token class regressed a surface).

- [ ] **Step 3: Brand-firewall + contract checks**

Run: `cd frontend && grep -rniE "FinBERT|Alpha158|TFT|LightGBM|HMM|Qlib" app/ components/ | grep -vi "node_modules" | head`
Expected: no real model names rendered in user-facing copy (the Copilot work must not have leaked any). Confirm `lib/api.ts` is unchanged: `git diff --stat main -- frontend/lib/api.ts` → empty.

- [ ] **Step 4: Final review pass**

Invoke `vercel:react-best-practices` on the changed `.tsx` files, then `/code-review`. Apply high-confidence fixes.

- [ ] **Step 5: Milestone commit + hand back for north-star sign-off**

```bash
cd frontend && git commit --allow-empty -m "chore(redesign): Phase 1 complete — xAI design system + Copilot north-star"
```

Then present `/preview-design` and `/copilot` (screenshots in `test-results/`) for user approval before starting the roll-out waves.

---

## Notes for the implementer
- **Dark-only:** never add `dark:` variants or re-introduce `html.light`. The Settings → Appearance theme toggle is removed in a later wave (out of Phase 1 scope) — if it throws because `forcedTheme` ignores it, hide the control.
- **Duotone discipline:** color (`text-up`/`text-down`/`var(--color-up|down)`) appears ONLY on financial numbers/direction. If you reach for green/red on chrome, stop.
- **Motion:** every animation is authored per `emil-design-eng` and must pass `review-animations` before its commit.
- **Don't touch:** `lib/api.ts`, the streaming call in CopilotPanel, money-path guards, brand firewall. Re-skin only.
