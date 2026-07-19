I have what I need. The `next-upgrade`/`bootstrap` skill injections are auto-suggested from a `package.json` Read pattern — they don't apply to this task (synthesizing a research doc, not upgrading Next.js or deploying), so I'll skip them. The repo confirms the key facts: React 18.2.0, Next 14.1.0, Tailwind 3.4.1, framer-motion 10.18.0, lightweight-charts 4.2.0 (not v5), recharts 2.10.4, individual Radix primitives, cva, no `components.json` yet.

Here is the synthesized plan.

---

# Quant X — "AI Trading OS" Component Plan

Synthesized from 10 component-category scouts. Stack-verified against `frontend/package.json` on 2026-06-01: React 18.2.0, Next 14.1.0 (App Router), TypeScript 5.3.3, Tailwind 3.4.1 (config-based `tailwind.config.ts`), framer-motion 10.18.0, lightweight-charts **4.2.0**, recharts 2.10.4, individual `@radix-ui/*` primitives, cva 0.7, next-themes, lucide-react. **No `components.json` exists yet** — shadcn must be `init`'d.

---

## 1. Curated stack (layered)

| Layer | Need | THE PICK | Why one | Rejected (brief) |
|---|---|---|---|---|
| **Base** | Design system / primitives | **shadcn/ui (New York style, Tailwind v3 CSS-vars mode)** + existing Radix primitives + cva | Zero new runtime dep; slots onto cva/tailwind-merge/lucide/Radix you already have; copy-in so glass/radius are yours to gut. License MIT. | *Radix Themes* — rejected: competes with Tailwind, +40KB monolith duplicating your installed primitives, 12-step color lock-in that won't take #00E6A7. |
| **Base/theme bootstrap** | Generate the token block | **tweakcn** (one-time web tool, Apache-2.0) | Fastest way to emit the exact 18-token shadcn CSS-var block for the dark palette with live WCAG checks. Not a runtime dep. | *ui.jln.dev* — fine as dark-baseline reference only; no live sliders/contrast for dialing #00E6A7. |
| **Data — tables** | Dense sortable grids | **TanStack Table v8 + shadcn data-table pattern + @tanstack/react-virtual** | Headless 15KB, Tailwind-native cells (BUY/HOLD/SELL pills, conviction bars, engine chips), RSC page / `'use client'` table split, copy-in. MIT. | *AG Grid* — 298KB+ black-box, non-Tailwind chrome. *Material React Table* — MUI/Emotion fights Tailwind (their own docs say so). Both rejected. |
| **Data — table cell sparkline** | Inline trend in rows | **shadcn `tables-sparkline` block** (Recharts AreaChart cell) | Zero extra dep beyond Recharts (already installed); drops in as a TanStack cell renderer. MIT. | — (complement, not a competitor). |
| **Data — charts (price/equity)** | Candles, equity curve, real-time, drawdown | **lightweight-charts** (already at v4.2.0; bump to v5 optional) | Only canvas-native financial lib; 60fps on big OHLCV; programmable colors → palette. Apache-2.0. | SVG libs stutter on dense OHLCV. |
| **Data — charts (everything else)** | Donut, OI bars, payoff, gauges, sparklines | **Recharts v3 via shadcn chart registry** (already at v2.10.4 — upgrade to v3) | Backs shadcn `<ChartContainer>` with `--chart-*` token theming + dark mode. MIT. | *visx* — maintenance risk (Airbnb slowed, React 19 stuck in alpha); reserve for one-off bespoke paths only. |
| **Data — F&O OI heatmap** | Dense strike×expiry grid | **@nivo/heatmap (HeatMapCanvas)** | Canvas variant avoids SVG DOM bloat on 20×50 grids; themeable bg/font. MIT. | Recharts has no real heatmap; targeted add-on only. |
| **AI-chat** | Persistent Cursor-style copilot rail | **assistant-ui** (`AssistantSidebar` + Thread + tool-call/reasoning UI) | ONLY lib shipping a resizable persistent right-rail panel + generative tool-call UI + chain-of-thought + Vercel AI SDK/LangGraph runtime + Perplexity-clone example. MIT, React 18.2 + TW v3 confirmed. | *Vercel AI Elements* — **HARD REJECT**: targets React 19 + Tailwind v4. *prompt-kit* — keep as optional composer skin; no tool-call/step UI. *shadcn.io/ai* — loose, no sidebar/runtime. |
| **Motion** | Functional micro-interactions | **motion-primitives (ibelick)**, copy-in, on `motion/react` | Purpose-tuned recipes: AnimatedNumber, SlidingNumber (price odometer), TextEffect (streaming reveal), TextShimmer ("Analyzing…"), GlowEffect, InView. MIT, TW v3. | Foundational engine: install **`motion`** (Framer's successor) once — your repo has legacy `framer-motion@10`; migrate imports to `motion/react`. |
| **Premium / glass** | "AI is active" energy + glass surfaces | **Magic UI v3 (v3.magicui.design) Border Beam + Shimmer Button** + **handcrafted CSS glass utility** | Border Beam on Copilot rail/Studio edge = ambient AI signal; shimmer on emerald CTA. Glass recipe (one `backdrop-blur` per surface, never per row) for overlays only. MIT. | *glasscn-ui* — runtime dep, TW version unpinned. *Aceternity Aurora/Spotlight* — keep ONLY for marketing hero, never data planes. **Avoid Aceternity 3D/Three.js (~600KB) on inner pages.** |
| **Marketing** | Light-theme landing/pricing | **Magic UI Pro (v3 branch, $199 lifetime)** + free Magic UI v3 sections | Dark-first particle/gradient hero sections match FinStocks-grade premium; dedicated v3 branch removes the v4 blocker. | *Shadcnblocks* & *Tailark* — **rejected for now**: both fully migrated to Tailwind v4, no v3 branch → per-block backport tax. |

**Critical version note:** Use **`v3.magicui.design`** URLs for ALL Magic UI installs — the main site is now Tailwind v4 and will inject `@theme` directives that break your config.

---

## 2. Per-surface component map

| Surface | Components / library | Source |
|---|---|---|
| **AI Copilot cockpit (persistent right rail)** | assistant-ui `AssistantSidebar` (resizable 2-pane) + `Thread` + `Composer` + `ActionBar`; tool-call via `makeAssistantTool`/`ToolFallback` + `useToolArgsStatus`; chain-of-thought `ReasoningRoot` + `ToolGroupRoot`; streaming reveal = motion-primitives `TextEffect`; "Analyzing…" = `TextShimmer`; rail edge = Magic UI `BorderBeam`; references/citations from the bundled Perplexity-clone example. Runtime: `useChatRuntime` (Vercel AI SDK). | assistant-ui.com · v3.magicui.design/docs/components/border-beam |
| **Command Center bento dashboard** | Magic UI v3 `BentoGrid` + `BentoCard` (asymmetric `col-span`/`row-span` shell); each cell hosts a **Tremor KPI card** (label + metric + `SparkAreaChart` + `BadgeDelta`); reveal via motion-primitives `InView`; numbers via `AnimatedNumber`/Magic UI `NumberTicker`. | v3.magicui.design/docs/components/bento-grid · npm.tremor.so |
| **Signals feed (dense table)** | TanStack Table v8 + shadcn `data-table` + `@tanstack/react-virtual`; cell renderers: BUY/HOLD/SELL = shadcn `Badge` (cva variants), confidence = custom Tailwind bar, engine chips = `Badge`, sparkline = shadcn `tables-sparkline`; horizon tabs = Radix `Tabs`. | tanstack.com/table · shadcn.io/blocks/tables-sparkline |
| **Stock dossier** | Price chart = **lightweight-charts** (`'use client'` ref wrapper); KPI tiles = Tremor `Card`+`BadgeDelta`; tabs = Radix `Tabs`; AI verdict card = shadcn `Card` + Magic UI `BorderBeam` + motion-primitives `TextEffect` for the verdict reveal. | tradingview/lightweight-charts · npm.tremor.so |
| **Strategy Studio (Bolt-split + AI-Trade-Gate)** | Two-pane = shadcn `Resizable` (Radix); chat pane = assistant-ui `Thread`; DSL/code editor = shadcn `Textarea`/CodeBlock (or shadcn.io code-block block); live backtest equity curve = **lightweight-charts** Baseline series (benchmark-vs); AI-Trade-Gate metric banner = Tremor `Card` + `BadgeDelta` (Sharpe gate pass/fail), active-editor edge = Magic UI `BorderBeam`. | ui.shadcn.com · tradingview/lightweight-charts |
| **Scanner (NL search + results)** | NL input = shadcn `Input` (emerald focus ring) optionally with `Command` streaming suggestions (`shouldFilter={false}`); results = TanStack Table; row hover = motion-primitives `Spotlight`. | ui.shadcn.com/docs/components/command |
| **Portfolio + Doctor (health)** | Holdings = TanStack Table; allocation donut = Recharts `PieChart` via shadcn `<ChartContainer>`; P&L tiles = Tremor cards (`#00FF9D`/`#FF5B7F`); Doctor health = Tremor `ProgressCircle`/`ProgressBar` + shadcn `Alert`. | recharts.org + ui.shadcn.com/charts · npm.tremor.so |
| **AutoPilot (status hero + planned trades + drawdown + kill switch)** | Status hero = shadcn `Card`; planned-trades = TanStack Table; drawdown gauge = Recharts `ComposedChart`/`RadialBar` (shadcn chart); count = `NumberTicker`; **kill switch = Radix `Switch` + shadcn `AlertDialog` confirm** (destructive). | recharts.org · ui.shadcn.com |
| **F&O (OI heatmap + payoff + greeks + chain)** | OI heatmap = **@nivo/heatmap `HeatMapCanvas`** (palette-overridden scale); payoff diagram = Recharts `ComposedChart` (Area + ReferenceLine at breakeven); greeks = Tremor KPI cards (gold `#FFB547`); option chain = TanStack Table (virtualized), row drill-in = motion-primitives `MorphingDialog`. | nivo.rocks · recharts.org |
| **Watchlist (live tape + table)** | TanStack Table + virtual; live price = motion-primitives `SlidingNumber` (digit odometer); sparkline cell = shadcn `tables-sparkline`; card hover = Magic UI `MagicCard`. | tanstack.com/table · motion-primitives.com |
| **Inbox** | TanStack Table or shadcn list; unread = `Badge`; detail = shadcn `Sheet`/`Dialog`; toasts already covered by installed `sonner`. | ui.shadcn.com |
| **Settings (tabbed)** | shadcn `Tabs` + `Card` + `Switch` + `Select` + `Input` (all Radix-backed, already installed); glass on overlay panels only via glass utility. | ui.shadcn.com |
| **Command palette (Cmd-K)** | shadcn `Command` (cmdk wrapper) in `CommandDialog`; global keydown in a `'use client'` provider in `layout.tsx`; groups Markets/Signals/Strategies/Agents/Portfolio/Settings; `CommandShortcut` inline. Left nav = shadcn `Sidebar` `collapsible="icon"` (Cmd-B, override `SIDEBAR_KEYBOARD_SHORTCUT`). | ui.shadcn.com/docs/components/command + /sidebar |
| **Marketing landing / pricing (LIGHT theme)** | Magic UI Pro v3 hero/feature/pricing/FAQ/footer sections; `ShimmerButton`/`AnimatedShinyText` CTAs; Aceternity `Spotlight`/`Aurora` for ONE hero moment only; `SparklesText` for premium tier. | pro.magicui.design (v3) · ui.aceternity.com |

---

## 3. AI Trading OS theme tokens

Dark-first. shadcn semantic tokens use **bare HSL channels** (its native convention) so `hsl(var(--x) / <alpha>)` opacity modifiers work. Brand-extra tokens (gold/bull/bear/elevated) live in a parallel `--qx-*` rgb namespace so they never collide with shadcn's 18 tokens.

### `app/globals.css` — paste into `@layer base`

```css
@layer base {
  :root,
  .dark {
    /* shadcn 18-token set — HSL channels, dark-first ("AI Trading OS") */
    --background: 218 43% 6%;      /* #080B12 bg */
    --foreground: 210 20% 96%;     /* near-white text */
    --card: 222 36% 11%;           /* #121826 card */
    --card-foreground: 210 20% 96%;
    --popover: 218 30% 15%;        /* #182132 elevated */
    --popover-foreground: 210 20% 96%;
    --primary: 162 100% 45%;       /* #00E6A7 emerald-teal — SIGNATURE */
    --primary-foreground: 218 43% 6%;
    --secondary: 222 30% 16%;
    --secondary-foreground: 210 20% 96%;
    --muted: 222 24% 18%;
    --muted-foreground: 215 16% 62%;
    --accent: 258 90% 66%;         /* #8B5CF6 AI purple — AI surfaces only */
    --accent-foreground: 210 20% 96%;
    --destructive: 344 100% 68%;   /* #FF5B7F bearish */
    --destructive-foreground: 210 20% 96%;
    --border: 218 25% 18%;         /* subtle structural border */
    --input: 218 25% 18%;
    --ring: 162 100% 45%;          /* emerald focus ring */
    --radius: 0.5rem;              /* 8px */

    /* shadcn chart tokens → brand series */
    --chart-1: 162 100% 45%;       /* teal */
    --chart-2: 217 91% 60%;        /* electric blue #3B82F6 */
    --chart-3: 258 90% 66%;        /* AI purple */
    --chart-4: 36 100% 64%;        /* gold #FFB547 */
    --chart-5: 152 100% 50%;       /* bull green */

    /* shadcn sidebar tokens */
    --sidebar-background: 222 36% 11%;
    --sidebar-foreground: 210 20% 96%;
    --sidebar-primary: 162 100% 45%;
    --sidebar-primary-foreground: 218 43% 6%;
    --sidebar-accent: 218 30% 15%;
    --sidebar-accent-foreground: 210 20% 96%;
    --sidebar-border: 218 25% 18%;
    --sidebar-ring: 162 100% 45%;

    /* ── Quant X brand extras (rgb channels, parallel namespace) ── */
    --qx-bg: 8 11 18;             /* #080B12 */
    --qx-card: 18 24 38;          /* #121826 */
    --qx-elevated: 24 33 50;      /* #182132 */
    --qx-teal: 0 230 167;         /* #00E6A7 */
    --qx-blue: 59 130 246;        /* #3B82F6 */
    --qx-purple: 139 92 246;      /* #8B5CF6 */
    --qx-bull: 0 255 157;         /* #00FF9D — P&L only */
    --qx-bear: 255 91 127;        /* #FF5B7F — P&L only */
    --qx-gold: 255 181 71;        /* #FFB547 — F&O/premium */

    /* purposeful gradients */
    --gradient-ai: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%);
    --gradient-trading: linear-gradient(135deg, #00E6A7 0%, #00B8FF 100%);
    --gradient-premium: linear-gradient(135deg, #FFB547 0%, #FF7A00 100%);
  }
}

@layer utilities {
  /* Glassmorphism 2.0 — overlays/floating/composer ONLY, never dense data planes.
     Requires a colored glow orb behind it or the blur refracts flat black. */
  .glass-panel {
    @apply backdrop-blur-[14px] saturate-150 transform-gpu;
    background-color: rgb(var(--qx-card) / 0.6);
    border: 1px solid rgb(255 255 255 / 0.08);
  }
  .glass-overlay {
    @apply backdrop-blur-[20px] saturate-180 transform-gpu;
    background-color: rgb(var(--qx-elevated) / 0.5);
    border: 1px solid rgb(255 255 255 / 0.12);
  }
  .tnum { font-variant-numeric: tabular-nums; } /* apply on ALL numbers */
}
```

### `tailwind.config.ts` — `theme.extend`

```ts
// theme.extend.colors — merge into your existing extend block
colors: {
  // shadcn semantic (consumes the HSL vars above, opacity-modifier safe)
  border: "hsl(var(--border))",
  input: "hsl(var(--input))",
  ring: "hsl(var(--ring))",
  background: "hsl(var(--background))",
  foreground: "hsl(var(--foreground))",
  primary: { DEFAULT: "hsl(var(--primary))", foreground: "hsl(var(--primary-foreground))" },
  secondary: { DEFAULT: "hsl(var(--secondary))", foreground: "hsl(var(--secondary-foreground))" },
  destructive: { DEFAULT: "hsl(var(--destructive))", foreground: "hsl(var(--destructive-foreground))" },
  muted: { DEFAULT: "hsl(var(--muted))", foreground: "hsl(var(--muted-foreground))" },
  accent: { DEFAULT: "hsl(var(--accent))", foreground: "hsl(var(--accent-foreground))" },
  popover: { DEFAULT: "hsl(var(--popover))", foreground: "hsl(var(--popover-foreground))" },
  card: { DEFAULT: "hsl(var(--card))", foreground: "hsl(var(--card-foreground))" },
  chart: {
    1: "hsl(var(--chart-1))", 2: "hsl(var(--chart-2))", 3: "hsl(var(--chart-3))",
    4: "hsl(var(--chart-4))", 5: "hsl(var(--chart-5))",
  },
  sidebar: {
    DEFAULT: "hsl(var(--sidebar-background))",
    foreground: "hsl(var(--sidebar-foreground))",
    primary: "hsl(var(--sidebar-primary))",
    "primary-foreground": "hsl(var(--sidebar-primary-foreground))",
    accent: "hsl(var(--sidebar-accent))",
    "accent-foreground": "hsl(var(--sidebar-accent-foreground))",
    border: "hsl(var(--sidebar-border))",
    ring: "hsl(var(--sidebar-ring))",
  },
  // Quant X brand extras (rgb-channel, opacity-modifier safe)
  qx: {
    bg: "rgb(var(--qx-bg) / <alpha-value>)",
    card: "rgb(var(--qx-card) / <alpha-value>)",
    elevated: "rgb(var(--qx-elevated) / <alpha-value>)",
    teal: "rgb(var(--qx-teal) / <alpha-value>)",
    blue: "rgb(var(--qx-blue) / <alpha-value>)",
    purple: "rgb(var(--qx-purple) / <alpha-value>)",
    bull: "rgb(var(--qx-bull) / <alpha-value>)",
    bear: "rgb(var(--qx-bear) / <alpha-value>)",
    gold: "rgb(var(--qx-gold) / <alpha-value>)",
  },
},
backgroundImage: {
  "gradient-ai": "var(--gradient-ai)",
  "gradient-trading": "var(--gradient-trading)",
  "gradient-premium": "var(--gradient-premium)",
},
borderRadius: { lg: "var(--radius)", md: "calc(var(--radius) - 2px)", sm: "calc(var(--radius) - 4px)" },
```

> Before accepting `shadcn init`'s `globals.css` write, diff it against your existing `--color-*` block so it does not overwrite anything. Generate/verify these exact triplets in **tweakcn** (tweakcn.com/editor/theme) for WCAG contrast on #080B12, then hand-add the `--qx-*` extras it won't emit.

---

## 4. Dependencies to install

```bash
# ── runtime npm deps ──
cd frontend

# Charts + tables (recharts already present at v2 → upgrade to v3; lightweight-charts at v4)
npm i recharts@^3                       # upgrade from 2.10.4 for shadcn chart registry
npm i @tanstack/react-table @tanstack/react-virtual
npm i @nivo/core @nivo/heatmap          # F&O OI heatmap only
npm i @tremor/react                     # KPI/delta/sparkline cards (Tailwind v3-pinned npm pkg)

# AI copilot (assistant-ui) + Vercel AI SDK runtime
npm i @assistant-ui/react @assistant-ui/react-ai-sdk @assistant-ui/styles
npm i ai @ai-sdk/react                  # Vercel AI SDK runtime (useChatRuntime)

# Motion: install successor `motion`; migrate framer-motion@10 imports → motion/react
npm i motion

# (optional) upgrade lightweight-charts 4.2.0 → 5 for Baseline benchmark series
# npm i lightweight-charts@^5
```

> Add Tremor to `tailwind.config.ts` `content`: `"./node_modules/@tremor/**/*.{js,ts,jsx,tsx}"`.

```bash
# ── shadcn init + registry copy-in (no runtime deps; code lands in repo) ──
npx shadcn@latest init        # New York style · CSS variables: yes · darkMode: class · baseColor: slate

npx shadcn@latest add button card badge table tabs dialog dropdown-menu \
  tooltip select switch progress input textarea separator sheet alert \
  alert-dialog command sidebar resizable skeleton sonner

# data-table sparkline cell + chart container
npx shadcn@latest add tables-sparkline
npx shadcn@latest add chart            # shadcn <ChartContainer> for Recharts

# AI components (assistant-ui registry — Thread/Composer/etc.)
npx shadcn@latest add "https://r.assistant-ui.com/thread.json"

# motion-primitives (copy-in; needs `motion` peer already installed)
npx shadcn@latest add "https://motion-primitives.com/c/animated-number.json"
npx shadcn@latest add "https://motion-primitives.com/c/sliding-number.json"
npx shadcn@latest add "https://motion-primitives.com/c/text-effect.json"
npx shadcn@latest add "https://motion-primitives.com/c/text-shimmer.json"
npx shadcn@latest add "https://motion-primitives.com/c/glow-effect.json"
npx shadcn@latest add "https://motion-primitives.com/c/in-view.json"
npx shadcn@latest add "https://motion-primitives.com/c/morphing-dialog.json"

# Magic UI v3 (TAILWIND v3 BRANCH — pin v3.magicui.design or you get v4 @theme)
npx shadcn@latest add "https://v3.magicui.design/r/bento-grid.json"
npx shadcn@latest add "https://v3.magicui.design/r/border-beam.json"
npx shadcn@latest add "https://v3.magicui.design/r/shimmer-button.json"
npx shadcn@latest add "https://v3.magicui.design/r/number-ticker.json"
npx shadcn@latest add "https://v3.magicui.design/r/magic-card.json"
```

**Theme bootstrap (no install):** open **tweakcn.com/editor/theme**, dial the palette, export → paste into `globals.css`. Optional Pro: **pro.magicui.design** ($199 lifetime, v3 branch) for assembled marketing sections.

---

## 5. 21st.dev Magic MCP

**What it is:** a "v0-in-your-IDE" MCP server. Type `/ui <description>` in Claude Code → it generates 5 dark-mode-first, TypeScript-typed, Tailwind+Motion component variants in a browser preview, then writes the chosen one to a real file. Backed by a 1,400+ founder-reviewed component catalog skewing to dark AI/SaaS UI, plus an **Agent Elements** sub-library (chat shell, streaming markdown, tool-call cards: Bash/Edit/Search/Todo/Plan/Approval) that maps onto the Copilot rail. MIT.

**Connect command (Claude Code):**
```bash
# 1. Get key at https://21st.dev/magic/console (GitHub/Google sign-in)
# 2. One-liner install:
npx @21st-dev/cli@latest install claude --api-key <YOUR_KEY>
# …or add to project .mcp.json:
#   { "mcpServers": { "@21st-dev/magic": {
#       "command": "npx",
#       "args": ["-y", "@21st-dev/magic@latest", "API_KEY=\"<YOUR_KEY>\""] } } }
# 3. Restart Claude Code → /mcp to verify → trigger with: /ui <description>
```

**Free-tier limits:** ~100 credits/month (each `/ui` generation consumes credits, ≈5 generations); Pro $20/mo = 400 credits. Free tier is enough to prototype individual surfaces, **not** a full cockpit sprint.

**WHEN to use it vs copy-in libs:**
- **Use Magic MCP** to break blank-canvas cold starts on *bespoke* surfaces with no off-the-shelf match — F&O payoff layout, AI-Trade-Gate banner, Doctor health panel, dossier verdict card. Treat output as a **draft**.
- **Prefer copy-in libs** (shadcn / assistant-ui / Tremor / Magic UI v3) for anything they already cover — copilot rail, bento, tables, Cmd-K. They're versioned, reviewed, and Tailwind-v3-safe.
- **MANDATORY post-step:** Magic + Agent Elements author against **Tailwind v4** (`@import "tailwindcss"`, `@variant dark`, v4-only utilities). **Audit and convert every generated file to v3** before committing, wire any custom tokens into `tailwind.config.ts`, and add `"use client"` (all generated components are client + Motion).

**shadcn registry MCP worth adding (free, official):**
```bash
pnpm dlx shadcn@latest mcp init --client claude
# or .mcp.json: { "mcpServers": { "shadcn": { "command": "npx", "args": ["shadcn@latest","mcp"] } } }
```
Gives Claude Code live search/preview/install of shadcn blocks (no hallucinated props), pinned to Tailwind v3. Zero cost, no account. **Use the official `ui.shadcn.com/docs/mcp` version — NOT the paid `shadcn.io/mcp` ($19/mo).**

---

## 6. Risks & compatibility notes

**Tailwind v3-vs-v4 (the dominant risk):**
- **Magic UI:** main `magicui.design` is now v4 → **always use `v3.magicui.design`** registry URLs. Wrong branch injects `@theme` and breaks `tailwind.config.ts`.
- **Tremor:** the copy-paste site `tremor.so` migrated to v4; **only the npm package `@tremor/react` (≥3.18) stays v3** → use `npm.tremor.so` docs, not the site.
- **HARD-BLOCK — do NOT adopt:** Vercel AI Elements (React 19 + TW v4), Shadcnblocks (v4-only, no v3 branch), Tailark (v4-only). These will produce CSS-var/React-API mismatches on our React 18.2 + TW v3 stack.
- **21st.dev Magic MCP / Agent Elements:** author against v4 → manual v3 backport on every generated file (see §5).

**RSC / `'use client'` boundaries:**
- RSC-safe (pure server): shadcn `Card`, `Badge`, `Table` (markup), `Separator`; bento *outer shell*.
- Must be `'use client'`: ALL of assistant-ui, TanStack Table data-table file, **all Tremor components**, all Recharts, **all** lightweight-charts (canvas), Nivo, motion/motion-primitives/Magic UI animations, shadcn `Command`/`Sidebar`/`Dialog`/`Tabs`.
- Pattern: keep `page.tsx` a Server Component that fetches data; isolate the interactive grid/chart/copilot into `'use client'` leaf components wrapped in `<Suspense>`. Drop `CommandProvider` + `SidebarProvider` (+ assistant-ui runtime provider) at the `layout.tsx` boundary; pages below stay RSC.

**License flags:** MIT — shadcn, assistant-ui, motion-primitives, Magic UI (free), TanStack, Nivo, Tremor, cmdk, tweakcn (Apache-2.0). **Apache-2.0 attribution** — lightweight-charts (retain TradingView copyright; v5 `attributionLogo` can satisfy on-chart, or credit on /about — decide whether the TV badge is acceptable in a fintech UI). **Paid/lifetime** — Magic UI Pro $199; 21st.dev Pro $20/mo; Tailwind UI Plus $299 (skip — shadcn `Command` replaces it). Aceternity free tier is MIT — **stay on free, avoid Pro redistribution terms.**

**Bundle-weight cautions:**
- AG Grid (298KB), Material React Table+MUI, **Aceternity 3D/Three.js (~600KB)** — **never on Signals/Scanner/AutoPilot/inner data pages.**
- `motion` full declarative = ~34KB → use `LazyMotion` + `m` components or `useAnimate` mini (~2.3KB) on dense pages (Signals table).
- Recharts naive import ~136KB gzip → named imports only (`import { AreaChart, Area } from "recharts"`) for tree-shaking.
- Nivo is additive per package → install `@nivo/core` + `@nivo/heatmap` only.
- assistant-ui markdown: use the newer `@assistant-ui/react-markdown` path is **deprecated** — follow current UI-package migration notes.

**What to AVOID (brand-firewall + SEBI + MEMORY constraints):**
- **Glass discipline:** glassmorphism ONLY on overlays/floating/composer. **Never `backdrop-filter` per table row** (one per surface, GPU-accelerated). Every glass surface needs a colored glow orb behind it or it renders flat black on #080B12. Avoid `feTurbulence`/`feSpecularLighting` backdrop SVG filters (broken in Safari) — use `backdrop-blur` + low-opacity noise PNG/SVG.
- **No neon overload, no square corners, no countdowns** (per brief). Reserve high-drama (Aceternity Spotlight/Aurora, SparklesText) for **marketing only** and ≤3 moments per surface.
- **Color semantics:** `#00FF9D`/`#FF5B7F` are **P&L-only**; `#8B5CF6` purple is **AI-surfaces-only**; `#FFB547` gold is **F&O/premium-only**. Don't bleed these into generic UI chrome.
- **Brand firewall (MEMORY):** UI strings must use public engine names (SwingLens/AlphaRank/RegimeIQ/AutoPilot) — **never** real model names (TFT, Qlib, FinBERT). The engine chips in the Signals table must render branded names only; a brand-firewall test pins this.
- **SEBI-safe:** no countdown timers, no urgency/guaranteed-return framing in CTAs or marketing blocks — keep ShimmerButton/AnimatedShinyText copy factual.
- **assistant-ui is for explanation/narration/tool-routing UI only** — per the (now-reversed) gating memory, trade execution still flows through the backtest+Sharpe gate; the Copilot rail surfaces it, it does not bypass it.