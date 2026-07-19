-- ============================================================================
-- 2026-07-07 — Paper-trading evaluation window for the style engines
-- (momentum H=20 / swing H=10).
--
-- Replaces the JSON-snapshot-only record from the 15:55 IST
-- generate_style_signals cron (backend/platform/scheduler.py) with real
-- tables:
--   * style_signals          — daily persisted top-book per engine
--   * style_signal_outcomes  — matured H-bar forward returns + the
--                              equal-weight-universe benchmark (the frozen
--                              comparator pre-registered in
--                              data/paper/baseline_expectations.json)
--
-- Written by the scheduler via the service role (bypasses RLS); read by
-- GET /api/signals/style/paper-window. Idempotent (IF NOT EXISTS).
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. style_signals — one row per (engine, trade_date, symbol) in the top book.
--    Same-day cron reruns UPSERT over the PK (idempotent).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.style_signals (
    engine            text NOT NULL CHECK (engine IN ('momentum','swing')),
    trade_date        date NOT NULL,
    symbol            text NOT NULL,
    rank              int,
    percentile        double precision,
    confidence        double precision,
    direction         text,
    entry_price       double precision,
    stop_loss         double precision,
    target            double precision,
    risk_reward       double precision,
    expected_return   double precision,
    top_decile_prob   double precision,
    status            text,
    forecast_degraded boolean DEFAULT false,
    generated_at      timestamptz DEFAULT now(),
    PRIMARY KEY (engine, trade_date, symbol)
);

CREATE INDEX IF NOT EXISTS style_signals_engine_date_idx
    ON public.style_signals (engine, trade_date DESC);

-- ----------------------------------------------------------------------------
-- 2. style_signal_outcomes — H-bar forward return per persisted signal row,
--    written by the 23:30 IST style_paper_eval cron once the panel has H
--    trading bars after trade_date. bench_fwd_return_h is duplicated per row
--    (it is a per-date scalar: equal-weight mean forward return over ALL
--    universe symbols with valid closes at t and t+H).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.style_signal_outcomes (
    engine             text NOT NULL CHECK (engine IN ('momentum','swing')),
    trade_date         date NOT NULL,
    symbol             text NOT NULL,
    rank               int,
    fwd_return_h       double precision,
    bench_fwd_return_h double precision,
    excess_h           double precision,
    horizon_days       int,
    matured_at         timestamptz DEFAULT now(),
    PRIMARY KEY (engine, trade_date, symbol)
);

CREATE INDEX IF NOT EXISTS style_signal_outcomes_engine_date_idx
    ON public.style_signal_outcomes (engine, trade_date DESC);

-- ----------------------------------------------------------------------------
-- 3. RLS — authenticated read-only trust surface (mirrors the platform tables
--    in 2026_04_19_pr2_v1_ai_stack.sql). No INSERT/UPDATE policies: the
--    scheduler writes through the service role, which bypasses RLS.
-- ----------------------------------------------------------------------------
ALTER TABLE public.style_signals         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.style_signal_outcomes ENABLE ROW LEVEL SECURITY;

-- Drop-and-recreate pattern for idempotency (CREATE POLICY has no
-- IF NOT EXISTS in Postgres 14).
DO $$
BEGIN
    DROP POLICY IF EXISTS "authenticated read style_signals" ON public.style_signals;
    CREATE POLICY "authenticated read style_signals" ON public.style_signals
        FOR SELECT TO authenticated USING (true);

    DROP POLICY IF EXISTS "authenticated read style_signal_outcomes" ON public.style_signal_outcomes;
    CREATE POLICY "authenticated read style_signal_outcomes" ON public.style_signal_outcomes
        FOR SELECT TO authenticated USING (true);
END$$;

-- ----------------------------------------------------------------------------
-- 4. RECORD MIGRATION
-- ----------------------------------------------------------------------------
INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_07_07_pr_style_signals_paper_window',
    'Paper window: style_signals + style_signal_outcomes (momentum/swing top-book persistence + matured H-bar outcomes vs equal-weight universe), authenticated read-only RLS'
)
ON CONFLICT DO NOTHING;
