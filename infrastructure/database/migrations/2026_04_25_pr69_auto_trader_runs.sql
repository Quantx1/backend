-- ============================================================================
-- PR 69 — auto_trader_runs rebalance log
-- ============================================================================
-- Each row = one daily 15:45 IST rebalance tick. The /auto-trader page
-- already shows executed trades, but a user with allow_fno=false or a
-- bear-regime VIX overlay may go days without a single trade firing —
-- they need a surface that says "engine ran, here's what it considered".
--
-- Persisted state per run:
--   ran_at             — UTC when the rebalance tick fired
--   regime             — bull|sideways|bear at run time
--   vix                — INDIA VIX value used for the equity overlay
--   vix_band           — derived band (calm..panic) per Step 1 §F4 table
--   equity_scaler_pct  — derived from vix_band (15..100)
--   actions_count      — how many rebalance instructions were emitted
--   trades_executed    — how many of those reached the broker
--   decisions          — JSONB array; per-instruction summary for audit
--   summary            — short narrative line shown on the page
--
-- Idempotent — safe to re-run.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.auto_trader_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    ran_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    regime TEXT,
    vix NUMERIC(8, 2),
    vix_band TEXT,
    equity_scaler_pct SMALLINT,
    actions_count SMALLINT NOT NULL DEFAULT 0,
    trades_executed SMALLINT NOT NULL DEFAULT 0,
    decisions JSONB NOT NULL DEFAULT '[]'::jsonb,
    summary TEXT
);

-- Per-user reverse chronological lookup powers the page's rebalance log.
CREATE INDEX IF NOT EXISTS auto_trader_runs_user_ran_idx
    ON public.auto_trader_runs (user_id, ran_at DESC);

-- Cohort analytics: "how many rebalance ticks fired yesterday across all
-- Elite users?" — supports admin dashboards without scanning per-user.
CREATE INDEX IF NOT EXISTS auto_trader_runs_ran_idx
    ON public.auto_trader_runs (ran_at DESC);

-- RLS: users see their own ticks, service role writes everything.
ALTER TABLE public.auto_trader_runs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS auto_trader_runs_self_read ON public.auto_trader_runs;
CREATE POLICY auto_trader_runs_self_read
    ON public.auto_trader_runs
    FOR SELECT
    TO authenticated
    USING (user_id = auth.uid());

INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_25_pr69_auto_trader_runs',
    'Add auto_trader_runs rebalance-log table — PR 69 dashboard rebalance history.'
) ON CONFLICT (version) DO NOTHING;
