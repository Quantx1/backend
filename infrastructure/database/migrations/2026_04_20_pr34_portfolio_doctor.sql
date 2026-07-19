-- ============================================================================
-- PR 34 — Portfolio Doctor (F7) storage + quota
-- ============================================================================
-- One row per Doctor run. Reports are durable so users can revisit
-- prior assessments. ``position_count`` + ``composite_score`` keep
-- list-views cheap without reading the full narrative blob.
--
-- Tier gates:
--   Free   — one-off ₹199 product (handled at checkout; row still stored)
--   Pro    — monthly rerun included; quota enforced by ``RateLimiter``
--   Elite  — unlimited reruns
--
-- Idempotent — safe to re-run.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.portfolio_doctor_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Per-run inputs snapshot.
    source TEXT NOT NULL CHECK (source IN ('manual', 'broker', 'csv')),
    position_count INTEGER NOT NULL DEFAULT 0,
    capital DECIMAL(15, 2),

    -- 4-agent composite output (InsightAI).
    composite_score INTEGER CHECK (composite_score BETWEEN 0 AND 100),
    action TEXT CHECK (action IN ('rebalance', 'hold', 'reduce_risk', 'increase_risk')),
    narrative TEXT,
    per_position JSONB DEFAULT '[]'::jsonb,   -- [{symbol, weight, score, action, narrative}]
    risk_flags JSONB DEFAULT '[]'::jsonb,     -- concentration / sector / drawdown / stale-stop
    agents JSONB DEFAULT '{}'::jsonb,         -- raw agent output keyed by role

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS portfolio_doctor_reports_user_idx
    ON public.portfolio_doctor_reports (user_id, created_at DESC);

-- Row-Level Security: a user can only read their own reports.
ALTER TABLE public.portfolio_doctor_reports ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
    DROP POLICY IF EXISTS "users own doctor reports" ON public.portfolio_doctor_reports;
    CREATE POLICY "users own doctor reports" ON public.portfolio_doctor_reports
        FOR ALL USING (auth.uid() = user_id);
EXCEPTION WHEN OTHERS THEN NULL; END $$;

-- Log this migration.
INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_20_pr34_portfolio_doctor',
    'Add portfolio_doctor_reports — PR 34 Portfolio Doctor (F7) InsightAI run storage.'
) ON CONFLICT (version) DO NOTHING;
