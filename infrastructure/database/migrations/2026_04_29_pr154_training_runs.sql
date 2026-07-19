-- ============================================================================
-- PR 154 — training_runs table
-- ============================================================================
-- Persists every unified-runner invocation (PR 128/129) so /admin/training
-- shows history across server restarts. The in-memory dict was fine for
-- single-process dev but loses everything on deploy.
--
-- Idempotent — safe to re-run.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',  -- running | ok | partial | failed
    triggered_by TEXT,                       -- admin email
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    reports JSONB NOT NULL DEFAULT '[]'::jsonb,
    error TEXT
);

CREATE INDEX IF NOT EXISTS training_runs_started_idx
    ON public.training_runs (started_at DESC);

INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_29_pr154_training_runs',
    'Add training_runs table for persistent admin training-run history.'
) ON CONFLICT (version) DO NOTHING;
