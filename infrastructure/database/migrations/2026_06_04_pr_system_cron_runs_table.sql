-- ============================================================================
-- 2026-06-04 — system_cron_runs: cron idempotency ledger (migration was missing).
--
-- This table exists in production but had NO migration, so a clean DB rebuild
-- (fresh install from complete_schema.sql) would silently omit it — and with it
-- the load-bearing UNIQUE(job_id, run_date) constraint that prevents a daily
-- cron (e.g. the AutoPilot rebalance that places real trades) from double-firing
-- if two scheduler instances run or a job is retried within the same day.
--
-- Mirrors the live prod schema exactly. Idempotent.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.system_cron_runs (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id       TEXT        NOT NULL,
    run_date     DATE        NOT NULL,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status       TEXT        NOT NULL DEFAULT 'running'
                 CHECK (status IN ('running', 'completed', 'failed', 'skipped')),
    pid          TEXT,
    duration_ms  INTEGER,
    error        TEXT,
    metadata     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT system_cron_runs_job_id_run_date_key UNIQUE (job_id, run_date)
);

CREATE INDEX IF NOT EXISTS system_cron_runs_job_date_idx
    ON public.system_cron_runs (job_id, run_date DESC);

-- Backend-only (service_role). Consistent with the other ops/telemetry tables.
ALTER TABLE public.system_cron_runs ENABLE ROW LEVEL SECURITY;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname='public' AND tablename='system_cron_runs'
          AND policyname='system_cron_runs_service_only'
    ) THEN
        CREATE POLICY system_cron_runs_service_only ON public.system_cron_runs
            FOR ALL TO service_role USING (true) WITH CHECK (true);
    END IF;
END $$;
