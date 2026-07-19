-- ============================================================================
-- 2026-06-04 — Add service_role policies to RLS-enabled / no-policy tables.
--
-- These tables had RLS ENABLED but ZERO policies (Supabase advisor: "RLS
-- Enabled No Policy"). That state is already SAFE — with no policy, anon/
-- authenticated are fully blocked, and the backend (service_role) bypasses RLS.
-- But the advisor flags it as possibly-unintentional. Adding an explicit
-- service_role-only policy documents intent and clears the note WITHOUT
-- changing access (service_role bypasses RLS regardless; anon/authenticated
-- stay blocked). The frontend never reads these directly (it only reads
-- user_profiles), and the backend reads them via the service key.
--
-- Idempotent.
-- ============================================================================

DO $$
DECLARE
    t text;
    tbls text[] := ARRAY[
        'admin_audit_log', 'alpha_scores', 'audit_log', 'broker_connections',
        'candles', 'daily_universe', 'earnings_predictions', 'eod_scan_runs',
        'features', 'forecast_scores', 'market_data', 'model_outcomes',
        'model_performance', 'model_versions', 'news_sentiment',
        'scheduler_job_runs', 'schema_migrations', 'signal_debates', 'stocks',
        'subscription_plans', 'system_flags', 'training_runs', 'vix_forecasts'
    ];
BEGIN
    FOREACH t IN ARRAY tbls LOOP
        -- skip any table that doesn't exist (defensive)
        IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename=t) THEN
            EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY;', t);
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies
                WHERE schemaname='public' AND tablename=t
                  AND policyname = t || '_service_only'
            ) THEN
                EXECUTE format(
                    'CREATE POLICY %I ON public.%I FOR ALL TO service_role '
                    'USING (true) WITH CHECK (true);',
                    t || '_service_only', t
                );
            END IF;
        END IF;
    END LOOP;
END $$;
