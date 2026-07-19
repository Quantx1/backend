-- ============================================================================
-- 2026-06-04 — Enable RLS on 16 previously-unprotected tables.
--
-- Supabase advisor flagged these with Row Level Security DISABLED, i.e. fully
-- exposed to the anon + authenticated roles (anyone with the anon key could
-- read/modify every row — including user chat data in copilot_conversations /
-- copilot_messages).
--
-- The frontend reaches NONE of these directly — it uses Supabase only for auth
-- (auth.getUser/getSession); all table access goes through the FastAPI backend
-- with the service key. service_role BYPASSES RLS, so enabling RLS + a
-- service_role-only policy closes the hole without breaking any code path.
-- (Same pattern as analytics_events / llm_usage_events in the PR-V migration.)
--
-- If any of these later needs a DIRECT authenticated frontend read, add a
-- per-table owner policy (e.g. USING (auth.uid() = user_id)) at that point.
--
-- Idempotent: ENABLE RLS is a no-op if already on; policies are guarded.
-- ============================================================================

DO $$
DECLARE
    t text;
    tbls text[] := ARRAY[
        'strategy_runner_runs', 'tick_data', 'tick_collector_runs',
        'paper_option_positions', 'paper_option_legs', 'paper_option_trades',
        'copilot_conversations', 'copilot_messages', 'strategy_search_runs',
        'discovered_strategies', 'saved_scans', 'saved_scan_alerts',
        'scanner_outcomes', 'scanner_stats', 'system_cron_runs',
        'autopilot_track_record_daily'
    ];
BEGIN
    FOREACH t IN ARRAY tbls LOOP
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY;', t);
        IF NOT EXISTS (
            SELECT 1 FROM pg_policies
            WHERE schemaname = 'public'
              AND tablename  = t
              AND policyname = t || '_service_only'
        ) THEN
            EXECUTE format(
                'CREATE POLICY %I ON public.%I FOR ALL TO service_role '
                'USING (true) WITH CHECK (true);',
                t || '_service_only', t
            );
        END IF;
    END LOOP;
END $$;
