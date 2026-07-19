-- ============================================================================
-- 2026-06-04 — Lock 26 legacy "exposed" tables to service-role-only.
--
-- Audit (frontend supabase-js + backend anon client) found the app reads only
-- 4 tables via the anon/authenticated role: user_profiles, strategy_catalog,
-- strategy_backtests, user_strategy_deployments. The other 26 carried anon/
-- authenticated/public RLS policies from an earlier design where the frontend
-- read tables directly — but the app now reads them via the FastAPI service key.
--
-- Worse, several had `USING (true) TO public` policies — i.e. WORLD read (and
-- some world write/delete) of USER data once populated:
--   portfolio_doctor_reports, user_autopilot_streams, research_reports,
--   user_ai_overlay_settings (user data); model_rolling_performance,
--   regime_history (global, non-user).
--
-- This converts all 26 to service-role-only: drop the permissive policies,
-- ensure RLS + a service_role policy, revoke anon/authenticated grants. Safe —
-- service_role bypasses RLS, and none of the 26 are read via anon/authenticated.
-- The 4 genuinely-used tables are intentionally NOT in this list.
--
-- Idempotent.
-- ============================================================================

DO $$
DECLARE
    t   text;
    pol record;
    tbls text[] := ARRAY[
        'ai_portfolio_holdings','auto_trader_runs','model_rolling_performance','notifications',
        'paper_portfolios','paper_positions','paper_snapshots','paper_trades','payments',
        'portfolio_doctor_reports','portfolio_history','positions','regime_history',
        'research_reports','signals','strategy_executions','strategy_outcomes','strategy_positions',
        'trades','user_ai_overlay_settings','user_autopilot_streams','user_digest_deliveries',
        'user_referrals','user_strategies','user_weekly_reviews','watchlist'
    ];
BEGIN
    FOREACH t IN ARRAY tbls LOOP
        IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename=t) THEN
            -- drop every policy that grants anon/authenticated/public
            FOR pol IN
                SELECT policyname FROM pg_policies
                WHERE schemaname='public' AND tablename=t
                  AND roles && ARRAY['anon','authenticated','public']::name[]
            LOOP
                EXECUTE format('DROP POLICY %I ON public.%I;', pol.policyname, t);
            END LOOP;

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

            EXECUTE format('REVOKE ALL ON public.%I FROM anon, authenticated;', t);
        END IF;
    END LOOP;
END $$;
