-- ============================================================================
-- PR sec-1 — enable RLS on 24 previously-open tables (2026-05-12)
--
-- Supabase advisor flagged 25 tables with RLS DISABLED including
-- model_versions, signals (already enabled separately), payments
-- (already enabled), broker_connections, model_performance, alpha_scores,
-- forecast_scores, etc. With anon key exposure (frontend), this meant
-- anyone could read/write/drop every row in these tables.
--
-- Locked decision (2026-05-12): enable RLS everywhere, no policies for
-- these server-only tables. service_role bypasses RLS so backend reads
-- and writes continue unaffected. Frontend never queries these tables
-- directly — every read goes through /api/* which authenticates against
-- service_role.
--
-- If a future feature genuinely needs anon/authenticated direct read on
-- one of these tables (e.g., public /track-record page reading from
-- model_performance), add a targeted SELECT policy in a follow-up PR —
-- never disable RLS to "fix" access.
--
-- sector_scores was dropped in 2026_05_12_pr_remove_f10_sector_rotation.sql
-- so it's excluded from this list (24 tables, not 25).
-- ============================================================================

-- ── 1. Subscription / broker / users (sensitive) ──────────────────────────
ALTER TABLE IF EXISTS public.subscription_plans       ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.broker_connections       ENABLE ROW LEVEL SECURITY;

-- ── 2. Scanner / market caches (server-side compute) ──────────────────────
ALTER TABLE IF EXISTS public.daily_universe           ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.eod_scan_runs            ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.market_data              ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.stocks                   ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.candles                  ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.features                 ENABLE ROW LEVEL SECURITY;

-- ── 3. Audit + system ─────────────────────────────────────────────────────
ALTER TABLE IF EXISTS public.audit_log                ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.admin_audit_log          ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.schema_migrations        ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.system_flags             ENABLE ROW LEVEL SECURITY;

-- ── 4. ML pipeline (server-side training + inference writes) ──────────────
ALTER TABLE IF EXISTS public.model_versions           ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.model_performance        ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.model_outcomes           ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.alpha_scores             ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.forecast_scores          ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.vix_forecasts            ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.news_sentiment           ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.earnings_predictions     ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.signal_debates           ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.training_runs            ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.scheduler_job_runs       ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.gemini_call_log          ENABLE ROW LEVEL SECURITY;
