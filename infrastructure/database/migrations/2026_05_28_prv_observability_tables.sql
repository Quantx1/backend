-- ============================================================================
-- PR-V (2026-05-28) — Observability persistence layer.
--
-- Two tables that the admin observability surfaces have been quietly
-- depending on, but no migration created them:
--
--   1. analytics_events  — referenced by backend/api/admin/observability.py:100
--      The comment claims PostHog events are "mirrored" here, but no
--      writer existed. PR-V wires the dual-write in posthog_events.track().
--
--   2. llm_usage_events  — new. One row per LLM API call (Copilot,
--      Lab agent, Doctor, debate, ...). Carries model, input/output
--      tokens, and a precomputed micros_usd cost so the cost-per-user
--      rollup on /admin/system can scan a single column.
--
-- Both tables are append-only. They never participate in product reads
-- on the hot path — only admin dashboards + monthly billing rollups.
-- ============================================================================

BEGIN;

-- ─────────────────────────────────────────────────────────────────────────
-- 1. analytics_events — PostHog mirror
-- ─────────────────────────────────────────────────────────────────────────
-- Schema mirrors what backend/api/admin/observability.py:95-114
-- already expects: event TEXT, properties JSONB, ts TIMESTAMPTZ,
-- user_id UUID NULL.

CREATE TABLE IF NOT EXISTS public.analytics_events (
    id          BIGSERIAL PRIMARY KEY,
    event       TEXT        NOT NULL,
    properties  JSONB       NOT NULL DEFAULT '{}'::jsonb,
    user_id     UUID        REFERENCES auth.users(id) ON DELETE SET NULL,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS analytics_events_event_ts_idx
    ON public.analytics_events (event, ts DESC);

CREATE INDEX IF NOT EXISTS analytics_events_user_ts_idx
    ON public.analytics_events (user_id, ts DESC)
    WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS analytics_events_ts_idx
    ON public.analytics_events (ts DESC);

-- Service role only — admin reads come through SUPABASE_SERVICE_KEY.
ALTER TABLE public.analytics_events ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename  = 'analytics_events'
          AND policyname = 'analytics_events_service_only'
    ) THEN
        CREATE POLICY analytics_events_service_only
            ON public.analytics_events
            FOR ALL
            TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END $$;

-- ─────────────────────────────────────────────────────────────────────────
-- 2. llm_usage_events — per-call cost log
-- ─────────────────────────────────────────────────────────────────────────
-- One row per LLM API call. ``micros_usd`` is precomputed so the daily
-- cost panel can SUM(micros_usd) WHERE ts > now() - interval '24h' in a
-- single index scan, rather than re-tokenizing.

CREATE TABLE IF NOT EXISTS public.llm_usage_events (
    id              BIGSERIAL PRIMARY KEY,
    user_id         UUID        REFERENCES auth.users(id) ON DELETE SET NULL,
    feature         TEXT        NOT NULL,        -- 'copilot' | 'lab' | 'doctor' | 'debate' | 'finrobot'
    provider        TEXT        NOT NULL,        -- 'anthropic' | 'google' | 'openai'
    model           TEXT        NOT NULL,        -- 'claude-sonnet-4-6', 'gemini-2.5-pro', ...
    input_tokens    INTEGER     NOT NULL DEFAULT 0,
    output_tokens   INTEGER     NOT NULL DEFAULT 0,
    cache_read_tokens  INTEGER  NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER  NOT NULL DEFAULT 0,
    micros_usd      BIGINT      NOT NULL DEFAULT 0,  -- USD * 1_000_000, so 0.000001 USD precision
    latency_ms      INTEGER,
    request_id      TEXT,                            -- provider's request_id when available
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS llm_usage_events_user_ts_idx
    ON public.llm_usage_events (user_id, ts DESC)
    WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS llm_usage_events_feature_ts_idx
    ON public.llm_usage_events (feature, ts DESC);

CREATE INDEX IF NOT EXISTS llm_usage_events_ts_idx
    ON public.llm_usage_events (ts DESC);

ALTER TABLE public.llm_usage_events ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename  = 'llm_usage_events'
          AND policyname = 'llm_usage_events_service_only'
    ) THEN
        CREATE POLICY llm_usage_events_service_only
            ON public.llm_usage_events
            FOR ALL
            TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END $$;

COMMIT;
