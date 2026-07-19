-- ============================================================================
-- PR 64 — user_digest_deliveries audit table
-- ============================================================================
-- The morning/evening digest scheduler jobs (PR 61) fan out to Telegram
-- and WhatsApp but have nowhere to record what was sent. This table
-- lets us answer "did user X receive the morning brief on 2026-04-21?"
-- without tailing logs, and supports per-user cohort analytics + support
-- when someone says "I didn't get my digest".
--
-- One row per (user, channel) send attempt. We intentionally record
-- only a short `body_preview` — the full body regenerates deterministically
-- from the same day's market data, so retaining 1000-char messages has
-- no compliance upside and inflates the table over months.
--
-- Idempotent — safe to re-run.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.user_digest_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('morning', 'evening')),
    channel TEXT NOT NULL CHECK (channel IN ('telegram', 'whatsapp')),
    status TEXT NOT NULL CHECK (status IN ('sent', 'failed', 'skipped')),
    body_preview TEXT,
    error_detail TEXT,
    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Per-user lookup: "show me this user's last 30 days of digest deliveries".
CREATE INDEX IF NOT EXISTS user_digest_deliveries_user_sent_idx
    ON public.user_digest_deliveries (user_id, sent_at DESC);

-- Cohort analytics: "how many successful WhatsApp sends yesterday?"
CREATE INDEX IF NOT EXISTS user_digest_deliveries_sent_channel_idx
    ON public.user_digest_deliveries (sent_at DESC, channel, status);

-- RLS: users see their own rows, service role writes everything.
ALTER TABLE public.user_digest_deliveries ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS user_digest_deliveries_self_read ON public.user_digest_deliveries;
CREATE POLICY user_digest_deliveries_self_read
    ON public.user_digest_deliveries
    FOR SELECT
    TO authenticated
    USING (user_id = auth.uid());

INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_20_pr64_digest_deliveries',
    'Add user_digest_deliveries audit table — PR 64 digest delivery audit.'
) ON CONFLICT (version) DO NOTHING;
