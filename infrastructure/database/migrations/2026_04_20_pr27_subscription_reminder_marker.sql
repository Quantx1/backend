-- ============================================================================
-- PR 27 — subscription_reminder_sent_at column
-- ============================================================================
-- Scheduler job `subscription_lifecycle_check` (06:15 IST daily) uses this
-- column to avoid sending the 3-day renewal reminder more than once per
-- billing cycle. NULL = never reminded. Updated to NOW() each time the
-- job fires a reminder for the user.
--
-- Idempotent — IF NOT EXISTS guard lets this re-run safely.
-- ============================================================================

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS subscription_reminder_sent_at TIMESTAMPTZ;

-- Log this migration.
INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_20_pr27_subscription_reminder_marker',
    'Add user_profiles.subscription_reminder_sent_at — PR 27 renewal reminder dedupe marker.'
) ON CONFLICT (version) DO NOTHING;
