-- ============================================================================
-- PR 79 — persist signal filter defaults from the onboarding quiz
-- ============================================================================
-- The N5 onboarding quiz derives a `suggested_filters` payload (min
-- confidence, include_intraday, include_fno, segment) from the answers
-- but PR 37 never persisted it — it was returned on submit and shown
-- on the result screen, then thrown away.
--
-- Adding `signal_filter_defaults` JSONB so future signal-list surfaces
-- (signals, screener results, watchlist alerts) can use the user's
-- onboarded filter preset as their default state. Pre-existing users
-- whose quiz never ran will see NULL and fall through to the page
-- defaults, so this is fully backwards-compatible.
--
-- Idempotent — safe to re-run.
-- ============================================================================

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS signal_filter_defaults JSONB
        DEFAULT '{}'::jsonb;

INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_25_pr79_signal_filter_defaults',
    'Add user_profiles.signal_filter_defaults — persists N5 quiz preset for downstream consumption.'
) ON CONFLICT (version) DO NOTHING;
