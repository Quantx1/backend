-- ============================================================================
-- PR 55 — Telegram connect flow (onboarding activation funnel)
-- ============================================================================
-- We already ship the free Telegram daily digest + per-event alerts, but
-- nothing in the UI asks the user to link their account after signup.
-- This PR adds a short-lived link token so the onboarding step can issue
-- a ``/start <token>`` deep link to the bot; the webhook handler swaps
-- the token for ``telegram_chat_id``.
--
-- Idempotent — safe to re-run.
-- ============================================================================

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS telegram_link_token TEXT;

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS telegram_link_expires_at TIMESTAMPTZ;

-- Fast lookup during the webhook path. Uniqueness prevents collisions;
-- NULLs are allowed (every un-linking user nulls the token).
CREATE UNIQUE INDEX IF NOT EXISTS user_profiles_telegram_link_token_idx
    ON public.user_profiles (telegram_link_token)
    WHERE telegram_link_token IS NOT NULL;

INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_20_pr55_telegram_link',
    'Add telegram_link_token + expiry for the onboarding Telegram connect flow.'
) ON CONFLICT (version) DO NOTHING;
