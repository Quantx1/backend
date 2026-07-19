-- ============================================================================
-- PR 123 — UI preferences JSONB
-- ============================================================================
-- Cross-device persistence for in-app UI state that belongs to the user
-- (not the device). First consumer: watchlist alert preset pins
-- (PR 122) — sessionStorage was good for a single tab but disappears
-- across devices. Schema is open-ended JSONB so future UI prefs (saved
-- screener filters, dashboard widget order, etc.) reuse this column.
--
-- Shape after PR 123:
--   {
--     "watchlist_preset_pins": {
--       "RELIANCE":   "atr2",
--       "NIFTYBEES":  "pct5",
--       ...
--     }
--   }
--
-- Idempotent — safe to re-run.
-- ============================================================================

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS ui_preferences JSONB DEFAULT '{}'::jsonb;

-- Log this migration.
INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_29_pr123_ui_preferences',
    'Add ui_preferences JSONB on user_profiles — PR 123 cross-device UI state.'
) ON CONFLICT (version) DO NOTHING;
