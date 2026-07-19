-- ============================================================================
-- PR-U (2026-05-28) — Align subscription_plans seed with locked 3-tier model.
--
-- complete_schema.sql:82-89 still seeds Free / Starter / Pro at
-- ₹0 / ₹499 / ₹1,499 and then DELETEs any row named 'elite'. This
-- contradicts:
--   * the user_profiles.tier CHECK constraint added in
--     2026_04_19_pr2_v1_ai_stack.sql:33 (tier IN ('free','pro','elite'))
--   * the locked tier model captured in memory
--     project_tier_model_locked_2026_05_28.md
--     (Free ₹0 / Pro ₹999 / Elite ₹1,999)
--   * the frontend STATIC_PLANS in app/pricing/page.tsx:83-132
--
-- The drift hasn't broken production because payment_routes.py:334-360
-- defensively pattern-matches plan names ("elite" in name → elite,
-- "pro" in name → pro), but a "starter" purchase would map to free —
-- silently downgrading paying users.
--
-- This migration is idempotent. It is safe to re-run.
-- ============================================================================

BEGIN;

-- 1. Insert / update the three canonical rows.
--    Prices in INR paise: ₹0 / ₹999 / ₹1,999 monthly.
--    Annual reflects 20% discount (12× monthly × 0.8) per memory.
INSERT INTO public.subscription_plans (
    name,
    display_name,
    description,
    price_monthly,
    price_quarterly,
    price_yearly,
    max_signals_per_day,
    max_positions,
    max_capital,
    signal_only,
    semi_auto,
    full_auto,
    equity_trading,
    futures_trading,
    options_trading,
    telegram_alerts,
    priority_support,
    api_access,
    sort_order
) VALUES
    ('free',  'Free',  'Paper trade + 1 signal/day. No card required.',
        0, 0, 0,
        1, 1, 500000,
        TRUE,  FALSE, FALSE,  TRUE, FALSE, FALSE,  FALSE, FALSE, FALSE,  1),
    ('pro',   'Pro',   'Real signals, Scanner Lab, Alerts Studio.',
        99900, 269900, 959900,
        -1, 15, -1,
        TRUE,  TRUE,  FALSE,  TRUE, FALSE, FALSE,  TRUE,  FALSE, FALSE,  2),
    ('elite', 'Elite', 'AutoPilot auto-trading + AI SIP + F&O.',
        199900, 539900, 1919900,
        -1, -1, -1,
        TRUE,  TRUE,  TRUE,   TRUE, TRUE,  TRUE,   TRUE,  TRUE,  TRUE,   3)
ON CONFLICT (name) DO UPDATE SET
    display_name        = EXCLUDED.display_name,
    description         = EXCLUDED.description,
    price_monthly       = EXCLUDED.price_monthly,
    price_quarterly     = EXCLUDED.price_quarterly,
    price_yearly        = EXCLUDED.price_yearly,
    max_signals_per_day = EXCLUDED.max_signals_per_day,
    max_positions       = EXCLUDED.max_positions,
    max_capital         = EXCLUDED.max_capital,
    signal_only         = EXCLUDED.signal_only,
    semi_auto           = EXCLUDED.semi_auto,
    full_auto           = EXCLUDED.full_auto,
    equity_trading      = EXCLUDED.equity_trading,
    futures_trading     = EXCLUDED.futures_trading,
    options_trading     = EXCLUDED.options_trading,
    telegram_alerts     = EXCLUDED.telegram_alerts,
    priority_support    = EXCLUDED.priority_support,
    api_access          = EXCLUDED.api_access,
    sort_order          = EXCLUDED.sort_order,
    is_active           = TRUE;

-- 2. Retire the legacy 'starter' row if it slipped in via the old seed.
--    Anyone with an active starter subscription is migrated to 'pro' so
--    they keep their feature set (starter@₹499 → pro@₹999 is a price
--    bump, but no paying-customer claim has surfaced; rather than
--    downgrade silently to free we honour the paid status until renewal).
UPDATE public.user_profiles
SET subscription_plan_id = (SELECT id FROM public.subscription_plans WHERE name = 'pro'),
    tier = 'pro'
WHERE subscription_plan_id = (SELECT id FROM public.subscription_plans WHERE name = 'starter');

DELETE FROM public.subscription_plans WHERE name = 'starter';

COMMIT;
