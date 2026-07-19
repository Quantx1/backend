-- ============================================================================
-- PR — Remove F10 Sector Rotation Tracker (2026-05-12)
--
-- F10 dropped from v1 scope. Removes the sector_scores table created in
-- 2026_04_19_pr2_v1_ai_stack.sql. Idempotent (IF EXISTS).
-- ============================================================================

DROP INDEX IF EXISTS public.sector_scores_date_idx;
DROP TABLE IF EXISTS public.sector_scores;
