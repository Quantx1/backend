-- ============================================================================
-- 2026-06-04 — Drop the dead gemini_call_log table.
--
-- Legacy per-call Gemini telemetry table (created in 2026_04_19_pr2_v1_ai_stack).
-- Superseded by llm_usage_events; 0 rows; zero references in application code
-- after the Gemini->OpenRouter migration. Dropping it also clears its
-- "RLS enabled, no policy" advisory note.
-- ============================================================================

DROP TABLE IF EXISTS public.gemini_call_log;
