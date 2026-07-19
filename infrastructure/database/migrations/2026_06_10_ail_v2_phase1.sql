-- AIL v2 Phase 1 — persistent grounded-answer cache + per-feature LLM usage.
-- Idempotent. Service-role only (RLS on, no anon/auth policies).

CREATE TABLE IF NOT EXISTS public.llm_response_cache (
    cache_key   text PRIMARY KEY,
    surface     text NOT NULL DEFAULT '',
    payload     jsonb NOT NULL,
    model       text NOT NULL DEFAULT '',
    created_at  timestamptz NOT NULL DEFAULT now(),
    expires_at  timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_response_cache_expires
    ON public.llm_response_cache (expires_at);
ALTER TABLE public.llm_response_cache ENABLE ROW LEVEL SECURITY;

CREATE TABLE IF NOT EXISTS public.llm_feature_usage (
    user_id     uuid NOT NULL,
    feature     text NOT NULL,
    window_key  text NOT NULL,
    used        integer NOT NULL DEFAULT 0,
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, feature, window_key)
);
ALTER TABLE public.llm_feature_usage ENABLE ROW LEVEL SECURITY;
