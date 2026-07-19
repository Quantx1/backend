-- ============================================================================
-- PR 2 — Consolidated v1 AI-stack migration
-- ============================================================================
-- Adds every schema change Steps 1-3 need (except is_admin which shipped in
-- PR 1). Single idempotent file — safe to re-run.
--
-- Touches:
--   * signals              : model-output columns (tft_p10/p50/p90, lgbm/qlib/
--                            timesfm/chronos/hgnc scores, finbert_sentiment,
--                            regime_at_signal, explanation_text)
--   * user_profiles        : tier, openalgo creds, whatsapp, feature flags
--   * 17 new tables        : regime_history, alpha_scores, forecast_scores,
--                            vix_forecasts, news_sentiment,
--                            earnings_predictions, paper_*, ai_portfolio_*,
--                            signal_debates, user_weekly_reviews,
--                            model_versions, model_rolling_performance,
--                            gemini_call_log, scheduler_job_runs
--   * Existing candles     : kept as-is — fills the tick-bar role
--   * Existing model_performance : kept as-is — adds new
--                            model_rolling_performance table for Step-4 N4
--                            public /models page
-- ============================================================================

-- Ensure uuid-ossp (already installed in base schema, but idempotent safety).
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- 1. USER_PROFILES — tier + OpenAlgo + WhatsApp + feature flags
-- ============================================================================

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS tier TEXT NOT NULL DEFAULT 'free'
        CHECK (tier IN ('free', 'pro', 'elite'));

-- OpenAlgo credentials (Fernet-encrypted at rest by BROKER_ENCRYPTION_KEY)
ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS openalgo_url TEXT;
ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS openalgo_api_key TEXT;

-- Already has telegram_chat_id + telegram_connected in base schema.
-- Add WhatsApp parity.
ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS whatsapp_phone TEXT;
ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS whatsapp_verified BOOLEAN DEFAULT FALSE;

-- Feature flags per user (Elite-tier opt-ins)
ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS auto_trader_enabled BOOLEAN DEFAULT FALSE;
ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS ai_portfolio_enabled BOOLEAN DEFAULT FALSE;

-- Index for tier gating hot path (checked on nearly every authenticated request).
CREATE INDEX IF NOT EXISTS user_profiles_tier_idx ON public.user_profiles (tier);

-- ============================================================================
-- 2. SIGNALS — multi-model score columns
-- ============================================================================
-- Base schema already has: catboost_score, tft_score, stockformer_score,
-- model_agreement, strategy_names, tft_prediction JSONB. We ADD:

ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS tft_p10 DECIMAL(15, 4);
ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS tft_p50 DECIMAL(15, 4);
ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS tft_p90 DECIMAL(15, 4);
ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS lgbm_buy_prob DECIMAL(5, 4);
ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS qlib_score DECIMAL(8, 4);
ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS qlib_rank INTEGER;
ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS timesfm_p50 DECIMAL(15, 4);
ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS chronos_p50 DECIMAL(15, 4);
ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS hgnc_up_prob DECIMAL(5, 4);
ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS finbert_sentiment DECIMAL(5, 4);
ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS regime_at_signal TEXT
        CHECK (regime_at_signal IS NULL OR regime_at_signal IN ('bull', 'sideways', 'bear'));
ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS explanation_text TEXT;
ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS explanation_generated_at TIMESTAMPTZ;

-- Indexes for signal-list hot paths.
CREATE INDEX IF NOT EXISTS signals_regime_at_signal_idx
    ON public.signals (regime_at_signal) WHERE regime_at_signal IS NOT NULL;
CREATE INDEX IF NOT EXISTS signals_qlib_rank_idx
    ON public.signals (qlib_rank) WHERE qlib_rank IS NOT NULL;

-- ============================================================================
-- 3. REGIME HISTORY — F8 HMM + Chronos-2 macro detector output
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.regime_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    regime TEXT NOT NULL CHECK (regime IN ('bull', 'sideways', 'bear')),
    prob_bull DECIMAL(5, 4) NOT NULL,
    prob_sideways DECIMAL(5, 4) NOT NULL,
    prob_bear DECIMAL(5, 4) NOT NULL,
    vix DECIMAL(8, 4),
    nifty_close DECIMAL(15, 2),
    -- Chronos-2 covariate-aware persistence probability (how long regime holds)
    persistence_prob DECIMAL(5, 4),
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS regime_history_detected_at_idx
    ON public.regime_history (detected_at DESC);

-- ============================================================================
-- 4. ALPHA SCORES — F2/F3/F5 Qlib Alpha158 rank per (symbol, date)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.alpha_scores (
    symbol TEXT NOT NULL,
    trade_date DATE NOT NULL,
    qlib_rank INTEGER NOT NULL,
    qlib_score_raw DECIMAL(10, 6) NOT NULL,
    sector_rank INTEGER,           -- rank within stock's sector
    quality_score DECIMAL(5, 2),   -- F5 multi-factor quality screen
    top_factors JSONB DEFAULT '{}'::jsonb,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, trade_date)
);
CREATE INDEX IF NOT EXISTS alpha_scores_date_rank_idx
    ON public.alpha_scores (trade_date DESC, qlib_rank);

-- ============================================================================
-- 5. FORECAST SCORES — F3 TimesFM + Chronos + ensemble
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.forecast_scores (
    symbol TEXT NOT NULL,
    trade_date DATE NOT NULL,
    horizon_days INTEGER NOT NULL CHECK (horizon_days IN (1, 5, 10, 15)),
    timesfm_p50 DECIMAL(15, 4),
    chronos_bolt_p50 DECIMAL(15, 4),
    chronos_2_p50 DECIMAL(15, 4),
    ensemble_p50 DECIMAL(15, 4),
    -- Quantile bands for the ensemble (for chart quantile overlays)
    ensemble_p10 DECIMAL(15, 4),
    ensemble_p90 DECIMAL(15, 4),
    direction TEXT CHECK (direction IS NULL OR direction IN ('bullish', 'bearish', 'neutral')),
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, trade_date, horizon_days)
);
CREATE INDEX IF NOT EXISTS forecast_scores_date_idx
    ON public.forecast_scores (trade_date DESC);

-- ============================================================================
-- 6. VIX FORECASTS — F6 TFT VIX variant (drives option-strategy selection)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.vix_forecasts (
    trade_date DATE NOT NULL,
    horizon_days INTEGER NOT NULL CHECK (horizon_days BETWEEN 1 AND 10),
    tft_p10 DECIMAL(8, 4),
    tft_p50 DECIMAL(8, 4),
    tft_p90 DECIMAL(8, 4),
    direction TEXT CHECK (direction IS NULL OR direction IN ('rising', 'falling', 'stable')),
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (trade_date, horizon_days)
);

-- ============================================================================
-- 7. NEWS SENTIMENT — FinBERT-India per-symbol daily aggregate
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.news_sentiment (
    symbol TEXT NOT NULL,
    trade_date DATE NOT NULL,
    mean_score DECIMAL(5, 4) NOT NULL,
    headline_count INTEGER NOT NULL DEFAULT 0,
    positive_count INTEGER NOT NULL DEFAULT 0,
    negative_count INTEGER NOT NULL DEFAULT 0,
    neutral_count INTEGER NOT NULL DEFAULT 0,
    sample_headlines JSONB DEFAULT '[]'::jsonb,
    sources TEXT[] DEFAULT ARRAY[]::TEXT[],
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, trade_date)
);
CREATE INDEX IF NOT EXISTS news_sentiment_date_idx
    ON public.news_sentiment (trade_date DESC);

-- ============================================================================
-- 8. EARNINGS PREDICTIONS — F9 XGBoost beat/miss model
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.earnings_predictions (
    symbol TEXT NOT NULL,
    announce_date DATE NOT NULL,
    beat_prob DECIMAL(5, 4) NOT NULL,
    confidence TEXT CHECK (confidence IN ('low', 'medium', 'high')),
    evidence JSONB DEFAULT '{}'::jsonb,
    strategy_recommendation TEXT,  -- markdown from FinRobot transcript agent
    actual_result TEXT CHECK (actual_result IS NULL OR actual_result IN ('beat', 'miss', 'inline')),
    actual_return DECIMAL(8, 4),   -- populated after announcement for model-outcome fine-tune
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, announce_date)
);
CREATE INDEX IF NOT EXISTS earnings_predictions_announce_date_idx
    ON public.earnings_predictions (announce_date);

-- ============================================================================
-- 10. PAPER TRADING — F11 acquisition engine (Free tier)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.paper_portfolios (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    cash DECIMAL(15, 2) NOT NULL DEFAULT 1000000,  -- ₹10L virtual seed
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS public.paper_positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    signal_id UUID REFERENCES public.signals(id) ON DELETE SET NULL,
    symbol TEXT NOT NULL,
    qty INTEGER NOT NULL,
    entry_price DECIMAL(15, 2) NOT NULL,
    entry_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'closed', 'expired')),
    stop_loss DECIMAL(15, 2),
    target DECIMAL(15, 2)
);
CREATE INDEX IF NOT EXISTS paper_positions_user_status_idx
    ON public.paper_positions (user_id, status);

CREATE TABLE IF NOT EXISTS public.paper_trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    position_id UUID REFERENCES public.paper_positions(id) ON DELETE SET NULL,
    signal_id UUID REFERENCES public.signals(id) ON DELETE SET NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('buy', 'sell')),
    qty INTEGER NOT NULL,
    price DECIMAL(15, 2) NOT NULL,
    pnl DECIMAL(15, 2),            -- realized P&L on sells
    pnl_pct DECIMAL(8, 4),
    exit_reason TEXT CHECK (exit_reason IS NULL OR exit_reason IN ('target', 'stop', 'time', 'manual')),
    ai_note TEXT,                  -- Gemini post-mortem on close
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS paper_trades_user_date_idx
    ON public.paper_trades (user_id, executed_at DESC);

CREATE TABLE IF NOT EXISTS public.paper_snapshots (
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    equity DECIMAL(15, 2) NOT NULL,
    cash DECIMAL(15, 2) NOT NULL,
    invested DECIMAL(15, 2) NOT NULL DEFAULT 0,
    drawdown_pct DECIMAL(8, 4),
    nifty_close DECIMAL(15, 2),    -- for benchmark overlay
    PRIMARY KEY (user_id, snapshot_date)
);

-- ============================================================================
-- 11. AI PORTFOLIO — F5 AI SIP Elite monthly-rebalanced holdings
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.ai_portfolio_holdings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    target_weight DECIMAL(5, 4) NOT NULL,    -- 0..1
    current_weight DECIMAL(5, 4),
    qty INTEGER DEFAULT 0,
    last_rebalanced_at TIMESTAMPTZ,
    UNIQUE (user_id, symbol)
);
CREATE INDEX IF NOT EXISTS ai_portfolio_holdings_user_idx
    ON public.ai_portfolio_holdings (user_id);

-- ============================================================================
-- 12. SIGNAL DEBATES — B1 TradingAgents Bull/Bear transcripts (Elite)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.signal_debates (
    signal_id UUID PRIMARY KEY REFERENCES public.signals(id) ON DELETE CASCADE,
    bull_case TEXT,
    bear_case TEXT,
    risk_assessment TEXT,
    trader_verdict TEXT,
    agent_cost_usd DECIMAL(8, 6),   -- Gemini token cost for this debate
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 13. USER WEEKLY REVIEWS — N10 Gemini-generated personal summary
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.user_weekly_reviews (
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    week_of DATE NOT NULL,                  -- Monday of the review week
    content_markdown TEXT NOT NULL,
    week_return_pct DECIMAL(8, 4),
    nifty_return_pct DECIMAL(8, 4),
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, week_of)
);

-- ============================================================================
-- 14. MODEL VERSIONS — Step 2 locked: B2 + Postgres registry (replaces MLflow)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name TEXT NOT NULL,               -- 'tft_swing', 'lstm_intraday', 'finrl_x_ppo', etc.
    version INTEGER NOT NULL,
    artifact_uri TEXT NOT NULL,             -- b2://swingai-models/tft_swing/v2/model.ckpt
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trained_by TEXT,                        -- 'rishi-colab', 'github-actions', etc.
    metrics JSONB DEFAULT '{}'::jsonb,      -- { "directional_acc": 0.58, "ic": 0.047, "sharpe": 1.74 }
    git_sha TEXT,
    is_prod BOOLEAN NOT NULL DEFAULT FALSE,
    is_shadow BOOLEAN NOT NULL DEFAULT FALSE,
    is_retired BOOLEAN NOT NULL DEFAULT FALSE,
    notes TEXT,
    UNIQUE (model_name, version)
);
CREATE INDEX IF NOT EXISTS model_versions_name_trained_idx
    ON public.model_versions (model_name, trained_at DESC);
-- Partial index for hot "find prod version" queries — one row per model_name.
CREATE UNIQUE INDEX IF NOT EXISTS model_versions_prod_unique_idx
    ON public.model_versions (model_name)
    WHERE is_prod = TRUE;

-- ============================================================================
-- 15. MODEL ROLLING PERFORMANCE — Step-4 N4 public /models dashboard
-- ============================================================================
-- NOTE: `model_performance` already exists in base schema with a different
-- shape (per-day accuracy per fixed model family). We keep that intact and
-- add this new table for per-model rolling stats exposed on the public
-- /models page.

CREATE TABLE IF NOT EXISTS public.model_rolling_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name TEXT NOT NULL,
    window_days INTEGER NOT NULL CHECK (window_days IN (7, 30, 90, 365)),
    win_rate DECIMAL(5, 4),
    avg_pnl_pct DECIMAL(8, 4),
    signal_count INTEGER NOT NULL DEFAULT 0,
    directional_accuracy DECIMAL(5, 4),
    sharpe_ratio DECIMAL(5, 2),
    max_drawdown_pct DECIMAL(5, 2),
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS model_rolling_performance_name_window_idx
    ON public.model_rolling_performance (model_name, window_days, computed_at DESC);

-- ============================================================================
-- 16. MODEL OUTCOMES — Month 2+ user-outcome fine-tune training data
-- ============================================================================
-- The "Layer 1 Data Moat" per ULTRA DEEP RESEARCH §Layer 1. Every closed
-- paper + live trade writes an outcome row used for monthly fine-tuning of
-- FinBERT / TFT / BreakoutMetaLabeler / Qlib.

CREATE TABLE IF NOT EXISTS public.model_outcomes (
    signal_id UUID PRIMARY KEY REFERENCES public.signals(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    entry_at TIMESTAMPTZ,
    exit_at TIMESTAMPTZ,
    exit_reason TEXT CHECK (exit_reason IS NULL OR exit_reason IN ('target', 'stop', 'time', 'manual')),
    pnl_pct DECIMAL(8, 4),
    holding_days INTEGER,

    -- Inputs at entry (frozen snapshot for retraining)
    tft_p50_at_entry DECIMAL(15, 4),
    qlib_rank_at_entry INTEGER,
    lstm_prob_at_entry DECIMAL(5, 4),
    finbert_score_at_entry DECIMAL(5, 4),
    regime_at_entry TEXT,
    chart_image_path TEXT,                -- B2 URI for PatternCNN training Month 4+
    news_headlines_at_entry JSONB DEFAULT '[]'::jsonb,

    -- Per-model correctness labels
    tft_correct BOOLEAN,
    qlib_correct BOOLEAN,
    lstm_correct BOOLEAN,
    finbert_correct BOOLEAN,

    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS model_outcomes_computed_at_idx
    ON public.model_outcomes (computed_at DESC);

-- ============================================================================
-- 17. GEMINI CALL LOG — cost accounting for single-LLM decision
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.gemini_call_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_type TEXT NOT NULL,              -- 'explain_signal' / 'copilot' / 'digest' / 'finrobot_*' / 'tradingagents_*' / 'hgnc' / 'weekly_review'
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd DECIMAL(10, 8),
    latency_ms INTEGER,
    cache_hit BOOLEAN DEFAULT FALSE,
    error TEXT,
    called_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS gemini_call_log_called_at_idx
    ON public.gemini_call_log (called_at DESC);
CREATE INDEX IF NOT EXISTS gemini_call_log_task_type_idx
    ON public.gemini_call_log (task_type, called_at DESC);

-- ============================================================================
-- 18. SCHEDULER JOB RUNS — 22-job observability per Step 3 §5
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.scheduler_job_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_name TEXT NOT NULL,
    triggered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'success', 'failed', 'skipped')),
    duration_ms INTEGER,
    items_processed INTEGER,
    error TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS scheduler_job_runs_job_triggered_idx
    ON public.scheduler_job_runs (job_name, triggered_at DESC);
CREATE INDEX IF NOT EXISTS scheduler_job_runs_status_idx
    ON public.scheduler_job_runs (status, triggered_at DESC) WHERE status != 'success';

-- ============================================================================
-- 19. ROW LEVEL SECURITY
-- ============================================================================

-- User-scoped tables: filter by auth.uid() = user_id
ALTER TABLE public.paper_portfolios     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.paper_positions      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.paper_trades         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.paper_snapshots      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_portfolio_holdings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_weekly_reviews  ENABLE ROW LEVEL SECURITY;

-- Drop-and-recreate pattern for idempotency (CREATE POLICY has no IF NOT EXISTS in Postgres 14).
DO $$
BEGIN
    -- paper_portfolios
    DROP POLICY IF EXISTS "users own paper portfolio" ON public.paper_portfolios;
    CREATE POLICY "users own paper portfolio" ON public.paper_portfolios
        FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

    DROP POLICY IF EXISTS "users own paper positions" ON public.paper_positions;
    CREATE POLICY "users own paper positions" ON public.paper_positions
        FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

    DROP POLICY IF EXISTS "users own paper trades" ON public.paper_trades;
    CREATE POLICY "users own paper trades" ON public.paper_trades
        FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

    DROP POLICY IF EXISTS "users own paper snapshots" ON public.paper_snapshots;
    CREATE POLICY "users own paper snapshots" ON public.paper_snapshots
        FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

    DROP POLICY IF EXISTS "users own ai portfolio" ON public.ai_portfolio_holdings;
    CREATE POLICY "users own ai portfolio" ON public.ai_portfolio_holdings
        FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

    DROP POLICY IF EXISTS "users own weekly reviews" ON public.user_weekly_reviews;
    CREATE POLICY "users own weekly reviews" ON public.user_weekly_reviews
        FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
END$$;

-- Public-read trust surfaces: /regime and /models (Step 4 §4.2 + §4.4).
ALTER TABLE public.regime_history             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_rolling_performance  ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    DROP POLICY IF EXISTS "public read regime_history" ON public.regime_history;
    CREATE POLICY "public read regime_history" ON public.regime_history
        FOR SELECT USING (true);

    DROP POLICY IF EXISTS "public read model_rolling_performance" ON public.model_rolling_performance;
    CREATE POLICY "public read model_rolling_performance" ON public.model_rolling_performance
        FOR SELECT USING (true);
END$$;

-- Authenticated-read shared tables: signals table already enforces via base
-- schema policies (no change needed).

-- ============================================================================
-- 20. RECORD MIGRATION
-- ============================================================================

INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_19_pr2_v1_ai_stack',
    'PR 2 consolidated v1 — 18 new tables + signals/user_profiles column adds for full Step 1-3 AI stack'
)
ON CONFLICT (version) DO NOTHING;
