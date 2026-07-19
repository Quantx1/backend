-- ============================================================================
-- QUANT X - ENHANCED AI CORE SCHEMA UPDATES
-- ============================================================================
-- Adds new fields for:
-- - Hierarchical Ensemble (5 models + adaptive weighting)
-- - Market Regime Detection
-- - Premium Signal Filter (8-point validation)
-- - Dynamic Risk Management (5 multipliers)
-- - Confidence Decay Tracking
-- ============================================================================

-- ============================================================================
-- UPDATE SIGNALS TABLE
-- ============================================================================

-- Add new ensemble model scores (replace 3 models with 5)
ALTER TABLE public.signals 
    ADD COLUMN IF NOT EXISTS lstm_score DECIMAL(5, 2),
    ADD COLUMN IF NOT EXISTS xgboost_score DECIMAL(5, 2),
    ADD COLUMN IF NOT EXISTS rf_score DECIMAL(5, 2),
    ADD COLUMN IF NOT EXISTS svm_score DECIMAL(5, 2);

-- Rename existing columns for consistency
ALTER TABLE public.signals 
    RENAME COLUMN catboost_score TO tft_score_old;

-- Add new model agreement (now 0-5 instead of 0-3)
ALTER TABLE public.signals 
    DROP CONSTRAINT IF EXISTS signals_model_agreement_check;

ALTER TABLE public.signals
    ADD CONSTRAINT signals_model_agreement_check CHECK (model_agreement >= 0 AND model_agreement <= 5);

-- Add ensemble metrics
ALTER TABLE public.signals 
    ADD COLUMN IF NOT EXISTS ensemble_agreement_score DECIMAL(5, 2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS ensemble_uncertainty DECIMAL(5, 2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS model_weights JSONB DEFAULT '{}';

-- Add market regime fields
ALTER TABLE public.signals 
    ADD COLUMN IF NOT EXISTS market_regime TEXT CHECK (market_regime IN ('BULLISH', 'BEARISH', 'RANGE', 'CHOPPY')),
    ADD COLUMN IF NOT EXISTS regime_confidence DECIMAL(5, 2),
    ADD COLUMN IF NOT EXISTS regime_characteristics JSONB DEFAULT '{}';

-- Add premium filter validation (8-point)
ALTER TABLE public.signals 
    ADD COLUMN IF NOT EXISTS signal_grade TEXT CHECK (signal_grade IN ('PREMIUM', 'EXCELLENT', 'GOOD', 'SKIP')),
    ADD COLUMN IF NOT EXISTS reliability_score DECIMAL(5, 2),
    ADD COLUMN IF NOT EXISTS validation_details JSONB DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS strategy_confluence DECIMAL(5, 2),
    ADD COLUMN IF NOT EXISTS active_strategies JSONB DEFAULT '[]';

-- Add dynamic risk management fields
ALTER TABLE public.signals 
    ADD COLUMN IF NOT EXISTS base_risk_percent DECIMAL(5, 2) DEFAULT 2.0,
    ADD COLUMN IF NOT EXISTS final_risk_percent DECIMAL(5, 2),
    ADD COLUMN IF NOT EXISTS risk_multipliers JSONB DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS position_size_shares INTEGER;

-- Add 70 features storage (compressed JSONB)
ALTER TABLE public.signals 
    ADD COLUMN IF NOT EXISTS features_snapshot JSONB DEFAULT '{}';

-- Add confidence decay tracking
ALTER TABLE public.signals 
    ADD COLUMN IF NOT EXISTS initial_confidence DECIMAL(5, 2),
    ADD COLUMN IF NOT EXISTS current_confidence DECIMAL(5, 2),
    ADD COLUMN IF NOT EXISTS confidence_decay_applied DECIMAL(5, 2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS days_held INTEGER DEFAULT 0;

-- Add execution priority
ALTER TABLE public.signals 
    ADD COLUMN IF NOT EXISTS execution_priority TEXT CHECK (execution_priority IN ('PREMIUM', 'EXCELLENT', 'GOOD', 'SKIP')) DEFAULT 'GOOD';

-- Create indexes for new fields
CREATE INDEX IF NOT EXISTS idx_signals_market_regime ON public.signals(market_regime);
CREATE INDEX IF NOT EXISTS idx_signals_signal_grade ON public.signals(signal_grade);
CREATE INDEX IF NOT EXISTS idx_signals_reliability_score ON public.signals(reliability_score DESC);
CREATE INDEX IF NOT EXISTS idx_signals_execution_priority ON public.signals(execution_priority);

-- ============================================================================
-- UPDATE TRADES TABLE
-- ============================================================================

-- Add fields to track confidence decay at trade time
ALTER TABLE public.trades 
    ADD COLUMN IF NOT EXISTS entry_confidence DECIMAL(5, 2),
    ADD COLUMN IF NOT EXISTS exit_confidence DECIMAL(5, 2),
    ADD COLUMN IF NOT EXISTS confidence_at_entry DECIMAL(5, 2),
    ADD COLUMN IF NOT EXISTS days_held_at_exit INTEGER;

-- Add dynamic risk info
ALTER TABLE public.trades 
    ADD COLUMN IF NOT EXISTS risk_multiplier_total DECIMAL(5, 3),
    ADD COLUMN IF NOT EXISTS risk_multipliers_breakdown JSONB DEFAULT '{}';

-- ============================================================================
-- CREATE NEW TABLE: MODEL_PERFORMANCE (Track ensemble performance)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Model info
    model_name TEXT NOT NULL CHECK (model_name IN ('TFT', 'LSTM', 'XGBoost', 'RandomForest', 'SVM')),
    
    -- Performance metrics
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy DECIMAL(5, 2) DEFAULT 0,
    
    -- By regime
    bullish_accuracy DECIMAL(5, 2) DEFAULT 0,
    bearish_accuracy DECIMAL(5, 2) DEFAULT 0,
    range_accuracy DECIMAL(5, 2) DEFAULT 0,
    choppy_accuracy DECIMAL(5, 2) DEFAULT 0,
    
    -- Error metrics
    mean_absolute_error DECIMAL(8, 4) DEFAULT 0,
    root_mean_squared_error DECIMAL(8, 4) DEFAULT 0,
    
    -- Weighting history
    average_weight DECIMAL(5, 4) DEFAULT 0,
    min_weight DECIMAL(5, 4) DEFAULT 0,
    max_weight DECIMAL(5, 4) DEFAULT 0,
    
    -- Time period
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_model_performance_model ON public.model_performance(model_name);
CREATE INDEX idx_model_performance_period ON public.model_performance(period_start, period_end);

-- ============================================================================
-- CREATE NEW TABLE: REGIME_HISTORY (Track market regime changes)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.regime_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Date
    date DATE NOT NULL UNIQUE,
    
    -- Regime detection
    regime TEXT NOT NULL CHECK (regime IN ('BULLISH', 'BEARISH', 'RANGE', 'CHOPPY')),
    confidence DECIMAL(5, 2) NOT NULL,
    
    -- Characteristics
    trend_strength DECIMAL(5, 2),
    volatility DECIMAL(5, 2),
    directional_bias DECIMAL(8, 4),
    
    -- Market data
    nifty_close DECIMAL(15, 2),
    nifty_change_percent DECIMAL(5, 2),
    vix_close DECIMAL(5, 2),
    
    -- Volume/Breadth
    advances INTEGER,
    declines INTEGER,
    advance_decline_ratio DECIMAL(5, 2),
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_regime_history_date ON public.regime_history(date DESC);
CREATE INDEX idx_regime_history_regime ON public.regime_history(regime);

-- ============================================================================
-- CREATE NEW TABLE: SIGNAL_VALIDATION_LOG (Track filter performance)
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.signal_validation_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Signal reference
    symbol TEXT NOT NULL,
    signal_id UUID REFERENCES public.signals(id),
    
    -- 8-point validation scores
    ai_confidence DECIMAL(5, 2),
    strategy_confluence DECIMAL(5, 2),
    smc_confirmation DECIMAL(5, 2),
    price_action_score DECIMAL(5, 2),
    technical_alignment DECIMAL(5, 2),
    regime_fit DECIMAL(5, 2),
    volume_confirmation BOOLEAN,
    entry_precision DECIMAL(5, 2),
    
    -- Results
    reliability_score DECIMAL(5, 2),
    signal_grade TEXT CHECK (signal_grade IN ('PREMIUM', 'EXCELLENT', 'GOOD', 'SKIP')),
    passed_validation BOOLEAN,
    
    -- Outcome (filled after trade closes)
    actual_outcome TEXT CHECK (actual_outcome IN ('WIN', 'LOSS', 'BREAKEVEN')),
    actual_return_percent DECIMAL(8, 4),
    
    -- Metadata
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_validation_log_symbol ON public.signal_validation_log(symbol);
CREATE INDEX idx_validation_log_grade ON public.signal_validation_log(signal_grade);
CREATE INDEX idx_validation_log_timestamp ON public.signal_validation_log(timestamp DESC);

-- ============================================================================
-- UPDATE EXISTING FUNCTIONS
-- ============================================================================

-- Update the update_updated_at trigger function to work with new tables
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add triggers for new tables
DROP TRIGGER IF EXISTS update_model_performance_updated_at ON public.model_performance;
CREATE TRIGGER update_model_performance_updated_at
    BEFORE UPDATE ON public.model_performance
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- CREATE VIEWS FOR ANALYTICS
-- ============================================================================

-- View: Signal Performance by Grade
CREATE OR REPLACE VIEW public.signal_performance_by_grade AS
SELECT 
    signal_grade,
    COUNT(*) as total_signals,
    COUNT(CASE WHEN result = 'win' THEN 1 END) as wins,
    COUNT(CASE WHEN result = 'loss' THEN 1 END) as losses,
    ROUND(COUNT(CASE WHEN result = 'win' THEN 1 END)::NUMERIC / 
          NULLIF(COUNT(CASE WHEN result IN ('win', 'loss') THEN 1 END), 0) * 100, 2) as win_rate,
    ROUND(AVG(actual_return), 2) as avg_return,
    ROUND(AVG(reliability_score), 2) as avg_reliability
FROM public.signals
WHERE result IS NOT NULL
GROUP BY signal_grade
ORDER BY 
    CASE signal_grade
        WHEN 'PREMIUM' THEN 1
        WHEN 'EXCELLENT' THEN 2
        WHEN 'GOOD' THEN 3
        ELSE 4
    END;

-- View: Model Performance Summary
CREATE OR REPLACE VIEW public.model_performance_summary AS
SELECT 
    model_name,
    ROUND(AVG(accuracy), 2) as avg_accuracy,
    ROUND(AVG(bullish_accuracy), 2) as bullish_accuracy,
    ROUND(AVG(bearish_accuracy), 2) as bearish_accuracy,
    ROUND(AVG(range_accuracy), 2) as range_accuracy,
    ROUND(AVG(choppy_accuracy), 2) as choppy_accuracy,
    ROUND(AVG(average_weight), 4) as avg_weight,
    SUM(total_predictions) as total_predictions
FROM public.model_performance
GROUP BY model_name
ORDER BY avg_accuracy DESC;

-- View: Regime Performance
CREATE OR REPLACE VIEW public.regime_performance AS
SELECT 
    market_regime,
    COUNT(*) as total_signals,
    COUNT(CASE WHEN result = 'win' THEN 1 END) as wins,
    ROUND(COUNT(CASE WHEN result = 'win' THEN 1 END)::NUMERIC / 
          NULLIF(COUNT(CASE WHEN result IN ('win', 'loss') THEN 1 END), 0) * 100, 2) as win_rate,
    ROUND(AVG(actual_return), 2) as avg_return,
    ROUND(AVG(confidence), 2) as avg_confidence
FROM public.signals
WHERE result IS NOT NULL AND market_regime IS NOT NULL
GROUP BY market_regime
ORDER BY win_rate DESC;

-- ============================================================================
-- GRANT PERMISSIONS (for authenticated users)
-- ============================================================================

-- Grant access to new tables
GRANT SELECT, INSERT, UPDATE ON public.model_performance TO authenticated;
GRANT SELECT, INSERT ON public.regime_history TO authenticated;
GRANT SELECT, INSERT, UPDATE ON public.signal_validation_log TO authenticated;

-- Grant access to views
GRANT SELECT ON public.signal_performance_by_grade TO authenticated;
GRANT SELECT ON public.model_performance_summary TO authenticated;
GRANT SELECT ON public.regime_performance TO authenticated;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Add comment to track migration
COMMENT ON TABLE public.signals IS 'Enhanced signals table with AI ensemble, regime detection, and premium filtering - Updated for Enhanced AI Core v2.0';
COMMENT ON TABLE public.model_performance IS 'Tracks performance of 5 AI models in the hierarchical ensemble';
COMMENT ON TABLE public.regime_history IS 'Historical record of market regime classifications';
COMMENT ON TABLE public.signal_validation_log IS 'Logs 8-point validation results for signals';

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Enhanced AI Core Schema Updates Applied Successfully!';
    RAISE NOTICE 'New Features:';
    RAISE NOTICE '  ✓ 5-Model Hierarchical Ensemble';
    RAISE NOTICE '  ✓ Market Regime Detection (4 regimes)';
    RAISE NOTICE '  ✓ Premium Signal Filter (8-point validation)';
    RAISE NOTICE '  ✓ Dynamic Risk Management (5 multipliers)';
    RAISE NOTICE '  ✓ Confidence Decay Tracking';
    RAISE NOTICE '  ✓ Performance Analytics Views';
END $$;
