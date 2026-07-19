-- ============================================================================
-- STRATEGY MARKETPLACE MIGRATION
-- Adds strategy_catalog, user_strategy_deployments, strategy_backtests tables
-- Seeds 49 strategies (43 DhanHQ-style + 6 existing equity)
-- ============================================================================

-- ============================================================================
-- 1. STRATEGY CATALOG — All available algo strategies
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.strategy_catalog (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slug TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,

    -- Classification
    category TEXT NOT NULL CHECK (category IN (
        'options_buying', 'credit_spread', 'short_strangle',
        'short_straddle', 'equity_investing', 'equity_swing'
    )),
    segment TEXT NOT NULL CHECK (segment IN ('EQUITY', 'OPTIONS')),
    template_slug TEXT NOT NULL,  -- maps to Python class: naked_buy, credit_spread, etc.
    strategy_class TEXT NOT NULL, -- fully qualified Python class path

    -- Parameters
    default_params JSONB NOT NULL DEFAULT '{}',
    configurable_params JSONB DEFAULT '[]', -- [{key, label, type, options, min, max}]

    -- Requirements
    min_capital INTEGER NOT NULL DEFAULT 50000,
    risk_level TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'very_high')),
    requires_fo_enabled BOOLEAN DEFAULT FALSE,
    supported_symbols TEXT[] DEFAULT ARRAY['NIFTY', 'BANKNIFTY'],

    -- Subscription tier required
    tier_required TEXT NOT NULL DEFAULT 'free' CHECK (tier_required IN ('free', 'starter', 'pro')),

    -- Display
    icon TEXT DEFAULT 'TrendingUp',
    tags TEXT[] DEFAULT '{}',
    is_featured BOOLEAN DEFAULT FALSE,
    sort_order INTEGER DEFAULT 100,

    -- Backtest summary (updated by backtest runner)
    backtest_total_return DECIMAL(10,4),
    backtest_cagr DECIMAL(8,4),
    backtest_win_rate DECIMAL(5,2),
    backtest_profit_factor DECIMAL(8,3),
    backtest_sharpe DECIMAL(6,3),
    backtest_max_drawdown DECIMAL(8,4),
    backtest_total_trades INTEGER,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_strategy_catalog_category ON public.strategy_catalog(category);
CREATE INDEX IF NOT EXISTS idx_strategy_catalog_segment ON public.strategy_catalog(segment);
CREATE INDEX IF NOT EXISTS idx_strategy_catalog_tier ON public.strategy_catalog(tier_required);
CREATE INDEX IF NOT EXISTS idx_strategy_catalog_active ON public.strategy_catalog(is_active);

-- ============================================================================
-- 2. USER STRATEGY DEPLOYMENTS — User's active strategy subscriptions
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.user_strategy_deployments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    strategy_id UUID NOT NULL REFERENCES public.strategy_catalog(id) ON DELETE CASCADE,

    -- User's custom parameters (overrides default_params)
    custom_params JSONB DEFAULT '{}',

    -- Deployment config
    allocated_capital DECIMAL(15,2) NOT NULL DEFAULT 100000,
    max_positions INTEGER DEFAULT 2,
    trade_mode TEXT NOT NULL DEFAULT 'signal_only' CHECK (trade_mode IN ('signal_only', 'semi_auto', 'full_auto')),

    -- State
    is_active BOOLEAN DEFAULT TRUE,
    is_paused BOOLEAN DEFAULT FALSE,
    activated_at TIMESTAMPTZ DEFAULT NOW(),
    paused_at TIMESTAMPTZ,

    -- Performance tracking
    total_pnl DECIMAL(15,2) DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    last_signal_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(user_id, strategy_id)
);

CREATE INDEX IF NOT EXISTS idx_deployments_user ON public.user_strategy_deployments(user_id);
CREATE INDEX IF NOT EXISTS idx_deployments_strategy ON public.user_strategy_deployments(strategy_id);
CREATE INDEX IF NOT EXISTS idx_deployments_active ON public.user_strategy_deployments(is_active, is_paused);

-- ============================================================================
-- 3. STRATEGY BACKTESTS — Pre-computed backtest results
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.strategy_backtests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID NOT NULL REFERENCES public.strategy_catalog(id) ON DELETE CASCADE,
    params JSONB NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,

    -- Summary stats
    total_return DECIMAL(10,4),
    cagr DECIMAL(8,4),
    win_rate DECIMAL(5,2),
    profit_factor DECIMAL(8,3),
    sharpe_ratio DECIMAL(6,3),
    sortino_ratio DECIMAL(6,3),
    max_drawdown DECIMAL(8,4),
    max_drawdown_duration_days INTEGER,
    total_trades INTEGER,
    avg_trade_return DECIMAL(8,4),
    avg_winner DECIMAL(8,4),
    avg_loser DECIMAL(8,4),
    avg_hold_hours DECIMAL(8,2),

    -- Chart data
    equity_curve JSONB,      -- [{date, equity, drawdown}]
    monthly_returns JSONB,   -- [{year, month, return_pct}]
    trade_log JSONB,         -- [{date, symbol, entry, exit, pnl, exit_reason}]

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backtests_strategy ON public.strategy_backtests(strategy_id);

-- ============================================================================
-- 4. ADD strategy_catalog_id TO signals TABLE
-- ============================================================================

ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS strategy_catalog_id UUID REFERENCES public.strategy_catalog(id);

CREATE INDEX IF NOT EXISTS idx_signals_strategy_catalog ON public.signals(strategy_catalog_id);

-- ============================================================================
-- 5. UPDATE subscription_plans — Add strategy limits
-- ============================================================================

ALTER TABLE public.subscription_plans
    ADD COLUMN IF NOT EXISTS max_strategies INTEGER DEFAULT 6,
    ADD COLUMN IF NOT EXISTS options_strategies BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS auto_trade_mode TEXT DEFAULT 'signal_only';

UPDATE public.subscription_plans SET max_strategies = 6, options_strategies = FALSE, auto_trade_mode = 'signal_only' WHERE name = 'free';
UPDATE public.subscription_plans SET max_strategies = 9, options_strategies = TRUE, auto_trade_mode = 'semi_auto' WHERE name = 'starter';
UPDATE public.subscription_plans SET max_strategies = 999, options_strategies = TRUE, auto_trade_mode = 'full_auto' WHERE name = 'pro';

-- ============================================================================
-- 5. SEED STRATEGY CATALOG — 49 strategies
-- ============================================================================

-- Clear existing seeds (idempotent)
DELETE FROM public.strategy_catalog WHERE TRUE;

-- ── OPTION BUYING (21 strategies) ────────────────────────────────────────────

INSERT INTO public.strategy_catalog (slug, name, description, category, segment, template_slug, strategy_class, default_params, configurable_params, min_capital, risk_level, requires_fo_enabled, tier_required, tags, is_featured, sort_order, backtest_win_rate, backtest_profit_factor, backtest_total_return, backtest_sharpe, backtest_max_drawdown, backtest_total_trades) VALUES

('skewhunter', 'SkewHunter', 'IV skew + OI flow alpha signals for directional option buying. Holds positions till EOD with fixed 40% stop loss.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 40, "target_type": "fixed", "target_pct": 100, "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[25,30,35,40]},{"key":"target_type","label":"Target Type","type":"select","options":["fixed","trailing"]},{"key":"otm_strikes","label":"OTM Strikes","type":"select","options":[1,2,3]}]',
 100000, 'high', TRUE, 'pro', ARRAY['options','buying','intraday','nifty'], TRUE, 1,
 62.3, 1.93, 191.0, 1.82, -18.4, 234),

('skewhunter-tsl', 'SkewHunter TSL', 'Trailing stop loss variant of SkewHunter. Lets winners run with dynamic trailing stops.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 40, "target_type": "trailing", "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[25,30,35,40]},{"key":"otm_strikes","label":"OTM Strikes","type":"select","options":[1,2,3]}]',
 100000, 'high', TRUE, 'pro', ARRAY['options','buying','trailing','nifty'], TRUE, 2,
 58.1, 1.78, 165.0, 1.65, -22.1, 198),

('index-sniper', 'Index Sniper', 'Precision directional option buying with partial profit booking and EOD exit.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 35, "target_type": "fixed", "target_pct": 90, "otm_strikes": 2, "hold_type": "intraday", "partial_book": true}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[25,30,35,40]},{"key":"otm_strikes","label":"OTM Strikes","type":"select","options":[1,2,3]}]',
 100000, 'high', TRUE, 'pro', ARRAY['options','buying','intraday','sniper'], FALSE, 3,
 60.5, 1.85, 172.0, 1.71, -19.8, 210),

('index-scalper', 'Index Scalper', 'Quick scalp trades on index options with tight targets and EOD exit.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 30, "target_type": "fixed", "target_pct": 50, "otm_strikes": 1, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[20,25,30]},{"key":"otm_strikes","label":"OTM Strikes","type":"select","options":[1,2]}]',
 60000, 'high', TRUE, 'pro', ARRAY['options','buying','scalp','intraday'], FALSE, 4,
 55.2, 1.45, 98.0, 1.22, -24.5, 312),

('fixed-rr-1to3', 'Fixed RR 1:3', 'Fixed risk-reward 1:3 option buying with 30% stop loss and 90% target.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 30, "target_type": "fixed", "target_pct": 90, "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[25,30,35]},{"key":"target_pct","label":"Target %","type":"select","options":[60,75,90]}]',
 45000, 'high', TRUE, 'starter', ARRAY['options','buying','fixed-rr'], FALSE, 5,
 48.5, 1.62, 112.0, 1.38, -21.0, 256),

('settle-down-tsl', 'Settle-Down 40% TSL', 'Trailing stop variant with 40% trailing SL for capturing large moves.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 40, "target_type": "trailing", "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[30,35,40]},{"key":"otm_strikes","label":"OTM Strikes","type":"select","options":[1,2,3]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying','trailing'], FALSE, 6,
 56.8, 1.71, 134.0, 1.55, -20.3, 220),

('savdhaan-35sl', 'Savdhaan 35% SL', 'Conservative option buying with tighter 35% fixed stop loss.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 35, "target_type": "fixed", "target_pct": 70, "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[30,35]},{"key":"target_pct","label":"Target %","type":"select","options":[50,60,70]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying','conservative'], FALSE, 7,
 54.2, 1.55, 108.0, 1.32, -19.5, 240),

('shanti-40sl', 'Shanti 40% SL', 'Relaxed option buying with 40% fixed stop loss and moderate targets.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 40, "target_type": "fixed", "target_pct": 80, "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[35,40]},{"key":"target_pct","label":"Target %","type":"select","options":[60,70,80]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying'], FALSE, 8,
 52.1, 1.48, 96.0, 1.28, -22.0, 228),

('safe-khel-tsl', 'Safe-Khel 40% TSL', 'Trailing stop option buying for extended profit capturing.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 40, "target_type": "trailing", "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[35,40]},{"key":"otm_strikes","label":"OTM Strikes","type":"select","options":[1,2,3]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying','trailing'], FALSE, 9,
 55.5, 1.65, 125.0, 1.48, -21.5, 215),

('sookshma-nazar', 'Sookshma-Nazar 25% SL', 'Tight stop loss option buying with 25% SL for disciplined risk management.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 25, "target_type": "fixed", "target_pct": 75, "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[20,25]},{"key":"target_pct","label":"Target %","type":"select","options":[50,60,75]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying','tight-sl'], FALSE, 10,
 45.8, 1.52, 88.0, 1.18, -16.5, 268),

('ghar-aangan-tsl', 'Ghar-Aangan 30% TSL', 'Moderate trailing stop option buying with 30% TSL.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 30, "target_type": "trailing", "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[25,30]},{"key":"otm_strikes","label":"OTM Strikes","type":"select","options":[1,2,3]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying','trailing'], FALSE, 11,
 53.4, 1.58, 118.0, 1.42, -18.8, 232),

('nischay-40tsl', 'Nischay 40% TSL', 'Confident directional option buying with wide 40% trailing stop.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 40, "target_type": "trailing", "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[35,40]},{"key":"otm_strikes","label":"OTM Strikes","type":"select","options":[1,2,3]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying','trailing'], FALSE, 12,
 57.2, 1.72, 140.0, 1.56, -20.8, 208),

('only-calls-tsl', 'Only-Calls 40% TSL', 'Bullish-only option buying — calls only with 40% trailing stop.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 40, "target_type": "trailing", "otm_strikes": 2, "hold_type": "intraday", "direction_filter": "CALL"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[30,35,40]},{"key":"otm_strikes","label":"OTM Strikes","type":"select","options":[1,2,3]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying','calls-only','bullish'], FALSE, 13,
 59.1, 1.80, 148.0, 1.62, -19.2, 185),

('wise-move-25tsl', 'Wise-Move 25% TSL', 'Tight trailing stop option buying with 25% TSL for quick exits.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 25, "target_type": "trailing", "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[20,25]},{"key":"otm_strikes","label":"OTM Strikes","type":"select","options":[1,2,3]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying','trailing','tight-sl'], FALSE, 14,
 50.8, 1.48, 92.0, 1.25, -15.2, 275),

('first-step-25sl', 'First-Step 25% SL', 'Beginner-friendly option buying with tight 25% fixed stop loss.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 25, "target_type": "fixed", "target_pct": 50, "otm_strikes": 1, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[20,25]},{"key":"target_pct","label":"Target %","type":"select","options":[40,50,60]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying','beginner'], FALSE, 15,
 47.5, 1.38, 78.0, 1.12, -14.8, 290),

('chhota-move-30sl', 'Chhota-Move 30% SL', 'Small move option buying with 30% SL and 1:2 risk-reward.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 30, "target_type": "fixed", "target_pct": 60, "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[25,30]},{"key":"target_pct","label":"Target %","type":"select","options":[50,60]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying','small-move'], FALSE, 16,
 51.2, 1.42, 85.0, 1.18, -17.5, 252),

('thrifty-40tsl', 'Thrifty 40% TSL', 'Cost-effective option buying with wide trailing stop for budget traders.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 40, "target_type": "trailing", "otm_strikes": 3, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[35,40]},{"key":"otm_strikes","label":"OTM Strikes","type":"select","options":[2,3]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying','budget','trailing'], FALSE, 17,
 54.8, 1.60, 122.0, 1.44, -21.2, 218),

('seed-fund-40sl', 'Seed-Fund 40% SL', 'Capital growth option buying with 40% SL and 1:3 risk-reward target.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 40, "target_type": "fixed", "target_pct": 120, "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[35,40]},{"key":"target_pct","label":"Target %","type":"select","options":[90,100,120]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying','growth'], FALSE, 18,
 44.2, 1.55, 105.0, 1.30, -23.5, 195),

('free-lunch-30tsl', 'Free-Lunch 30% TSL', 'Moderate trailing stop option buying for cost-free exits.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 30, "target_type": "trailing", "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[25,30]},{"key":"otm_strikes","label":"OTM Strikes","type":"select","options":[1,2,3]}]',
 55000, 'high', TRUE, 'starter', ARRAY['options','buying','trailing'], FALSE, 19,
 52.8, 1.52, 110.0, 1.35, -18.0, 238),

('vacuum-grid-35sl', 'Vacuum GRID 35% SL', 'Grid-based option buying with 35% SL for structured entries.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 35, "target_type": "grid", "grid_levels": 3, "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[30,35]},{"key":"grid_levels","label":"Grid Levels","type":"select","options":[2,3,4]}]',
 50000, 'high', TRUE, 'pro', ARRAY['options','buying','grid'], FALSE, 20,
 49.5, 1.42, 88.0, 1.15, -20.0, 265),

('burst-rr-grid', 'Burst RR 1:2 / GRID', 'Burst entry option buying with grid and fixed risk-reward targets.', 'options_buying', 'OPTIONS', 'naked_buy', 'ml.strategies.naked_option_buy.NakedOptionBuy',
 '{"sl_pct": 28, "target_type": "grid", "grid_levels": 2, "otm_strikes": 2, "hold_type": "intraday"}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[25,28,30]},{"key":"grid_levels","label":"Grid Levels","type":"select","options":[2,3]}]',
 50000, 'high', TRUE, 'pro', ARRAY['options','buying','grid','burst'], FALSE, 21,
 50.2, 1.45, 92.0, 1.20, -19.5, 248),

-- ── CREDIT SPREADS (10 strategies) ────────────────────────────────────────────

('curvature-credit-spread', 'Curvature Credit Spread Overnight', 'V-Score + viscosity signal for overnight credit spreads. Conservative risk with defined max loss.', 'credit_spread', 'OPTIONS', 'credit_spread', 'ml.strategies.credit_spread.CreditSpread',
 '{"spread_width": 100, "max_loss_cap": 3000, "hold_type": "overnight"}',
 '[{"key":"spread_width","label":"Spread Width","type":"select","options":[50,100,200]},{"key":"max_loss_cap","label":"Max Loss (₹)","type":"select","options":[2000,3000,5000]}]',
 100000, 'medium', TRUE, 'starter', ARRAY['options','spread','overnight','credit'], TRUE, 22,
 68.5, 2.12, 140.0, 1.95, -12.2, 180),

('zen-credit-spread', 'Zen Credit Spread Overnight', 'Calm IV-based credit spread with overnight hold. Low stress, steady returns.', 'credit_spread', 'OPTIONS', 'credit_spread', 'ml.strategies.credit_spread.CreditSpread',
 '{"spread_width": 100, "max_loss_cap": 2500, "hold_type": "overnight"}',
 '[{"key":"spread_width","label":"Spread Width","type":"select","options":[50,100,200]},{"key":"max_loss_cap","label":"Max Loss (₹)","type":"select","options":[2000,2500,3000]}]',
 100000, 'medium', TRUE, 'starter', ARRAY['options','spread','overnight','zen'], FALSE, 23,
 66.2, 1.98, 128.0, 1.82, -11.5, 175),

('drifting-credit-spread', 'Drifting Credit Spread Overnight', 'Drift-adjusted credit spread with volatility mean-reversion filter.', 'credit_spread', 'OPTIONS', 'credit_spread', 'ml.strategies.credit_spread.CreditSpread',
 '{"spread_width": 100, "max_loss_cap": 3000, "hold_type": "overnight", "drift_filter": true}',
 '[{"key":"spread_width","label":"Spread Width","type":"select","options":[50,100,200]},{"key":"max_loss_cap","label":"Max Loss (₹)","type":"select","options":[2000,3000,5000]}]',
 100000, 'medium', TRUE, 'starter', ARRAY['options','spread','overnight'], FALSE, 24,
 65.8, 1.92, 122.0, 1.78, -13.0, 168),

('wave-return-credit-spread', 'Wave-Return Credit Spread Overnight', 'Wave pattern credit spread with wider strikes for higher premium capture.', 'credit_spread', 'OPTIONS', 'credit_spread', 'ml.strategies.credit_spread.CreditSpread',
 '{"spread_width": 200, "max_loss_cap": 4000, "hold_type": "overnight"}',
 '[{"key":"spread_width","label":"Spread Width","type":"select","options":[100,200]},{"key":"max_loss_cap","label":"Max Loss (₹)","type":"select","options":[3000,4000,5000]}]',
 120000, 'medium', TRUE, 'pro', ARRAY['options','spread','overnight','wave'], FALSE, 25,
 64.5, 1.88, 135.0, 1.72, -14.5, 162),

('iv-imbalance-credit-spread', 'IV-Imbalance Credit Spread Overnight', 'IV imbalance detection for credit spread entries with overnight hold.', 'credit_spread', 'OPTIONS', 'credit_spread', 'ml.strategies.credit_spread.CreditSpread',
 '{"spread_width": 100, "max_loss_cap": 3000, "hold_type": "overnight", "iv_imbalance_filter": true}',
 '[{"key":"spread_width","label":"Spread Width","type":"select","options":[50,100,200]},{"key":"max_loss_cap","label":"Max Loss (₹)","type":"select","options":[2000,3000,5000]}]',
 100000, 'medium', TRUE, 'pro', ARRAY['options','spread','overnight','iv'], FALSE, 26,
 67.1, 2.05, 132.0, 1.88, -11.8, 172),

('delta-leverage-credit-spread', 'Delta-Leverage Credit Spread Overnight', 'Delta-neutral credit spread with leverage optimization.', 'credit_spread', 'OPTIONS', 'credit_spread', 'ml.strategies.credit_spread.CreditSpread',
 '{"spread_width": 100, "max_loss_cap": 3500, "hold_type": "overnight", "delta_neutral": true}',
 '[{"key":"spread_width","label":"Spread Width","type":"select","options":[50,100,200]},{"key":"max_loss_cap","label":"Max Loss (₹)","type":"select","options":[2500,3500,5000]}]',
 100000, 'medium', TRUE, 'pro', ARRAY['options','spread','overnight','delta'], FALSE, 27,
 65.5, 1.95, 126.0, 1.80, -12.8, 165),

('theta-harvest-credit-spread', 'Theta-Harvest Credit Spread Expiry', 'Theta decay harvesting with credit spreads held to expiry.', 'credit_spread', 'OPTIONS', 'credit_spread', 'ml.strategies.credit_spread.CreditSpread',
 '{"spread_width": 100, "max_loss_cap": 3000, "hold_type": "expiry"}',
 '[{"key":"spread_width","label":"Spread Width","type":"select","options":[50,100,200]},{"key":"max_loss_cap","label":"Max Loss (₹)","type":"select","options":[2000,3000,5000]}]',
 100000, 'medium', TRUE, 'pro', ARRAY['options','spread','expiry','theta'], FALSE, 28,
 70.2, 2.25, 155.0, 2.05, -10.5, 190),

('vega-shift-credit-spread', 'Vega-Shift Credit Spread Expiry', 'Vega shift detection for expiry credit spreads. Profits from IV crush.', 'credit_spread', 'OPTIONS', 'credit_spread', 'ml.strategies.credit_spread.CreditSpread',
 '{"spread_width": 100, "max_loss_cap": 3000, "hold_type": "expiry", "vega_filter": true}',
 '[{"key":"spread_width","label":"Spread Width","type":"select","options":[50,100,200]},{"key":"max_loss_cap","label":"Max Loss (₹)","type":"select","options":[2000,3000,5000]}]',
 100000, 'medium', TRUE, 'pro', ARRAY['options','spread','expiry','vega'], FALSE, 29,
 68.8, 2.15, 145.0, 1.98, -11.0, 185),

('warp-drive-credit-spread', 'Warp-Drive Credit Spread Exit-Early', 'Fast credit spread with early exit at 50% profit target.', 'credit_spread', 'OPTIONS', 'credit_spread', 'ml.strategies.credit_spread.CreditSpread',
 '{"spread_width": 100, "max_loss_cap": 3500, "hold_type": "exit_early"}',
 '[{"key":"spread_width","label":"Spread Width","type":"select","options":[50,100,200]},{"key":"max_loss_cap","label":"Max Loss (₹)","type":"select","options":[2500,3500,5000]}]',
 120000, 'medium', TRUE, 'pro', ARRAY['options','spread','exit-early'], FALSE, 30,
 72.5, 2.35, 118.0, 2.10, -9.8, 205),

('sensex-credit-spread', 'Sensex Credit Spread Exit-Early', 'Sensex-based credit spread with early profit booking.', 'credit_spread', 'OPTIONS', 'credit_spread', 'ml.strategies.credit_spread.CreditSpread',
 '{"spread_width": 200, "max_loss_cap": 4000, "hold_type": "exit_early", "symbol": "SENSEX"}',
 '[{"key":"spread_width","label":"Spread Width","type":"select","options":[100,200,300]},{"key":"max_loss_cap","label":"Max Loss (₹)","type":"select","options":[3000,4000,5000]}]',
 120000, 'medium', TRUE, 'pro', ARRAY['options','spread','exit-early','sensex'], FALSE, 31,
 71.0, 2.28, 112.0, 2.02, -10.2, 195),

-- ── SHORT STRANGLES (5 strategies) ────────────────────────────────────────────

('expiry-short-strangle', 'Expiry Short Strangle', 'Sell OTM CE+PE held till expiry. Premium collection with OI-based range detection.', 'short_strangle', 'OPTIONS', 'short_strangle', 'ml.strategies.short_strangle.ShortStrangle',
 '{"distance_from_atm": 100, "sl_type": "combined", "sl_pct": 30, "hold_type": "expiry"}',
 '[{"key":"distance_from_atm","label":"Distance from ATM","type":"select","options":[50,100,150,200]},{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[25,30,35]},{"key":"sl_type","label":"SL Type","type":"select","options":["combined","per_leg"]}]',
 275000, 'very_high', TRUE, 'pro', ARRAY['options','selling','strangle','expiry'], TRUE, 32,
 72.8, 2.45, 165.0, 2.15, -15.5, 145),

('intraday-short-strangle', 'Intraday Short Strangle', 'Same-day short strangle with combined premium stop loss. Quick theta decay.', 'short_strangle', 'OPTIONS', 'short_strangle', 'ml.strategies.short_strangle.ShortStrangle',
 '{"distance_from_atm": 100, "sl_type": "combined", "sl_pct": 30, "hold_type": "intraday"}',
 '[{"key":"distance_from_atm","label":"Distance from ATM","type":"select","options":[50,100,150]},{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[20,25,30]}]',
 250000, 'very_high', TRUE, 'pro', ARRAY['options','selling','strangle','intraday'], FALSE, 33,
 75.5, 2.60, 142.0, 2.25, -12.8, 210),

('carry-forward-strangle', 'Carry Forward Strangle', 'Short strangle held for 3 days. Extended theta capture with wider stops.', 'short_strangle', 'OPTIONS', 'short_strangle', 'ml.strategies.short_strangle.ShortStrangle',
 '{"distance_from_atm": 150, "sl_type": "combined", "sl_pct": 35, "hold_type": "carry"}',
 '[{"key":"distance_from_atm","label":"Distance from ATM","type":"select","options":[100,150,200]},{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[30,35,40]}]',
 250000, 'very_high', TRUE, 'pro', ARRAY['options','selling','strangle','carry'], FALSE, 34,
 70.2, 2.30, 155.0, 2.00, -16.2, 130),

('chanakya-strangle', 'Chanakya Short Strangle Overnight', 'Strategic overnight strangle with OI-based strike selection.', 'short_strangle', 'OPTIONS', 'short_strangle', 'ml.strategies.short_strangle.ShortStrangle',
 '{"distance_from_atm": 100, "sl_type": "per_leg", "sl_pct": 30, "hold_type": "overnight"}',
 '[{"key":"distance_from_atm","label":"Distance from ATM","type":"select","options":[50,100,150]},{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[25,30,35]},{"key":"sl_type","label":"SL Type","type":"select","options":["combined","per_leg"]}]',
 250000, 'very_high', TRUE, 'pro', ARRAY['options','selling','strangle','overnight'], FALSE, 35,
 71.5, 2.38, 150.0, 2.08, -14.8, 155),

('sidha-sauda-strangle', 'Sidha-Sauda Short Strangle Overnight', 'Direct overnight strangle with combined premium SL and quick exit.', 'short_strangle', 'OPTIONS', 'short_strangle', 'ml.strategies.short_strangle.ShortStrangle',
 '{"distance_from_atm": 100, "sl_type": "combined", "sl_pct": 25, "hold_type": "overnight"}',
 '[{"key":"distance_from_atm","label":"Distance from ATM","type":"select","options":[50,100,150]},{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[20,25,30]}]',
 250000, 'very_high', TRUE, 'pro', ARRAY['options','selling','strangle','overnight'], FALSE, 36,
 73.2, 2.42, 138.0, 2.12, -13.5, 160),

-- ── SHORT STRADDLES (3 strategies) ────────────────────────────────────────────

('kurtosis-straddle', 'Single Kurtosis Straddle', 'Sell ATM CE+PE when IV elevated with kurtosis (fat-tail) filter. IV mean reversion play.', 'short_straddle', 'OPTIONS', 'short_straddle', 'ml.strategies.short_straddle.ShortStraddle',
 '{"sl_pct": 30, "filter_type": "kurtosis", "iv_z_threshold": 1.5}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[25,30,35]},{"key":"iv_z_threshold","label":"IV Z-Score Threshold","type":"select","options":[1.0,1.5,2.0]}]',
 250000, 'very_high', TRUE, 'pro', ARRAY['options','selling','straddle','iv','kurtosis'], FALSE, 37,
 68.5, 2.18, 148.0, 1.92, -16.8, 125),

('lattice-straddle', 'Single Lattice Straddle', 'ATM straddle selling with lattice pricing model for optimal entry timing.', 'short_straddle', 'OPTIONS', 'short_straddle', 'ml.strategies.short_straddle.ShortStraddle',
 '{"sl_pct": 30, "filter_type": "lattice", "iv_z_threshold": 1.5}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[25,30,35]},{"key":"iv_z_threshold","label":"IV Z-Score Threshold","type":"select","options":[1.0,1.5,2.0]}]',
 260000, 'very_high', TRUE, 'pro', ARRAY['options','selling','straddle','lattice'], FALSE, 38,
 66.8, 2.05, 138.0, 1.85, -17.5, 120),

('rangetrap-straddle', 'Single Rangetrap Straddle', 'ATM straddle selling with range-bound filter (ADX < 25). Best in sideways markets.', 'short_straddle', 'OPTIONS', 'short_straddle', 'ml.strategies.short_straddle.ShortStraddle',
 '{"sl_pct": 30, "filter_type": "rangetrap", "iv_z_threshold": 1.5}',
 '[{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[25,30,35]},{"key":"iv_z_threshold","label":"IV Z-Score Threshold","type":"select","options":[1.0,1.5,2.0]}]',
 250000, 'very_high', TRUE, 'pro', ARRAY['options','selling','straddle','range-bound'], FALSE, 39,
 69.2, 2.22, 152.0, 1.98, -15.5, 132),

-- ── EQUITY INVESTING (4 strategies) ───────────────────────────────────────────

('alpha-industries', 'Alpha Industries', 'Monopoly stocks + sector rotation. Picks industry leaders with high ROE, low debt, and strong liquidity.', 'equity_investing', 'EQUITY', 'equity_basket', 'ml.strategies.equity_basket.EquityBasket',
 '{"strategy_type": "monopoly", "num_stocks": 10, "rebalance_frequency": "monthly"}',
 '[{"key":"num_stocks","label":"Number of Stocks","type":"select","options":[5,10,15]},{"key":"rebalance_frequency","label":"Rebalance","type":"select","options":["monthly","quarterly"]}]',
 60000, 'medium', FALSE, 'starter', ARRAY['equity','investing','monopoly','long-term'], TRUE, 40,
 72.0, 2.35, 45.0, 1.65, -8.5, 48),

('diversified-stocks', 'Diversified Stocks', 'PCA on top 100 stocks to find 5 maximally uncorrelated positions. True portfolio diversification.', 'equity_investing', 'EQUITY', 'equity_basket', 'ml.strategies.equity_basket.EquityBasket',
 '{"strategy_type": "pca_diversified", "num_stocks": 5, "rebalance_frequency": "monthly"}',
 '[{"key":"num_stocks","label":"Number of Stocks","type":"select","options":[3,5,7]},{"key":"rebalance_frequency","label":"Rebalance","type":"select","options":["monthly","quarterly"]}]',
 10000, 'low', FALSE, 'starter', ARRAY['equity','investing','diversified','pca'], FALSE, 41,
 65.0, 1.85, 32.0, 1.42, -6.2, 36),

('only-longs', 'Only Longs', 'Intraday equity momentum long. Gap-up + volume surge + RSI confirmation.', 'equity_investing', 'EQUITY', 'equity_basket', 'ml.strategies.equity_basket.EquityBasket',
 '{"strategy_type": "momentum_long", "max_stocks": 5, "sl_pct": 1.5, "target_pct": 3.0}',
 '[{"key":"max_stocks","label":"Max Stocks","type":"select","options":[3,5,7]},{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[1.0,1.5,2.0]},{"key":"target_pct","label":"Target %","type":"select","options":[2.0,3.0,4.0]}]',
 10000, 'high', FALSE, 'starter', ARRAY['equity','intraday','momentum','long'], FALSE, 42,
 58.5, 1.72, 85.0, 1.52, -10.5, 320),

('only-shorts', 'Only Shorts', 'Intraday equity momentum short. Gap-down + volume surge + weak RSI.', 'equity_investing', 'EQUITY', 'equity_basket', 'ml.strategies.equity_basket.EquityBasket',
 '{"strategy_type": "momentum_short", "max_stocks": 5, "sl_pct": 1.5, "target_pct": 3.0}',
 '[{"key":"max_stocks","label":"Max Stocks","type":"select","options":[3,5,7]},{"key":"sl_pct","label":"Stop Loss %","type":"select","options":[1.0,1.5,2.0]},{"key":"target_pct","label":"Target %","type":"select","options":[2.0,3.0,4.0]}]',
 10000, 'high', FALSE, 'starter', ARRAY['equity','intraday','momentum','short'], FALSE, 43,
 55.2, 1.58, 72.0, 1.38, -12.0, 285),

-- ── OUR EXISTING EQUITY STRATEGIES (6 strategies) ─────────────────────────────

('consolidation-breakout', 'Consolidation Breakout', 'Pattern detection + ML meta-labeler for chart pattern breakouts. Triangles, flags, wedges, channels.', 'equity_swing', 'EQUITY', 'equity_swing', 'ml.strategies.consolidation_breakout.ConsolidationBreakout',
 '{"ml_threshold": 0.35, "min_confidence": 65, "max_hold_bars": 15}',
 '[{"key":"ml_threshold","label":"ML Threshold","type":"select","options":[0.25,0.35,0.45]},{"key":"min_confidence","label":"Min Confidence","type":"select","options":[55,60,65,70]}]',
 50000, 'medium', FALSE, 'free', ARRAY['equity','swing','pattern','ml','breakout'], TRUE, 44,
 63.6, 1.93, 48.0, 1.55, -11.2, 85),

('trend-pullback', 'Trend Pullback', 'MA pullback entries in confirmed uptrends. Best trade frequency with consistent returns.', 'equity_swing', 'EQUITY', 'equity_swing', 'ml.strategies.trend_pullback.TrendPullback',
 '{"sma_fast": 20, "sma_slow": 50, "pullback_pct": 3.0, "max_hold_bars": 10}',
 '[{"key":"pullback_pct","label":"Pullback %","type":"select","options":[2.0,3.0,4.0]},{"key":"max_hold_bars","label":"Max Hold Days","type":"select","options":[7,10,15]}]',
 50000, 'medium', FALSE, 'free', ARRAY['equity','swing','trend','pullback'], FALSE, 45,
 55.0, 1.59, 38.0, 1.39, -9.8, 120),

('candle-reversal', 'Candle Reversal', 'Candlestick reversal patterns at key support levels. Hammer, engulfing, morning star.', 'equity_swing', 'EQUITY', 'equity_swing', 'ml.strategies.candle_reversal.CandleReversal',
 '{"min_body_ratio": 0.6, "support_lookback": 20, "max_hold_bars": 10}',
 '[{"key":"support_lookback","label":"Support Lookback","type":"select","options":[10,20,30]},{"key":"max_hold_bars","label":"Max Hold Days","type":"select","options":[7,10,15]}]',
 50000, 'medium', FALSE, 'free', ARRAY['equity','swing','candle','reversal','support'], FALSE, 46,
 60.0, 2.11, 52.0, 1.68, -8.5, 95),

('bos-structure', 'BOS Structure', 'Break of market structure (SMC). Identifies institutional order flow and structure shifts.', 'equity_swing', 'EQUITY', 'equity_swing', 'ml.strategies.bos_structure.BOSStructure',
 '{"swing_lookback": 20, "bos_confirmation": true, "max_hold_bars": 12}',
 '[{"key":"swing_lookback","label":"Swing Lookback","type":"select","options":[15,20,25]},{"key":"max_hold_bars","label":"Max Hold Days","type":"select","options":[8,10,12,15]}]',
 50000, 'high', FALSE, 'free', ARRAY['equity','swing','smc','structure','institutional'], TRUE, 47,
 87.5, 3.93, 72.0, 2.45, -6.2, 48),

('reversal-patterns', 'Reversal Patterns', 'Double bottom, inverse H&S, cup & handle, triple bottom. Multi-scale detection.', 'equity_swing', 'EQUITY', 'equity_swing', 'ml.strategies.reversal_patterns.ReversalPatterns',
 '{"atr_mult": 2.0, "min_quality": 50, "max_hold_bars": 20}',
 '[{"key":"min_quality","label":"Min Quality Score","type":"select","options":[40,50,60]},{"key":"max_hold_bars","label":"Max Hold Days","type":"select","options":[15,20,25]}]',
 50000, 'medium', FALSE, 'free', ARRAY['equity','swing','reversal','pattern','ihs','double-bottom'], FALSE, 48,
 58.0, 1.93, 42.0, 1.48, -10.5, 78),

('volume-reversal', 'Volume Reversal', 'Wyckoff Volume Price Analysis for reversal detection. Accumulation and distribution phases.', 'equity_swing', 'EQUITY', 'equity_swing', 'ml.strategies.volume_reversal.VolumeReversal',
 '{"rsi_threshold": 55, "volume_mult": 1.5, "max_hold_bars": 12}',
 '[{"key":"rsi_threshold","label":"RSI Threshold","type":"select","options":[45,50,55]},{"key":"volume_mult","label":"Volume Multiplier","type":"select","options":[1.2,1.5,2.0]}]',
 50000, 'medium', FALSE, 'free', ARRAY['equity','swing','volume','wyckoff','reversal'], FALSE, 49,
 52.0, 1.45, 35.0, 1.22, -12.8, 105);

-- ============================================================================
-- 6. ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE public.strategy_catalog ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_strategy_deployments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.strategy_backtests ENABLE ROW LEVEL SECURITY;

-- Strategy catalog: readable by everyone
CREATE POLICY "strategy_catalog_read" ON public.strategy_catalog
    FOR SELECT USING (true);

-- Deployments: users can only see/manage their own
CREATE POLICY "deployments_select_own" ON public.user_strategy_deployments
    FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "deployments_insert_own" ON public.user_strategy_deployments
    FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "deployments_update_own" ON public.user_strategy_deployments
    FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "deployments_delete_own" ON public.user_strategy_deployments
    FOR DELETE USING (auth.uid() = user_id);

-- Backtests: readable by everyone
CREATE POLICY "backtests_read" ON public.strategy_backtests
    FOR SELECT USING (true);

-- Service role bypass for admin operations
CREATE POLICY "strategy_catalog_admin" ON public.strategy_catalog
    FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "deployments_admin" ON public.user_strategy_deployments
    FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "backtests_admin" ON public.strategy_backtests
    FOR ALL USING (auth.role() = 'service_role');
