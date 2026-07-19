-- ============================================================================
-- QUANT X - PRODUCTION DATABASE MIGRATIONS
-- ============================================================================
-- Run this SQL in Supabase SQL Editor to set up production-ready:
-- 1. Row Level Security (RLS) policies
-- 2. Performance indexes
-- 3. Auto-create profile trigger
-- 4. Stats/portfolio history updates
-- ============================================================================

-- ============================================================================
-- 1. ENABLE RLS ON ALL TABLES
-- ============================================================================

ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE broker_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE payment_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE payment_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_history ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- 2. USER PROFILES RLS POLICIES
-- ============================================================================

-- Users can read their own profile
CREATE POLICY "Users can view own profile" ON user_profiles
    FOR SELECT USING (auth.uid() = id);

-- Users can update their own profile
CREATE POLICY "Users can update own profile" ON user_profiles
    FOR UPDATE USING (auth.uid() = id);

-- Service role can manage all profiles
CREATE POLICY "Service role full access to profiles" ON user_profiles
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- 3. SIGNALS RLS POLICIES  
-- ============================================================================

-- All authenticated users can view signals
CREATE POLICY "Authenticated users can view signals" ON signals
    FOR SELECT USING (auth.role() = 'authenticated');

-- Only service role can insert/update signals
CREATE POLICY "Service role can manage signals" ON signals
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- 4. TRADES RLS POLICIES
-- ============================================================================

-- Users can view their own trades
CREATE POLICY "Users can view own trades" ON trades
    FOR SELECT USING (auth.uid() = user_id);

-- Users can insert their own trades
CREATE POLICY "Users can insert own trades" ON trades
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Service role can manage all trades
CREATE POLICY "Service role can manage trades" ON trades
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- 5. POSITIONS RLS POLICIES
-- ============================================================================

-- Users can view their own positions
CREATE POLICY "Users can view own positions" ON positions
    FOR SELECT USING (auth.uid() = user_id);

-- Service role can manage all positions
CREATE POLICY "Service role can manage positions" ON positions
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- 6. WATCHLIST RLS POLICIES
-- ============================================================================

-- Users can view their own watchlist
CREATE POLICY "Users can view own watchlist" ON watchlist
    FOR SELECT USING (auth.uid() = user_id);

-- Users can manage their own watchlist
CREATE POLICY "Users can manage own watchlist" ON watchlist
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================================
-- 7. NOTIFICATIONS RLS POLICIES
-- ============================================================================

-- Users can view their own notifications
CREATE POLICY "Users can view own notifications" ON notifications
    FOR SELECT USING (auth.uid() = user_id);

-- Users can update their own notifications (mark as read)
CREATE POLICY "Users can update own notifications" ON notifications
    FOR UPDATE USING (auth.uid() = user_id);

-- Service role can manage all notifications
CREATE POLICY "Service role can manage notifications" ON notifications
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- 8. BROKER CONNECTIONS RLS POLICIES
-- ============================================================================

-- Users can view their own broker connections
CREATE POLICY "Users can view own broker connections" ON broker_connections
    FOR SELECT USING (auth.uid() = user_id);

-- Service role can manage all broker connections
CREATE POLICY "Service role can manage broker connections" ON broker_connections
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- 9. SUBSCRIPTIONS RLS POLICIES
-- ============================================================================

-- Users can view their own subscriptions
CREATE POLICY "Users can view own subscriptions" ON subscriptions
    FOR SELECT USING (auth.uid() = user_id);

-- Service role can manage all subscriptions
CREATE POLICY "Service role can manage subscriptions" ON subscriptions
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- 10. PAYMENT ORDERS RLS POLICIES
-- ============================================================================

-- Users can view their own payment orders
CREATE POLICY "Users can view own payment orders" ON payment_orders
    FOR SELECT USING (auth.uid() = user_id);

-- Service role can manage all payment orders
CREATE POLICY "Service role can manage payment orders" ON payment_orders
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- 11. PAYMENT TRANSACTIONS RLS POLICIES
-- ============================================================================

-- Users can view their own transactions
CREATE POLICY "Users can view own transactions" ON payment_transactions
    FOR SELECT USING (auth.uid() = user_id);

-- Service role can manage all transactions
CREATE POLICY "Service role can manage transactions" ON payment_transactions
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- 12. PORTFOLIO HISTORY RLS POLICIES
-- ============================================================================

-- Users can view their own portfolio history
CREATE POLICY "Users can view own portfolio history" ON portfolio_history
    FOR SELECT USING (auth.uid() = user_id);

-- Service role can manage all portfolio history
CREATE POLICY "Service role can manage portfolio history" ON portfolio_history
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- 13. PERFORMANCE INDEXES
-- ============================================================================

-- Signals indexes
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
CREATE INDEX IF NOT EXISTS idx_signals_date_status ON signals(date DESC, status);
CREATE INDEX IF NOT EXISTS idx_signals_confidence ON signals(confidence DESC);

-- Trades indexes
CREATE INDEX IF NOT EXISTS idx_trades_user_id ON trades(user_id);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_user_status ON trades(user_id, status);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);

-- Positions indexes
CREATE INDEX IF NOT EXISTS idx_positions_user_id ON positions(user_id);
CREATE INDEX IF NOT EXISTS idx_positions_is_active ON positions(is_active);
CREATE INDEX IF NOT EXISTS idx_positions_user_active ON positions(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);

-- Watchlist indexes
CREATE INDEX IF NOT EXISTS idx_watchlist_user_id ON watchlist(user_id);

-- Notifications indexes
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_user_read ON notifications(user_id, is_read);
CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON notifications(created_at DESC);

-- Broker connections indexes
CREATE INDEX IF NOT EXISTS idx_broker_connections_user_id ON broker_connections(user_id);
CREATE INDEX IF NOT EXISTS idx_broker_connections_user_status ON broker_connections(user_id, status);

-- Subscriptions indexes
CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);

-- Payment orders indexes
CREATE INDEX IF NOT EXISTS idx_payment_orders_user_id ON payment_orders(user_id);
CREATE INDEX IF NOT EXISTS idx_payment_orders_razorpay_order_id ON payment_orders(razorpay_order_id);

-- Payment transactions indexes  
CREATE INDEX IF NOT EXISTS idx_payment_transactions_user_id ON payment_transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_payment_transactions_razorpay_payment_id ON payment_transactions(razorpay_payment_id);

-- Portfolio history indexes
CREATE INDEX IF NOT EXISTS idx_portfolio_history_user_id ON portfolio_history(user_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_history_user_date ON portfolio_history(user_id, date DESC);

-- ============================================================================
-- 14. AUTO-CREATE PROFILE TRIGGER
-- ============================================================================

-- Function to create user profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (
        id,
        email,
        full_name,
        phone,
        capital,
        risk_profile,
        trading_mode,
        max_positions,
        risk_per_trade,
        fo_enabled,
        subscription_status,
        broker_connected,
        total_trades,
        winning_trades,
        total_pnl,
        created_at
    ) VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', ''),
        COALESCE(NEW.raw_user_meta_data->>'phone', ''),
        100000,  -- Default 1 lakh capital
        'moderate',
        'signal_only',
        5,
        2,
        false,
        'trial',
        false,
        0,
        0,
        0,
        NOW()
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Drop existing trigger if exists
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

-- Create trigger
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============================================================================
-- 15. UPDATE USER STATS FUNCTION
-- ============================================================================

-- Function to update user stats after trade close
CREATE OR REPLACE FUNCTION public.update_user_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'closed' AND OLD.status != 'closed' THEN
        UPDATE public.user_profiles
        SET 
            total_trades = total_trades + 1,
            winning_trades = winning_trades + CASE WHEN NEW.net_pnl > 0 THEN 1 ELSE 0 END,
            total_pnl = total_pnl + COALESCE(NEW.net_pnl, 0),
            updated_at = NOW()
        WHERE id = NEW.user_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Drop existing trigger if exists
DROP TRIGGER IF EXISTS on_trade_closed ON trades;

-- Create trigger
CREATE TRIGGER on_trade_closed
    AFTER UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION public.update_user_stats();

-- ============================================================================
-- 16. BROKER CONNECTIONS TABLE (if not exists)
-- ============================================================================

CREATE TABLE IF NOT EXISTS broker_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    broker_name TEXT NOT NULL,
    status TEXT DEFAULT 'disconnected',
    account_id TEXT,
    access_token TEXT,  -- Encrypted
    refresh_token TEXT, -- Encrypted
    connected_at TIMESTAMPTZ,
    disconnected_at TIMESTAMPTZ,
    last_synced_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, broker_name)
);

-- ============================================================================
-- 17. PAYMENT ORDERS TABLE (if not exists)
-- ============================================================================

CREATE TABLE IF NOT EXISTS payment_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    razorpay_order_id TEXT UNIQUE NOT NULL,
    plan_id TEXT NOT NULL,
    billing_period TEXT NOT NULL,
    amount INTEGER NOT NULL,
    currency TEXT DEFAULT 'INR',
    status TEXT DEFAULT 'created',
    razorpay_payment_id TEXT,
    error_code TEXT,
    error_description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    paid_at TIMESTAMPTZ,
    failed_at TIMESTAMPTZ
);

-- ============================================================================
-- 18. PAYMENT TRANSACTIONS TABLE (if not exists)
-- ============================================================================

CREATE TABLE IF NOT EXISTS payment_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    razorpay_order_id TEXT,
    razorpay_payment_id TEXT,
    razorpay_refund_id TEXT,
    amount INTEGER NOT NULL,
    currency TEXT DEFAULT 'INR',
    status TEXT NOT NULL,
    plan_id TEXT,
    error_code TEXT,
    error_description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- 19. SUBSCRIPTION PLANS TABLE (seed data)
-- ============================================================================

CREATE TABLE IF NOT EXISTS subscription_plans (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT,
    price_monthly INTEGER DEFAULT 0,
    price_quarterly INTEGER DEFAULT 0,
    price_yearly INTEGER DEFAULT 0,
    max_signals_per_day INTEGER DEFAULT 2,
    max_positions INTEGER DEFAULT 2,
    max_capital INTEGER DEFAULT 10000000,
    signal_only BOOLEAN DEFAULT true,
    semi_auto BOOLEAN DEFAULT false,
    full_auto BOOLEAN DEFAULT false,
    equity_trading BOOLEAN DEFAULT true,
    futures_trading BOOLEAN DEFAULT false,
    options_trading BOOLEAN DEFAULT false,
    telegram_alerts BOOLEAN DEFAULT false,
    priority_support BOOLEAN DEFAULT false,
    api_access BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default plans (3 tiers: Free, Starter, Pro)
INSERT INTO subscription_plans (id, name, display_name, description, price_monthly, price_quarterly, price_yearly, max_signals_per_day, max_positions, max_capital, signal_only, semi_auto, full_auto, equity_trading, futures_trading, options_trading, telegram_alerts, priority_support, api_access)
VALUES
    ('free', 'free', 'Free', 'Get started with basic signals', 0, 0, 0, 5, 3, 10000000, true, false, false, true, false, false, false, false, false),
    ('starter', 'starter', 'Starter', 'For beginners starting their trading journey', 49900, 129900, 399900, 20, 5, 50000000, true, true, false, true, false, false, false, false, false),
    ('pro', 'pro', 'Pro', 'For serious traders who want full edge', 149900, 399900, 1199900, -1, 15, -1, true, true, true, true, true, true, true, true, true)
ON CONFLICT (id) DO UPDATE SET
    price_monthly = EXCLUDED.price_monthly,
    price_quarterly = EXCLUDED.price_quarterly,
    price_yearly = EXCLUDED.price_yearly;

-- Remove elite plan if it exists from previous migration
DELETE FROM subscription_plans WHERE id = 'elite';

-- ============================================================================
-- 20. GRANT PERMISSIONS
-- ============================================================================

GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated, service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated, service_role;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO anon, authenticated, service_role;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
