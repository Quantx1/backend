-- ============================================================================
-- P2 MIGRATION: Missing Indexes & Race-Condition Constraints
-- Run via Supabase SQL editor or psql
-- ============================================================================

-- -------------------------------------------------------
-- 1. Composite indexes on hot query paths
-- -------------------------------------------------------

-- Trades: user + created_at (portfolio history, recent trades)
CREATE INDEX IF NOT EXISTS idx_trades_user_created
    ON public.trades(user_id, created_at DESC);

-- Trades: user + status (open/pending trades lookup)
CREATE INDEX IF NOT EXISTS idx_trades_user_status
    ON public.trades(user_id, status);

-- Trades: symbol + status (signal-to-trade join)
CREATE INDEX IF NOT EXISTS idx_trades_symbol_status
    ON public.trades(symbol, status);

-- Positions: user + is_active + symbol (position lookup)
CREATE INDEX IF NOT EXISTS idx_positions_user_active_symbol
    ON public.positions(user_id, is_active, symbol);

-- Signals: date + status (today's active signals — powers v_today_signals view)
CREATE INDEX IF NOT EXISTS idx_signals_date_status
    ON public.signals(date, status);

-- Signals: symbol + date (signal history per stock)
CREATE INDEX IF NOT EXISTS idx_signals_symbol_date
    ON public.signals(symbol, date DESC);

-- Payments: user + created_at (payment history)
CREATE INDEX IF NOT EXISTS idx_payments_user_created
    ON public.payments(user_id, created_at DESC);

-- Payments: user + status (active subscription lookup)
CREATE INDEX IF NOT EXISTS idx_payments_user_status
    ON public.payments(user_id, status);

-- Notifications: user + created_at (notification feed)
CREATE INDEX IF NOT EXISTS idx_notifications_user_created
    ON public.notifications(user_id, created_at DESC);

-- Watchlist: user_id (user's watchlist)
CREATE INDEX IF NOT EXISTS idx_watchlist_user
    ON public.watchlist(user_id);

-- User profiles: subscription_status (admin queries)
CREATE INDEX IF NOT EXISTS idx_user_profiles_sub_status
    ON public.user_profiles(subscription_status);

-- -------------------------------------------------------
-- 2. UNIQUE constraint on razorpay_order_id (prevents payment race condition)
-- -------------------------------------------------------

-- Add UNIQUE constraint so concurrent webhook + verify cannot double-process
-- Using CREATE UNIQUE INDEX to be idempotent with IF NOT EXISTS
CREATE UNIQUE INDEX IF NOT EXISTS idx_payments_razorpay_order_id_unique
    ON public.payments(razorpay_order_id)
    WHERE razorpay_order_id IS NOT NULL;

-- Also add unique on razorpay_payment_id for refund lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_payments_razorpay_payment_id_unique
    ON public.payments(razorpay_payment_id)
    WHERE razorpay_payment_id IS NOT NULL;

-- -------------------------------------------------------
-- 3. Assistant credit persistence columns on user_profiles
-- -------------------------------------------------------

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS assistant_credits_used INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS assistant_credits_date DATE;
