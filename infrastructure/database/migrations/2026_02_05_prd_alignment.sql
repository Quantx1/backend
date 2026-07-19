-- PRD Alignment Migration (Alpha/Beta)

-- User profile gating
ALTER TABLE public.user_profiles
  ADD COLUMN IF NOT EXISTS paper_trading_started_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE public.user_profiles
  ADD COLUMN IF NOT EXISTS live_trading_whitelisted BOOLEAN DEFAULT FALSE;
ALTER TABLE public.user_profiles
  ADD COLUMN IF NOT EXISTS kill_switch_active BOOLEAN DEFAULT FALSE;

-- Signals explainability
ALTER TABLE public.signals
  ADD COLUMN IF NOT EXISTS strategy_names TEXT[] DEFAULT ARRAY[]::TEXT[];
ALTER TABLE public.signals
  ADD COLUMN IF NOT EXISTS tft_prediction JSONB DEFAULT '{}'::jsonb;

-- Execution mode split
ALTER TABLE public.trades
  ADD COLUMN IF NOT EXISTS execution_mode TEXT DEFAULT 'paper';
ALTER TABLE public.positions
  ADD COLUMN IF NOT EXISTS execution_mode TEXT DEFAULT 'paper';

DO $$
BEGIN
  ALTER TABLE public.trades
    ADD CONSTRAINT trades_execution_mode_check CHECK (execution_mode IN ('paper','live'));
EXCEPTION WHEN duplicate_object THEN NULL;
END$$;

DO $$
BEGIN
  ALTER TABLE public.positions
    ADD CONSTRAINT positions_execution_mode_check CHECK (execution_mode IN ('paper','live'));
EXCEPTION WHEN duplicate_object THEN NULL;
END$$;

-- Broker connections table
CREATE TABLE IF NOT EXISTS public.broker_connections (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE,
  broker_name TEXT NOT NULL CHECK (broker_name IN ('zerodha', 'angelone', 'upstox')),
  status TEXT DEFAULT 'connected' CHECK (status IN ('connected', 'disconnected', 'error')),
  account_id TEXT,
  access_token TEXT,
  refresh_token TEXT,
  connected_at TIMESTAMPTZ DEFAULT NOW(),
  last_synced_at TIMESTAMPTZ,
  disconnected_at TIMESTAMPTZ,
  metadata JSONB DEFAULT '{}'::jsonb,
  UNIQUE(user_id, broker_name)
);

CREATE INDEX IF NOT EXISTS idx_broker_connections_user ON public.broker_connections(user_id);
CREATE INDEX IF NOT EXISTS idx_broker_connections_status ON public.broker_connections(status);

-- Stocks metadata
CREATE TABLE IF NOT EXISTS public.stocks (
  symbol TEXT PRIMARY KEY,
  name TEXT,
  sector TEXT,
  market_cap_cat TEXT CHECK (market_cap_cat IN ('large', 'mid', 'small')),
  nifty50 BOOLEAN DEFAULT FALSE,
  niftybank BOOLEAN DEFAULT FALSE,
  liquidity_score DECIMAL(6, 2),
  beta DECIMAL(6, 3),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stocks_sector ON public.stocks(sector);

-- Candles cache
CREATE TABLE IF NOT EXISTS public.candles (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  stock_symbol TEXT NOT NULL,
  exchange TEXT DEFAULT 'NSE',
  interval TEXT DEFAULT '1d',
  timestamp TIMESTAMPTZ NOT NULL,
  open DECIMAL(15, 4),
  high DECIMAL(15, 4),
  low DECIMAL(15, 4),
  close DECIMAL(15, 4),
  volume BIGINT,
  source TEXT DEFAULT 'yfinance',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(stock_symbol, interval, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol_time ON public.candles(stock_symbol, timestamp DESC);

-- Feature store
CREATE TABLE IF NOT EXISTS public.features (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  stock_symbol TEXT NOT NULL,
  interval TEXT DEFAULT '1d',
  timestamp TIMESTAMPTZ NOT NULL,
  feature_set TEXT DEFAULT 'prd_v1',
  features JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(stock_symbol, interval, timestamp, feature_set)
);

CREATE INDEX IF NOT EXISTS idx_features_symbol_time ON public.features(stock_symbol, timestamp DESC);

