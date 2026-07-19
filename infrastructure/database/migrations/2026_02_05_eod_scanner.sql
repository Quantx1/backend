-- Add EOD scanner tables for daily universe + run logs

CREATE TABLE IF NOT EXISTS public.daily_universe (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    source TEXT DEFAULT 'pkscreener_github',
    scan_type TEXT DEFAULT 'swing',
    run_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (trade_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_daily_universe_date ON public.daily_universe(trade_date);
CREATE INDEX IF NOT EXISTS idx_daily_universe_symbol ON public.daily_universe(symbol);

CREATE TABLE IF NOT EXISTS public.eod_scan_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_date DATE NOT NULL,
    status TEXT DEFAULT 'running' CHECK (status IN ('running', 'success', 'failed')),
    source TEXT DEFAULT 'pkscreener_github',
    scan_type TEXT DEFAULT 'swing',
    min_price DECIMAL(12, 2),
    max_price DECIMAL(12, 2),
    min_volume BIGINT,
    candidate_count INTEGER DEFAULT 0,
    signal_count INTEGER DEFAULT 0,
    error TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_eod_scan_runs_date ON public.eod_scan_runs(trade_date);
CREATE INDEX IF NOT EXISTS idx_eod_scan_runs_status ON public.eod_scan_runs(status);
