-- F4: fundamentals history (free screener.in source, backend-only). Idempotent.
CREATE TABLE IF NOT EXISTS public.fundamentals_history (
    snapshot_date DATE NOT NULL, symbol TEXT NOT NULL,
    pe NUMERIC, roe NUMERIC, roce NUMERIC, market_cap_cr NUMERIC,
    book_value NUMERIC, dividend_yield NUMERIC, current_price NUMERIC,
    debt_to_equity NUMERIC, eps NUMERIC,
    sales_growth NUMERIC, profit_growth NUMERIC, promoter_pct NUMERIC,
    source TEXT DEFAULT 'screener.in', created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (snapshot_date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol ON public.fundamentals_history (symbol, snapshot_date DESC);

ALTER TABLE public.fundamentals_history ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.fundamentals_history FROM anon, authenticated;
