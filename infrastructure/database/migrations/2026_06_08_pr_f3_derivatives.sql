-- F3: derivatives EOD (free nselib F&O bhavcopy, backend-only). Idempotent.
CREATE TABLE IF NOT EXISTS public.options_chain_eod (
    date DATE NOT NULL, symbol TEXT NOT NULL, expiry DATE NOT NULL,
    strike NUMERIC NOT NULL, option_type TEXT NOT NULL,      -- CE|PE
    oi BIGINT, oi_change BIGINT, volume BIGINT, ltp NUMERIC,
    source TEXT DEFAULT 'nselib', created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (date, symbol, expiry, strike, option_type)
);
CREATE INDEX IF NOT EXISTS idx_options_chain_sym_date ON public.options_chain_eod (symbol, date DESC);

CREATE TABLE IF NOT EXISTS public.futures_eod (
    date DATE NOT NULL, symbol TEXT NOT NULL, expiry DATE NOT NULL,
    open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC,
    oi BIGINT, oi_change BIGINT, volume BIGINT,
    source TEXT DEFAULT 'nselib', created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (date, symbol, expiry)
);
CREATE INDEX IF NOT EXISTS idx_futures_sym_date ON public.futures_eod (symbol, date DESC);

CREATE TABLE IF NOT EXISTS public.derivatives_metrics_eod (
    date DATE NOT NULL, symbol TEXT NOT NULL, expiry DATE NOT NULL,
    pcr_oi NUMERIC, pcr_volume NUMERIC, max_pain NUMERIC,
    total_ce_oi BIGINT, total_pe_oi BIGINT,
    source TEXT DEFAULT 'nselib', created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (date, symbol, expiry)
);
CREATE INDEX IF NOT EXISTS idx_deriv_metrics_sym_date ON public.derivatives_metrics_eod (symbol, date DESC);

ALTER TABLE public.options_chain_eod ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.futures_eod ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.derivatives_metrics_eod ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.options_chain_eod, public.futures_eod, public.derivatives_metrics_eod FROM anon, authenticated;
