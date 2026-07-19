-- F5: forward news_items corpus (free RSS/Google-News + classifier). Idempotent.
CREATE TABLE IF NOT EXISTS public.news_items (
    trade_date DATE NOT NULL, symbol TEXT NOT NULL, url_hash TEXT NOT NULL,
    title TEXT, url TEXT, source TEXT, published_at TEXT,
    sentiment_label TEXT, sentiment_score NUMERIC,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (trade_date, symbol, url_hash)
);
CREATE INDEX IF NOT EXISTS idx_news_items_symbol_date ON public.news_items (symbol, trade_date DESC);

ALTER TABLE public.news_items ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.news_items FROM anon, authenticated;
