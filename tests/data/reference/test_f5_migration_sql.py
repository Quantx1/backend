from pathlib import Path
SQL = Path("infrastructure/database/migrations/2026_06_08_pr_f5_news_items.sql").read_text()

def test_defines_news_items():
    assert "CREATE TABLE IF NOT EXISTS public.news_items" in SQL
    assert "PRIMARY KEY (trade_date, symbol, url_hash)" in SQL

def test_idempotent_plain_pk_rls():
    assert "COALESCE(" not in SQL
    assert "ENABLE ROW LEVEL SECURITY" in SQL
    assert "REVOKE ALL" in SQL
