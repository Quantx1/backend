from pathlib import Path
SQL = Path("infrastructure/database/migrations/2026_06_08_pr_f4_fundamentals.sql").read_text()

def test_defines_fundamentals_table():
    assert "CREATE TABLE IF NOT EXISTS public.fundamentals_history" in SQL
    assert "PRIMARY KEY (snapshot_date, symbol)" in SQL

def test_idempotent_plain_pk_rls():
    assert "COALESCE(" not in SQL
    assert "ENABLE ROW LEVEL SECURITY" in SQL
    assert "REVOKE ALL" in SQL
