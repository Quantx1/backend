# tests/data/reference/test_f3_migration_sql.py
from pathlib import Path
SQL = Path("infrastructure/database/migrations/2026_06_08_pr_f3_derivatives.sql").read_text()

def test_defines_derivatives_tables():
    for t in ["public.options_chain_eod", "public.futures_eod", "public.derivatives_metrics_eod"]:
        assert f"CREATE TABLE IF NOT EXISTS {t}" in SQL, t

def test_idempotent_plain_pk_rls():
    assert "COALESCE(" not in SQL
    assert SQL.count("ENABLE ROW LEVEL SECURITY") >= 3
    assert "REVOKE ALL" in SQL
