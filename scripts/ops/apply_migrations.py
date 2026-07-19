#!/usr/bin/env python3
"""
Apply Quant X SQL migrations to Supabase Postgres — idempotent + ledger-tracked.

Why this exists: ``scripts/run_migration.py`` is hardcoded to the single
marketplace file and can't apply arbitrary migrations. This runner applies any
migration file(s) from ``infrastructure/database/migrations/`` in chronological
order, records each in the project's existing ``schema_migrations`` ledger, and
skips ones already recorded. Every migration uses ``IF NOT EXISTS`` guards, so
re-application is a safe no-op regardless of the ledger.

DDL needs a DIRECT Postgres connection — the Supabase *service key* CANNOT run
raw DDL. Get the connection string from:
    Supabase Dashboard -> Settings -> Database -> Connection string -> URI
Prefer the DIRECT connection (port 5432) over the transaction pooler (6543) for
DDL, then:

    export DATABASE_URL='postgresql://postgres.[ref]:[PWD]@db.[ref].supabase.co:5432/postgres'

Usage:
    # Apply the F0-F5 data-foundation batch (substring matches the 5 files):
    python scripts/ops/apply_migrations.py 2026_06_08_pr_f

    # Apply specific files (path, basename, or substring):
    python scripts/ops/apply_migrations.py 2026_06_08_pr_f0f1_reference_ohlc.sql 2026_06_08_pr_f2_orderflow

    # Apply everything not yet recorded in the ledger:
    python scripts/ops/apply_migrations.py --pending

    # Inspect:
    python scripts/ops/apply_migrations.py --list
    python scripts/ops/apply_migrations.py 2026_06_08_pr_f --dry-run
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("migrate")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MIG_DIR = os.path.join(ROOT, "infrastructure", "database", "migrations")
BOOTSTRAP = "000_schema_migrations.sql"

LEDGER_DDL = (
    "CREATE TABLE IF NOT EXISTS public.schema_migrations ("
    " id SERIAL PRIMARY KEY, version TEXT UNIQUE NOT NULL,"
    " description TEXT, applied_at TIMESTAMPTZ DEFAULT NOW())"
)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)


def _conn_str() -> str | None:
    return os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_DB_URL")


def _all_files() -> list[str]:
    return sorted(
        p for p in glob.glob(os.path.join(MIG_DIR, "*.sql"))
        if os.path.basename(p) != BOOTSTRAP
    )


def _version(path: str) -> str:
    return os.path.basename(path)[:-4]  # strip ".sql"


def _resolve(tokens: list[str]) -> list[str]:
    """Resolve file tokens (absolute path / basename / substring / glob) to paths."""
    out: list[str] = []
    for tok in tokens:
        if os.path.isfile(tok):
            out.append(os.path.abspath(tok))
            continue
        cand = os.path.join(MIG_DIR, tok if tok.endswith(".sql") else tok + ".sql")
        if os.path.isfile(cand):
            out.append(cand)
            continue
        matches = [
            p for p in _all_files()
            if tok in os.path.basename(p) or fnmatch.fnmatch(os.path.basename(p), tok)
        ]
        if not matches:
            log.error("No migration matches %r", tok)
            sys.exit(2)
        out.extend(matches)
    # de-dup, keep chronological (filename sorts by date prefix)
    return sorted(dict.fromkeys(out), key=os.path.basename)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply Quant X migrations (idempotent, ledger-tracked).")
    ap.add_argument("files", nargs="*", help="migration files: path, basename, or substring")
    ap.add_argument("--pending", action="store_true", help="apply every dated file not yet in schema_migrations")
    ap.add_argument("--list", action="store_true", help="list migrations + applied status, then exit")
    ap.add_argument("--dry-run", action="store_true", help="print the plan without executing")
    ap.add_argument("--force", action="store_true", help="apply even if already recorded in the ledger")
    args = ap.parse_args()
    _load_env()

    conn_str = _conn_str()
    conn = None
    applied: set[str] = set()

    if conn_str:
        try:
            import psycopg2
        except ImportError:
            log.error("psycopg2 not installed. Run: pip install psycopg2-binary")
            sys.exit(1)
        conn = psycopg2.connect(conn_str)
        conn.autocommit = False
        with conn.cursor() as cur:           # ensure the ledger exists
            cur.execute(LEDGER_DDL)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT version FROM public.schema_migrations")
            applied = {r[0] for r in cur.fetchall()}

    # Resolve which files to act on.
    if args.files:
        targets = _resolve(args.files)
    elif args.pending:
        targets = [p for p in _all_files() if _version(p) not in applied]
    elif args.list:
        targets = _all_files()
    else:
        ap.error("specify migration file(s), or --pending, or --list")
        return  # unreachable

    if args.list:
        log.info("%d migration file(s):", len(targets))
        for p in targets:
            mark = "applied" if _version(p) in applied else "PENDING"
            print(f"  [{mark:7s}] {os.path.basename(p)}")
        if not conn_str:
            log.warning("(no DATABASE_URL set — 'applied' status is unknown; all shown PENDING)")
        if conn:
            conn.close()
        return

    if not targets:
        log.info("Nothing to apply.")
        if conn:
            conn.close()
        return

    log.info("Plan (%d):", len(targets))
    for p in targets:
        if _version(p) in applied and not args.force:
            state = "skip"
        elif _version(p) in applied:
            state = "FORCE"
        else:
            state = "apply"
        log.info("  [%-5s] %s", state, os.path.basename(p))

    if args.dry_run:
        log.info("DRY RUN — nothing executed.")
        if conn:
            conn.close()
        return

    if not conn_str:
        log.error(
            "No Postgres connection. DDL needs a DIRECT connection (the service key cannot run DDL).\n"
            "  Supabase -> Settings -> Database -> Connection string -> URI (prefer port 5432):\n"
            "  export DATABASE_URL='postgresql://postgres.[ref]:[PWD]@db.[ref].supabase.co:5432/postgres'"
        )
        sys.exit(1)

    n_applied = 0
    try:
        for p in targets:
            ver, name = _version(p), os.path.basename(p)
            if ver in applied and not args.force:
                log.info("SKIP  %s (already applied)", name)
                continue
            with open(p, encoding="utf-8") as f:
                sql = f.read()
            log.info("APPLY %s (%d bytes)...", name, len(sql))
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    cur.execute(
                        "INSERT INTO public.schema_migrations(version, description) VALUES (%s, %s) "
                        "ON CONFLICT (version) DO UPDATE SET applied_at = NOW()",
                        (ver, "applied via scripts/ops/apply_migrations.py"),
                    )
                conn.commit()
                applied.add(ver)
                n_applied += 1
                log.info("  OK   %s", name)
            except Exception as exc:           # noqa: BLE001 — surface + roll back, then stop
                conn.rollback()
                log.error("  FAIL %s — rolled back, stopping: %s", name, exc)
                sys.exit(1)
        log.info("Done. Applied %d migration(s).", n_applied)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
