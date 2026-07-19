#!/usr/bin/env python3
"""
Schema-consolidation audit (P1-6).

For every per-PR migration in ``infrastructure/database/migrations/``
that's NOT one of the base files, extract the schema objects it
creates / alters and check whether ``complete_schema.sql`` already
contains the same form.

Outputs a report grouped by migration with rows like:
    [OK]      pr1_security  user_profiles.is_admin
    [MISSING] pr27_…        user_profiles.subscription_renewal_notified_at

Exits non-zero if any object is missing — so this can run in CI.

Heuristics used (good enough for our schema; not a full SQL parser):
  * ``CREATE TABLE ...``                          → table name
  * ``ALTER TABLE ... ADD COLUMN [IF NOT EXISTS] X`` → table + column
  * ``CREATE INDEX [IF NOT EXISTS] name ...``     → index name

We accept the consolidated schema mentioning the column/table anywhere
(case-insensitive, comments stripped). Indexes are matched on name only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "infrastructure" / "database" / "migrations"
CONSOLIDATED = REPO_ROOT / "infrastructure" / "database" / "complete_schema.sql"

# Files NOT to audit — the consolidated schema itself + bootstrap.
SKIP_FILES = {"000_schema_migrations.sql"}

# The base migrations recorded in schema_migrations are pre-consolidation —
# their content is already cumulative in complete_schema.sql by definition
# (those rows seed the tracker). We still audit any post-consolidation
# PR migration.

CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:public\.)?(\w+)",
    re.IGNORECASE,
)
ADD_COLUMN_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:public\.)?(\w+)"
    r"(?:\s+ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+))?",
    re.IGNORECASE,
)
ADD_COLUMN_INLINE_RE = re.compile(
    r"ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
    re.IGNORECASE,
)
CREATE_INDEX_RE = re.compile(
    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?"
    r"(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
    re.IGNORECASE,
)


def strip_comments(sql: str) -> str:
    """Remove `--` line comments and `/* */` blocks.

    Line comments are stripped FIRST so a `/*` or `*/` fragment that happens to
    live inside a `--` comment (e.g. a ``/api/*`` path or ``delivery_*/adj_close``)
    can't be mistaken for a block-comment delimiter and silently delete a whole
    region of the schema.
    """
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def parse_migration(path: Path) -> List[Tuple[str, str]]:
    """Return [(kind, identifier)] tuples for one migration file.
    kind is one of: ``table``, ``column:<table>``, ``index``."""
    text = strip_comments(path.read_text(encoding="utf-8"))
    items: List[Tuple[str, str]] = []

    for m in CREATE_TABLE_RE.finditer(text):
        items.append(("table", m.group(1).lower()))

    # ALTER TABLE blocks may add multiple columns. Walk each ALTER
    # statement up to the next semicolon and pull every ADD COLUMN.
    for stmt in re.split(r";", text):
        if "alter table" not in stmt.lower():
            continue
        m_table = re.search(
            r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:public\.)?(\w+)",
            stmt, re.IGNORECASE,
        )
        if not m_table:
            continue
        table = m_table.group(1).lower()
        for m_col in ADD_COLUMN_INLINE_RE.finditer(stmt):
            col = m_col.group(1).lower()
            items.append((f"column:{table}", col))

    for m in CREATE_INDEX_RE.finditer(text):
        items.append(("index", m.group(1).lower()))

    return items


def consolidated_schema_text() -> str:
    return strip_comments(CONSOLIDATED.read_text(encoding="utf-8")).lower()


def audit() -> int:
    consolidated = consolidated_schema_text()
    files = sorted(p for p in MIGRATIONS_DIR.iterdir()
                   if p.name.endswith(".sql") and p.name not in SKIP_FILES)

    total = 0
    missing = 0
    rows: List[str] = []

    for f in files:
        items = parse_migration(f)
        if not items:
            continue
        for kind, ident in items:
            total += 1
            if kind == "table":
                # Just confirm the table name appears in a CREATE TABLE
                # form in the consolidated schema.
                ok = re.search(
                    rf"create\s+table\s+(?:if\s+not\s+exists\s+)?"
                    rf"(?:public\.)?{re.escape(ident)}\b",
                    consolidated,
                ) is not None
                obj = ident
            elif kind.startswith("column:"):
                table = kind.split(":", 1)[1]
                # Match the column near the table — either inside the
                # CREATE TABLE block or as an ALTER ... ADD COLUMN. We
                # use a permissive scan: column name must appear, and
                # the table name must appear somewhere reasonable.
                col_seen = re.search(
                    rf"\b{re.escape(ident)}\b", consolidated,
                ) is not None
                table_seen = re.search(
                    rf"\b{re.escape(table)}\b", consolidated,
                ) is not None
                ok = col_seen and table_seen
                obj = f"{table}.{ident}"
            else:  # index
                ok = re.search(
                    rf"\b{re.escape(ident)}\b", consolidated,
                ) is not None
                obj = ident
            tag = "[OK]" if ok else "[MISSING]"
            rows.append(f"{tag:11s} {f.name:55s} {kind:20s} {obj}")
            if not ok:
                missing += 1

    print(f"Schema audit: {total} objects across "
          f"{sum(1 for f in files)} migration files")
    print(f"  {missing} missing from complete_schema.sql\n")
    for r in rows:
        if r.startswith("[MISSING]"):
            print(r)
    if missing == 0:
        print("All PR-migration objects present in consolidated schema.")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(audit())
