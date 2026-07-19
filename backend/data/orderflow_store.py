"""Upsert helpers for the F2 order-flow tables (honest-empty on failure)."""
from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def upsert_rows(supabase, table: str, rows: List[Dict], on_conflict: str) -> int:
    if not rows:
        return 0
    try:
        supabase.table(table).upsert(rows, on_conflict=on_conflict).execute()
        return len(rows)
    except Exception as e:
        logger.debug("upsert %s failed (%d rows): %s", table, len(rows), e)
        return 0
