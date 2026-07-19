"""
Cross-sectional ranking helper.

QlibEngine produces one raw score per (symbol, date). Turning those into
integer ranks for the ``alpha_scores`` table is a small, isolated piece
— and small enough to unit-test without any LightGBM dependency.
"""

from __future__ import annotations

from typing import Dict, List


def rank_cross_section(scores: Dict[str, float]) -> List[dict]:
    """Convert {symbol: raw_score} to rank-annotated records, highest
    score = rank 1.

    Returns
    -------
    list[{"symbol", "qlib_score_raw", "qlib_rank"}]
        Sorted by rank ascending (best first).
    """
    if not scores:
        return []

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    rows: List[dict] = []
    for idx, (symbol, raw) in enumerate(ordered, start=1):
        rows.append({
            "symbol": symbol,
            "qlib_score_raw": round(float(raw), 6),
            "qlib_rank": idx,
        })
    return rows
