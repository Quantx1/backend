"""Portfolio analytics (#19) — holdings correlation + actionable rebalancing.

`compute_correlation` (pairwise Pearson over return series) and
`rebalancing_suggestions` (deterministic: trim overweight, cut the weakest,
diversify a concentrated sector, de-risk a highly-correlated pair) are pure
(tested). The grounded rationale is layered on top, user-triggered + cached.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_correlation(returns_by_symbol: Dict[str, List[float]]) -> Dict[str, Any]:
    """Pairwise Pearson correlation across holdings' daily returns."""
    import numpy as np
    syms = [s for s, r in returns_by_symbol.items() if r and len(r) >= 20]
    if len(syms) < 2:
        return {"avg_corr": None, "pairs": [], "symbols": syms}
    n = min(len(returns_by_symbol[s]) for s in syms)
    mat = np.array([returns_by_symbol[s][-n:] for s in syms], dtype=float)
    corr = np.corrcoef(mat)
    pairs: List[Dict[str, Any]] = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            c = corr[i][j]
            if c == c:  # not NaN
                pairs.append({"a": syms[i], "b": syms[j], "corr": round(float(c), 2)})
    pairs.sort(key=lambda p: p["corr"], reverse=True)
    vals = [p["corr"] for p in pairs]
    return {"avg_corr": round(sum(vals) / len(vals), 2) if vals else None,
            "pairs": pairs[:8], "symbols": syms}


def rebalancing_suggestions(positions: List[Dict[str, Any]], *,
                            sector_by_symbol: Optional[Dict[str, str]] = None,
                            scores: Optional[Dict[str, int]] = None,
                            top_corr_pair: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Concrete actions: trim overweight, cut weakest, diversify a >40% sector,
    de-risk a >0.85-correlated pair. positions: [{symbol, weight(0..1)}]."""
    total = sum(p["weight"] for p in positions) or 1.0
    sugg: List[Dict[str, Any]] = []
    flagged = set()

    for p in sorted(positions, key=lambda x: x["weight"], reverse=True):
        w = p["weight"] / total
        if w >= 0.25:
            sugg.append({"action": "trim", "symbol": p["symbol"], "from_pct": round(w * 100),
                         "to_pct": 18, "reason": f"Overweight at {w * 100:.0f}% — trim toward an 18% cap."})
            flagged.add(p["symbol"])

    if scores:
        for p in sorted(positions, key=lambda x: scores.get(x["symbol"], 50))[:2]:
            sc = scores.get(p["symbol"], 50)
            if sc < 45 and p["symbol"] not in flagged:
                sugg.append({"action": "reduce", "symbol": p["symbol"], "from_pct": round(p["weight"] / total * 100),
                             "to_pct": None, "reason": f"Weakest holding (score {sc}/100) — reduce or replace."})
                flagged.add(p["symbol"])

    if sector_by_symbol:
        by_sec: Dict[str, float] = {}
        for p in positions:
            sec = sector_by_symbol.get(p["symbol"])
            if sec:
                by_sec[sec] = by_sec.get(sec, 0.0) + p["weight"] / total
        for sec, w in sorted(by_sec.items(), key=lambda kv: kv[1], reverse=True):
            if w >= 0.40:
                sugg.append({"action": "diversify", "symbol": None, "sector": sec,
                             "reason": f"{sec} is {w * 100:.0f}% of the book — add exposure outside it."})
                break

    if top_corr_pair and top_corr_pair.get("corr", 0) >= 0.85:
        a, b = top_corr_pair["a"], top_corr_pair["b"]
        sugg.append({"action": "de-risk", "symbol": None, "pair": [a, b],
                     "reason": f"{a} and {b} move together ({top_corr_pair['corr']}) — they don't diversify each other."})
    return sugg
