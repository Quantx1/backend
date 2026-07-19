"""
Finance sentiment lexicon — a transparent, zero-cost cross-check model.

A compact Loughran-McDonald-style polarity word list for financial headlines.
This is NOT a fallback masquerading as the trained model (the no-fallback rule
targets silent substitution) — it is an EXPLICIT, separately-labelled third
opinion in the sentiment ensemble, shown to the user as "lexicon". The LLM
classifier stays primary; this adds an instant, deterministic, offline cross-
check so the UI can show model AGREEMENT ("3/3 models agree").

Pure, 0 tokens, no network, no model download.
"""

from __future__ import annotations

import re
from typing import Dict

_POSITIVE = frozenset("""
beat beats beating surge surged surges jump jumps jumped soar soared rally
rallies rallied gain gains gained rise rises rose climb climbs climbed
record high highs profit profitable growth grew grows upgrade upgraded
upbeat outperform outperforms bullish strong stronger strength robust
expansion expand expands wins win won bags bagged order orders deal deals
breakthrough approval approved boost boosts boosted recovery rebound
rebounds dividend bonus buyback acquire acquires acquisition stake raises
raised raise top-line bottom-line beat-estimates multibagger rerating
""".split())

_NEGATIVE = frozenset("""
miss misses missed fall falls fell drop drops dropped plunge plunged slump
slumped crash crashed tumble tumbled decline declines declined loss losses
weak weaker weakness downgrade downgraded bearish cut cuts slashed slash
probe investigation fraud scam default defaults lawsuit sue sued penalty
fined fine ban banned recall halt halted resign resigns resigned exit
warning warns warned concern concerns risk risky debt distress downturn
selloff sell-off underperform underperforms layoff layoffs shut shutdown
stake-sale offload dumps dumped raid sebi-probe insolvency bankruptcy
""".split())

_WORD_RE = re.compile(r"[a-z][a-z-]+")


def lexicon_score(title: str) -> float:
    """Polarity in [-1, 1] from finance term counts. 0 when no terms hit."""
    toks = _WORD_RE.findall((title or "").lower())
    pos = sum(1 for t in toks if t in _POSITIVE)
    neg = sum(1 for t in toks if t in _NEGATIVE)
    if pos == 0 and neg == 0:
        return 0.0
    return round((pos - neg) / (pos + neg), 4)


def lexicon_label(title: str) -> str:
    s = lexicon_score(title)
    if s > 0.0:
        return "positive"
    if s < 0.0:
        return "negative"
    return "neutral"


def scores_for(titles) -> Dict[str, float]:
    return {t: lexicon_score(t) for t in titles}


__all__ = ["lexicon_score", "lexicon_label", "scores_for"]
