#!/usr/bin/env python3
"""Resolve a COMPLETE NSE ticker -> brand-domain map via logo.dev Brand Search.

Reads company names from the `instruments` table (instrument_type='EQ'),
resolves each name to its brand domain, and writes
`frontend/lib/symbol-domains.json` (ticker -> domain). The frontend then
renders real logos for every resolvable ticker (logo.dev by domain), with a
monogram fallback for the rest.

Resumable: keeps already-resolved entries, so it can be re-run / survive a
rate-limit interruption. The logo.dev SECRET key is read from LOGODEV_SECRET
and is NEVER hard-coded or committed.

Usage:
  LOGODEV_SECRET=sk_xxx .venv/bin/python scripts/data/resolve_logos.py
"""
import os
import sys
import json
import time
import urllib.parse
import urllib.request
import urllib.error
import concurrent.futures

sys.path.insert(0, os.getcwd())
try:
    from dotenv import load_dotenv
    load_dotenv('.env')
    load_dotenv('backend/.env')
except Exception:
    pass
from backend.data.ohlc_store import pg_connect

SK = os.environ.get('LOGODEV_SECRET', '')
assert SK.startswith('sk_'), 'set LOGODEV_SECRET=sk_... (the logo.dev secret key)'
OUT = 'frontend/lib/symbol-domains.json'


def fetch_names():
    c = pg_connect()
    cur = c.cursor()
    cur.execute(
        "SELECT symbol, name FROM instruments "
        "WHERE instrument_type='EQ' AND name IS NOT NULL"
    )
    rows = cur.fetchall()
    c.close()
    return {r[0].strip().upper(): (r[1] or '').strip() for r in rows if r[0]}


# TLD signals used only to re-rank results that ALREADY have a name match.
_IN_TLDS = ('.in', '.co.in', '.net.in', '.org.in', '.ind.in', '.firm.in')
_FOREIGN_TLDS = ('.com.br', '.co.uk', '.de', '.fr', '.ru', '.cn', '.jp', '.au',
                 '.com.au', '.ca', '.es', '.it', '.nl', '.pl', '.tr', '.id',
                 '.my', '.sg', '.ph', '.vn', '.kr', '.tw', '.mx', '.ar')
_STOP = {'limited', 'ltd', 'the', 'india', 'indian', 'company', 'co', 'corp',
         'corporation', 'inc', 'plc', 'and', 'of', 'pvt', 'private', 'group'}


def _tokens(s):
    out, cur = [], []
    for ch in (s or '').lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append(''.join(cur))
            cur = []
    if cur:
        out.append(''.join(cur))
    return out


def _clean(s):
    """Alnum-squash with common suffix words removed (for ==/startswith tests)."""
    return ''.join(t for t in _tokens(s) if t not in _STOP)


def _query(name):
    """Search query = brand tokens with corporate-suffix words dropped, so
    'Infosys Limited' searches as 'infosys' (-> infosys.com, not the literal
    'infosyslimited.com' spam match) and 'Tata Motors Limited' as 'tata motors'."""
    toks = _tokens(name)
    kept = [t for t in toks if t not in _STOP] or toks
    return ' '.join(kept)


def _best_domain(results, company):
    """Pick the result whose brand name best matches the Indian company name.
    Returns None when NO result shows real name evidence -> the frontend falls
    back to a monogram, which beats a confidently-wrong foreign logo."""
    cn = _clean(company)
    ctoks = {t for t in _tokens(company) if t not in _STOP}
    best, best_score = None, 0
    for r in results:
        dom = (r.get('domain') or '').lower()
        if not dom:
            continue
        rn = _clean(r.get('name') or '')
        stem = dom.split('.')[0]
        score, evidence = 0, False
        if rn and cn:
            if rn == cn:
                score += 100
                evidence = True
            elif len(rn) >= 4 and (cn.startswith(rn) or rn.startswith(cn)):
                score += 70
                evidence = True
            else:
                overlap = len({t for t in _tokens(r.get('name') or '')} & ctoks)
                if overlap:
                    score += 25 * overlap
                    evidence = True
        if len(stem) >= 3 and stem == cn:
            score += 60
            evidence = True
        elif len(stem) >= 4 and (stem in cn or cn.startswith(stem)):
            score += 35
            evidence = True
        if not evidence:
            continue
        # A foreign-country TLD on an Indian NSE listing is almost always a
        # wrong-company match (IRFC->railwayfinance.co.uk) -> skip it; if nothing
        # else matches we return None and the frontend shows a monogram.
        if dom.endswith(_FOREIGN_TLDS):
            continue
        # Indian-TLD is only a weak tiebreaker; never enough to beat an exact
        # domain-stem match (hdfcbank.com must win over hdfc.bank.in).
        if dom.endswith(_IN_TLDS):
            score += 8
        # Prefer the cleaner/shorter domain on ties (infosys.com > infosyslimited.com).
        if score > best_score or (score == best_score and best and len(dom) < len(best)):
            best, best_score = dom, score
    return best


def search(name):
    q = urllib.parse.quote(name)
    req = urllib.request.Request(
        f'https://api.logo.dev/search?q={q}',
        headers={
            'Authorization': f'Bearer {SK}',
            # logo.dev sits behind Cloudflare, which 403s the default urllib UA.
            'User-Agent': 'Mozilla/5.0 (compatible; QuantX-logo-resolver/1.0)',
        },
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r) or []
        except urllib.error.HTTPError as e:
            # 403 = Cloudflare rate-limit under concurrency; back off and retry.
            if e.code in (429, 503, 403):
                time.sleep(1.5 * (attempt + 1))
                continue
            return []
        except Exception:
            time.sleep(1)
            continue
    return []


def resolve(item):
    sym, name = item
    name = name or sym
    return sym, _best_domain(search(_query(name)), name)


def main():
    existing = {}
    if os.path.exists(OUT):
        try:
            existing = json.load(open(OUT))
        except Exception:
            existing = {}
    names = fetch_names()
    todo = [(s, n) for s, n in names.items() if s not in existing]
    print(f'{len(names)} EQ names; {len(existing)} cached; {len(todo)} to resolve', flush=True)
    res = dict(existing)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for sym, dom in ex.map(resolve, todo):
            done += 1
            if dom:
                res[sym] = dom
            if done % 100 == 0:
                json.dump(dict(sorted(res.items())), open(OUT, 'w'), separators=(',', ':'))
                print(f'{done}/{len(todo)}  resolved_total={len(res)}', flush=True)
    json.dump(dict(sorted(res.items())), open(OUT, 'w'), separators=(',', ':'))
    print(f'DONE wrote {OUT}: {len(res)} domains', flush=True)


if __name__ == '__main__':
    main()
