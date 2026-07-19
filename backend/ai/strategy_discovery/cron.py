"""Discovery cron handlers — nightly batch + email digest.

Schedule (registered in scheduler.py):
  22:30 IST Mon-Fri — `nightly_discovery()`
       Runs 4 discovery batches back-to-back (equity_swing, equity_position,
       fo_weekly, fo_monthly) on a small budget; each batch persists
       candidates into Supabase like a user-triggered run.
  07:30 IST Tue-Sat — `morning_digest()`
       Emails Elite + admin users a summary of overnight discoveries:
       count per kind + top-3 candidates by score with a deep link to
       the Discovered tab.

Both jobs are safe to retry — they create new rows each invocation.
The digest is idempotent within a calendar day via a metadata flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from backend.ai.strategy_discovery import DiscoveryConfig, run_discovery
from backend.core.database import get_supabase_admin

logger = logging.getLogger(__name__)


# Cron-mode batch size — small so the whole sweep finishes inside the
# market-closed window without hammering the data API.
_CRON_DEFAULTS: Dict[str, DiscoveryConfig] = {
    "equity_swing": DiscoveryConfig(
        kind="equity_swing", mode="ga",
        universe="nifty50", symbols_per_candidate=5,
        history_period="3y", seed=0,         # seed=0 → freshly random each night
        ga_pop_size=10, ga_generations=3, ga_elite=3, ga_children_per_elite=2,
        walk_forward_folds=3,
    ),
    "equity_position": DiscoveryConfig(
        kind="equity_position", mode="ga",
        universe="nifty100", symbols_per_candidate=6,
        history_period="5y", seed=0,
        ga_pop_size=10, ga_generations=3, ga_elite=3, ga_children_per_elite=2,
        walk_forward_folds=3,
    ),
    "fo_weekly": DiscoveryConfig(
        kind="fo_weekly", mode="random",
        universe="NIFTY", sample_size=12, history_period="2y", seed=0,
        walk_forward_folds=3,
    ),
    "fo_monthly": DiscoveryConfig(
        kind="fo_monthly", mode="random",
        universe="NIFTY", sample_size=12, history_period="2y", seed=0,
        walk_forward_folds=3,
    ),
    # PR-H1 — intraday batches use 30-day windows (yfinance free-tier
    # limits 5m/15m data to ~60 days; we keep it conservative). Smaller
    # universe to stay within the nightly time budget.
    "intraday_5m": DiscoveryConfig(
        kind="intraday_5m", mode="random",
        universe="nifty50", symbols_per_candidate=4,
        sample_size=10, history_period="30d", seed=0,
        walk_forward_folds=0,                  # short window = no WF
    ),
    "intraday_15m": DiscoveryConfig(
        kind="intraday_15m", mode="random",
        universe="nifty50", symbols_per_candidate=4,
        sample_size=10, history_period="60d", seed=0,
        walk_forward_folds=0,
    ),
}


def nightly_discovery() -> Dict[str, Any]:
    """Run all 4 kinds back-to-back. Returns a summary for telemetry.

    Each batch's failure is isolated — one borked search doesn't kill
    the others.
    """
    import random
    seed = random.randint(1, 999_999)
    logger.info("nightly discovery starting (seed=%d)", seed)
    summary: Dict[str, Any] = {"started_at": datetime.now(timezone.utc).isoformat(),
                               "results": {}}
    for kind, base_cfg in _CRON_DEFAULTS.items():
        cfg = DiscoveryConfig(**{**base_cfg.to_dict(), "seed": seed + hash(kind) % 1000})
        try:
            run_id = run_discovery(cfg)
            summary["results"][kind] = {"run_id": str(run_id), "status": "ok"}
            logger.info("nightly discovery %s → run %s", kind, run_id)
        except Exception as e:
            logger.exception("nightly discovery %s failed: %s", kind, e)
            summary["results"][kind] = {"status": "failed", "error": str(e)[:200]}
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    return summary


@dataclass
class DigestEntry:
    """One row in the morning digest email."""
    kind: str
    run_id: str
    total: int
    viable: int
    best_score: Optional[float]
    top_candidates: List[Dict[str, Any]]      # [{label, score, sharpe, max_dd, ...}]


def _format_digest_html(entries: List[DigestEntry], app_base_url: str) -> str:
    """Render an HTML email body. Plain inline styles — most email clients
    strip <style> blocks and external CSS."""
    rows = []
    for e in entries:
        if e.total == 0:
            rows.append(f"""
            <tr><td style="padding:12px;border-top:1px solid #E5E1D5;">
              <strong>{e.kind}</strong> — no candidates produced
            </td></tr>""")
            continue
        kind_label = {
            "equity_swing": "Equity · Swing",
            "equity_position": "Equity · Position",
            "fo_weekly": "F&amp;O · Weekly",
            "fo_monthly": "F&amp;O · Monthly",
        }.get(e.kind, e.kind)
        top_html = "".join(
            f"<li>{c['label'][:90]} — score <b>{c['score']:.2f}</b>"
            f" · Sharpe {c.get('sharpe', 0):.2f}"
            f" · trades {c.get('trade_count', 0)}</li>"
            for c in e.top_candidates
        )
        rows.append(f"""
        <tr><td style="padding:14px;border-top:1px solid #E5E1D5;">
          <div style="font-weight:600;color:#0A0D14;">
            {kind_label} — {e.viable}/{e.total} viable · best {e.best_score:.2f if e.best_score is not None else 0.0}
          </div>
          <ul style="margin:8px 0 0 18px;padding:0;color:#5B5F6B;font-size:13px;line-height:1.5;">
            {top_html}
          </ul>
          <a href="{app_base_url}/strategies?tab=discovered&run={e.run_id}"
             style="display:inline-block;margin-top:10px;padding:6px 12px;
                    background:#3D80FF;color:#fff;text-decoration:none;
                    border-radius:6px;font-size:12px;">
            Open run →
          </a>
        </td></tr>""")

    return f"""<!doctype html>
<html><body style="margin:0;padding:24px;background:#F7F5F0;font-family:-apple-system,sans-serif;">
  <table style="max-width:600px;margin:0 auto;background:#fff;border-radius:8px;border:1px solid #E5E1D5;">
    <tr><td style="padding:18px 20px;background:#0A0D14;color:#fff;border-radius:8px 8px 0 0;">
      <div style="font-size:18px;font-weight:600;">Overnight Discoveries — Quant X</div>
      <div style="font-size:12px;color:#8B92A5;margin-top:4px;">
        {datetime.now().strftime('%A, %d %b %Y')}
      </div>
    </td></tr>
    {''.join(rows)}
    <tr><td style="padding:14px 20px;border-top:1px solid #E5E1D5;
                   color:#8B92A5;font-size:11px;">
      Promote a candidate from the Discovered tab to send it to Paper or Live.
    </td></tr>
  </table>
</body></html>"""


def morning_digest(app_base_url: str = "https://quantx.app") -> Dict[str, Any]:
    """Summarise the last 18 hours of discovery runs + email Elite users.

    Idempotent within a calendar day — checks for an existing digest
    metadata stamp before sending.
    """
    sb = get_supabase_admin()

    today_iso = datetime.now(timezone.utc).date().isoformat()
    # Check for prior dispatch today via metadata stamp on a sentinel row.
    # We piggy-back on strategy_search_runs.metadata — one "digest" row
    # per day, kind='digest_marker'? Actually that violates the CHECK
    # constraint on kind. Use a tiny in-process guard instead: if any
    # of the last 4 runs already has metadata.digest_sent=true skip.
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=18)).isoformat()
    runs_res = sb.table("strategy_search_runs").select(
        "id,kind,status,candidates_total,candidates_viable,best_score,"
        "best_candidate_id,started_at,completed_at,metadata"
    ).gte("started_at", cutoff).eq("status", "completed").execute()

    runs = runs_res.data or []
    if not runs:
        logger.info("morning_digest: nothing to send (no completed runs in last 18h)")
        return {"sent": False, "reason": "no recent runs"}

    if any((r.get("metadata") or {}).get("digest_sent_at", "").startswith(today_iso)
           for r in runs):
        logger.info("morning_digest: already sent today — skipping")
        return {"sent": False, "reason": "already sent today"}

    # Build digest entries — one per kind (latest run wins on tie).
    by_kind: Dict[str, Dict[str, Any]] = {}
    for r in runs:
        k = r["kind"]
        if k not in by_kind or r["started_at"] > by_kind[k]["started_at"]:
            by_kind[k] = r

    entries: List[DigestEntry] = []
    for kind, run in by_kind.items():
        # Top-3 candidates by score
        c_res = sb.table("discovered_strategies").select(
            "label,score,sharpe,max_drawdown_pct,trade_count"
        ).eq("run_id", run["id"]).eq("status", "candidate").order(
            "score", desc=True
        ).limit(3).execute()
        entries.append(DigestEntry(
            kind=kind,
            run_id=run["id"],
            total=run["candidates_total"] or 0,
            viable=run["candidates_viable"] or 0,
            best_score=run.get("best_score"),
            top_candidates=c_res.data or [],
        ))

    # Recipients — Elite users + admin
    users_res = sb.table("user_profiles").select(
        "user_id,email,full_name,tier,is_admin"
    ).or_("tier.eq.elite,is_admin.eq.true").execute()
    recipients = [u for u in (users_res.data or [])
                  if u.get("email")]
    if not recipients:
        logger.info("morning_digest: no recipients with email")
        return {"sent": False, "reason": "no recipients"}

    html = _format_digest_html(entries, app_base_url)
    subject = (
        f"Overnight Discoveries — "
        f"{sum(e.viable for e in entries)} viable strategies"
    )

    sent_count = 0
    try:
        import asyncio
        import os
        from backend.platform.push import EmailService
        api_key = os.environ.get("RESEND_API_KEY", "")
        from_email = os.environ.get(
            "EMAIL_FROM", "Quant X <no-reply@quantx.app>",
        )
        emailer = EmailService(api_key=api_key, from_email=from_email)
        if not emailer.is_available:
            logger.warning("morning_digest: EmailService not configured")
            return {"sent": False, "reason": "RESEND_API_KEY missing"}

        # EmailService.send is async; we're in a sync cron handler so
        # drive it via a fresh event loop. Per-recipient try/except so
        # a single bad address doesn't drop the whole digest.
        async def _send_all() -> int:
            n = 0
            for u in recipients:
                try:
                    ok = await emailer.send(to=u["email"], subject=subject, html=html)
                    if ok:
                        n += 1
                except Exception as e:
                    logger.warning(
                        "digest email failed for %s: %s", u.get("email"), e,
                    )
            return n

        sent_count = asyncio.run(_send_all())
    except Exception as e:
        logger.exception("digest dispatch failed: %s", e)
        return {"sent": False, "reason": str(e)[:200]}

    # Stamp metadata on the runs so we don't double-send
    stamp_ts = datetime.now(timezone.utc).isoformat()
    for r in runs:
        new_md = {**(r.get("metadata") or {}), "digest_sent_at": stamp_ts}
        sb.table("strategy_search_runs").update({"metadata": new_md}).eq("id", r["id"]).execute()

    logger.info("morning_digest sent to %d recipients", sent_count)
    return {"sent": True, "recipients": sent_count, "kinds": list(by_kind.keys())}
