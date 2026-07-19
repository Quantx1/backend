"""Saved Scans + Alerts service (PR-S6).

A saved scan is a frozen screener configuration the user can re-run on
a schedule. When new symbols appear vs the previous run, an alert fires
and notifies the user via their selected channels (push/email/WhatsApp/
telegram).

Three pieces:
  * `runner.run_saved_scan(scan_row)` — executes one saved scan, returns
    matched symbols + diff vs last run
  * `cron.sweep_due_scans()` — picks all scans due for a re-run based
    on schedule + market hours, runs them, persists alerts
  * `notifications.dispatch_alert(alert_row)` — sends the alert through
    the user's selected channels (reuses existing push/email/whatsapp)
"""

from .runner import run_saved_scan, RunResult
from .cron import sweep_due_scans

__all__ = ["run_saved_scan", "RunResult", "sweep_due_scans"]
