"""AutoPilot package — PR-M Supervisor + existing rebalancer.

The original ``services.autopilot_service.AutoPilotService`` is the
EXECUTION layer — supervised ML stack (Qlib + HMM + VIX + Kelly) that
emits trades at 15:50 IST. That stays as the trade-decisioning module.

This package adds the SUPERVISOR layer — a time-windowed orchestration
loop that wraps the executor with continuous monitoring + reporting
across all 24 hours. Indian markets only execute 9:15-15:30 IST, so
"24/7" here means continuous oversight, not continuous trading:

    Pre-market   (06:00-09:15 IST)  → prefetch data, regime check, signal prep
    Market open  (09:15-15:30 IST)  → position watchdog (every 5 min),
                                       SL/TP enforcement, exit alerts
    Post-market  (15:30-18:00 IST)  → daily P&L report, journal write,
                                       send digest, mark unfilled orders
    Overnight    (18:00-06:00 IST)  → regime model refresh trigger,
                                       signal cache invalidation, EOD scan

Honours the memory lock — every decision in the supervisor delegates
to AutoPilotService (supervised ML) or RiskManagementEngine (rules).
No LLM agent gates a trade.
"""

from .supervisor import (
    AutoPilotSupervisor,
    SupervisorWindow,
    WindowReport,
)

__all__ = ["AutoPilotSupervisor", "SupervisorWindow", "WindowReport"]
