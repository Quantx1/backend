"""F12 daily digest — morning brief + evening summary.

Builds a per-user message for both free Telegram delivery and Pro
WhatsApp delivery. Morning brief runs pre-market (7:30 IST); evening
summary runs post-close (17:30 IST). The generator returns a short
plain-text body that renders the same on both channels.
"""

from .generator import (
    DigestPayload,
    build_morning_brief,
    build_evening_summary,
)
from .delivery import (
    deliver_morning_all,
    deliver_evening_all,
)

__all__ = [
    "DigestPayload",
    "build_morning_brief",
    "build_evening_summary",
    "deliver_morning_all",
    "deliver_evening_all",
]
