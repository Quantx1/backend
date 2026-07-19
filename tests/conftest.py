"""Pytest root config — adds the project root to sys.path so test modules
can do ``from backend...`` without needing PYTHONPATH set externally.

This conftest was removed alongside the 137 legacy tests in PR-A2.1;
restored here as a minimal one-purpose shim. Keep it ≤20 lines.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
