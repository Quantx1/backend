"""Quant X services namespace — remaining unmoved subpackages.

Post-PR-A8: this package only holds subpackages that weren't migrated:
``assistant/``, ``autopilot/``, ``copilot/``, ``doctor/``, ``lab/``,
``regime/``, ``strategy_runner/``. Everything else now lives under
``ai/`` / ``trading/`` / ``data/`` / ``platform/``.

Do not add new modules here. New code that conceptually belongs in
``services/`` (i.e. an agent or orchestrator) should go under one of
the new top-level packages.
"""
