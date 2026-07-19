"""
Compatibility shim — resolve a model's files to a local directory,
trying the registry first and falling back to the legacy ``artifacts/models/``
path when B2 credentials aren't configured (dev / CI).

This lets services migrate to ``registry.resolve()`` incrementally
without breaking local development where B2 isn't wired up.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from backend.core.config import settings

logger = logging.getLogger(__name__)


def resolve_model_dir(
    model_name: str,
    disk_fallback: Path,
    *,
    shadow: bool = False,
) -> Optional[Path]:
    """Return the directory containing the model's artifact files.

    Preference order:
        1. Registry (B2) if ``B2_APPLICATION_KEY_ID`` + ``B2_APPLICATION_KEY``
           are set and a ``model_versions`` row exists.
        2. ``disk_fallback`` — the historical ``artifacts/models/`` directory.
           Used when B2 isn't configured, or when the registry is empty
           (e.g. before running the PR 3 migration script).

    Returns ``None`` if neither source is available. Callers should
    treat ``None`` as "skip this model" and log appropriately.
    """
    use_registry = bool(
        settings.B2_APPLICATION_KEY_ID and settings.B2_APPLICATION_KEY
    )
    if use_registry:
        try:
            from . import get_registry  # local import — avoids cycles

            reg = get_registry()
            return reg.resolve(model_name, shadow=shadow)
        except LookupError:
            logger.info(
                "Registry has no %s row for %s — falling back to disk",
                "shadow" if shadow else "prod", model_name,
            )
        except Exception as e:
            logger.warning(
                "Registry resolve for %s failed (%s) — falling back to disk",
                model_name, e,
            )

    disk_fallback = Path(disk_fallback)
    if disk_fallback.exists() and disk_fallback.is_dir():
        return disk_fallback
    if disk_fallback.exists():
        return disk_fallback.parent
    return None


def resolve_model_file(
    model_name: str,
    filename: str,
    disk_fallback: Path,
    *,
    shadow: bool = False,
) -> Optional[Path]:
    """Like ``resolve_model_dir`` but returns the path to a specific
    file inside the resolved directory.

    ``disk_fallback`` should be the direct file path under the legacy
    ``artifacts/models/`` layout (e.g. ``ROOT/artifacts/models/regime_hmm.pkl``).
    """
    disk_fallback = Path(disk_fallback)
    fallback_dir = disk_fallback.parent if disk_fallback.is_file() or disk_fallback.suffix else disk_fallback
    folder = resolve_model_dir(model_name, fallback_dir, shadow=shadow)
    if folder is None:
        return None
    target = folder / filename
    if target.exists():
        return target
    # Last-resort: the disk fallback path (even if folder came back empty).
    if disk_fallback.exists():
        return disk_fallback
    return None
