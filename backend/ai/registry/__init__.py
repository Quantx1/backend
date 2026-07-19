"""
Model registry — B2 object store + Postgres ``model_versions`` metadata.

Public surface:
    from backend.ai.registry import get_registry, ModelRegistry

    reg = get_registry()
    model_dir = reg.resolve("regime_hmm")   # local Path, cached
    reg.register("tft_swing_v2", [ckpt_path, config_path], metrics={...})
    reg.promote("tft_swing_v2", 2)

See ``model_registry.py`` for the full API.
"""

from .b2_client import B2Client, B2ConfigError, get_b2_client
from .compat import resolve_model_dir, resolve_model_file
from .model_registry import ModelRegistry, get_registry

__all__ = [
    "B2Client",
    "B2ConfigError",
    "ModelRegistry",
    "get_b2_client",
    "get_registry",
    "resolve_model_dir",
    "resolve_model_file",
]
