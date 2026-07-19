"""
ModelRegistry — public API for loading, registering, and promoting
model artifacts.

Storage layout in B2::

    <bucket>/
        <model_name>/
            v<version>/
                <file_1>
                <file_2>
                ...

Each logical model can ship with multiple files (TFT is the canonical
multi-file case: ``tft_model.ckpt`` + ``tft_config.json`` +
``tft_config.pt``). ``resolve(model_name)`` returns the *local directory*
containing every file for the target version — callers find the primary
artifact by the well-known filename they expect.

Version metadata lives in Postgres (``public.model_versions``). B2 holds
bytes; Postgres holds the ``is_prod`` / ``is_shadow`` / metrics truth.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.config import settings

from . import versions as _versions
from .b2_client import B2Client, B2ConfigError, get_b2_client

logger = logging.getLogger(__name__)


def _remote_prefix(model_name: str, version: int) -> str:
    return f"{model_name}/v{version}/"


def _materialize_for_upload(src: Path) -> Path:
    """Return a single file path suitable for B2 upload.

    If ``src`` is already a file, return it unchanged. If ``src`` is a
    directory (NF.save / AutoGluon predictor / huggingface snapshot),
    tar.gz it into ``<src>.tar.gz`` next to the original and return that.
    Idempotent — won't re-tar if the .tar.gz already exists and is newer
    than the directory.
    """
    if src.is_file():
        return src
    if not src.exists():
        raise FileNotFoundError(f"artifact missing: {src}")
    tar_path = src.with_suffix(src.suffix + ".tar.gz") if src.suffix else src.parent / f"{src.name}.tar.gz"
    # Re-tar if missing or stale
    needs_tar = (
        not tar_path.exists()
        or tar_path.stat().st_mtime < src.stat().st_mtime
    )
    if needs_tar:
        logger.info("registry: tarballing dir %s -> %s", src, tar_path)
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(src, arcname=src.name)
    return tar_path


def _cache_dir(model_name: str, version: int) -> Path:
    base = Path(settings.MODEL_CACHE_DIR).expanduser().resolve()
    return base / model_name / f"v{version}"


class ModelRegistry:
    """Resolve artifacts, register new versions, promote/shadow/retire.

    ``resolve()`` is the hot-path read operation called by services at
    startup — it downloads to the local cache once and returns the path.
    All other methods are admin operations used by training scripts and
    the admin UI.
    """

    def __init__(self, b2: Optional[B2Client] = None):
        self._b2_explicit = b2

    @property
    def b2(self) -> B2Client:
        return self._b2_explicit if self._b2_explicit is not None else get_b2_client()

    # -------------------------------------------------------------- resolve

    def resolve(
        self,
        model_name: str,
        version: Optional[int] = None,
        *,
        shadow: bool = False,
    ) -> Path:
        """Return the local directory containing all files for a version.

        If ``version`` is None and ``shadow=False`` we resolve the
        current prod version. If ``shadow=True`` we resolve the latest
        shadow version. If ``version`` is given, we resolve that
        explicit version regardless of flags.

        Raises ``LookupError`` if the target version doesn't exist.
        Downloads any missing files from B2 to the local cache on first
        access.
        """
        row = self._pick_row(model_name, version=version, shadow=shadow)
        if row is None:
            target = (
                f"v{version}" if version is not None else ("shadow" if shadow else "prod")
            )
            raise LookupError(f"No model_versions row for {model_name} ({target})")

        v = int(row["version"])
        cache = _cache_dir(model_name, v)
        cache.mkdir(parents=True, exist_ok=True)
        prefix = _remote_prefix(model_name, v)

        remote_files = self.b2.list_prefix(prefix)
        if not remote_files:
            raise FileNotFoundError(
                f"B2 prefix {prefix} is empty for {model_name} v{v}"
            )

        for key in remote_files:
            filename = key.removeprefix(prefix)
            if not filename:
                continue
            local_path = cache / filename
            if local_path.exists() and local_path.stat().st_size > 0:
                continue
            self.b2.download_file(key, local_path)

        return cache

    def _pick_row(
        self,
        model_name: str,
        version: Optional[int],
        shadow: bool,
    ) -> Optional[Dict[str, Any]]:
        if version is not None:
            return _versions.get_version(model_name, version)
        if shadow:
            return _versions.get_shadow(model_name)
        return _versions.get_prod(model_name)

    # ------------------------------------------------------------- register

    def register(
        self,
        model_name: str,
        local_files: List[Path],
        *,
        version: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        trained_by: Optional[str] = None,
        git_sha: Optional[str] = None,
        notes: Optional[str] = None,
        is_prod: bool = False,
        is_shadow: bool = False,
    ) -> Dict[str, Any]:
        """Upload ``local_files`` to B2 and insert a ``model_versions`` row.

        ``version`` auto-increments if None. ``is_prod=True`` does NOT
        atomically demote the previous prod row — call ``promote()``
        after successful regression checks for that semantic.
        """
        if not local_files:
            raise ValueError("register() requires at least one file")

        v = version if version is not None else _versions.next_version(model_name)
        prefix = _remote_prefix(model_name, v)

        uploaded: List[str] = []
        for src in local_files:
            src = Path(src)
            # Auto-tarball directories (NF, AutoGluon, HF snapshots).
            # b2_client.upload_file only accepts single files.
            upload_src = _materialize_for_upload(src)
            remote_key = prefix + upload_src.name
            self.b2.upload_file(upload_src, remote_key)
            uploaded.append(remote_key)

        artifact_uri = f"b2://{self.b2.bucket_name}/{prefix}"
        row = _versions.insert_version(
            model_name=model_name,
            version=v,
            artifact_uri=artifact_uri,
            metrics=metrics,
            trained_by=trained_by,
            git_sha=git_sha,
            notes=notes,
            is_prod=is_prod,
            is_shadow=is_shadow,
        )
        logger.info(
            "Registered %s v%s with %d files at %s",
            model_name, v, len(uploaded), artifact_uri,
        )
        return row

    # -------------------------------------------------------------- promote

    def promote(self, model_name: str, version: int) -> Dict[str, Any]:
        """Atomically promote ``version`` to prod, demoting prior prod."""
        return _versions.promote_version(model_name, version)

    def shadow(self, model_name: str, version: int) -> Dict[str, Any]:
        return _versions.mark_shadow_version(model_name, version)

    def retire(self, model_name: str, version: int) -> Dict[str, Any]:
        return _versions.retire_version(model_name, version)

    def rollback(self, model_name: str, to_version: int) -> Dict[str, Any]:
        """Rollback = promote an older version back to prod."""
        row = _versions.get_version(model_name, to_version)
        if row is None:
            raise LookupError(f"Cannot rollback: {model_name} v{to_version} not found")
        if row.get("is_retired"):
            raise ValueError(
                f"Cannot rollback to retired version {model_name} v{to_version}"
            )
        return self.promote(model_name, to_version)

    # --------------------------------------------------------------- list

    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        return _versions.list_versions(model_name)

    def get_prod(self, model_name: str) -> Optional[Dict[str, Any]]:
        return _versions.get_prod(model_name)

    def get_shadow(self, model_name: str) -> Optional[Dict[str, Any]]:
        return _versions.get_shadow(model_name)


# ---------------------------------------------------------------- singleton

_registry_singleton: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Module-level singleton. B2 client is still lazy — no credentials
    are touched until the first upload/download."""
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = ModelRegistry()
    return _registry_singleton


__all__ = [
    "ModelRegistry",
    "get_registry",
    "B2ConfigError",
]
