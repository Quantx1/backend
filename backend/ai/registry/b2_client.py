"""
Thin Backblaze B2 wrapper used by the model registry.

Design notes:
- B2 is the canonical store for ALL production model artifacts. A single
  Application Key scoped to one bucket authorizes every operation.
- We expose three primitives — upload_file, download_file, list_prefix —
  plus a small helper, exists. Anything higher-level (resolving a
  ``model_name`` to an artifact directory, caching downloads) lives in
  ``ModelRegistry``.
- ``b2sdk`` is imported lazily so unit tests that don't touch B2 can
  import this module without the dependency installed.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional

from backend.core.config import settings

logger = logging.getLogger(__name__)


class B2ConfigError(RuntimeError):
    """Raised when B2 credentials or bucket name are missing."""


class B2Client:
    """Wraps a single authorized B2 bucket.

    Instantiation is cheap; the underlying authorization happens lazily
    on first operation. Thread-safe for concurrent uploads/downloads.
    """

    _lock = threading.Lock()

    def __init__(
        self,
        key_id: Optional[str] = None,
        app_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ):
        self._key_id = key_id or settings.B2_APPLICATION_KEY_ID
        self._app_key = app_key or settings.B2_APPLICATION_KEY
        self._bucket_name = bucket_name or settings.B2_BUCKET_MODELS
        self._bucket = None
        self._api = None

        if not (self._key_id and self._app_key and self._bucket_name):
            raise B2ConfigError(
                "B2 credentials missing. Set B2_APPLICATION_KEY_ID, "
                "B2_APPLICATION_KEY, B2_BUCKET_MODELS in the environment."
            )

    # ------------------------------------------------------------------ auth

    def _ensure_bucket(self):
        if self._bucket is not None:
            return self._bucket
        with self._lock:
            if self._bucket is not None:
                return self._bucket
            from b2sdk.v2 import B2Api, InMemoryAccountInfo

            info = InMemoryAccountInfo()
            api = B2Api(info)
            api.authorize_account("production", self._key_id, self._app_key)
            self._api = api
            self._bucket = api.get_bucket_by_name(self._bucket_name)
            logger.info("B2 authorized bucket=%s", self._bucket_name)
            return self._bucket

    # -------------------------------------------------------------- upload

    def upload_file(self, local_path: Path, remote_key: str) -> str:
        """Upload ``local_path`` to ``remote_key`` inside the bucket.

        Returns the ``b2://<bucket>/<key>`` URI on success.
        """
        bucket = self._ensure_bucket()
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"B2 upload source missing: {local_path}")

        bucket.upload_local_file(
            local_file=str(local_path),
            file_name=remote_key,
        )
        uri = f"b2://{self._bucket_name}/{remote_key}"
        logger.info("B2 upload %s → %s (%d bytes)", local_path.name, uri, local_path.stat().st_size)
        return uri

    # ------------------------------------------------------------ download

    def download_file(self, remote_key: str, local_path: Path) -> Path:
        """Download ``remote_key`` from the bucket into ``local_path``.

        Creates parent directories as needed. Returns the local path.
        """
        bucket = self._ensure_bucket()
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded = bucket.download_file_by_name(remote_key)
        downloaded.save_to(str(local_path))
        logger.info("B2 download %s → %s", remote_key, local_path)
        return local_path

    # --------------------------------------------------------------- list

    def list_prefix(self, prefix: str) -> List[str]:
        """Return every file name under ``prefix`` (non-recursive)."""
        bucket = self._ensure_bucket()
        names: List[str] = []
        for file_version, _ in bucket.ls(folder_to_list=prefix, recursive=True):
            names.append(file_version.file_name)
        return names

    # ------------------------------------------------------------- exists

    def exists(self, remote_key: str) -> bool:
        try:
            bucket = self._ensure_bucket()
            bucket.get_file_info_by_name(remote_key)
            return True
        except Exception:
            return False

    @property
    def bucket_name(self) -> str:
        return self._bucket_name


# ---------------------------------------------------------------- singleton

_client_singleton: Optional[B2Client] = None
_singleton_lock = threading.Lock()


def get_b2_client() -> B2Client:
    """Return a lazily-initialized module-level B2Client."""
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton
    with _singleton_lock:
        if _client_singleton is None:
            _client_singleton = B2Client()
    return _client_singleton
