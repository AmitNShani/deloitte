"""Small JSON disk cache for API results.

It exists for reliability and speed, not as a data source: every entry records when it was fetched
from the API, so answers can cite data freshness, and a failed refresh falls back to the stale copy
(flagged as stale) instead of failing the whole question.

Lookup order: runtime cache (data/cache/), then the committed read-only snapshot (data/snapshot/,
gzipped), then the API. Writes only ever go to the runtime cache.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src import config

log = logging.getLogger(__name__)


@dataclass
class CacheResult:
    value: Any
    fetched_at: float
    stale: bool = False  # True when a refresh failed and an expired copy was served


def _path(key: str):
    return config.CACHE_DIR / f"{key}.json"


def snapshot_path(key: str):
    return config.SNAPSHOT_DIR / f"{key}.json.gz"


def _read_entry(key: str) -> dict | None:
    path = _path(key)
    if path.exists():
        return json.loads(path.read_text())
    if config.SNAPSHOT_DIR is not None and snapshot_path(key).exists():
        with gzip.open(snapshot_path(key), "rt") as f:
            return json.load(f)
    return None


def cached(key: str, ttl: int | None, fetch: Callable[[], Any]) -> CacheResult:
    """Return the value for `key`, calling `fetch` when it's missing or older than `ttl` seconds.

    ttl=None means the entry never expires (data for a fixed past period).
    """
    path = _path(key)
    entry = _read_entry(key)
    if entry is not None and (ttl is None or time.time() - entry["fetched_at"] < ttl):
        return CacheResult(entry["value"], entry["fetched_at"])

    try:
        value = fetch()
    except Exception as exc:
        if entry is None:
            raise
        # One line, not a traceback: offline, every source falls back here at once.
        log.warning("refresh of %s failed (%s: %s); serving stale copy", key, type(exc).__name__, exc)
        log.debug("refresh failure detail for %s", key, exc_info=True)
        return CacheResult(entry["value"], entry["fetched_at"], stale=True)

    fetched_at = time.time()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Writer-specific temp file: concurrent refreshes of the same key (e.g. two browser sessions)
    # must not move each other's file. The final rename is atomic, so the last writer wins.
    tmp = path.with_suffix(f".{os.getpid()}-{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"fetched_at": fetched_at, "value": value}))
    tmp.replace(path)
    return CacheResult(value, fetched_at)
