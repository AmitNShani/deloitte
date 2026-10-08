"""Tool input validation and cache concurrency (no network)."""
import threading
import time

import pytest

from src import config
from src.agent.tools import ToolInputError, get_route_profile
from src.data import cache


def test_zero_long_haul_threshold_is_rejected_not_defaulted():
    with pytest.raises(ToolInputError, match="positive"):
        get_route_profile("ANC", long_haul_miles=0)


def test_concurrent_refreshes_of_one_key_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    errors = []

    def slow_fetch():
        time.sleep(0.05)  # make the writes overlap
        return {"ok": True}

    def worker():
        try:
            cache.cached("same_key", ttl=0, fetch=slow_fetch)
        except Exception as exc:  # noqa: BLE001 - the test records any failure
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert cache.cached("same_key", ttl=3600, fetch=lambda: pytest.fail("should be cached")).value == {"ok": True}
    assert not list(tmp_path.glob("*.tmp"))  # no temp files left behind


def test_failed_refresh_serves_stale_copy_and_flags_it(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    cache.cached("k", ttl=3600, fetch=lambda: {"v": 1})

    def offline():
        raise ConnectionError("no network")

    result = cache.cached("k", ttl=0, fetch=offline)
    assert result.value == {"v": 1}
    assert result.stale is True


# --- bundled snapshot (data/snapshot/) ---------------------------------------------------------

def _snapshot(tmp_path, monkeypatch, key, value, fetched_at=1_000.0):
    import gzip
    import json
    cache_dir, snap_dir = tmp_path / "cache", tmp_path / "snapshot"
    snap_dir.mkdir()
    with gzip.open(snap_dir / f"{key}.json.gz", "wt") as f:
        json.dump({"fetched_at": fetched_at, "value": value}, f)
    monkeypatch.setattr(config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(config, "SNAPSHOT_DIR", snap_dir)
    return cache_dir, snap_dir


def test_empty_cache_is_served_from_snapshot_without_fetching(tmp_path, monkeypatch):
    cache_dir, _ = _snapshot(tmp_path, monkeypatch, "ontime_v2_2026-04", {"SFO": 1})
    result = cache.cached("ontime_v2_2026-04", config.TTL_FOREVER, fetch=lambda: pytest.fail("no download"))
    assert result.value == {"SFO": 1}
    assert result.fetched_at == 1_000.0          # answers cite when the data was really fetched
    assert not cache_dir.exists()                # the snapshot is read-only; nothing copied


def test_expired_snapshot_entry_is_refreshed_into_runtime_cache(tmp_path, monkeypatch):
    cache_dir, snap_dir = _snapshot(tmp_path, monkeypatch, "t100_latest_month", [2026, 4])
    before = (snap_dir / "t100_latest_month.json.gz").read_bytes()
    result = cache.cached("t100_latest_month", ttl=60, fetch=lambda: [2026, 5])
    assert result.value == [2026, 5]
    assert (cache_dir / "t100_latest_month.json").exists()
    assert (snap_dir / "t100_latest_month.json.gz").read_bytes() == before  # snapshot untouched


def test_snapshot_can_be_disabled(tmp_path, monkeypatch):
    _snapshot(tmp_path, monkeypatch, "k", "from snapshot")
    monkeypatch.setattr(config, "SNAPSHOT_DIR", None)  # what fetch_data --from-apis does
    assert cache.cached("k", config.TTL_FOREVER, fetch=lambda: "from api").value == "from api"
