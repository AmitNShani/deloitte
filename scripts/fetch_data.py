"""Fetch all the public data the agent uses, and optionally rebuild the committed snapshot.

    python -m scripts.fetch_data               # make sure all data is available (uses the snapshot)
    python -m scripts.fetch_data --from-apis   # ignore the snapshot and download everything (~5 min)
    python -m scripts.fetch_data --from-apis --snapshot   # ...then rebuild data/snapshot/ from it

The repo ships a compressed snapshot (data/snapshot/), so a fresh clone works without downloading.
Use --from-apis to check the API path end to end, or to pick up newer BTS months for the snapshot.
--from-apis skips only the snapshot: anything already in the runtime cache (data/cache/, which only
ever holds API responses) is reused, so delete that folder first to re-download everything.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import time

from src import config
from src.data import cache, repository

LIVE_KEYS = {"faa_nasstatus"}  # changes by the minute; never worth shipping


def fetch() -> None:
    t0 = time.time()
    print("1/3 Airport list, BTS traffic data and 12 months of BTS on-time data "
          "(from the APIs, the on-time files are ~30 MB each; this is the slow part)...", flush=True)
    ds = repository.dataset()
    print(f"    {len(ds.airports)} US airports with traffic, window {ds.window_label} "
          f"({time.time() - t0:.0f}s)")

    t1 = time.time()
    print("2/3 BTS route data, used for long-haul questions...", flush=True)
    try:
        repository.route_profile("ANC")
        print(f"    done ({time.time() - t1:.0f}s)")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"    FAILED: {exc}\n    Everything else works; long-haul answers will be labeled approximate.")

    print("3/3 FAA live airport status...", flush=True)
    try:
        status = repository.live_status()
        print(f"    {len(status['events'])} active events (feed updated {status['feed_update_time']})")
    except OSError as exc:
        print(f"    unavailable ({type(exc).__name__}); live status is optional context")
    print(f"Done in {time.time() - t0:.0f}s.")


def write_snapshot() -> None:
    """Compress the current runtime cache (minus live data) into data/snapshot/."""
    entries = [p for p in sorted(config.CACHE_DIR.glob("*.json")) if p.stem not in LIVE_KEYS]
    if config.SNAPSHOT_DIR.exists():
        shutil.rmtree(config.SNAPSHOT_DIR)
    config.SNAPSHOT_DIR.mkdir(parents=True)
    total = 0
    for p in entries:
        out = cache.snapshot_path(p.stem)
        with p.open("rb") as src, gzip.open(out, "wb", compresslevel=9) as dst:
            shutil.copyfileobj(src, dst)
        total += out.stat().st_size
    print(f"Snapshot: {len(entries)} files, {total / 1e6:.1f} MB in {config.SNAPSHOT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-apis", action="store_true",
                        help="ignore the bundled snapshot and download from the public APIs")
    parser.add_argument("--snapshot", action="store_true",
                        help="after fetching, rebuild data/snapshot/ from the runtime cache")
    args = parser.parse_args()
    if args.snapshot and not args.from_apis:
        # Without --from-apis, entries served from the old snapshot never reach data/cache/, so the
        # new snapshot would be incomplete.
        parser.error("--snapshot needs --from-apis, so every entry is fetched into data/cache/ first")
    if args.from_apis:
        config.SNAPSHOT_DIR = None  # cache lookups skip the snapshot
    fetch()
    if args.snapshot:
        config.SNAPSHOT_DIR = config.ROOT / "data" / "snapshot"
        write_snapshot()


if __name__ == "__main__":
    main()
