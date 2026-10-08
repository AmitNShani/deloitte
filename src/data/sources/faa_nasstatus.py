"""FAA NAS Status: live airport closures, ground stops, ground delay programs and delays.

Context only, never a score input: it changes minute to minute and would make rankings
irreproducible.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from src import config
from src.data import http
from src.data.cache import CacheResult, cached


def _flatten(elem: ET.Element) -> dict:
    out = {}
    for child in elem:
        if len(child):
            for k, v in _flatten(child).items():
                out[f"{child.tag}.{k}"] = v
            out.update({f"{child.tag}@{k}": v for k, v in child.attrib.items()})
        elif child.tag != "ARPT":
            out[child.tag] = (child.text or "").strip()
    return out


def _fetch() -> dict:
    root = ET.fromstring(http.get(config.FAA_STATUS_URL).content)
    events = []
    for delay_type in root.findall("Delay_type"):
        kind = delay_type.findtext("Name", "").strip()
        # Each delay type nests its entries differently; any element with an ARPT child is one.
        for entry in delay_type.iter():
            airport = entry.findtext("ARPT")
            if airport is not None and entry.find("ARPT") is not None:
                events.append({"airport": airport.strip(), "type": kind, "details": _flatten(entry)})
    return {"update_time": root.findtext("Update_Time", "").strip(), "events": events}


def status() -> CacheResult:
    """All currently active FAA airport events, with the feed's update time."""
    return cached("faa_nasstatus", config.TTL_LIVE, _fetch)
