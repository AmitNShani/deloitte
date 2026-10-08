"""BTS T-100 Segment, all carriers (TranStats table FMG), via its download form.

Route-level (origin -> destination) monthly departures, passengers, freight and distance for
domestic and international flights by US and foreign carriers. Used for route profiles such as the
long-haul share. TranStats only offers this table through an ASP.NET form, so we replay the form
POST. This is the most fragile source; callers must handle failure.
"""
from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

from src import config
from src.data import http, periods
from src.data.cache import CacheResult, cached

_FIELDS = ["DEPARTURES_PERFORMED", "SEATS", "PASSENGERS", "FREIGHT", "DISTANCE",
           "ORIGIN", "DEST", "DEST_COUNTRY", "CLASS", "YEAR", "MONTH"]
_GROUP = ["ORIGIN", "DEST", "DEST_COUNTRY", "CLASS", "DISTANCE", "YEAR", "MONTH"]


def _download_year(year: int) -> pd.DataFrame:
    s = http.session()
    page = s.get(config.SEGMENT_FORM_URL, timeout=config.HTTP_TIMEOUT)
    page.raise_for_status()
    form = dict(re.findall(r'type="hidden" name="([^"]+)" id="[^"]*" value="([^"]*)"', page.text))
    form.update({"cboGeography": "All", "cboYear": str(year), "cboPeriod": "All",
                 "chkDownloadZip": "on", "btnDownload": "Download"})
    form.update({f: "on" for f in _FIELDS})

    resp = s.post(config.SEGMENT_FORM_URL, data=form, timeout=600)
    resp.raise_for_status()
    if not resp.content.startswith(b"PK"):
        raise RuntimeError("TranStats form did not return a zip file; the form may have changed")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = next(n for n in z.namelist() if n.startswith("T_T100"))
        with z.open(csv_name) as f:
            return pd.read_csv(f, usecols=_FIELDS, keep_default_na=False, na_values=[""])


def _fetch_window(months: list[periods.Month]) -> list[dict]:
    frames = [_download_year(y) for y in sorted({m[0] for m in months})]
    df = pd.concat(frames)
    wanted = {m[0] * 100 + m[1] for m in months}
    df = df[(df["YEAR"] * 100 + df["MONTH"]).isin(wanted) & (df["DEPARTURES_PERFORMED"] > 0)]
    agg = (df.groupby(_GROUP, as_index=False, dropna=False)[["DEPARTURES_PERFORMED", "SEATS", "PASSENGERS", "FREIGHT"]]
           .sum())
    agg.columns = [c.lower() for c in agg.columns]
    return agg.to_dict(orient="records")


def route_rows(months: list[periods.Month]) -> CacheResult:
    """Route-level rows (summed across carriers) for the given months."""
    key = f"segment_{periods.label(months[0])}_{periods.label(months[-1])}"
    return cached(key, config.TTL_FOREVER, lambda: _fetch_window(months))
