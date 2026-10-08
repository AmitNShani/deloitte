"""Central configuration: data sources, cache lifetimes, regions, and scoring parameters."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"        # runtime cache (git-ignored); all writes go here
# Read-only compressed copy of the data, committed so a fresh clone starts without the ~5-minute
# download. Read only when the runtime cache has no entry. scripts/fetch_data.py rebuilds it.
SNAPSHOT_DIR = ROOT / "data" / "snapshot"

# --- Public data sources -------------------------------------------------------------------
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
T100_ORIGIN_URL = "https://data.bts.gov/resource/r495-tyji.json"  # Socrata: T-100 by origin airport
ONTIME_ZIP_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
# TranStats download form for table FMG: T-100 Segment (All Carriers), domestic + international
SEGMENT_FORM_URL = (
    "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4%20Pn44vr45"
)
FAA_STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"

HTTP_TIMEOUT = 60

# Local speech-to-text model for voice questions (faster-whisper): tiny.en, base.en, small.en, ...
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
USER_AGENT = "airport-investment-agent/0.1 (take-home prototype)"

# --- Cache lifetimes (seconds) -------------------------------------------------------------
TTL_FOREVER = None        # data for a fixed past month or window doesn't change once published
TTL_STATIC = 30 * 86400   # airport metadata
TTL_MONTHLY = 7 * 86400   # "what is the latest BTS month": new data arrives through this check
TTL_LIVE = 5 * 60         # FAA live status

# --- Analysis window -----------------------------------------------------------------------
TTM_MONTHS = 12  # trailing-twelve-month window, compared with the twelve months before it

# --- Regions (US Census divisions, plus a few common aliases) ------------------------------
CENSUS_DIVISIONS = {
    "new england": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "middle atlantic": ["NJ", "NY", "PA"],
    "east north central": ["IL", "IN", "MI", "OH", "WI"],
    "west north central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "south atlantic": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "east south central": ["AL", "KY", "MS", "TN"],
    "west south central": ["AR", "LA", "OK", "TX"],
    "mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "pacific": ["AK", "CA", "HI", "OR", "WA"],
}
REGION_ALIASES = {
    "west coast": ["CA", "OR", "WA"],
    "east coast": ["ME", "NH", "MA", "RI", "CT", "NY", "NJ", "DE", "MD", "VA", "NC", "SC", "GA", "FL"],
    "mid-atlantic": CENSUS_DIVISIONS["middle atlantic"],
    "midwest": CENSUS_DIVISIONS["east north central"] + CENSUS_DIVISIONS["west north central"],
    "southwest": ["AZ", "NM", "NV", "TX", "OK"],
}

# --- Scoring -------------------------------------------------------------------------------
WEIGHTS = {"congestion": 0.35, "utilization": 0.25, "growth": 0.25, "scale": 0.15}

GATE_MIN_PASSENGER_GROWTH_PCT = 0.0   # must be growing
GATE_MIN_LOAD_FACTOR_PCT = 70.0       # deliberately loose floor

REFERENCE_SET_SIZE = 75  # percentiles are computed against the top-N US airports by passengers
MISSING_COMPONENT_PERCENTILE = 0.5  # neutral assumption (the median) when a component has no data

UNMET_MIN_LOAD_FACTOR_PCT = 80.0
UNMET_MIN_CONGESTION_PCTL = 0.60
# Absolute floor on NAS delay per arriving flight. Most top-75 airports sit between 2 and 3 minutes,
# so the percentile alone flags airports that are barely above the median (the 60th percentile is
# ~2.7 min). 4.0 sits in the largest gap in the upper range (IAH 4.5 vs MIA 3.7) and keeps the
# clearly constrained hubs (EWR, LGA, BOS, ORD, DCA, SFO, JFK, DEN, FLL, MCO, IAH).
UNMET_MIN_NAS_DELAY_MIN = 4.0

LONG_HAUL_MILES = 3000
CARGO_SERVICE_CLASSES = {"G", "P"}  # T-100 service classes for all-cargo flights
PFC_PER_PASSENGER = 4.50            # Passenger Facility Charge cap, used for an illustrative figure
